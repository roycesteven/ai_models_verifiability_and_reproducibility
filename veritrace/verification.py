"""SMT verification of the explicit arithmetic profile and a real baseline.

UNSAT trusts Z3 and this encoding; an SMT-LIB query is not a proof certificate.
SAT counterexamples are independently replayed by the rational software engine.
The floating-point claim covers finite binary32 inputs with canonical +0 only.
"""

from __future__ import annotations

from fractions import Fraction
import time

import z3

from .model import execute, real_execute
from .numeric import bits_to_fraction, fraction_to_bits
from .properties import Property


def _real(value: Fraction):
    return z3.RealVal(f"{value.numerator}/{value.denominator}")


def _fp(bits: int):
    return z3.fpBVToFP(z3.BitVecVal(bits, 32), z3.Float32())


def _finite(expr):
    return z3.And(z3.Not(z3.fpIsNaN(expr)), z3.Not(z3.fpIsInf(expr)))


def _canonical_zero(expr):
    return z3.If(z3.fpIsZero(expr), _fp(0), expr)


def _fp_compare(expr, operator: str, threshold: Fraction):
    """Compare a finite FP term to an exact rational, without fp.to_real.

    After RNE rounding, the threshold is either exact or adjacent to the true
    boundary. Adjusting strictness makes the comparison exact on binary32.
    This also handles rational bounds with no representable point between them.
    """
    maximum = bits_to_fraction(0x7F7FFFFF)
    if threshold > maximum:
        return z3.BoolVal(operator in ("<", "<="))
    if threshold < -maximum:
        return z3.BoolVal(operator in (">", ">="))
    rounded_bits = fraction_to_bits(threshold)
    rounded = bits_to_fraction(rounded_bits)
    boundary = _fp(rounded_bits)
    if operator == "<":
        return z3.fpLEQ(expr, boundary) if rounded < threshold else z3.fpLT(expr, boundary)
    if operator == ">":
        return z3.fpGEQ(expr, boundary) if rounded > threshold else z3.fpGT(expr, boundary)
    if operator == "<=":
        return z3.fpLT(expr, boundary) if rounded > threshold else z3.fpLEQ(expr, boundary)
    if operator == ">=":
        return z3.fpGT(expr, boundary) if rounded < threshold else z3.fpGEQ(expr, boundary)
    raise ValueError(f"Unsupported comparison: {operator}")


def _float_graph(model, inputs, indices):
    outputs, intermediate = [], []
    for column in range(model.output_size):
        total = _fp(0)
        for index in indices:
            product = _canonical_zero(z3.fpMul(z3.RNE(), inputs[index], _fp(model.weights[index][column])))
            intermediate.append(product)
            total = _canonical_zero(z3.fpAdd(z3.RNE(), total, product))
            intermediate.append(total)
        total = _canonical_zero(z3.fpAdd(z3.RNE(), total, _fp(model.bias[column])))
        intermediate.append(total)
        output = z3.If(z3.fpLEQ(total, _fp(0)), _fp(0), total)
        outputs.append(output)
    return outputs, intermediate


def _real_graph(model, inputs, indices):
    outputs = []
    for column in range(model.output_size):
        total = sum((inputs[i] * _real(bits_to_fraction(model.weights[i][column])) for i in indices), z3.RealVal(0))
        total = total + _real(bits_to_fraction(model.bias[column]))
        outputs.append(z3.If(total <= 0, z3.RealVal(0), total))
    return outputs


def _rational_value(value) -> Fraction:
    if not z3.is_rational_value(value):
        raise ValueError("Solver returned a non-rational value for linear real arithmetic")
    return Fraction(value.numerator_as_long(), value.denominator_as_long())


def verify(model, prop: Property, semantics: str = "float32", order: str = "canonical", timeout_ms: int = 10_000) -> dict:
    """Check existence of an unsafe input; SAT means a replayed counterexample.

    ``real`` quantifies exact real inputs over the box. ``float32`` quantifies
    only finite binary32 encodings and checks that *every intermediate* remains
    finite throughout that box before returning a property result. Reachable
    overflow or solver incompleteness returns UNKNOWN, never a safety claim.
    ``timeout_ms`` is an overall budget shared by all solver checks.
    """
    if semantics not in ("float32", "real"):
        raise ValueError("semantics must be 'float32' or 'real'")
    if order not in ("canonical", "cancellation_first"):
        raise ValueError("Unsupported reduction order")
    if order == "cancellation_first" and model.input_size != 3:
        raise ValueError("cancellation_first is defined only for three inputs")
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("timeout_ms must be a positive integer")
    if len(prop.input_bounds) != model.input_size or prop.output_index >= model.output_size:
        raise ValueError("Property dimensions do not match the model")
    indices = list(range(model.input_size)) if order == "canonical" else [0, 2, 1]
    started = time.monotonic()
    deadline = started + timeout_ms / 1000
    result = {
        "status": "unknown",
        "semantics": semantics,
        "order": order,
        "model_digest": model.digest,
        "solver": f"Z3 {z3.get_version_string()}",
        "proof_certificate": False,
        "input_domain_status": "unknown",
        "overflow_check": "not_applicable" if semantics == "real" else "not_checked",
    }

    def finish(**fields):
        result.update(fields)
        result["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return result

    def check(solver):
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            return z3.unknown, "overall timeout exhausted"
        solver.set(timeout=max(1, remaining))
        answer = solver.check()
        return answer, solver.reason_unknown() if answer == z3.unknown else None

    try:
        if semantics == "float32":
            inputs = [z3.FP(f"X_{i}", z3.Float32()) for i in range(model.input_size)]
            constraints = []
            for value, (lower, upper) in zip(inputs, prop.input_bounds):
                constraints.extend([
                    _finite(value),
                    z3.Not(z3.And(z3.fpIsZero(value), z3.fpIsNegative(value))),
                    _fp_compare(value, ">=", lower),
                    _fp_compare(value, "<=", upper),
                ])
            outputs, intermediate = _float_graph(model, inputs, indices)
            unsafe = _fp_compare(outputs[prop.output_index], prop.operator, prop.threshold)
        else:
            inputs = [z3.Real(f"X_{i}") for i in range(model.input_size)]
            constraints = [constraint for value, (lower, upper) in zip(inputs, prop.input_bounds)
                           for constraint in (value >= _real(lower), value <= _real(upper))]
            outputs = _real_graph(model, inputs, indices)
            selected = outputs[prop.output_index]
            unsafe = selected < _real(prop.threshold) if prop.operator == "<" else selected > _real(prop.threshold)

        solver = z3.Solver()
        solver.add(*constraints)
        # The retained property query contains the exact input domain and model.
        property_solver = z3.Solver()
        property_solver.add(*constraints, unsafe)
        result["smt2"] = property_solver.to_smt2()
        domain_status, reason = check(solver)
        result["input_domain_status"] = str(domain_status)
        if domain_status == z3.unknown:
            return finish(reason=f"input domain check: {reason}")
        if domain_status == z3.unsat:
            return finish(status="unsat", vacuous=True, reason="No representable inputs satisfy the property box")

        if semantics == "float32":
            solver.push()
            solver.add(z3.Or(*[z3.Not(_finite(term)) for term in intermediate]))
            result["overflow_smt2"] = solver.to_smt2()
            overflow_status, reason = check(solver)
            result["overflow_check"] = str(overflow_status)
            if overflow_status == z3.unknown:
                return finish(reason=f"finite-intermediate check: {reason}")
            if overflow_status == z3.sat:
                assignment = solver.model()
                overflow_inputs = [assignment.eval(z3.fpToIEEEBV(value), model_completion=True).as_long() for value in inputs]
                return finish(reason="A nonfinite intermediate is reachable; execution leaves the supported profile",
                              overflow_witness={"input_bits": overflow_inputs})
            solver.pop()

        solver.add(unsafe)
        status, reason = check(solver)
        if status == z3.unknown:
            return finish(reason=f"unsafe-output check: {reason}")
        if status == z3.unsat:
            return finish(status="unsat", vacuous=False)
        assignment = solver.model()
        if semantics == "float32":
            input_bits = [assignment.eval(z3.fpToIEEEBV(value), model_completion=True).as_long() for value in inputs]
            solver_output_bits = [assignment.eval(z3.fpToIEEEBV(value), model_completion=True).as_long() for value in outputs]
            replay = execute(model, input_bits, order=order)
            output_bits = replay["output_bits"]
            input_values = [bits_to_fraction(value) for value in input_bits]
            output_values = [bits_to_fraction(value) for value in output_bits]
            if output_bits != solver_output_bits or not prop.contains(input_values) or not prop.is_unsafe(output_values):
                return finish(reason="Solver witness failed independent software replay")
            witness = {
                "input_bits": input_bits,
                "output_bits": output_bits,
                "input_values": [str(value) for value in input_values],
                "output_values": [str(value) for value in output_values],
                "replay_validated": True,
            }
        else:
            input_values = [_rational_value(assignment.eval(value, model_completion=True)) for value in inputs]
            solver_outputs = [_rational_value(assignment.eval(value, model_completion=True)) for value in outputs]
            output_values = real_execute(model, input_values)
            if output_values != solver_outputs or not prop.contains(input_values) or not prop.is_unsafe(output_values):
                return finish(reason="Real-arithmetic witness failed independent rational replay")
            witness = {
                "input_values": [str(value) for value in input_values],
                "output_values": [str(value) for value in output_values],
                "replay_validated": True,
            }
        return finish(status="sat", vacuous=False, witness=witness)
    except (z3.Z3Exception, OverflowError, ArithmeticError) as exc:
        return finish(reason=f"Verification error: {type(exc).__name__}: {exc}")
