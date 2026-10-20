"""End-to-end experiment: model -> contract -> verification -> independent evidence."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .evidence import digest, environment, make_manifest, write_json
from .model import execute, load_onnx, real_execute, write_demo_onnx
from .native import c_source, run_native, run_onnx_runtime
from .numeric import bits_to_float, bits_to_fraction, fraction_to_bits, validate_bits
from .properties import load_vnnlib
from .verification import verify


PROPERTY_TEXT = """; Legacy VNN-COMP Real-variable subset: this formula describes UNSAFE outputs.
; All scalar inputs to the implementation must be finite canonical binary32.
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)
(assert (>= X_0 1))
(assert (<= X_0 1))
(assert (>= X_1 (/ 1 2)))
(assert (<= X_1 1))
(assert (>= X_2 1))
(assert (<= X_2 1))
(assert (< Y_0 (/ 1 2)))
"""


def execution_contract() -> dict:
    return {
        "schema": "veritrace-execution-contract-v1",
        "graph_profile": "One float32 batch-1 MatMul -> Add -> Relu; ONNX opset 13",
        "format": "IEEE-754 binary32 finite subset",
        "rounding": "round nearest, ties to even, after every multiply and add",
        "subnormals": "gradual underflow; no flush to zero",
        "zero": "Canonical +0 results; -0 inputs and constants rejected",
        "nonfinite": "NaN and infinity rejected; reachable intermediate overflow invalidates proof",
        "contraction": "Fused multiply-add forbidden",
        "canonical_reduction": "Start at +0; multiply then add in increasing input index order",
        "bias": "Add after the completed MatMul reduction, rounding once",
        "activation": "Relu: return value if positive; otherwise +0",
        "alternative": "cancellation_first uses index order [0,2,1]; a distinct execution contract",
    }


def make_witness(model, prop, result: dict) -> dict:
    if result["status"] != "sat" or result["semantics"] != "float32":
        raise ValueError("A portable binary32 witness requires a float32 SAT result")
    witness = {
        **result["witness"], "schema": "veritrace-witness-v1",
        "model_digest": model.digest, "property_digest": digest(prop.to_dict()),
        "contract_digest": digest(execution_contract()), "order": result["order"],
    }
    witness["independent_replay"] = replay_witness(model, prop, witness)
    return witness


def replay_witness(model, prop, witness: dict) -> dict:
    if not isinstance(witness, dict):
        raise ValueError("Witness must be a JSON object")
    if witness.get("schema") != "veritrace-witness-v1":
        raise ValueError("Unsupported witness schema")
    if witness.get("model_digest") != model.digest or witness.get("property_digest") != digest(prop.to_dict()):
        raise ValueError("Witness belongs to a different model or property")
    if witness.get("contract_digest") != digest(execution_contract()):
        raise ValueError("Witness execution contract differs")
    for field, width in (("input_bits", model.input_size), ("output_bits", model.output_size)):
        values = witness.get(field)
        if not isinstance(values, list) or len(values) != width:
            raise ValueError(f"Witness {field} must be a list of width {width}")
        for value in values:
            validate_bits(value)
    inputs = witness["input_bits"]
    if not prop.contains([bits_to_fraction(v) for v in inputs]):
        raise ValueError("Witness inputs are outside the property domain")
    result = execute(model, inputs, order=witness["order"])
    outputs = result["output_bits"]
    if outputs != witness["output_bits"]:
        raise ValueError("Witness outputs do not match independent execution")
    if not prop.is_unsafe([bits_to_fraction(v) for v in outputs]):
        raise ValueError("Witness does not violate the engineering requirement")
    return {"status": "validated", "method": "Exact-rational software binary32 replay", **result}


def run_demo(out: Path, *, native: bool = True, runtime: bool = True, timeout_ms: int = 10000) -> dict:
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("timeout_ms must be a positive integer")
    out.mkdir(parents=True, exist_ok=True)
    # Only these known generated files are replaced. In particular, an UNKNOWN
    # rerun must never leave an earlier SAT witness or complete report behind.
    generated = ["model.onnx", "model.json", "property.vnnlib", "execution-contract.json",
                 "witness.json", "report.json", "report.html", "manifest.json",
                 "native.c", "native-O0", "native-O2"]
    for label in ("real", "canonical", "cancellation_first"):
        generated.extend((f"{label}.smt2", f"{label}-overflow.smt2"))
    for name in generated:
        (out / name).unlink(missing_ok=True)
    write_demo_onnx(out / "model.onnx")
    (out / "property.vnnlib").write_text(PROPERTY_TEXT)
    model = load_onnx(out / "model.onnx")
    prop = load_vnnlib(out / "property.vnnlib", model.input_size, model.output_size)
    contract = execution_contract()
    write_json(out / "execution-contract.json", contract)
    write_json(out / "model.json", model.to_dict())

    checks = {}
    for label, semantics, order in (("real", "real", "canonical"),
                                     ("canonical", "float32", "canonical"),
                                     ("cancellation_first", "float32", "cancellation_first")):
        result = verify(model, prop, semantics=semantics, order=order, timeout_ms=timeout_ms)
        (out / f"{label}.smt2").write_text(result.pop("smt2", "; No property query available\n"))
        # A distinct overflow query is kept when the verifier supplies one.
        if "overflow_smt2" in result:
            (out / f"{label}-overflow.smt2").write_text(result.pop("overflow_smt2"))
        checks[label] = result

    witness = None
    if checks["canonical"]["status"] == "sat":
        witness = make_witness(model, prop, checks["canonical"])
        write_json(out / "witness.json", witness)

    inputs, samples = [], []
    for numerator in range(8, 17):
        x = Fraction(numerator, 16)
        row = [fraction_to_bits(Fraction(1)), fraction_to_bits(x), fraction_to_bits(Fraction(1))]
        inputs.append(row)
        canonical = execute(model, row, order="canonical")
        alternative = execute(model, row, order="cancellation_first")
        samples.append({
            "x": float(x), "x_exact": str(x), "inputs": row,
            "real": [float(v) for v in real_execute(model, [bits_to_fraction(v) for v in row])],
            "canonical": canonical, "cancellation_first": alternative,
        })

    backends = {f"software:{order}": [s[order]["output_bits"] for s in samples]
                for order in ("canonical", "cancellation_first")}
    (out / "native.c").write_text(c_source(model, inputs))
    native_result = run_native(model, inputs, out) if native else {"status": "skipped", "backends": {}}
    runtime_result = run_onnx_runtime(out / "model.onnx", inputs) if runtime else {"status": "skipped", "backends": {}}
    backends.update(native_result["backends"])
    backends.update(runtime_result["backends"])
    native_matches = None
    if native_result["status"] == "observed":
        native_matches = all(rows == backends["software:" + key.split(":")[-1]]
                             for key, rows in native_result["backends"].items())
    native_result["matches_software_on_samples"] = native_matches
    manifest = make_manifest(model.digest, contract, inputs, backends)
    write_json(out / "manifest.json", manifest)

    expected = {"real": "unsat", "canonical": "sat", "cancellation_first": "unsat"}
    complete = all(checks[k]["status"] == v for k, v in expected.items())
    complete = complete and native_result["status"] != "error" and runtime_result["status"] != "error"
    complete = complete and native_matches is not False
    report = {
        "schema": "veritrace-report-v1",
        "title": "When the implementation changes the answer",
        "status": "complete" if complete else "incomplete",
        "model_digest": model.digest, "contract_digest": digest(contract),
        "property": prop.to_dict(), "contract": contract,
        "verification": checks, "witness": witness, "samples": samples,
        "backend_bits": backends,
        "backends": {key: [[bits_to_float(v) for v in row] for row in rows] for key, rows in backends.items()},
        "native": native_result, "onnx_runtime": runtime_result,
        "environment": environment(), "manifest_digest": manifest["payload_sha256"],
        "limitations": [
            "A hand-built inference example, not a trained aerospace controller.",
            "The real-number and float32 verifiers prove claims only for the accepted graph and property subset.",
            "The alternative reduction is a different contract, not an automatic safe rewrite.",
            "Native and ONNX Runtime results are observations of nine vectors on one host.",
            "UNSAT relies on Z3 and this encoding; SMT-LIB files are replayable queries, not proof certificates.",
            "No SONNX compliance, VNN-COMP qualification, or aerospace certification is claimed.",
        ],
    }
    write_json(out / "report.json", report)
    from .report import render_report
    (out / "report.html").write_text(render_report(report))
    return report
