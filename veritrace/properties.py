"""A deliberately small, fail-closed VNN-LIB property reader.

Accepted properties describe a closed input box and one strict unsafe output
inequality. This is a subset of VNN-LIB, not a general SMT-LIB interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re


class PropertyError(ValueError):
    """The requested property is outside the supported, bounded fragment."""


@dataclass(frozen=True)
class Property:
    input_bounds: tuple[tuple[Fraction, Fraction], ...]
    output_index: int
    operator: str
    threshold: Fraction

    def __post_init__(self) -> None:
        try:
            bounds = tuple((Fraction(lo), Fraction(hi)) for lo, hi in self.input_bounds)
            threshold = Fraction(self.threshold)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise PropertyError("Bounds and threshold must be finite rational numbers") from exc
        if not bounds:
            raise PropertyError("At least one bounded input is required")
        if any(lo > hi for lo, hi in bounds):
            raise PropertyError("Input bounds are inconsistent")
        if type(self.output_index) is not int or self.output_index < 0:
            raise PropertyError("Output index must be a nonnegative integer")
        if self.operator not in ("<", ">"):
            raise PropertyError("The unsafe output comparison must be strict: < or >")
        object.__setattr__(self, "input_bounds", bounds)
        object.__setattr__(self, "threshold", threshold)

    def contains(self, inputs) -> bool:
        return len(inputs) == len(self.input_bounds) and all(
            lo <= value <= hi for value, (lo, hi) in zip(inputs, self.input_bounds)
        )

    def is_unsafe(self, outputs) -> bool:
        """Evaluate the unsafe predicate on an output sequence or a scalar."""
        value = outputs[self.output_index] if isinstance(outputs, (list, tuple)) else outputs
        return value < self.threshold if self.operator == "<" else value > self.threshold

    def to_dict(self) -> dict:
        return {
            "input_bounds": [
                {"lower": str(lo), "upper": str(hi)} for lo, hi in self.input_bounds
            ],
            "unsafe_output": {
                "index": self.output_index,
                "operator": self.operator,
                "threshold": str(self.threshold),
            },
        }


_NUMERAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
_VARIABLE = re.compile(r"([XY])_(0|[1-9][0-9]*)\Z")


def _parse(text: str) -> list:
    if len(text) > 100_000:
        raise PropertyError("Property exceeds the 100,000-character limit")
    text = re.sub(r";[^\n]*", "", text)
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    if len(tokens) > 25_000:
        raise PropertyError("Property contains too many tokens")
    result: list = []
    stack = [result]
    for token in tokens:
        if token == "(":
            if len(stack) >= 64:
                raise PropertyError("Property nesting is too deep")
            child: list = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise PropertyError("Unmatched closing parenthesis")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise PropertyError("Unclosed parenthesis")
    if any(not isinstance(command, list) for command in result):
        raise PropertyError("Expected parenthesized SMT-LIB commands")
    return result


def _rational(expr) -> Fraction:
    if isinstance(expr, str) and _NUMERAL.fullmatch(expr):
        if len(expr) > 256:
            raise PropertyError("Numeral exceeds the 256-character limit")
        return Fraction(expr)
    if isinstance(expr, list) and len(expr) == 2 and expr[0] == "-":
        return -_rational(expr[1])
    if isinstance(expr, list) and len(expr) == 3 and expr[0] == "/":
        numerator, denominator = _rational(expr[1]), _rational(expr[2])
        if denominator == 0:
            raise PropertyError("Rational denominator must be nonzero")
        return numerator / denominator
    raise PropertyError(f"Expected a rational constant, found {expr!r}")


def _conjuncts(expr):
    if isinstance(expr, list) and expr and expr[0] == "and":
        if len(expr) < 2:
            raise PropertyError("Empty conjunction is unsupported")
        for child in expr[1:]:
            yield from _conjuncts(child)
    else:
        yield expr


def parse_vnnlib(text: str, input_size: int, output_size: int) -> Property:
    """Read declare-const, closed X bounds, and a single strict Y inequality.

    Constants can be decimal integers, decimals, unary minus, or rational (/ p q).
    Conjunctions are accepted; disjunctions and arbitrary arithmetic are rejected.
    All X_i and Y_i must be declared Real exactly once, before their use.
    """
    if type(input_size) is not int or type(output_size) is not int or min(input_size, output_size) < 1:
        raise PropertyError("Model dimensions must be positive integers")
    expected = {f"X_{i}" for i in range(input_size)} | {f"Y_{i}" for i in range(output_size)}
    declared: set[str] = set()
    bounds: list[list[Fraction | None]] = [[None, None] for _ in range(input_size)]
    unsafe = None
    for command in _parse(text):
        if not command:
            raise PropertyError("Empty SMT-LIB command")
        if command[0] == "declare-const":
            if len(command) != 3 or not isinstance(command[1], str) or command[2] != "Real":
                raise PropertyError("Only (declare-const X_i/Y_i Real) is supported")
            name = command[1]
            if name not in expected or name in declared:
                raise PropertyError(f"Unexpected or duplicate declaration: {name}")
            declared.add(name)
            continue
        if command[0] != "assert" or len(command) != 2:
            raise PropertyError(f"Unsupported command: {command[0]!r}")
        for expr in _conjuncts(command[1]):
            if not isinstance(expr, list) or len(expr) != 3 or expr[0] not in ("<=", ">=", "=", "<", ">"):
                raise PropertyError("Expected a supported comparison against a rational constant")
            op, left, right = expr
            if isinstance(left, str) and _VARIABLE.fullmatch(left):
                name, value = left, _rational(right)
            elif isinstance(right, str) and _VARIABLE.fullmatch(right):
                name, value = right, _rational(left)
                op = {"<=": ">=", ">=": "<=", "<": ">", ">": "<", "=": "="}[op]
            else:
                raise PropertyError("Each comparison must contain one declared X_i or Y_i")
            if name not in declared:
                raise PropertyError(f"Undeclared variable: {name}")
            index = int(name[2:])
            if name.startswith("X_"):
                if op not in ("<=", ">=", "="):
                    raise PropertyError("Input bounds must be closed (<=, >=, or =)")
                if op in (">=", "="):
                    bounds[index][0] = value if bounds[index][0] is None else max(bounds[index][0], value)
                if op in ("<=", "="):
                    bounds[index][1] = value if bounds[index][1] is None else min(bounds[index][1], value)
            else:
                if op not in ("<", ">") or unsafe is not None:
                    raise PropertyError("Exactly one strict unsafe output inequality is supported")
                unsafe = (index, op, value)
    if declared != expected:
        raise PropertyError(f"Missing declarations: {', '.join(sorted(expected - declared))}")
    if any(lo is None or hi is None for lo, hi in bounds):
        raise PropertyError("Every input must have a finite lower and upper bound")
    if unsafe is None:
        raise PropertyError("Missing unsafe output inequality")
    return Property(tuple((lo, hi) for lo, hi in bounds), *unsafe)


def load_vnnlib(path: str | Path, input_size: int, output_size: int) -> Property:
    with Path(path).open(encoding="utf-8") as stream:
        text = stream.read(100_001)
    return parse_vnnlib(text, input_size, output_size)
