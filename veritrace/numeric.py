"""Exact software semantics for a finite, canonical-zero binary32 profile.

Arithmetic never delegates rounding to the host floating-point unit.  Each
operation converts its operands to rational numbers, performs exact arithmetic,
then rounds once to IEEE 754 binary32 (nearest, ties to even).  Subnormals are
gradual.  NaN and infinity are outside the profile; overflow raises rather than
silently introducing infinity.  Every zero result is represented by +0.

The float conversion helpers are I/O adapters only.  Model and execution inputs
must contain canonical +0; ``validate_bits`` enforces that restriction.
"""

from fractions import Fraction
import math
import struct

SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
FRACTION_MASK = 0x007FFFFF


def validate_bits(bits: int, *, canonical_zero: bool = True) -> int:
    """Validate a finite unsigned 32-bit encoding, optionally allowing -0."""
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("binary32 bits must be a Python integer")
    if not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("binary32 encoding must fit an unsigned 32-bit integer")
    if bits & EXP_MASK == EXP_MASK:
        raise ValueError("NaN and infinity are outside the finite binary32 profile")
    if canonical_zero and bits == SIGN_MASK:
        raise ValueError("negative zero is outside the canonical +0 input profile")
    return bits


def bits_to_fraction(bits: int) -> Fraction:
    """Decode finite binary32 exactly; either signed zero becomes rational zero."""
    validate_bits(bits, canonical_zero=False)
    exponent = (bits >> 23) & 0xFF
    significand = bits & FRACTION_MASK
    if exponent:
        significand |= 1 << 23
        power = exponent - 150
    else:
        power = -149
    if bits & SIGN_MASK:
        significand = -significand
    if power >= 0:
        return Fraction(significand << power)
    return Fraction(significand, 1 << -power)


def _round_integer(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and quotient & 1):
        quotient += 1
    return quotient


def _round_scaled(numerator: int, denominator: int, shift: int) -> int:
    if shift >= 0:
        return _round_integer(numerator << shift, denominator)
    return _round_integer(numerator, denominator << -shift)


def fraction_to_bits(value: Fraction) -> int:
    """Round an exact rational to finite binary32, nearest with ties to even."""
    if not isinstance(value, (Fraction, int)) or isinstance(value, bool):
        raise TypeError("rounding requires a Fraction or integer, not a host float")
    value = Fraction(value)
    if not value:
        return 0
    sign = SIGN_MASK if value < 0 else 0
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    if exponent < -126:
        significand = _round_scaled(numerator, denominator, 149)
        # Rounding may reach the smallest normal (encoding 0x00800000).
        return (sign | significand) if significand else 0
    if exponent > 127:
        raise OverflowError("binary32 rounding overflow is outside the finite profile")
    significand = _round_scaled(numerator, denominator, 23 - exponent)
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        raise OverflowError("binary32 rounding overflow is outside the finite profile")
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def float_to_bits(value: float) -> int:
    """I/O adapter: round a finite Python float to binary32, canonicalizing zero."""
    if not math.isfinite(value):
        raise ValueError("NaN and infinity are outside the finite binary32 profile")
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    validate_bits(bits, canonical_zero=False)
    return 0 if bits == SIGN_MASK else bits


def bits_to_float(bits: int) -> float:
    """Decode for presentation only; verification uses bits_to_fraction."""
    validate_bits(bits, canonical_zero=False)
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def add(a: int, b: int) -> int:
    return fraction_to_bits(bits_to_fraction(a) + bits_to_fraction(b))


def mul(a: int, b: int) -> int:
    return fraction_to_bits(bits_to_fraction(a) * bits_to_fraction(b))


def relu(a: int) -> int:
    validate_bits(a, canonical_zero=False)
    return 0 if a & SIGN_MASK else a


def hex_bits(bits: int) -> str:
    validate_bits(bits, canonical_zero=False)
    return f"0x{bits:08x}"
