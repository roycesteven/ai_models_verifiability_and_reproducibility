"""Boundary checks for independently rounded finite binary32 operations."""

from fractions import Fraction
import math
import random
import unittest

from veritrace.numeric import (add, bits_to_float, bits_to_fraction, float_to_bits,
                              fraction_to_bits, hex_bits, mul, relu, validate_bits)


class NumericTests(unittest.TestCase):
    def test_exact_decode(self):
        self.assertEqual(bits_to_fraction(0x00000001), Fraction(1, 2**149))
        self.assertEqual(bits_to_fraction(0x00800000), Fraction(1, 2**126))
        self.assertEqual(bits_to_fraction(0xBF800000), -1)
        self.assertEqual(bits_to_fraction(0x7F7FFFFF), (2**24 - 1) * 2**104)

    def test_roundtrips_cover_exponent_and_sign_classes(self):
        generator = random.Random(782)
        encodings = [sign | exponent << 23 | mantissa
                     for sign in (0, 0x80000000)
                     for exponent in range(255)
                     for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFF)]
        encodings.extend(generator.randrange(0x7F800000) for _ in range(1024))
        for bits in encodings:
            expected = 0 if bits == 0x80000000 else bits
            self.assertEqual(fraction_to_bits(bits_to_fraction(bits)), expected, hex(bits))

    def test_normal_halfway_ties_to_even(self):
        self.assertEqual(fraction_to_bits(Fraction(1) + Fraction(1, 2**24)), 0x3F800000)
        self.assertEqual(fraction_to_bits(Fraction(1) + Fraction(3, 2**24)), 0x3F800002)
        self.assertEqual(fraction_to_bits(-Fraction(1) - Fraction(3, 2**24)), 0xBF800002)
        midpoint = Fraction(1) + Fraction(1, 2**24)
        self.assertEqual(fraction_to_bits(midpoint + Fraction(1, 2**200)), 0x3F800001)
        self.assertEqual(fraction_to_bits(midpoint - Fraction(1, 2**200)), 0x3F800000)

    def test_subnormal_halfway_and_normal_boundary(self):
        self.assertEqual(fraction_to_bits(Fraction(1, 2**150)), 0)
        self.assertEqual(fraction_to_bits(Fraction(-1, 2**150)), 0)
        self.assertEqual(fraction_to_bits(Fraction(3, 2**150)), 2)
        self.assertEqual(fraction_to_bits(Fraction(5, 2**150)), 2)
        boundary = Fraction(1, 2**126) - Fraction(1, 2**150)
        self.assertEqual(fraction_to_bits(boundary), 0x00800000)
        self.assertEqual(fraction_to_bits(boundary - Fraction(1, 2**200)), 0x007FFFFF)

    def test_rounding_carry_between_binades(self):
        self.assertEqual(fraction_to_bits(Fraction(2) - Fraction(1, 2**24)), 0x40000000)

    def test_cancellation_separate_rounding_and_gradual_underflow(self):
        one, large = float_to_bits(1), float_to_bits(2**24)
        self.assertEqual(add(large, one), large)
        self.assertEqual(add(large, float_to_bits(-(2**24))), 0)
        self.assertEqual(mul(1, float_to_bits(1)), 1)
        self.assertEqual(mul(3, float_to_bits(0.5)), 2)
        self.assertEqual(relu(float_to_bits(-1)), 0)
        self.assertEqual(relu(1), 1)
        self.assertEqual(relu(0x80000000), 0)

    def test_rejects_nonfinite_and_overflow(self):
        for bits in (0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001):
            with self.assertRaises(ValueError):
                bits_to_fraction(bits)
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                float_to_bits(value)
        maximum = bits_to_fraction(0x7F7FFFFF)
        self.assertEqual(fraction_to_bits(maximum + 2**103 - 1), 0x7F7FFFFF)
        for value in (maximum + 2**103, -(maximum + 2**103), Fraction(2**128)):
            with self.assertRaises(OverflowError):
                fraction_to_bits(value)
        with self.assertRaises(OverflowError):
            add(0x7F7FFFFF, 0x7F7FFFFF)

    def test_profile_input_validation_and_presentation(self):
        for value in (-1, 2**32, 0x80000000):
            with self.assertRaises(ValueError):
                validate_bits(value)
        for value in (True, 1.0, "0x3f800000"):
            with self.assertRaises(TypeError):
                validate_bits(value)
        with self.assertRaises(TypeError):
            fraction_to_bits(0.5)
        self.assertEqual(float_to_bits(-0.0), 0)
        self.assertEqual(bits_to_float(0x3FC00000), 1.5)
        self.assertEqual(hex_bits(1), "0x00000001")


if __name__ == "__main__":
    unittest.main()
