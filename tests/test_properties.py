from fractions import Fraction as F
import unittest

from veritrace.properties import Property, PropertyError, parse_vnnlib


DECLARATIONS = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"


class PropertyTests(unittest.TestCase):
    def test_comments_conjunction_reversed_bounds_and_exact_rationals(self):
        prop = parse_vnnlib(DECLARATIONS + """
            ; The inequalities describe the unsafe region.
            (assert (and (>= X_0 (/ 1 3)) (>= 1 X_0)))
            (assert (< Y_0 (/ 1 2)))
        """, 1, 1)
        self.assertEqual(prop.input_bounds, ((F(1, 3), F(1)),))
        self.assertEqual(prop.threshold, F(1, 2))
        self.assertTrue(prop.contains([F(1, 2)]))
        self.assertFalse(prop.contains([F(0)]))
        self.assertFalse(prop.contains([]))
        self.assertTrue(prop.is_unsafe([F(1, 3)]))
        self.assertFalse(prop.is_unsafe([F(1, 2)]))

    def test_equality_decimal_negative_and_reversed_output(self):
        prop = parse_vnnlib(DECLARATIONS + """
            (assert (= X_0 (- 0.125)))
            (assert (< (/ 1 3) Y_0))
        """, 1, 1)
        self.assertEqual(prop.input_bounds, ((F(-1, 8), F(-1, 8)),))
        self.assertEqual(prop.operator, ">")
        self.assertTrue(prop.is_unsafe([F(1)]))
        self.assertEqual(prop.to_dict()["unsafe_output"]["threshold"], "1/3")

    def test_repeated_bounds_are_intersected(self):
        prop = parse_vnnlib(DECLARATIONS + """
            (assert (>= X_0 0)) (assert (>= X_0 1))
            (assert (<= X_0 3)) (assert (<= X_0 2))
            (assert (> Y_0 0))
        """, 1, 1)
        self.assertEqual(prop.input_bounds, ((F(1), F(2)),))

    def test_unsupported_or_invalid_properties_fail_closed(self):
        suffix = "(assert (< Y_0 0))"
        examples = [
            "(assert (>= X_0 0))" + suffix,  # missing upper bound
            "(assert (>= X_0 1)) (assert (<= X_0 0))" + suffix,
            "(assert (and (>= X_0 0) (< X_0 1)))" + suffix,
            "(assert (or (>= X_0 0) (<= X_0 1)))" + suffix,
            "(assert (= (+ X_0 1) 0))" + suffix,
            "(assert (= X_0 (/ 1 0)))" + suffix,
            "(assert (= X_0 0)) (assert (<= Y_0 0))",
            "(assert (= X_0 0))" + suffix + suffix,
            "(assert (= X_0 0)) (assert (and))" + suffix,
            "(assert (= X_0 0)) (check-sat)" + suffix,
            "(assert (= X_0 0)) (assert (< Y_1 0))",
            "(declare-const X_1 Real) (assert (= X_0 0))" + suffix,
            "(declare-const X_0 Real) (assert (= X_0 0))" + suffix,
            "(assert (= X_0 __import__('os').system('touch /tmp/unwanted')))" + suffix,
            "(assert (= X_0 0))",
            "(assert (= X_0 0))" + suffix + ")",
            "(assert (= X_0 0))" + suffix + "(",
            "naked-token",
        ]
        for text in examples:
            with self.subTest(text=text):
                with self.assertRaises(PropertyError):
                    parse_vnnlib(DECLARATIONS + text, 1, 1)

    def test_missing_or_wrong_declarations_rejected(self):
        for declaration in ["", "(declare-const X_0 Int)", "(declare-fun X_0 () Real)"]:
            with self.subTest(declaration=declaration), self.assertRaises(PropertyError):
                parse_vnnlib(declaration + "(assert (= X_0 0)) (assert (< Y_0 0))", 1, 1)

    def test_programmatic_property_checks(self):
        with self.assertRaises(PropertyError):
            Property(((F(1), F(0)),), 0, "<", F(1))
        with self.assertRaises(PropertyError):
            Property(((F(0), F(1)),), 0, "<=", F(1))
        with self.assertRaises(PropertyError):
            Property(((F(0), F(1)),), -1, "<", F(1))


if __name__ == "__main__":
    unittest.main()
