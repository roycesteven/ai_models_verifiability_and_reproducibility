from fractions import Fraction as F
import operator
import unittest
from unittest.mock import patch

import z3

from veritrace.model import DenseReluModel, demo_model
from veritrace.numeric import bits_to_fraction, fraction_to_bits
from veritrace.properties import Property
from veritrace.verification import _fp, _fp_compare, verify


def box(lower, upper, op="<", threshold=F(1, 2)):
    return Property(((F(lower), F(upper)),), 0, op, F(threshold))


class VerificationTests(unittest.TestCase):
    def test_demo_real_safe_float_counterexample_and_fixed_order_safe(self):
        prop = Property(((F(1), F(1)), (F(1, 2), F(1)), (F(1), F(1))), 0, "<", F(1, 2))
        model = demo_model()
        real = verify(model, prop, semantics="real")
        rounded = verify(model, prop)
        reordered = verify(model, prop, order="cancellation_first")
        self.assertEqual(real["status"], "unsat", real)
        self.assertEqual(rounded["status"], "sat", rounded)
        self.assertEqual(reordered["status"], "unsat", reordered)
        self.assertTrue(rounded["witness"]["replay_validated"])
        self.assertEqual(rounded["witness"]["output_bits"], [0])
        self.assertEqual(rounded["overflow_check"], "unsat")
        self.assertFalse(reordered["proof_certificate"])
        self.assertIn("fp.mul", rounded["smt2"])
        self.assertIn("fp.add", rounded["smt2"])
        self.assertIn("roundNearestTiesToEven", rounded["smt2"])

    def test_reachable_overflow_is_unknown_even_when_relu_would_hide_it(self):
        model = DenseReluModel([[fraction_to_bits(-2)]], [0])
        maximum = bits_to_fraction(0x7F7FFFFF)
        result = verify(model, box(maximum, maximum, "<", 0))
        self.assertEqual(result["status"], "unknown", result)
        self.assertEqual(result["overflow_check"], "sat")
        self.assertEqual(result["overflow_witness"]["input_bits"], [0x7F7FFFFF])

    def test_nonrepresentable_exact_input_is_explicitly_vacuous(self):
        model = DenseReluModel([[fraction_to_bits(1)]], [0])
        result = verify(model, box(F(1, 3), F(1, 3)))
        self.assertEqual(result["status"], "unsat", result)
        self.assertTrue(result["vacuous"])
        real = verify(model, box(F(1, 3), F(1, 3)), semantics="real")
        self.assertEqual(real["status"], "sat", real)
        self.assertEqual(real["witness"]["input_values"], ["1/3"])
        self.assertTrue(real["witness"]["replay_validated"])

    def test_nonrepresentable_rational_threshold_is_exact(self):
        model = DenseReluModel([[fraction_to_bits(1)]], [0])
        rounded_third = bits_to_fraction(fraction_to_bits(F(1, 3)))
        self.assertGreater(rounded_third, F(1, 3))
        below = verify(model, box(rounded_third, rounded_third, "<", F(1, 3)))
        above = verify(model, box(rounded_third, rounded_third, ">", F(1, 3)))
        self.assertEqual(below["status"], "unsat", below)
        self.assertEqual(above["status"], "sat", above)

    def test_signed_zero_canonicalized_and_subnormal_preserved(self):
        model = DenseReluModel([[fraction_to_bits(-1)]], [0])
        zero = verify(model, box(0, 0, "<", 1))
        self.assertEqual(zero["status"], "sat", zero)
        self.assertEqual(zero["witness"]["input_bits"], [0])
        self.assertEqual(zero["witness"]["output_bits"], [0])
        tiny = bits_to_fraction(1)
        subnormal = verify(model, box(-tiny, -tiny, ">", 0))
        self.assertEqual(subnormal["status"], "sat", subnormal)
        self.assertEqual(subnormal["witness"]["output_bits"], [1])

    def test_multioutput_index_and_real_replay(self):
        model = DenseReluModel([[fraction_to_bits(1), fraction_to_bits(2)]], [0, fraction_to_bits(1)])
        prop = Property(((F(1), F(1)),), 1, ">", F(2))
        for semantics in ("float32", "real"):
            with self.subTest(semantics=semantics):
                result = verify(model, prop, semantics=semantics)
                self.assertEqual(result["status"], "sat", result)
                self.assertEqual(result["witness"]["output_values"], ["1", "3"])

    def test_replay_mismatch_does_not_claim_counterexample(self):
        model = DenseReluModel([[fraction_to_bits(1)]], [0])
        with patch("veritrace.verification.execute", return_value={"output_bits": [fraction_to_bits(1)]}):
            result = verify(model, box(0, 0))
        self.assertEqual(result["status"], "unknown", result)
        self.assertIn("failed independent", result["reason"])
        self.assertNotIn("witness", result)

    def test_solver_unknown_propagates_without_safety_claim(self):
        model = DenseReluModel([[fraction_to_bits(1)]], [0])
        with patch.object(z3.Solver, "check", return_value=z3.unknown), patch.object(z3.Solver, "reason_unknown", return_value="test timeout"):
            result = verify(model, box(0, 1))
        self.assertEqual(result["status"], "unknown", result)
        self.assertIn("timeout", result["reason"])

    def test_rational_to_float_comparison_boundaries(self):
        bits = [0, 1, 0x80000001, 0x3F000000, 0xBF000000,
                fraction_to_bits(F(1, 3)), fraction_to_bits(F(-1, 3)), 0x7F7FFFFF, 0xFF7FFFFF]
        maximum = bits_to_fraction(0x7F7FFFFF)
        thresholds = [F(0), F(1, 3), F(-1, 3), F(1, 2), F(-1, 2),
                      F(1, 2**150), F(-1, 2**150), maximum, -maximum, maximum * 2, -maximum * 2]
        operations = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}
        for encoding in bits:
            value = bits_to_fraction(encoding)
            for threshold in thresholds:
                for symbol, compare in operations.items():
                    with self.subTest(bits=encoding, threshold=threshold, comparison=symbol):
                        actual = z3.simplify(_fp_compare(_fp(encoding), symbol, threshold))
                        self.assertEqual(z3.is_true(actual), compare(value, threshold))

    def test_invalid_options_raise(self):
        model = DenseReluModel([[fraction_to_bits(1)]], [0])
        for kwargs in ({"semantics": "float64"}, {"order": "random"}, {"order": "cancellation_first"}, {"timeout_ms": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                verify(model, box(0, 1), **kwargs)


if __name__ == "__main__":
    unittest.main()
