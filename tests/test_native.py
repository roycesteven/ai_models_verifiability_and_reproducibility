from fractions import Fraction
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from veritrace.model import DenseReluModel, demo_model, execute
from veritrace.native import c_source, run_native
from veritrace.numeric import float_to_bits, fraction_to_bits


class NativeTests(unittest.TestCase):
    def test_missing_compiler_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("veritrace.native.shutil.which", return_value=None):
                result = run_native(demo_model(), [[float_to_bits(1)] * 3], Path(directory))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["backends"], {})

    @unittest.skipUnless(any(shutil.which(name) for name in ("cc", "gcc", "clang")), "C compiler is not installed")
    def test_c_both_optimizations_agree_with_software_bit_patterns(self):
        one = float_to_bits(1)
        models_and_inputs = [
            (demo_model(), [[one, fraction_to_bits(Fraction(i, 16)), one] for i in range(8, 17)] + [[0, 1, 0]]),
            (DenseReluModel([[float_to_bits(0.5)], [0], [0]], [0]), [[1, 0, 0], [3, 0, 0], [0x80000001, 0, 0]]),
            (DenseReluModel([[float_to_bits(1 + 2**-23)], [0], [0]], [float_to_bits(-(1 + 2**-22))]),
             [[float_to_bits(1 + 2**-23), 0, 0]]),
        ]
        for model, inputs in models_and_inputs:
            with self.subTest(model=model.to_dict()):
                with tempfile.TemporaryDirectory() as directory:
                    result = run_native(model, inputs, Path(directory))
                    self.assertEqual(result["status"], "observed", result)
                    self.assertEqual(len(result["backends"]), 4)
                    for name, outputs in result["backends"].items():
                        order = name.rsplit(":", 1)[-1]
                        expected = [execute(model, row, order=order)["output_bits"] for row in inputs]
                        self.assertEqual(outputs, expected, name)
                    for command in result["commands"]:
                        self.assertIn("-ffp-contract=off", command)
                        self.assertIn("-fno-fast-math", command)

    @unittest.skipUnless(any(shutil.which(name) for name in ("cc", "gcc", "clang")), "C compiler is not installed")
    def test_native_overflow_is_an_error_without_partial_evidence(self):
        model = DenseReluModel([[0x7F7FFFFF], [0], [0]], [0])
        with tempfile.TemporaryDirectory() as directory:
            result = run_native(model, [[float_to_bits(2), 0, 0]], Path(directory))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["backends"], {})

    def test_rejects_unsupported_input_dimension(self):
        with self.assertRaises(ValueError):
            c_source(DenseReluModel([[0]], [0]), [[0]])

    def test_rejects_invalid_corpus_before_writing_source_or_invoking_compiler(self):
        invalid_inputs = (None, [], {}, [None], [[]], [[0, 0]], [[0, 0, 0, 0]],
                          [[0, 0, 0], [0, 0]], [(0, 0, 0)])
        invalid_inputs += tuple([[bits, 0, 0]] for bits in
                                (-1, 2**32, True, 1.0, "0", None, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000))
        for inputs in invalid_inputs:
            with self.subTest(inputs=inputs), tempfile.TemporaryDirectory() as directory:
                with patch("veritrace.native.subprocess.run") as process:
                    with self.assertRaises(ValueError):
                        run_native(demo_model(), inputs, Path(directory))
                    process.assert_not_called()
                self.assertFalse((Path(directory) / "native.c").exists())

    def test_revalidates_mutated_model_before_generation(self):
        model = demo_model()
        model.weights[0][0] = 0x80000000
        with self.assertRaises(ValueError):
            c_source(model, [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
