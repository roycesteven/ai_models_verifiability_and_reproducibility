from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from veritrace.model import DenseReluModel, demo_model, execute, load_onnx, real_execute, write_demo_onnx
from veritrace.numeric import float_to_bits


class ModelTests(unittest.TestCase):
    def test_demo_distinguishes_arithmetic_semantics(self):
        model = demo_model()
        inputs = [float_to_bits(1)] * 3
        canonical = execute(model, inputs)
        alternative = execute(model, inputs, order="cancellation_first")
        self.assertEqual(canonical["output_bits"], [0])
        self.assertEqual(alternative["output_bits"], [float_to_bits(1)])
        self.assertEqual(real_execute(model, [Fraction(1)] * 3), [Fraction(1)])
        self.assertEqual(len(canonical["trace"]), 8)
        self.assertEqual(canonical["trace"][-1]["operation"], "relu")
        json.dumps(canonical)

    def test_bias_and_multiple_outputs(self):
        encode = float_to_bits
        model = DenseReluModel([[encode(2), encode(-3)], [encode(1), encode(1)]], [encode(-1), encode(2)])
        outputs = execute(model, [encode(3), encode(2)])["output_bits"]
        self.assertEqual(outputs, [encode(7), 0])
        self.assertEqual(real_execute(model, [Fraction(3), Fraction(2)]), [7, 0])

    def test_multiply_and_add_are_rounded_separately(self):
        a = float_to_bits(1 + 2**-23)
        model = DenseReluModel([[a]], [float_to_bits(-(1 + 2**-22))])
        self.assertEqual(execute(model, [a])["output_bits"], [0])
        self.assertEqual(real_execute(model, [Fraction(1) + Fraction(1, 2**23)]), [Fraction(1, 2**46)])

    def test_model_and_input_validation(self):
        for weights, bias in (([], [0]), ([[0]], []), ([[0], [0, 0]], [0]), ([[0x80000000]], [0]), ([[0]], [0x7F800000])):
            with self.assertRaises(ValueError):
                DenseReluModel(weights, bias)
        for inputs in ([0, 0], [0x80000000, 0, 0], [0x7FC00000, 0, 0]):
            with self.assertRaises(ValueError):
                execute(demo_model(), inputs)
        with self.assertRaises(ValueError):
            execute(demo_model(), [0, 0, 0], order="random")
        with self.assertRaises(ValueError):
            execute(DenseReluModel([[0]], [0]), [0], order="cancellation_first")

    def test_digest_is_stable_and_tracks_model_mutation(self):
        first, second = demo_model(), demo_model()
        self.assertEqual(first.digest, second.digest)
        original = first.digest
        first.bias[0] = float_to_bits(1)
        self.assertNotEqual(first.digest, original)
        self.assertEqual(len(original), 64)


class OnnxProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.onnx"
        write_demo_onnx(self.path)

    def test_roundtrip_preserves_constants_and_results(self):
        loaded = load_onnx(self.path)
        self.assertEqual(loaded.to_dict(), demo_model().to_dict())
        self.assertEqual(loaded.digest, demo_model().digest)
        self.assertEqual(execute(loaded, [float_to_bits(1)] * 3)["output_bits"], [0])

    def test_strict_rejection_of_unsupported_models(self):
        def extra_node(model):
            model.graph.node.append(helper.make_node("Identity", ["input"], ["unused"]))

        def external(model):
            tensor = model.graph.initializer[0]
            tensor.data_location = TensorProto.EXTERNAL
            tensor.external_data.add(key="location", value="untrusted.bin")

        def negative_zero(model):
            model.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array([-0.0], dtype=np.float32), "bias"))

        mutations = {
            "opset": lambda m: setattr(m.opset_import[0], "version", 14),
            "IR": lambda m: setattr(m, "ir_version", 11),
            "custom_domain": lambda m: setattr(m.graph.node[0], "domain", "custom"),
            "attribute": lambda m: m.graph.node[0].attribute.append(helper.make_attribute("transpose", 1)),
            "branch": extra_node,
            "extra_output": lambda m: m.graph.output.append(helper.make_tensor_value_info("linear", TensorProto.FLOAT, [1, 1])),
            "dynamic_input": lambda m: setattr(m.graph.input[0].type.tensor_type.shape.dim[1], "dim_param", "N"),
            "empty_dimension": lambda m: setattr(m.graph.input[0].type.tensor_type.shape.dim[1], "dim_value", 0),
            "wrong_output_shape": lambda m: setattr(m.graph.output[0].type.tensor_type.shape.dim[1], "dim_value", 2),
            "float64_input": lambda m: setattr(m.graph.input[0].type.tensor_type, "elem_type", TensorProto.DOUBLE),
            "float64_constant": lambda m: setattr(m.graph.initializer[0], "data_type", TensorProto.DOUBLE),
            "rank2_bias": lambda m: m.graph.initializer[1].dims.append(1),
            "external_data": external,
            "negative_zero": negative_zero,
            "infinity": lambda m: m.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array([np.inf], dtype=np.float32), "bias")),
            "nan": lambda m: m.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array([np.nan], dtype=np.float32), "bias")),
            "overridable_initializer": lambda m: m.graph.input.append(helper.make_tensor_value_info("weights", TensorProto.FLOAT, [3, 1])),
            "wrong_edge": lambda m: m.graph.node[1].input.__setitem__(0, "input"),
            "duplicate_definition": lambda m: m.graph.node[1].output.__setitem__(0, "linear"),
            "training_graph": lambda m: m.training_info.add(),
            "sparse_initializer": lambda m: m.graph.sparse_initializer.add(),
            "inconsistent_intermediate": lambda m: m.graph.value_info.append(helper.make_tensor_value_info("linear", TensorProto.FLOAT, [1, 2])),
            "unused_initializer": lambda m: m.graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.float32), "unused")),
        }
        original = self.path.read_bytes()
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                model = onnx.load_model_from_string(original)
                mutate(model)
                self.path.write_bytes(model.SerializeToString())
                with self.assertRaises(ValueError):
                    load_onnx(self.path)

    def test_valid_explicit_intermediate_shapes(self):
        model = onnx.load(self.path)
        for name in ("linear", "biased"):
            model.graph.value_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1]))
        onnx.save_model(model, self.path)
        self.assertEqual(load_onnx(self.path).digest, demo_model().digest)


if __name__ == "__main__":
    unittest.main()
