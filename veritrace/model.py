"""Restricted ONNX graph import and an explicit scalar execution schedule.

Accepted ONNX profile: IR 7–10, default-domain opset 13, one static float32
input [1,N], MatMul with W[N,M], Add with b[M], then Relu and output [1,M].
All tensors are finite and use canonical +0.  Training, external/sparse data,
custom operators, attributes, extra branches, and overridable weights fail
closed.  This profile is a prototype and does not claim SONNX conformance.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .numeric import add, bits_to_fraction, float_to_bits, hex_bits, mul, relu, validate_bits

SEMANTICS = "binary32-rne-gradual-canonical-zero-separate-mul-add-v1"


@dataclass
class DenseReluModel:
    weights: list[list[int]]
    bias: list[int]
    name: str = "model"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("model name must be a string")
        if not isinstance(self.weights, list) or not self.weights:
            raise ValueError("weights must be a nonempty list of input rows")
        if not isinstance(self.bias, list) or not self.bias:
            raise ValueError("bias must be a nonempty list")
        for row in self.weights:
            if not isinstance(row, list) or len(row) != len(self.bias):
                raise ValueError("weights must be rectangular [input_size, output_size]")
            for bits in row:
                validate_bits(bits)
        for bits in self.bias:
            validate_bits(bits)

    @property
    def input_size(self) -> int:
        return len(self.weights)

    @property
    def output_size(self) -> int:
        return len(self.bias)

    def to_dict(self) -> dict:
        self.__post_init__()
        return {
            "name": self.name,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights_bits": [[hex_bits(value) for value in row] for row in self.weights],
            "bias_bits": [hex_bits(value) for value in self.bias],
            "semantics": SEMANTICS,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def execute(model: DenseReluModel, inputs: list[int], order: str = "canonical") -> dict:
    """Execute separate rounded multiply/add operations in a declared order."""
    model.__post_init__()
    if len(inputs) != model.input_size:
        raise ValueError(f"expected {model.input_size} inputs, received {len(inputs)}")
    for bits in inputs:
        validate_bits(bits)
    if order == "canonical":
        indices = list(range(model.input_size))
    elif order == "cancellation_first" and model.input_size == 3:
        indices = [0, 2, 1]
    else:
        raise ValueError("order must be canonical, or cancellation_first for exactly 3 inputs")
    trace = []

    def record(node: str, operation: str, operands: list[int], result: int) -> None:
        trace.append({"node": node, "operation": operation,
                      "input_bits": [hex_bits(value) for value in operands],
                      "output_bits": hex_bits(result)})

    outputs = []
    for j in range(model.output_size):
        accumulator = 0
        for i in indices:
            product = mul(inputs[i], model.weights[i][j])
            record(f"MatMul[{j}].mul[{i}]", "mul", [inputs[i], model.weights[i][j]], product)
            updated = add(accumulator, product)
            record(f"MatMul[{j}].add[{i}]", "add", [accumulator, product], updated)
            accumulator = updated
        biased = add(accumulator, model.bias[j])
        record(f"Add[{j}]", "add", [accumulator, model.bias[j]], biased)
        output = relu(biased)
        record(f"Relu[{j}]", "relu", [biased], output)
        outputs.append(output)
    return {"output_bits": outputs, "trace": trace}


def real_execute(model: DenseReluModel, inputs: list[Fraction]) -> list[Fraction]:
    """Evaluate the same graph over exact reals (rational inputs and constants)."""
    model.__post_init__()
    if len(inputs) != model.input_size:
        raise ValueError(f"expected {model.input_size} inputs, received {len(inputs)}")
    if any(not isinstance(value, (Fraction, int)) for value in inputs):
        raise TypeError("real_execute requires exact Fraction or integer inputs")
    return [max(Fraction(0), bits_to_fraction(model.bias[j]) + sum(
        (Fraction(inputs[i]) * bits_to_fraction(model.weights[i][j])
         for i in range(model.input_size)), Fraction(0))) for j in range(model.output_size)]


def demo_model() -> DenseReluModel:
    return DenseReluModel([[float_to_bits(2**24)], [float_to_bits(1)],
                           [float_to_bits(-(2**24))]], [0], "cancellation_demo")


def write_demo_onnx(path: str | Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = demo_model()
    graph = helper.make_graph([
        helper.make_node("MatMul", ["input", "weights"], ["linear"], name="dense"),
        helper.make_node("Add", ["linear", "bias"], ["biased"], name="bias_add"),
        helper.make_node("Relu", ["biased"], ["output"], name="activation"),
    ], model.name, [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.array(model.weights, dtype=np.uint32).view(np.float32), "weights"),
         numpy_helper.from_array(np.array(model.bias, dtype=np.uint32).view(np.float32), "bias")])
    exported = helper.make_model(graph, producer_name="veritrace",
                                 opset_imports=[helper.make_opsetid("", 13)])
    exported.ir_version = 9
    onnx.checker.check_model(exported, full_check=True)
    onnx.save_model(exported, str(path))


def load_onnx(path: str | Path) -> DenseReluModel:
    """Import only the documented profile, refusing all unsupported semantics."""
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    try:
        source = onnx.load_model(str(path), load_external_data=False)
    except Exception as error:
        raise ValueError(f"cannot read ONNX model: {error}") from error
    if not 7 <= source.ir_version <= 10:
        raise ValueError("supported ONNX IR versions are 7 through 10")
    if len(source.opset_import) != 1 or source.opset_import[0].domain != "" or source.opset_import[0].version != 13:
        raise ValueError("only default-domain ONNX opset 13 is supported")
    if source.functions or source.training_info:
        raise ValueError("ONNX functions and training graphs are unsupported")
    if getattr(source, "configuration", []):
        raise ValueError("ONNX device configurations are unsupported")
    graph = source.graph
    if graph.sparse_initializer or graph.quantization_annotation:
        raise ValueError("sparse initializers and quantization annotations are unsupported")
    if len(graph.input) != 1 or len(graph.output) != 1 or len(graph.node) != 3:
        raise ValueError("expected one input, one output, and exactly MatMul -> Add -> Relu")
    if len(graph.initializer) != 2:
        raise ValueError("expected exactly two constant initializers: weights and bias")
    for node, expected in zip(graph.node, ["MatMul", "Add", "Relu"]):
        if node.domain != "" or node.op_type != expected or node.attribute:
            raise ValueError("only attribute-free default-domain MatMul -> Add -> Relu is supported")
        if getattr(node, "overload", "") or getattr(node, "device_configurations", []):
            raise ValueError("operator overloads and device configurations are unsupported")
        arity = 1 if expected == "Relu" else 2
        if len(node.input) != arity or len(node.output) != 1 or not all([*node.input, *node.output]):
            raise ValueError(f"invalid {expected} inputs or outputs")
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    if len(initializers) != 2 or not all(initializers):
        raise ValueError("initializer names must be nonempty and unique")
    input_name = graph.input[0].name
    matmul, bias_add, activation = graph.node
    produced = [input_name, *initializers, *(node.output[0] for node in graph.node)]
    if not input_name or len(set(produced)) != len(produced):
        raise ValueError("overridable initializers and duplicate tensor definitions are unsupported")
    if (matmul.input[0] != input_name or matmul.input[1] not in initializers
            or bias_add.input[0] != matmul.output[0] or bias_add.input[1] not in initializers
            or bias_add.input[1] == matmul.input[1]
            or activation.input[0] != bias_add.output[0]
            or graph.output[0].name != activation.output[0]):
        raise ValueError("graph must connect input -> MatMul(weights) -> Add(bias) -> Relu -> output")

    def static_shape(value) -> list[int]:
        if value.type.WhichOneof("value") != "tensor_type":
            raise ValueError("only float32 tensors are supported")
        tensor = value.type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT or not tensor.HasField("shape"):
            raise ValueError("tensors must have static float32 shapes")
        if any(dim.WhichOneof("value") != "dim_value" or dim.dim_value <= 0 for dim in tensor.shape.dim):
            raise ValueError("dynamic or empty tensor dimensions are unsupported")
        return [dim.dim_value for dim in tensor.shape.dim]

    input_shape, output_shape = static_shape(graph.input[0]), static_shape(graph.output[0])
    if len(input_shape) != 2 or input_shape[0] != 1 or len(output_shape) != 2 or output_shape[0] != 1:
        raise ValueError("input and output shapes must be [1, features]")
    input_size, output_size = input_shape[1], output_shape[1]
    seen = set()
    for info in graph.value_info:
        if info.name not in [matmul.output[0], bias_add.output[0]] or info.name in seen or static_shape(info) != output_shape:
            raise ValueError("only consistent intermediate tensor shape declarations are supported")
        seen.add(info.name)

    def constant(name: str, shape: list[int]):
        tensor = initializers[name]
        if tensor.data_type != TensorProto.FLOAT or list(tensor.dims) != shape:
            raise ValueError(f"initializer {name} must be float32 with shape {shape}")
        if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL:
            raise ValueError("external tensor data is unsupported; provide a self-contained ONNX file")
        try:
            values = numpy_helper.to_array(tensor)
            bits = np.asarray(values, dtype=np.float32).view(np.uint32).tolist()
        except Exception as error:
            raise ValueError(f"invalid initializer {name}: {error}") from error
        return bits

    weights = constant(matmul.input[1], [input_size, output_size])
    bias = constant(bias_add.input[1], [output_size])
    model = DenseReluModel(weights, bias, graph.name or "model")
    try:
        onnx.checker.check_model(source, full_check=True)
    except Exception as error:
        raise ValueError(f"invalid ONNX model: {error}") from error
    return model
