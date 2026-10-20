"""Independent generated C execution and optional ONNX Runtime observations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .model import DenseReluModel
from .numeric import validate_bits


def c_source(model: DenseReluModel, inputs: list[list[int]]) -> str:
    model.__post_init__()
    if model.input_size != 3:
        raise ValueError("Native demonstration expects exactly three inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Native inputs must be a nonempty list of input rows")
    for index, row in enumerate(inputs):
        if not isinstance(row, list) or len(row) != model.input_size:
            raise ValueError(f"Native input row {index} must contain exactly {model.input_size} values")
        for bits in row:
            try:
                validate_bits(bits)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Native input row {index} contains an invalid binary32 encoding: {error}") from error
    initializer = lambda rows: ",\n".join("{" + ",".join(f"0x{x:08x}u" for x in row) + "}" for row in rows)
    return r'''/* Generated from the model and test inputs; inspectable independent runner.
   Contract: binary32, nearest/ties-even, separate multiply and add, +0. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON
static float from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t to_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float product(float a, float b) { volatile float r = a * b; return r == 0 ? 0.0f : r; }
static float sum(float a, float b) { volatile float r = a + b; return r == 0 ? 0.0f : r; }
static const uint32_t weights[3][@OUTPUTS@] = {@WEIGHTS@};
static const uint32_t biases[@OUTPUTS@] = {@BIAS@};
static const uint32_t inputs[@SAMPLES@][3] = {@INPUTS@};
int main(void) {
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
        FLT_MAX_EXP != 128 || fesetround(FE_TONEAREST) != 0) return 2;
    if (to_bits(product(from_bits(1u), 1.0f)) != 1u) return 3;
    const int orders[2][3] = {{0,1,2},{0,2,1}};
    for (int p = 0; p < 2; ++p) {
        for (int r = 0; r < @SAMPLES@; ++r) {
            printf("%d %d", p, r);
            for (int j = 0; j < @OUTPUTS@; ++j) {
                float accum = 0.0f;
                for (int t = 0; t < 3; ++t) {
                    int k = orders[p][t];
                    float term = product(from_bits(inputs[r][k]), from_bits(weights[k][j]));
                    if (!isfinite(term)) return 4;
                    accum = sum(accum, term);
                    if (!isfinite(accum)) return 4;
                }
                accum = sum(accum, from_bits(biases[j]));
                if (!isfinite(accum)) return 4;
                float y = accum > 0.0f ? accum : 0.0f;
                printf(" %08x", (unsigned)to_bits(y));
            }
            printf("\n");
        }
    }
    return 0;
}
'''.replace("@OUTPUTS@", str(model.output_size)).replace("@SAMPLES@", str(len(inputs))).replace(
        "@WEIGHTS@", initializer(model.weights)
    ).replace("@BIAS@", ",".join(f"0x{x:08x}u" for x in model.bias)).replace("@INPUTS@", initializer(inputs))


def run_native(model: DenseReluModel, inputs: list[list[int]], out: Path) -> dict:
    source = out / "native.c"
    source.write_text(c_source(model, inputs))
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        return {"status": "unavailable", "reason": "No C compiler found", "backends": {}}
    backends, commands = {}, []
    try:
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        for optimization in ("O0", "O2"):
            binary = out / f"native-{optimization}"
            command = [compiler, f"-{optimization}", "-std=c11", "-ffp-contract=off", "-fno-fast-math",
                       "-frounding-math", str(source.resolve()), "-lm", "-o", str(binary.resolve())]
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
            commands.append(command)
            lines = subprocess.run([str(binary.resolve())], capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
            for order in ("canonical", "cancellation_first"):
                backends[f"c:{optimization}:{order}"] = []
            if len(lines) != 2 * len(inputs):
                raise ValueError("Native runner returned an incomplete observation set")
            for expected, line in enumerate(lines):
                fields = line.split()
                p, r = map(int, fields[:2])
                if (p, r) != divmod(expected, len(inputs)) or len(fields) != model.output_size + 2:
                    raise ValueError("Native observation indices or output width differ")
                order = ("canonical", "cancellation_first")[p]
                backends[f"c:{optimization}:{order}"].append([int(x, 16) for x in fields[2:]])
        return {"status": "observed", "compiler": version, "commands": commands, "backends": backends}
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return {"status": "error", "reason": str(exc), "backends": {}, "commands": commands}


def run_onnx_runtime(path: Path, inputs: list[list[int]]) -> dict:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return {"status": "unavailable", "reason": "Install the runtime extra or requirements.txt", "backends": {}}
    backends = {}
    try:
        for label, optimization in (("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                                    ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
            options = ort.SessionOptions()
            options.graph_optimization_level = optimization
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            observations = []
            for row in inputs:
                x = np.array(row, dtype=np.uint32).view(np.float32).reshape(1, -1)
                output = session.run(None, {session.get_inputs()[0].name: x})[0]
                observations.append([int(v) for v in output.astype(np.float32).view(np.uint32).flat])
            backends[f"onnxruntime:cpu:{label}"] = observations
        return {"status": "observed", "version": ort.__version__, "provider": "CPUExecutionProvider",
                "threads": 1, "ordering": "Runtime-selected; no contract conformance assumed", "backends": backends}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "backends": {}}
