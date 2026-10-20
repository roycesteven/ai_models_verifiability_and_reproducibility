# VeriTrace

**A proof of concept for verified and reproducible AI execution.**

VeriTrace demonstrates a concrete gap between a neural network's mathematical
specification and its finite-precision execution. It imports a real ONNX model,
defines an explicit float32 execution contract, checks an engineering requirement
with Z3, independently replays a counterexample, and records reproducibility
evidence. The generated HTML report runs offline.

Inspired by the University of Manchester PhD project
[Verified and reproducible execution of AI models](https://www.findaphd.com/phds/project/school-of-engineering-phd-scholarships-verified-and-reproducible-execution-of-ai-models/?p198772),
supervised by Dr Edoardo Manino and Dr Lucas Cordeiro. This repository is an
independent research prototype, with no affiliation or certification claim.

## Run the demo

Requires Python 3.11+; a C compiler enables the independent native checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m veritrace demo --out artifacts/demo
```

Open **`artifacts/demo/report.html`** in your browser. No web server, account,
GPU, training data, or network connection is needed after installation.

Expected verification outcomes:

```text
real                 UNSAT   requirement holds within scope
canonical            SAT     requirement violated
cancellation_first   UNSAT   requirement holds within scope
```

The query asks whether an **unsafe output exists**. SAT means a counterexample;
UNSAT means no such input exists under the selected arithmetic semantics.
Timeouts and reachable arithmetic outside the supported profile remain UNKNOWN.

The development environment already contains `.venv` and a generated report;
from this workspace you can run `.venv/bin/python -m veritrace demo` directly.
For a minimal installation without ONNX Runtime, use `pip install -e .` and
run the demo with `--skip-runtime`. `--skip-native` skips compiler execution.
Missing optional backends are reported explicitly.

## The experiment

The hand-built, one-neuron network has three operators:

```text
[1, x, 1] → MatMul([2²⁴, 1, −2²⁴]) → Add(0) → ReLU → y

Input domain: x ∈ [1/2, 1]
Requirement: y ≥ 1/2
```

| Execution semantics | Output over this domain | Unsafe-query result |
| --- | --- | --- |
| Exact real arithmetic | `x` | UNSAT |
| Float32, accumulate inputs in order `[0,1,2]` | `0` | SAT |
| Float32, accumulate in order `[0,2,1]` | `x` | UNSAT |

At `2²⁴`, adjacent float32 values above it are two units apart. Adding `x ≤ 1`
first rounds back to `2²⁴`, including the halfway tie when `x = 1`. Subtracting
`2²⁴` then returns zero. Canceling the large terms first preserves `x`.

The alternate order is a **different execution contract**, not a generally safe
optimization. Neither contract is implied by ONNX MatMul alone. See the
[published operator specification](https://onnx.ai/onnx/operators/onnx__MatMul.html).

## What is implemented

- **Strict ONNX import/export:** static batch-one `MatMul → Add → Relu`, float32
  weights and bias, opset 13, IR 7–10. Unsupported constructs fail closed.
- **Independent concrete arithmetic:** exact rational decoding, multiplication
  and addition, explicit nearest/ties-even binary32 rounding, gradual
  subnormals, canonical positive zero, and rejection of nonfinite arithmetic.
- **Symbolic verification:** Z3 real arithmetic and Float32 encodings with an
  explicit operation order and no fused multiply-add. Before a float32 safety
  result, separate queries establish a nonempty domain and exclude reachable
  nonfinite intermediates. Empty representable domains are labeled vacuous.
- **Property interchange:** a bounded legacy VNN-COMP-style fragment with real
  `X_i` / `Y_i` declarations, closed rational input bounds, and one strict unsafe
  output inequality. Rational thresholds are interpreted exactly.
- **Witness replay:** compare solver outputs with an independent software
  execution and verify the actual requirement violation.
- **Measured execution:** generated C at `-O0` and `-O2`, plus optional ONNX
  Runtime CPU observations with graph optimization disabled and enabled.
- **Portable evidence:** saved model, contract, input/output bit patterns,
  SMT-LIB queries, environment details, and SHA-256-bound observation manifests.

The report includes a sample selector, operation-level bit traces, solver
outcomes, backend observations, and links to the underlying artifacts.

## Inspect and reproduce

Verify the generated files directly:

```bash
python -m veritrace verify \
  --model artifacts/demo/model.onnx \
  --property artifacts/demo/property.vnnlib \
  --order canonical --out artifacts/check.json
```

`verify` exits **0** for UNSAT, **1** for SAT, and **2** for UNKNOWN or invalid
input. A successful `demo` exits 0 because finding the counterexample is an
expected outcome of this experiment. Use `--semantics real` or
`--order cancellation_first` to change the verification semantics.
With `--out`, a float32 SAT result also saves `check.witness.json` for replay.

Replay the witness:

```bash
python -m veritrace replay \
  --model artifacts/demo/model.onnx \
  --property artifacts/demo/property.vnnlib \
  --witness artifacts/demo/witness.json
```

Run on a second machine, copy the first manifest there, and compare:

```bash
python -m veritrace demo --out artifacts/machine-b
python -m veritrace compare artifacts/demo/manifest.json artifacts/machine-b/manifest.json
```

`compare` exits 0 for identical observations, 1 for observed drift, and 2 for
incompatible or invalid manifests. It requires matching model, contract,
input vectors, and backend names. Package versions and host details remain
visible in each manifest. Hashes detect accidental changes; they are not signatures.

All files generated by `demo` in its output directory are replaced on rerun,
so an inconclusive rerun cannot retain a previous counterexample as current evidence.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover halfway rounding, subnormal boundaries, overflow, signed zero,
separate multiply/add, model-import rejection, rational property boundaries,
solver UNKNOWN propagation, counterexample replay, compiled C agreement,
manifest drift and tampering, and the integrated experiment. A CI workflow
is included for Linux/macOS and Python 3.11/3.12; remote CI execution is not
claimed by the local demo.

On the development host, both C optimizations match the independent arithmetic
on all nine demo vectors. ONNX Runtime's two observed configurations return
zero on those vectors. These are measurements on **one host**, not evidence
of an actual discrepancy between two different machines.

## Documentation and research limits

- [Precise semantics, supported formats, and trusted components](docs/SEMANTICS.md)
- [Research positioning, primary sources, and next experiments](docs/RESEARCH.md)

This prototype covers a deliberately small **inference** graph. It does not
implement training, arbitrary ONNX graphs, quantized operators, general error
propagation, ESBMC/Why3 integration, full VNN-LIB 2.0, or a SONNX conformance
checker. The cancellation phenomenon is established knowledge; the prototype's
value is an inspectable verification and reproduction workflow.

SAT witnesses are independently replayed. UNSAT results trust the Z3 solver
and the encoding; the generated SMT-LIB files are reproducible queries, **not
proof certificates**. Sampled backend agreement does not prove universal
equivalence. No aerospace safety or certification guarantee is claimed.

A standalone unsafe-output SMT query is only one part of a float32 safety
claim: the input domain must be inhabited and the corresponding overflow
query must also be UNSAT. The CLI performs and records these prerequisite checks.
