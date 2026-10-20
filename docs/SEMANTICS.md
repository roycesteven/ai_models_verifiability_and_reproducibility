# Execution semantics and the verification claim

VeriTrace is an executable experiment in specifying a small neural network at the arithmetic-operation level. Its result applies to the accepted graph, property, input domain, and selected execution contract together. An ONNX file alone does not specify this project's reduction schedule.

## The profile

The prototype accepts exactly one `MatMul → Add → Relu` inference chain: input `[1,N]`, constant weights `[N,M]`, constant bias `[M]`, and output `[1,M]`, all binary32. Supported files use ONNX IR versions 7–10 and default-domain opset 13. Dynamic shapes, extra branches or operators, attributes, external or sparse tensor data, overridable weights, and training graphs are rejected. This is a project-specific profile, not a SONNX implementation or a claim of conformance to that working group's specifications.

ONNX's `MatMul` documentation describes a matrix product with NumPy-like behavior but does not give the scalar accumulation schedule used below. The conclusion that this schedule is an additional contract is an interpretation of that published operator description. [ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)

The contract fixes these decisions:

| Decision | Prototype semantics |
| --- | --- |
| Storage and arithmetic format | Binary32, with 8 exponent bits and 24 bits of significand precision including the implicit bit |
| Rounding | Round to nearest, with ties to even, after every multiply and every add |
| Dot product | An explicit sequential order starting from positive zero |
| Fused operations | No fused multiply-add |
| Subnormals | Gradual underflow; no flush-to-zero |
| Zero | Input and constant negative zero are rejected; zero results are canonicalized to positive zero |
| Nonfinite values | NaN and infinity are outside the accepted finite execution contract |
| Graph evaluation | Follow the accepted graph's dependencies; preserve the named arithmetic operations |

The floating-point solver uses the SMT-LIB binary floating-point theory. `Float32` denotes `(_ FloatingPoint 8 24)` and `RNE` denotes the selected rounding mode. SMT-LIB distinguishes separate `fp.mul`/`fp.add` from `fp.fma`; these are different operations. [SMT-LIB floating-point theory](https://smt-lib.org/theories-FloatingPoint.shtml)

Canonical positive zero is an intentional profile rule. It means this is not an implementation of every observable feature of IEEE 754: signed-zero propagation, NaN payloads, exception flags, and alternative rounding modes are outside scope.

## Arithmetic definition

For a finite binary32 bit pattern with sign bit `s`, exponent field `e`, and fraction field `m`, let `value` be its exact rational value:

```text
e = 0:       value(s,e,m) = (-1)^s × m × 2^-149
1 ≤ e ≤ 254: value(s,e,m) = (-1)^s × (2^23 + m) × 2^(e-150)
```

Exponent field 255 is rejected. Let `R(q)` round an exact rational `q` to binary32 using nearest-even, preserving subnormal magnitudes and canonicalizing a zero result to `0x00000000`. If the IEEE-style rounded result would overflow to infinity, finite execution fails; it must not silently saturate to the maximum finite value.

```text
add(a,b) = R(value(a) + value(b))
mul(a,b) = R(value(a) × value(b))
relu(a)  = a if value(a) > 0 else +0
```

For an input row `A` of length `n`, weights `B` of shape `[n,m]`, and a permutation `π` of the reduction indices:

```text
acc[0,j]   = +0
p[t,j]     = mul(A[π(t)], B[π(t),j])
acc[t+1,j] = add(acc[t,j], p[t,j])
Y[j]       = acc[n,j]
```

`canonical` uses increasing index order. The demo's `cancellation_first` contract uses `[0,2,1]`. Both are deterministic contracts; choosing the first does not make it more numerically accurate.

The concrete software interpreter computes exact intermediate rational values using Python integers and `fractions.Fraction`, then explicitly rounds them. Its arithmetic is independent of the Z3 floating-point evaluator and the host's floating-point reductions. The implementation itself remains trusted software; testing it is not a formal proof that it implements this definition.

## The cancellation benchmark

The network has input `[1,x,1]`, weight column `[2^24,1,-2^24]`, a zero bias, and a ReLU output. Its engineering-style requirement is:

```text
Input:  X_0 = 1, 1/2 ≤ X_1 ≤ 1, X_2 = 1
Safety: Y_0 ≥ 1/2
```

These values are chosen to expose an arithmetic boundary, not to model an aircraft controller.

Over real numbers, multiplication and addition are exact, so the network is `ReLU(x) = x` throughout the input interval. Therefore its safety condition holds.

At `2^24`, the next greater binary32 value is `2^24 + 2`. For every binary32 `x` in `[1/2,1]`, nearest-even rounds `2^24 + x` to `2^24`. The endpoint `x = 1` is exactly halfway between the two neighboring representable values, and the lower one's significand is even. Consequently:

```text
canonical:          R(R(2^24 + x) - 2^24) = +0
cancellation_first: R(R(2^24 - 2^24) + x) = x
```

The omitted initial `+0`, exact multiplies, zero bias, and final ReLU do not change those results in this domain.

| Semantics | Output on the stated domain | Violation query |
| --- | --- | --- |
| Exact reals | `x` | UNSAT |
| Binary32, canonical order | `0` | SAT |
| Binary32, cancellation first | `x` | UNSAT |

This analytical explanation concerns the entire specified domain. The floating-point solver's input space consists of representable binary32 values in the interval; the real solver's input space contains every real value in it. Distinguishing those domains matters when comparing other properties.

## Property interpretation

The demo uses a restricted legacy VNN-COMP-style property file:

```smt2
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const Y_0 Real)
(assert (>= X_0 1))
(assert (<= X_0 1))
(assert (>= X_1 (/ 1 2)))
(assert (<= X_1 1))
(assert (>= X_2 1))
(assert (<= X_2 1))
(assert (< Y_0 (/ 1 2)))
```

The last assertion describes the **unsafe set**. The verifier searches for an input satisfying all input constraints whose computed output lies in that set. The file is not an assertion that a safe output should be below one half. For the floating-point interpretation, the input declarations are bound to finite binary32 values; they are not an instruction to perform graph arithmetic over Z3 reals.

Every `X_i` and `Y_i` must be declared `Real` exactly once before use. The accepted constraints are a finite closed box on every input and exactly one strict unsafe output comparison (`<` or `>`) against a rational constant. Conjunctions, decimal constants, unary minus, and rational `(/ p q)` constants are supported. Disjunctions, comparisons between variables, and arbitrary arithmetic are rejected. Rational bounds retain their exact meaning: when a bound lies between binary32 values, the floating-point comparison adjusts its strictness around the rounded boundary rather than silently changing the interval.

Before checking the unsafe output, the verifier checks that the input domain is inhabited and that no nonfinite arithmetic intermediate is reachable anywhere in that domain. Reachable overflow or an inconclusive finite-intermediate check returns `unknown`; it cannot produce a safety result by assuming overflow away. An empty representable input domain produces an explicitly vacuous UNSAT result. An overall time budget is shared across these checks.

This input format is not full current VNN-LIB. As checked on 13 October 2026, the official project reports VNN-LIB 2.0, with a formal syntax and semantics and typed network declarations. The prototype's compatibility claim is limited to the legacy fragment it actually parses. [VNN-LIB project](https://www.vnnlib.org/), [standard repository](https://github.com/VNNLIB/VNNLIB-Standard)

## Replaying the solver queries

For a nonvacuous float32 safety result, all three checks must succeed: the input domain is inhabited, the overflow query is UNSAT, and the unsafe-output query is UNSAT. The JSON result records the first two checks as `input_domain_status` and `overflow_check`. The separate `canonical-overflow.smt2` and `cancellation_first-overflow.smt2` files search for a nonfinite intermediate anywhere in the input box. Rerun these alongside their corresponding unsafe-output queries; a standalone unsafe-output UNSAT result does not establish that execution stays within the supported profile.

For example, a negative multiplication overflow followed by ReLU can produce zero in the unrestricted IEEE-754 expression. Its unsafe-output query might be UNSAT even though the multiplication left the finite profile. VeriTrace returns UNKNOWN for that case because the overflow query is SAT. A timeout in either check also prevents a safety claim. An empty representable input domain is reported separately with `vacuous: true`.

The real-arithmetic baseline has no finite-precision overflow check. Replaying a SAT query establishes solver satisfiability; use the separate witness replay command to check the concrete float32 counterexample with the exact-rational software interpreter. All saved SMT-LIB files are solver queries, not proof certificates.

## What each artifact establishes

| Artifact or result | Meaning | Limit |
| --- | --- | --- |
| `model.onnx` plus `execution-contract.json` | Graph bytes and selected arithmetic decisions | Does not establish that an arbitrary backend follows the contract |
| `canonical.smt2`, `real.smt2`, `cancellation_first.smt2` | Unsafe-output queries that can be inspected and rerun | Standalone replay omits the verifier's prerequisite checks; query files are not proof certificates |
| `canonical-overflow.smt2`, `cancellation_first-overflow.smt2` | Separate queries for reachable nonfinite arithmetic, when generated | Must be UNSAT before claiming safety under the finite profile |
| SAT result and `witness.json` | A specific violating input and its replay evidence | A witness for one contract need not violate another |
| Independently replayed witness | The separate rational interpreter reproduces the reported violation | Shares parsing and contract assumptions with the main tool |
| Nonvacuous UNSAT result | Z3 found no violation after checking domain feasibility and finite intermediates | Trusts the encoding, solver, and supported domain |
| Unknown result | A query timed out, left the finite profile, or could not be validated | Establishes neither safety nor a validated violation |
| ONNX Runtime and compiled C results | Observations on recorded inputs on the current host | Sampling is not universal equivalence or cross-machine evidence |
| Manifest and hashes | A means to identify and compare recorded artifacts and environments | Hash equality does not prove semantic correctness |

Prefer `python -m veritrace verify` when reproducing a complete result. For a nonvacuous Float32 safety claim, its JSON must report `input_domain_status: "sat"`, `overflow_check: "unsat"`, and `status: "unsat"`. Running an unsafe-output `.smt2` file alone does not establish that the input domain is inhabited or that every arithmetic intermediate remains finite. The separate overflow query must also be checked; if it is absent or inconclusive, do not infer a completed prerequisite. Files emitted by `verify --out check.json` use `check.smt2` and `check.overflow.smt2`; demo artifacts use the names in the table.

A successful demo does not prove the compiler, the loader, Z3, or ONNX Runtime correct. It does not verify training, quantized arithmetic, preprocessing, an aerospace system, or compliance with ED-324/ARP 6983. Those are separate research and assurance obligations.
