# Research positioning and next experiments

This proof of concept targets the project’s two-stage structure: first describe the computation precisely, then check a requirement against that computation. It implements a small inference-only experiment. It is a starting point for discussion, not a complete implementation of the proposed PhD or a novel verification algorithm.

The advertised project identifies finite-precision semantics, reproducibility testing, witness checking, and implementation-aware assurance as potential contributions. VeriTrace connects those ideas in one inspectable example. [Project description](https://www.findaphd.com/phds/project/school-of-engineering-phd-scholarships-verified-and-reproducible-execution-of-ai-models/?p198772)

## Connection to the literature

| Source | What it contributes to this prototype's framing |
| --- | --- |
| [Cordeiro et al., *Neural Network Verification is a Programming Language Challenge*, 2025](https://arxiv.org/html/2501.05867v2) | Distinguishes mathematical neural-network properties from implementation behavior and motivates explicit interfaces and trustworthy verification evidence. |
| [ONNX MatMul operator](https://onnx.ai/onnx/operators/onnx__MatMul.html) | Defines the imported operation at a higher level than our explicit scalar reduction schedule. |
| [Project-linked ONNX safety-related profile working group](https://github.com/ericjenn/working-groups/tree/ericjenn-srpwg-wg1/safety-related-profile) | Provides existing work to study and extend rather than treating a local profile as a replacement standard. |
| [SMT-LIB floating-point theory](https://smt-lib.org/theories-FloatingPoint.shtml) | Provides the logical vocabulary for explicit rounding and binary32 operations. |
| [Official Z3 Python API](https://z3prover.github.io/api/html/namespacez3py.html) | Exposes the floating-point expressions used to construct the verification query. |
| [VNN-LIB](https://www.vnnlib.org/) and its [standard repository](https://github.com/VNNLIB/VNNLIB-Standard) | Supply the current specification context; migration from the prototype's legacy property fragment is future work. |
| [VNN-COMP](https://vnn-comp.github.io/) | Provides a community and benchmark context for evaluating future verification tools. |

Sources were checked on 13 October 2026. In particular, the current VNN-LIB site reports release 2.0 and formal syntax and semantics. The limitations of the older language discussed in the January 2025 paper should not be presented as an unchanged description of current VNN-LIB.

The project-linked SONNX repository describes itself as work in progress and already includes formal operator specifications using Why3. A useful first research task is to compare its actual definitions, numerical assumptions, and open issues with this prototype. Merely naming a local JSON contract “SONNX” would not establish interoperability or a contribution. [Working-group README](https://raw.githubusercontent.com/ericjenn/working-groups/ericjenn-srpwg-wg1/safety-related-profile/README.md)

## Contribution boundary

What this repository supplies is an end-to-end integration experiment: an accepted graph, a concrete arithmetic contract, real and floating-point verification queries, replayable counterexample evidence, and host measurements. Floating-point non-associativity, SMT-based floating-point verification, and specification-driven assurance are established ideas.

A defensible future research contribution would require one or more of:

- A precise semantics for a useful operator subset, with a demonstrated correspondence to existing standards and implementations.
- An efficient method to establish numerical error bounds or contract preservation beyond small examples.
- A smaller trusted witness checker or independently checkable proof evidence.
- A benchmark suite that exposes implementation gaps under clearly defined execution contracts and includes independently validated results.

Novelty would need a fresh related-work review before being claimed.

## Staged research plan

These stages are proposed experiments, not completed work or a promised PhD schedule.

| Stage | Research question | Concrete experiment | Completion criterion |
| --- | --- | --- | --- |
| 1. Strengthen the foundation | Do the loader, interpreter, and solver encode the same accepted program? | Specify typing and shape rules; compare rounding against an independent reference; reject unsupported graphs and properties; handle all reachable nonfinite outcomes explicitly. | A documented conformance suite and clear obligations for every translation. |
| 2. Connect to SONNX and VNN-LIB | Which parts of the local contract correspond to current community specifications? | Pin versions of SONNX and VNN-LIB; compare operator definitions; adopt the current property parser; propose a narrowly scoped test or specification improvement. | A traceable mapping, identified gaps, and an upstream-ready contribution candidate. |
| 3. Establish deployment evidence | When does a deployment preserve the selected contract? | Run identical artifacts on actual x86-64 and Arm hosts; vary compiler, optimization, provider, and fusion settings one factor at a time; retain raw output bits. | Reproducible comparisons with every changed variable recorded; no machine claims based only on simulated schedules. |
| 4. Scale verification | Which combinations of static bounds and exact floating-point solving retain soundness while improving cost? | Add sound interval or affine error bounds, partition difficult domains, and reserve bit-precise reasoning for inconclusive regions. Compare against direct SMT and a real-only baseline. | Defined soundness argument plus measured solve rates, runtime, memory, and bound precision. |
| 5. Check implementations and evidence | How can trust in the translation and solver be reduced? | Verify generated C with ESBMC; study reusable Why3 operator contracts; evaluate another SMT solver and proof/witness checking options. | Independently validated counterexamples or checked proof evidence, with the remaining trusted base stated. |
| 6. Extend the computation | What extra semantics are needed for quantization and training? | Specify quantization scale, zero point, rounding, saturation, and accumulator width; later cover optimizer state, update order, and random choices in a bounded training example. | Separate validated semantics and benchmarks for each extension; no inference result relabeled as a training result. |

ESBMC supports verification of program assertions with floating-point reasoning, while Why3 provides a deductive verification platform and WhyML. These make them plausible next integration targets, but the prototype does not currently invoke them. [ESBMC](https://esbmc.github.io/), [Why3](https://why3.org/)

## A useful first experimental protocol

Freeze a model, property, input bit patterns, execution contract, and dependency versions before changing machines or backends. Include ordinary cases and deliberately difficult cases: cancellation, halfway rounding, subnormals, overflow boundaries, and ReLU decisions close to zero. Keep the constructed cancellation example separate from any realistic benchmark results.

For each execution, record the environment, runtime/provider, compiler and flags when applicable, input bits, output bits, and the first divergent intermediate operation when trace access exists. Repeat each configuration to distinguish variability within one configuration from differences across configurations. A deterministic backend may still implement a different contract.

For each verification task, record SAT, UNSAT, or unknown/timeout, elapsed time, resource limits, query size, and whether a witness was independently replayed. Report unsupported models, timed-out tasks, and failed replays explicitly in the denominator. Use maximum absolute error or ULP distance only with a defined domain and handling of zeros, subnormals, and exceptional values; also record whether the engineering decision actually changes.

An error-bound result needs a statement such as `|f_real(x) - f_impl(x)| ≤ ε` for every input in a specified domain. A decision-preservation result needs an additional margin argument. An equivalence result needs two fully specified executions. A shared output hash on sampled inputs establishes none of those universal claims by itself.

## Assurance scope

The project mentions aerospace assurance standards, but this repository does not interpret their clauses or claim compliance. A future standards-aligned effort would need access to the relevant editions, a concrete system and operational domain, allocated requirements, tool qualification considerations, and review with domain specialists. The prototype's narrower role is to make a numerical assumption and its verification consequences visible.
