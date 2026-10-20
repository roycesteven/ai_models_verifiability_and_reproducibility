"""Command-line entry points. SAT means a counterexample to the safety requirement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veritrace", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Build the entire experiment and offline report")
    demo.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--skip-native", action="store_true", help="Skip measured C runs")
    demo.add_argument("--skip-runtime", action="store_true", help="Skip optional ONNX Runtime runs")
    demo.add_argument("--timeout-ms", type=int, default=10000)
    verification = commands.add_parser("verify", help="Solve the unsafe-output query in a supported property")
    verification.add_argument("--model", type=Path, required=True)
    verification.add_argument("--property", type=Path, required=True)
    verification.add_argument("--order", choices=["canonical", "cancellation_first"], default="canonical")
    verification.add_argument("--semantics", choices=["float32", "real"], default="float32")
    verification.add_argument("--timeout-ms", type=int, default=10000)
    verification.add_argument("--out", type=Path, help="Write .json result, SMT queries, and a replayable .witness.json for float32 SAT")
    replay = commands.add_parser("replay", help="Check a counterexample with independent software arithmetic")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--property", type=Path, required=True)
    replay.add_argument("--witness", type=Path, required=True)
    comparison = commands.add_parser("compare", help="Compare observation manifests, including from other machines")
    comparison.add_argument("first", type=Path)
    comparison.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    try:
        if getattr(args, "timeout_ms", 1) <= 0:
            raise ValueError("--timeout-ms must be positive")
        if args.command == "demo":
            from .demo import run_demo
            report = run_demo(args.out, native=not args.skip_native, runtime=not args.skip_runtime,
                              timeout_ms=args.timeout_ms)
            print("VeriTrace — implementation-aware neural network verification")
            for name, check in report["verification"].items():
                meaning = {"sat": "requirement violated", "unsat": "requirement holds within scope"}.get(check["status"], "inconclusive")
                print(f"  {name:20} {check['status'].upper():7} {meaning}")
            print(f"  Native C: {report['native']['status']}; ONNX Runtime: {report['onnx_runtime']['status']}")
            print(f"  Experiment: {report['status']}")
            print(f"  Open {args.out.resolve() / 'report.html'}")
            return 0 if report["status"] == "complete" else 2
        if args.command == "compare":
            from .evidence import compare_manifests
            result = compare_manifests(json.loads(args.first.read_text()), json.loads(args.second.read_text()))
            print(json.dumps(result, indent=2))
            return {"identical": 0, "drift": 1, "incompatible": 2}[result["status"]]
        from .model import load_onnx
        from .properties import load_vnnlib
        model = load_onnx(args.model)
        prop = load_vnnlib(args.property, model.input_size, model.output_size)
        if args.command == "replay":
            from .demo import replay_witness
            result = replay_witness(model, prop, json.loads(args.witness.read_text()))
            print(json.dumps(result, indent=2))
            return 0
        from .verification import verify
        from .evidence import write_json
        if args.out and args.out.suffix != ".json":
            raise ValueError("--out must end in .json")
        result = verify(model, prop, semantics=args.semantics, order=args.order, timeout_ms=args.timeout_ms)
        smt2 = result.pop("smt2", "")
        overflow_smt2 = result.pop("overflow_smt2", "")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".overflow.smt2", ".witness.json"):
                args.out.with_suffix(suffix).unlink(missing_ok=True)
            if result["status"] == "sat" and args.semantics == "float32":
                from .demo import make_witness
                witness_path = args.out.with_suffix(".witness.json")
                write_json(witness_path, make_witness(model, prop, result))
                result["witness_file"] = str(witness_path)
            write_json(args.out, result)
            args.out.with_suffix(".smt2").write_text(smt2)
            if overflow_smt2:
                args.out.with_suffix(".overflow.smt2").write_text(overflow_smt2)
        print(json.dumps(result, indent=2))
        return {"unsat": 0, "sat": 1}.get(result["status"], 2)
    except (ValueError, OSError, KeyError, TypeError, OverflowError, ImportError) as exc:
        print(f"veritrace: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
