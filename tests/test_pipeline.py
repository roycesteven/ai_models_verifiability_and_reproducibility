"""Exercise retained evidence and command-line behavior as a user would."""

import copy
from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import z3

from veritrace.__main__ import main
from veritrace.demo import replay_witness, run_demo
from veritrace.model import load_onnx
from veritrace.numeric import fraction_to_bits
from veritrace.properties import load_vnnlib
from veritrace.report import render_report


class ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.tags = []
        self.scripts = []
        self._script = None

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "script":
            self._script = {"attributes": dict(attrs), "content": ""}

    def handle_data(self, data):
        if self._script is not None:
            self._script["content"] += data

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append(self._script)
            self._script = None


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="veritrace-pipeline-")
        cls.out = Path(cls.temporary.name)
        cls.report = run_demo(cls.out, native=False, runtime=False)
        cls.model = load_onnx(cls.out / "model.onnx")
        cls.prop = load_vnnlib(cls.out / "property.vnnlib", cls.model.input_size, cls.model.output_size)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_report_and_witness_are_actual_generated_evidence(self):
        report = json.loads((self.out / "report.json").read_text())
        witness = json.loads((self.out / "witness.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual({key: value["status"] for key, value in report["verification"].items()},
                         {"real": "unsat", "canonical": "sat", "cancellation_first": "unsat"})
        self.assertEqual(report["native"]["status"], "skipped")
        self.assertEqual(report["onnx_runtime"]["status"], "skipped")
        self.assertEqual(len(report["samples"]), 9)
        self.assertTrue(witness["replay_validated"])
        self.assertEqual(replay_witness(self.model, self.prop, witness)["status"], "validated")
        for sample in report["samples"]:
            self.assertEqual(sample["canonical"]["output_bits"], [0])
            self.assertEqual(sample["cancellation_first"]["output_bits"], [sample["inputs"][1]])
            self.assertEqual(sample["real"], [sample["x"]])

    def test_saved_smt_queries_reproduce_property_and_overflow_results(self):
        expected = {"real.smt2": z3.unsat, "canonical.smt2": z3.sat,
                    "cancellation_first.smt2": z3.unsat,
                    "canonical-overflow.smt2": z3.unsat,
                    "cancellation_first-overflow.smt2": z3.unsat}
        for filename, status in expected.items():
            with self.subTest(filename=filename):
                solver = z3.Solver()
                solver.set(timeout=10000)
                solver.from_file(str(self.out / filename))
                self.assertEqual(solver.check(), status)

    def test_tampered_witness_inputs_outputs_identity_and_order_are_rejected(self):
        original = json.loads((self.out / "witness.json").read_text())
        changes = {
            "schema": "unknown-schema",
            "model_digest": "0" * 64,
            "property_digest": "0" * 64,
            "contract_digest": "0" * 64,
            "input_bits": [0, original["input_bits"][1], fraction_to_bits(1)],
            "output_bits": [fraction_to_bits(1)],
            "order": "cancellation_first",
        }
        for key, value in changes.items():
            with self.subTest(field=key):
                witness = copy.deepcopy(original)
                witness[key] = value
                with self.assertRaises(ValueError):
                    replay_witness(self.model, self.prop, witness)

    def test_malformed_witness_values_cannot_pass_integer_equality(self):
        original = json.loads((self.out / "witness.json").read_text())
        for values in ([False], [0.0], [-1], [2**32], [0x80000000],
                       [0x7F800000], ["0"], [], None):
            with self.subTest(output_bits=values):
                witness = copy.deepcopy(original)
                witness["output_bits"] = values
                with self.assertRaises((TypeError, ValueError)):
                    replay_witness(self.model, self.prop, witness)
        for malformed in ([], None, "witness"):
            with self.subTest(witness=malformed), self.assertRaises(ValueError):
                replay_witness(self.model, self.prop, malformed)

    def test_replay_cli_returns_success_for_original_and_error_for_tampering(self):
        arguments = ("replay", "--model", str(self.out / "model.onnx"),
                     "--property", str(self.out / "property.vnnlib"), "--witness")
        code, stdout, stderr = self.invoke(*arguments, str(self.out / "witness.json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "validated")
        witness = json.loads((self.out / "witness.json").read_text())
        witness["output_bits"] = [fraction_to_bits(1)]
        path = self.out / "tampered-witness.json"
        path.write_text(json.dumps(witness))
        code, stdout, stderr = self.invoke(*arguments, str(path))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("do not match independent execution", stderr)

    def test_verify_cli_solves_changed_property_and_exports_queries(self):
        arguments = ("verify", "--model", str(self.out / "model.onnx"), "--property")
        result_path = self.out / "changed-result.json"
        code, stdout, stderr = self.invoke(*arguments, str(self.out / "property.vnnlib"), "--out", str(result_path))
        self.assertEqual(code, 1, stderr)
        initial = json.loads(stdout)
        self.assertEqual(initial["status"], "sat")
        witness_path = result_path.with_suffix(".witness.json")
        self.assertEqual(Path(initial["witness_file"]), witness_path)
        code, stdout, stderr = self.invoke("replay", "--model", str(self.out / "model.onnx"),
                                          "--property", str(self.out / "property.vnnlib"),
                                          "--witness", str(witness_path))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "validated")
        changed = self.out / "changed-property.vnnlib"
        changed.write_text((self.out / "property.vnnlib").read_text().replace("(< Y_0 (/ 1 2))", "(< Y_0 0)"))
        code, stdout, stderr = self.invoke(*arguments, str(changed), "--out", str(result_path))
        self.assertEqual(code, 0, stderr)
        result = json.loads(stdout)
        self.assertEqual(result["status"], "unsat")
        self.assertFalse(result["vacuous"])
        self.assertFalse(result["proof_certificate"])
        self.assertFalse(witness_path.exists())
        self.assertEqual(json.loads(result_path.read_text()), result)
        self.assertTrue(result_path.with_suffix(".smt2").exists())
        self.assertTrue(result_path.with_suffix(".overflow.smt2").exists())
        solver = z3.Solver()
        solver.set(timeout=10000)
        solver.from_file(str(result_path.with_suffix(".smt2")))
        self.assertEqual(solver.check(), z3.unsat)
        unknown = {"status": "unknown", "semantics": "float32", "order": "canonical",
                   "reason": "test timeout", "smt2": "; Inconclusive test run\n"}
        # Exercise a later UNKNOWN run over the same output path. Its artifacts
        # must not retain a prior overflow result or counterexample.
        witness_path.write_text("stale witness")
        with patch("veritrace.verification.verify", return_value=unknown):
            code, stdout, stderr = self.invoke(*arguments, str(changed), "--out", str(result_path))
        self.assertEqual(code, 2, stderr)
        self.assertEqual(json.loads(stdout)["status"], "unknown")
        self.assertFalse(witness_path.exists())
        self.assertFalse(result_path.with_suffix(".overflow.smt2").exists())
        self.assertIn("Inconclusive", result_path.with_suffix(".smt2").read_text())

    def test_inconclusive_demo_rerun_clears_previous_evidence_preserving_other_files(self):
        with tempfile.TemporaryDirectory(prefix="veritrace-rerun-") as directory:
            out = Path(directory)
            for filename in ("witness.json", "report.json", "report.html", "canonical-overflow.smt2",
                             "cancellation_first-overflow.smt2"):
                (out / filename).write_bytes((self.out / filename).read_bytes())
            for filename in ("native-O0", "native-O2"):
                (out / filename).write_text("stale executable")
            (out / "user-notes.txt").write_text("Keep this file")

            def inconclusive(model, prop, *, semantics, order, timeout_ms):
                return {"status": "unknown", "semantics": semantics, "order": order,
                        "reason": "test timeout", "smt2": "; Inconclusive test run\n"}

            with patch("veritrace.demo.verify", side_effect=inconclusive):
                report = run_demo(out, native=False, runtime=False)
            self.assertEqual(report["status"], "incomplete")
            self.assertIsNone(report["witness"])
            self.assertEqual(json.loads((out / "report.json").read_text())["status"], "incomplete")
            for filename in ("witness.json", "canonical-overflow.smt2", "cancellation_first-overflow.smt2",
                             "native-O0", "native-O2"):
                self.assertFalse((out / filename).exists(), filename)
            self.assertTrue((out / "native.c").exists())
            self.assertEqual((out / "user-notes.txt").read_text(), "Keep this file")
            parser = ScriptCollector()
            parser.feed((out / "report.html").read_text())
            embedded = next(script for script in parser.scripts if script["attributes"].get("id") == "report-data")
            self.assertEqual(json.loads(embedded["content"])["status"], "incomplete")
            self.assertIsNone(json.loads(embedded["content"])["witness"])

    def test_html_embeds_exact_json_without_executable_metadata(self):
        report = copy.deepcopy(self.report)
        injected = '</script><img src=x onerror="alert(1)"><script>alert(2)</script>&\u2028\u2029'
        report["title"] = injected
        report["limitations"].append(injected)
        parser = ScriptCollector()
        parser.feed(render_report(report))
        self.assertEqual(len(parser.scripts), 2)
        self.assertNotIn("img", parser.tags)
        data = [script for script in parser.scripts if script["attributes"].get("id") == "report-data"]
        self.assertEqual(len(data), 1)
        encoded = data[0]["content"]
        self.assertNotIn("<", encoded)
        self.assertNotIn(">", encoded)
        self.assertNotIn("&", encoded)
        self.assertEqual(json.loads(encoded), report)

    def test_compare_cli_reports_equal_then_drift_with_correct_exit_codes(self):
        original = self.out / "manifest.json"
        code, stdout, stderr = self.invoke("compare", str(original), str(original))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "identical")
        from veritrace.evidence import digest
        manifest = json.loads(original.read_text())
        manifest["payload"]["backends"]["software:canonical"][0][0] = 1
        manifest["payload_sha256"] = digest(manifest["payload"])
        changed = self.out / "drift-manifest.json"
        changed.write_text(json.dumps(manifest))
        code, stdout, stderr = self.invoke("compare", str(original), str(changed))
        self.assertEqual(code, 1, stderr)
        self.assertEqual(json.loads(stdout)["status"], "drift")


if __name__ == "__main__":
    unittest.main()
