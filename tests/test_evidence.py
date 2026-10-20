import copy
import unittest

from veritrace.evidence import compare_manifests, digest, make_manifest, validate_manifest
from veritrace.numeric import float_to_bits


class EvidenceTests(unittest.TestCase):
    def manifest(self):
        return make_manifest(digest("fixture"), {"rounding": "nearest-even"},
                             [[float_to_bits(1), float_to_bits(0.5), float_to_bits(1)]],
                             {"software:canonical": [[0]], "software:cancellation_first": [[float_to_bits(0.5)]]})

    def rehash(self, manifest):
        manifest["payload_sha256"] = digest(manifest["payload"])
        return manifest

    def test_identical_observations_can_come_from_different_hosts(self):
        first, second = self.manifest(), self.manifest()
        second["environment"]["machine"] = "independent-test-machine"
        result = compare_manifests(first, second)
        self.assertEqual(result["status"], "identical")
        self.assertEqual(result["vectors"], 1)
        self.assertEqual(result["differences"], [])

    def test_checksum_detects_changed_input_or_output(self):
        for field in ("inputs", "backends"):
            with self.subTest(field=field):
                manifest = self.manifest()
                if field == "inputs":
                    manifest["payload"][field][0][0] = 0
                else:
                    manifest["payload"][field]["software:canonical"][0][0] = 1
                with self.assertRaisesRegex(ValueError, "integrity"):
                    validate_manifest(manifest)

    def test_valid_independent_observations_report_one_bit_drift(self):
        first = self.manifest()
        second = copy.deepcopy(first)
        second["payload"]["backends"]["software:canonical"][0][0] = 1
        self.rehash(second)
        result = compare_manifests(first, second)
        self.assertEqual(result["status"], "drift")
        self.assertEqual([entry["backend"] for entry in result["differences"]], ["software:canonical"])
        self.assertEqual(result["differences"][0]["second"], [[1]])

    def test_different_comparison_context_is_incompatible(self):
        for field, value in (("model_digest", digest("different")),
                             ("contract", {"rounding": "toward-zero"}),
                             ("inputs", [[0, 0, 0]])):
            with self.subTest(field=field):
                first, second = self.manifest(), self.manifest()
                second["payload"][field] = value
                result = compare_manifests(first, self.rehash(second))
                self.assertEqual(result["status"], "incompatible")
                self.assertIn(field, result["reason"])
        first, second = self.manifest(), self.manifest()
        del second["payload"]["backends"]["software:cancellation_first"]
        self.assertEqual(compare_manifests(first, self.rehash(second))["status"], "incompatible")

    def test_invalid_observation_structure_cannot_be_blessed_by_rehashing(self):
        for rows in ([], [[]], [[0], [1]], [[True]], [[-1]], [[2**32]], [["0x0"]]):
            with self.subTest(rows=rows):
                manifest = self.manifest()
                manifest["payload"]["backends"]["software:canonical"] = rows
                with self.assertRaises(ValueError):
                    validate_manifest(self.rehash(manifest))

    def test_schema_and_canonical_serialization(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            digest({"number": float("nan")})
        manifest = self.manifest()
        manifest["schema"] = "unknown-schema"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_rejects_malformed_manifest_and_payload_fields_with_value_error(self):
        for value in (None, [], "manifest", 1, {}):
            with self.subTest(manifest=value), self.assertRaises(ValueError):
                validate_manifest(value)
        for field in self.manifest():
            manifest = self.manifest()
            del manifest[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_manifest(manifest)
        for payload in (None, [], "payload", 1, {}, {"model_digest": digest("fixture")}):
            manifest = self.manifest()
            manifest["payload"] = payload
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_manifest(self.rehash(manifest))
        for field in self.manifest()["payload"]:
            manifest = self.manifest()
            del manifest["payload"][field]
            with self.subTest(missing_payload=field), self.assertRaises(ValueError):
                validate_manifest(self.rehash(manifest))
        for field, value in (("model_digest", []), ("model_digest", "invalid"),
                             ("contract", []), ("contract", {}), ("backends", []),
                             ("backends", {}), ("backends", {"": [[0]]}),
                             ("backends", {"backend": [None]}),
                             ("backends", {"backend": "invalid"})):
            manifest = self.manifest()
            manifest["payload"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_manifest(self.rehash(manifest))

    def test_rejects_invalid_input_shapes_and_binary32_patterns(self):
        invalid_inputs = (None, [], {}, [None], [[]], [0], "rows", [[0, 0], [0]])
        invalid_inputs += tuple([[bits]] for bits in
                                (-1, 2**32, True, 0.0, "0", None, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000))
        for inputs in invalid_inputs:
            manifest = self.manifest()
            manifest["payload"]["inputs"] = inputs
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                validate_manifest(self.rehash(manifest))

    def test_output_width_is_consistent_across_backends(self):
        manifest = self.manifest()
        manifest["payload"]["backends"]["software:canonical"] = [[0, 0]]
        with self.assertRaisesRegex(ValueError, "widths"):
            validate_manifest(self.rehash(manifest))

    def test_nonfinite_and_negative_zero_output_observations_are_preserved(self):
        for bits in (0x7F800000, 0xFF800000, 0x7FC00000, 0x80000000):
            manifest = self.manifest()
            manifest["payload"]["backends"]["software:canonical"] = [[bits]]
            payload = validate_manifest(self.rehash(manifest))
            self.assertEqual(payload["backends"]["software:canonical"], [[bits]])


if __name__ == "__main__":
    unittest.main()
