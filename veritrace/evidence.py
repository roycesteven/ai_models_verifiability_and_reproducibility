"""Canonical serialization and portable, integrity-checked observation manifests.

Hashes detect accidental changes; they are not signatures or proof certificates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .numeric import validate_bits


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    packages = {}
    for name in ("numpy", "onnx", "onnxruntime", "z3-solver"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
    }


def make_manifest(model_digest: str, contract: dict, inputs: list, backends: dict) -> dict:
    payload = {
        "model_digest": model_digest,
        "contract": contract,
        "inputs": inputs,
        "backends": backends,
    }
    return {
        "schema": "veritrace-observations-v1",
        "environment": environment(),
        "payload": payload,
        "payload_sha256": digest(payload),
        "scope": "Observed vectors on this host only; no universal equivalence claim.",
    }


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("Observation manifest must be a JSON object")
    if manifest.get("schema") != "veritrace-observations-v1":
        raise ValueError("Unsupported observation manifest schema")
    if set(manifest) != {"schema", "environment", "payload", "payload_sha256", "scope"}:
        raise ValueError("Observation manifest has missing or unsupported fields")
    if not isinstance(manifest["environment"], dict) or not isinstance(manifest["scope"], str):
        raise ValueError("Manifest environment must be an object and scope must be a string")
    payload = manifest["payload"]
    if not isinstance(payload, dict) or set(payload) != {"model_digest", "contract", "inputs", "backends"}:
        raise ValueError("Manifest payload must be an object with model_digest, contract, inputs, and backends")
    try:
        payload_digest = digest(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Manifest payload must contain only finite JSON values") from error
    if payload_digest != manifest["payload_sha256"]:
        raise ValueError("Manifest integrity check failed: payload hash differs")
    model_digest = payload["model_digest"]
    if not isinstance(model_digest, str) or len(model_digest) != 64 or any(char not in "0123456789abcdef" for char in model_digest):
        raise ValueError("Manifest model_digest must be a lowercase SHA-256 digest")
    if not isinstance(payload["contract"], dict) or not payload["contract"] or any(not isinstance(key, str) for key in payload["contract"]):
        raise ValueError("Manifest contract must be a nonempty JSON object with string keys")
    inputs = payload["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Manifest inputs must be a nonempty list of rows")
    input_width = None
    for row in inputs:
        if not isinstance(row, list) or not row:
            raise ValueError("Manifest input rows must be nonempty lists")
        if input_width is None:
            input_width = len(row)
        elif len(row) != input_width:
            raise ValueError("Manifest input rows have inconsistent widths")
        for bits in row:
            try:
                validate_bits(bits)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Manifest inputs must contain finite canonical binary32 uint32 patterns: {error}") from error
    if not isinstance(payload["backends"], dict) or not payload["backends"]:
        raise ValueError("Manifest backends must be a nonempty JSON object")
    output_width = None
    for name, rows in payload["backends"].items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Manifest backend names must be nonempty strings")
        if not isinstance(rows, list) or len(rows) != len(inputs):
            raise ValueError(f"Invalid observations for {name}")
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ValueError(f"Output rows for {name} must be nonempty lists")
            if output_width is None:
                output_width = len(row)
            elif len(row) != output_width:
                raise ValueError(f"Inconsistent output widths for {name}")
            for bits in row:
                # Runtime observations may expose NaN/Inf or -0 even though the
                # specified execution profile forbids them. Preserve their bits.
                if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
                    raise ValueError("Manifest outputs must contain uint32 bit patterns")
    return payload


def compare_manifests(first: dict, second: dict) -> dict:
    a, b = validate_manifest(first), validate_manifest(second)
    for field in ("model_digest", "contract", "inputs"):
        if a[field] != b[field]:
            return {"status": "incompatible", "reason": f"Different {field}"}
    if a["backends"].keys() != b["backends"].keys():
        return {"status": "incompatible", "reason": "Backend sets differ; rerun with matching options"}
    differences = []
    for backend, rows in a["backends"].items():
        if rows != b["backends"][backend]:
            differences.append({"backend": backend, "first": rows, "second": b["backends"][backend]})
    return {
        "status": "drift" if differences else "identical",
        "differences": differences,
        "compared_backends": sorted(a["backends"]),
        "vectors": len(a["inputs"]),
        "scope": "Bitwise comparison of recorded outputs on the same input vectors only.",
    }
