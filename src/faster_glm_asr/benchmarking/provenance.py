"""Fail-closed local model and machine-readiness provenance bindings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_MANIFEST_SCHEMA = "cache-file-manifest-v1"
MODEL_MANIFEST_KIND = "model-snapshot"
MODEL_SNAPSHOT_FILE_COUNT = 10
MODEL_BINDING_SCHEMA = "asr-model-snapshot-binding-v1"
READINESS_REPORT_SCHEMA = "rtx3090-readiness-v1"
READINESS_BINDING_SCHEMA = "asr-readiness-binding-v1"
READINESS_CONTRACT_PATH = Path("configs/environments/rtx3090-ddp-v1.json")
READINESS_TOOL_PATH = Path("tools/preflight_rtx3090.py")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MODEL_BINDING_KEYS = {
    "schema_version",
    "repo_id",
    "revision",
    "file_count",
    "total_size_bytes",
    "manifest_sha256",
    "files_aggregate_sha256",
    "contract_sha256",
}
READINESS_BINDING_KEYS = {
    "schema_version",
    "readiness_schema_version",
    "report_sha256",
    "phase",
    "status",
    "training_ready",
    "day_rental_eligible",
    "contract_sha256",
    "tool_sha256",
    "gpu_count",
    "gpu_models",
    "gpu_identity_sha256",
    "topology_evidence_sha256",
    "clock_power_evidence_sha256",
    "hardware_evidence_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
        raw = resolved.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {label}: {path}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, resolved


def _safe_snapshot_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("model snapshot manifest path is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"model snapshot manifest path is unsafe: {value!r}")
    return path.as_posix()


def validate_model_snapshot_binding(
    value: Any,
    *,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != MODEL_BINDING_KEYS:
        raise ValueError("model snapshot binding schema is incomplete")
    if value.get("schema_version") != MODEL_BINDING_SCHEMA:
        raise ValueError("model snapshot binding schema is unsupported")
    if expected_repo_id is not None and value.get("repo_id") != expected_repo_id:
        raise ValueError("model snapshot binding repo_id mismatch")
    if expected_revision is not None and value.get("revision") != expected_revision:
        raise ValueError("model snapshot binding revision mismatch")
    if not isinstance(value.get("repo_id"), str) or not value["repo_id"]:
        raise ValueError("model snapshot binding repo_id is invalid")
    if (
        not isinstance(value.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value["revision"]) is None
    ):
        raise ValueError("model snapshot binding revision is not immutable")
    if value.get("file_count") != MODEL_SNAPSHOT_FILE_COUNT:
        raise ValueError(f"model snapshot binding must contain {MODEL_SNAPSHOT_FILE_COUNT} files")
    total_size = value.get("total_size_bytes")
    if not isinstance(total_size, int) or isinstance(total_size, bool) or total_size <= 0:
        raise ValueError("model snapshot binding total_size_bytes is invalid")
    for key in (
        "manifest_sha256",
        "files_aggregate_sha256",
        "contract_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA256_RE.fullmatch(value[key]) is None:
            raise ValueError(f"model snapshot binding {key} is invalid")
    return dict(value)


def validate_model_snapshot(
    root: Path,
    manifest_path: Path,
    *,
    expected_repo_id: str,
    expected_revision: str,
) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("model snapshot root cannot be a symbolic link")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"model snapshot root is missing: {root}") from exc
    if not root.is_dir():
        raise ValueError("model snapshot root must be a directory")
    manifest, resolved_manifest = _load_json_object(manifest_path, label="model snapshot manifest")
    try:
        resolved_manifest.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("model snapshot manifest must be outside the snapshot root")

    required_root_keys = {
        "schema_version",
        "kind",
        "created_at",
        "root_label",
        "metadata",
        "file_count",
        "total_size_bytes",
        "files",
    }
    if set(manifest) != required_root_keys:
        raise ValueError("model snapshot manifest root schema is incomplete")
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA:
        raise ValueError("model snapshot manifest schema is unsupported")
    if manifest.get("kind") != MODEL_MANIFEST_KIND:
        raise ValueError("model snapshot manifest kind mismatch")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("model snapshot manifest metadata is missing")
    if metadata.get("repo_id") != expected_repo_id:
        raise ValueError("model snapshot manifest repo_id mismatch")
    if metadata.get("revision") != expected_revision:
        raise ValueError("model snapshot manifest revision mismatch")
    contract_sha256 = metadata.get("contract_sha256")
    if not isinstance(contract_sha256, str) or SHA256_RE.fullmatch(contract_sha256) is None:
        raise ValueError("model snapshot manifest contract_sha256 is invalid")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != MODEL_SNAPSHOT_FILE_COUNT:
        raise ValueError(
            f"model snapshot manifest must list exactly {MODEL_SNAPSHOT_FILE_COUNT} files"
        )
    if manifest.get("file_count") != MODEL_SNAPSHOT_FILE_COUNT:
        raise ValueError("model snapshot manifest file_count mismatch")
    expected: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("model snapshot manifest file entry schema is invalid")
        relative = _safe_snapshot_path(item.get("path"))
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if relative in expected:
            raise ValueError(f"duplicate model snapshot manifest path: {relative}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"model snapshot manifest size is invalid: {relative}")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"model snapshot manifest SHA256 is invalid: {relative}")
        expected[relative] = {
            "size_bytes": size,
            "sha256": digest,
        }
    if [item["path"] for item in files] != sorted(expected):
        raise ValueError("model snapshot manifest files must be sorted by path")
    declared_total = sum(record["size_bytes"] for record in expected.values())
    if manifest.get("total_size_bytes") != declared_total or declared_total <= 0:
        raise ValueError("model snapshot manifest total_size_bytes mismatch")

    actual_paths: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"model snapshot symbolic link rejected: {path}")
        if path.is_file():
            actual_paths[path.relative_to(root).as_posix()] = path
    if set(actual_paths) != set(expected):
        missing = sorted(set(expected) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(expected))
        raise ValueError(
            f"model snapshot file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    actual: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected):
        path = actual_paths[relative]
        record = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if record != expected[relative]:
            raise ValueError(f"model snapshot byte mismatch: {relative}")
        actual[relative] = record

    binding = {
        "schema_version": MODEL_BINDING_SCHEMA,
        "repo_id": expected_repo_id,
        "revision": expected_revision,
        "file_count": len(actual),
        "total_size_bytes": sum(item["size_bytes"] for item in actual.values()),
        "manifest_sha256": sha256_file(resolved_manifest),
        "files_aggregate_sha256": canonical_sha256(actual),
        "contract_sha256": contract_sha256,
    }
    return validate_model_snapshot_binding(
        binding,
        expected_repo_id=expected_repo_id,
        expected_revision=expected_revision,
    )


def validate_readiness_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != READINESS_BINDING_KEYS:
        raise ValueError("readiness binding schema is incomplete")
    if value.get("schema_version") != READINESS_BINDING_SCHEMA:
        raise ValueError("readiness binding schema is unsupported")
    if value.get("readiness_schema_version") != READINESS_REPORT_SCHEMA:
        raise ValueError("readiness report schema is unsupported")
    if (
        value.get("phase") != "full"
        or value.get("status") != "pass"
        or value.get("training_ready") is not True
        or value.get("day_rental_eligible") is not True
    ):
        raise ValueError("readiness binding is not a passing full-phase decision")
    if value.get("gpu_count") != 4:
        raise ValueError("readiness binding must describe four GPUs")
    gpu_models = value.get("gpu_models")
    if (
        not isinstance(gpu_models, list)
        or not gpu_models
        or any(not isinstance(name, str) or not name for name in gpu_models)
    ):
        raise ValueError("readiness binding GPU models are invalid")
    for key in (
        "report_sha256",
        "contract_sha256",
        "tool_sha256",
        "gpu_identity_sha256",
        "topology_evidence_sha256",
        "clock_power_evidence_sha256",
        "hardware_evidence_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA256_RE.fullmatch(value[key]) is None:
            raise ValueError(f"readiness binding {key} is invalid")
    return dict(value)


def validate_readiness_report(
    report_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    report, resolved_report = _load_json_object(report_path, label="RTX 3090 readiness report")
    repository_root = repository_root.resolve()
    contract_path = (repository_root / READINESS_CONTRACT_PATH).resolve()
    tool_path = (repository_root / READINESS_TOOL_PATH).resolve()
    if not contract_path.is_file() or not tool_path.is_file():
        raise FileNotFoundError(
            "readiness validation requires the source checkout contract and preflight tool"
        )
    if report.get("schema_version") != READINESS_REPORT_SCHEMA:
        raise ValueError("readiness report schema is unsupported")
    if report.get("phase") != "full":
        raise ValueError("readiness report must be a full-phase report")
    overall = report.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("readiness report overall decision is missing")
    gates = report.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("readiness report gates are missing")
    contract = report.get("contract")
    tool = report.get("tool")
    actual_contract_sha256 = sha256_file(contract_path)
    actual_tool_sha256 = sha256_file(tool_path)
    if not isinstance(contract, dict) or contract.get("sha256") != actual_contract_sha256:
        raise ValueError("readiness report contract hash does not match this checkout")
    if not isinstance(tool, dict) or tool.get("sha256") != actual_tool_sha256:
        raise ValueError("readiness report tool hash does not match this checkout")

    contract_document, _ = _load_json_object(contract_path, label="RTX 3090 environment contract")
    spec = importlib.util.spec_from_file_location(
        "_faster_glm_asr_pinned_readiness_tool",
        tool_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the pinned readiness evaluator")
    readiness_tool = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(readiness_tool)
        recomputed_gates = readiness_tool.evaluate_full_gates(report, contract_document)
        recomputed_overall = readiness_tool._report_outcome(recomputed_gates, "full")
    except Exception as exc:
        raise ValueError("readiness report evidence cannot be reevaluated") from exc
    if gates != recomputed_gates:
        raise ValueError("readiness report gates do not match recomputed evidence")
    if overall != recomputed_overall:
        raise ValueError("readiness report overall decision does not match recomputed gates")
    if (
        overall.get("status") != "pass"
        or overall.get("ready") is not True
        or overall.get("training_ready") is not True
        or overall.get("day_rental_eligible") is not True
        or overall.get("failed_gates") != []
        or overall.get("warning_gates") != []
    ):
        raise ValueError("readiness report must be a warning-free passing decision")

    hardware = report.get("hardware")
    before = hardware.get("gpu_probe_before") if isinstance(hardware, dict) else None
    gpus = before.get("gpus") if isinstance(before, dict) else None
    if before is None or before.get("ok") is not True or not isinstance(gpus, list):
        raise ValueError("readiness report GPU evidence is missing")
    if len(gpus) != 4 or any(not isinstance(gpu, dict) for gpu in gpus):
        raise ValueError("readiness report must contain four GPU records")
    indices = [gpu.get("index") for gpu in gpus]
    if sorted(indices) != [0, 1, 2, 3]:
        raise ValueError("readiness report GPU indices are invalid")
    required_identity = (
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "compute_capability",
    )
    if any(any(gpu.get(key) in {None, ""} for key in required_identity) for gpu in gpus):
        raise ValueError("readiness report GPU identity is incomplete")
    identity_evidence = [
        {
            key: gpu.get(key)
            for key in (
                "index",
                "name",
                "uuid",
                "pci_bus_id",
                "driver_version",
                "memory_total_mib",
                "compute_capability",
            )
        }
        for gpu in sorted(gpus, key=lambda item: int(item["index"]))
    ]
    topology_evidence = {
        "gpu_pcie_links": [gpu.get("pcie_link") for gpu in gpus],
        "pcie_query": before.get("pcie_query"),
        "cpu_topology": hardware.get("cpu_topology"),
        "gpu_topology": hardware.get("topology"),
        "topology_p2p": hardware.get("topology_p2p"),
        "distributed_p2p": (
            report.get("distributed", {}).get("payload", {}).get("p2p")
            if isinstance(report.get("distributed"), dict)
            else None
        ),
    }
    after = hardware.get("gpu_probe_after")
    clock_power_evidence = {
        "before": [
            {
                key: gpu.get(key)
                for key in (
                    "index",
                    "temperature_c",
                    "pstate",
                    "power_draw_w",
                    "power_limit_w",
                    "utilization_percent",
                )
            }
            for gpu in gpus
        ],
        "after": after.get("gpus") if isinstance(after, dict) else None,
        "thermal": report.get("thermal"),
    }
    hardware_hashes = {
        "gpu_identity_sha256": canonical_sha256(identity_evidence),
        "topology_evidence_sha256": canonical_sha256(topology_evidence),
        "clock_power_evidence_sha256": canonical_sha256(clock_power_evidence),
    }
    binding = {
        "schema_version": READINESS_BINDING_SCHEMA,
        "readiness_schema_version": READINESS_REPORT_SCHEMA,
        "report_sha256": sha256_file(resolved_report),
        "phase": "full",
        "status": "pass",
        "training_ready": True,
        "day_rental_eligible": True,
        "contract_sha256": actual_contract_sha256,
        "tool_sha256": actual_tool_sha256,
        "gpu_count": len(gpus),
        "gpu_models": sorted({str(gpu["name"]) for gpu in gpus}),
        **hardware_hashes,
        "hardware_evidence_sha256": canonical_sha256(hardware_hashes),
    }
    return validate_readiness_binding(binding)
