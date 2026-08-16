#!/usr/bin/env python3
"""Fail-closed readiness checks for a single host with four RTX 3090 GPUs.

``host`` is the short hourly-billing gate. ``full`` adds exact package pins,
cache manifests, pre/post environment freezes, four-rank NCCL and BF16 work,
thermal sampling, and post-run health checks. Raw GPU UUIDs are retained exactly
in the private report; the tool never invents anonymized identifiers.

The tool does not download packages, models, or datasets. Optional endpoint
checks use HTTP HEAD only. It never dumps the process environment, credentials,
shell history, SSH configuration, or access tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX development host only
    resource = None  # type: ignore[assignment]


SCHEMA_VERSION = "rtx3090-readiness-v1"
MANIFEST_SCHEMA = "cache-file-manifest-v1"
CONTRACT_SCHEMA = "gpu-readiness-contract-v1"
PROVIDER_HEALTH_SCHEMA = "provider-gpu-health-evidence-v1"
WORKER_SENTINEL = "RTX3090_WORKER_JSON="
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs/environments/rtx3090-ddp-v1.json"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts/private/rtx3090-readiness"
REQUIRED_CORE_VERSIONS = {
    "torch": "2.10.0",
    "triton": "3.6.0",
    "transformers": "5.14.0",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.5.0",
    "safetensors": "0.8.0",
}
REQUIRED_BUILD_TOOL_VERSIONS = {
    "setuptools": "80.10.2",
    "wheel": "0.46.3",
}
WHEEL_CLOSURE_TARGET = {
    "implementation_name": "cpython",
    "implementation_version": "3.11.0",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.11.0",
    "python_version": "3.11",
    "sys_platform": "linux",
}
WHEEL_METADATA_MAX_BYTES = 4 * 1024 * 1024
PACKAGE_DISTRIBUTIONS = (
    "torch",
    "triton",
    "transformers",
    "tokenizers",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "datasets",
    "nvidia-ml-py",
    "setuptools",
    "wheel",
)
GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "memory.total",
    "memory.used",
    "driver_version",
    "pci.bus_id",
    "compute_cap",
    "temperature.gpu",
    "utilization.gpu",
    "pstate",
    "power.draw",
    "power.limit",
)
PCIE_QUERY_FIELDS = {
    "pcie.link.gen.current": "gen_current",
    "pcie.link.gen.max": "gen_max",
    "pcie.link.width.current": "width_current",
    "pcie.link.width.max": "width_max",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    nonce = uuid.uuid4().hex
    return DEFAULT_ARTIFACT_ROOT / f"{stamp}-pid{os.getpid()}-{nonce}" / "report.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported contract schema: {contract.get('schema_version')!r}")
    if contract.get("environment_id") != "rtx3090-ddp-v1":
        raise ValueError("unexpected environment_id")
    if contract.get("packages", {}).get("tokenizers") != "0.22.2":
        raise ValueError("contract must explicitly pin tokenizers==0.22.2")
    mismatches = {
        name: contract.get("packages", {}).get(name)
        for name, expected in REQUIRED_CORE_VERSIONS.items()
        if contract.get("packages", {}).get(name) != expected
    }
    if mismatches:
        raise ValueError(f"fixed core version mismatch: {sorted(mismatches)}")
    build_tool_mismatches = {
        name: contract.get("packages", {}).get(name)
        for name, expected in REQUIRED_BUILD_TOOL_VERSIONS.items()
        if contract.get("packages", {}).get(name) != expected
    }
    if build_tool_mismatches:
        raise ValueError(f"fixed build-tool version mismatch: {sorted(build_tool_mismatches)}")
    distributed = contract.get("distributed", {})
    if distributed.get("backend") != "nccl" or distributed.get("world_size") != 4:
        raise ValueError("contract must require NCCL with world_size=4")
    if distributed.get("minimum_stress_seconds", 0) < 300:
        raise ValueError("contract must require at least 300 seconds of GPU stress")
    if distributed.get("minimum_thermal_samples_per_gpu", 0) < 240:
        raise ValueError("contract must require at least 240 thermal samples per GPU")
    if contract.get("target", {}).get("nccl_min_bus_bandwidth_gb_per_s") != 1.0:
        raise ValueError("contract must keep the 1.0 GB/s NCCL functional sanity floor")
    health = contract.get("xid_health_evidence", {})
    if health.get("lookback_seconds", 0) < 900:
        raise ValueError("contract Xid lookback must be at least 900 seconds")
    if health.get("provider_schema_version") != PROVIDER_HEALTH_SCHEMA:
        raise ValueError("contract provider health evidence schema mismatch")
    for key in (
        "max_file_bytes",
        "max_future_skew_seconds",
        "max_reporting_delay_seconds",
        "max_finalization_lead_seconds",
    ):
        if not isinstance(health.get(key), int) or isinstance(health.get(key), bool):
            raise ValueError(f"contract xid_health_evidence.{key} must be an integer")
        if health[key] <= 0:
            raise ValueError(f"contract xid_health_evidence.{key} must be positive")
    rental = contract.get("rental_policy", {})
    if (
        rental.get("host_phase_authority") != "diagnostic-only;never-authorizes-day-rental"
        or rental.get("day_rental_requires_full_status") != "pass"
    ):
        raise ValueError("contract must reserve day-rental authority for a passing full phase")
    branch_names = [item.get("name") for item in contract.get("cuda_branches", [])]
    if branch_names != ["cu128", "cu126"]:
        raise ValueError("contract CUDA branch order must be cu128 then cu126")
    return contract


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish a mode-0600 artifact without replacing an existing path."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite evidence artifact: {path}") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, payload)


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe manifest path: {value!r}")
    return path


def build_file_manifest(root: Path, *, kind: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")
    if kind not in {"wheel-cache", "model-snapshot"}:
        raise ValueError("kind must be wheel-cache or model-snapshot")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"cache manifests reject symbolic links: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        safe_relative_path(relative)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"manifest root contains no files: {root}")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "kind": kind,
        "created_at": utc_now(),
        "root_label": root.name,
        "metadata": dict(metadata),
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }


def verify_file_manifest(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unsupported schema_version")
    kind = manifest.get("kind")
    if kind not in {"wheel-cache", "model-snapshot"}:
        errors.append("invalid manifest kind")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        errors.append("files must be a non-empty list")
        raw_files = []
    expected: dict[str, Mapping[str, Any]] = {}
    declared_total = 0
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            errors.append("invalid file entry schema")
            continue
        if not isinstance(item["path"], str):
            errors.append("file path must be a string")
            continue
        try:
            relative = safe_relative_path(item["path"]).as_posix()
        except ValueError as exc:
            errors.append(str(exc))
            continue
        size = item["size_bytes"]
        digest = item["sha256"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"invalid size: {relative}")
            continue
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"invalid sha256: {relative}")
            continue
        if relative in expected:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        expected[relative] = item
        declared_total += size
    if manifest.get("file_count") != len(expected):
        errors.append("file_count mismatch")
    if manifest.get("total_size_bytes") != declared_total:
        errors.append("total_size_bytes mismatch")

    root = root.resolve()
    actual: dict[str, Path] = {}
    if not root.is_dir():
        errors.append("cache root is missing or not a directory")
    else:
        for path in root.rglob("*"):
            if path.is_symlink():
                errors.append(f"symbolic link rejected: {path.relative_to(root).as_posix()}")
                continue
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = path
    errors.extend(f"missing: {item}" for item in sorted(set(expected) - set(actual)))
    errors.extend(f"unexpected: {item}" for item in sorted(set(actual) - set(expected)))
    verified = 0
    for relative in sorted(set(expected) & set(actual)):
        item = expected[relative]
        path = actual[relative]
        if path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != item["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")
            continue
        verified += 1
    return {
        "ok": not errors,
        "manifest_kind": kind,
        "metadata": metadata,
        "verified_files": verified,
        "expected_files": len(expected),
        "errors": errors[:50],
    }


def _requirement_inventory(
    contract: Mapping[str, Any], branch: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for value in (
        contract["common_requirements_file"],
        branch["requirements_file"],
    ):
        relative = safe_relative_path(str(value)).as_posix()
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"requirements file missing: {relative}")
        records.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def _normalized_distribution_name(value: str) -> str:
    return str(canonicalize_name(value))


def _exact_requirement_pins(
    contract: Mapping[str, Any], branch: Mapping[str, Any]
) -> dict[str, str]:
    pins: dict[str, str] = {}
    for relative in (contract["common_requirements_file"], branch["requirements_file"]):
        path = REPOSITORY_ROOT / safe_relative_path(str(relative))
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.partition("#")[0].strip()
            if not line or line.startswith("--"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
            if match is None:
                raise ValueError(
                    f"requirements must use unconditional exact pins: {relative}:{line_number}"
                )
            name = _normalized_distribution_name(match.group(1))
            version = match.group(2)
            previous = pins.setdefault(name, version)
            if previous != version:
                raise ValueError(f"conflicting exact requirement pins for {name}")
    return pins


def _target_wheel_tags() -> frozenset[Tag]:
    # Ubuntu 22.04 ships glibc 2.35. Build this tag set explicitly instead of
    # consulting packaging.tags.sys_tags(), which would describe the machine
    # creating the cache (often Windows), not the rental host.
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(35, 4, -1)]
    platforms.extend(
        (
            "manylinux2014_x86_64",
            "manylinux2010_x86_64",
            "manylinux1_x86_64",
            "linux_x86_64",
        )
    )
    tags = set(cpython_tags(python_version=(3, 11), abis=["cp311"], platforms=platforms))
    tags.update(
        compatible_tags(
            python_version=(3, 11),
            interpreter="cp311",
            platforms=platforms,
        )
    )
    return frozenset(tags)


TARGET_WHEEL_TAGS = _target_wheel_tags()


def _parse_wheel_filename(path: Path) -> dict[str, Any]:
    if path.suffix.casefold() != ".whl":
        raise ValueError(f"non-wheel cache entry rejected: {path.name}")
    try:
        distribution, version, _build, tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as exc:
        raise ValueError(f"invalid wheel filename: {path.name}: {exc}") from exc
    if not tags.intersection(TARGET_WHEEL_TAGS):
        raise ValueError(
            f"wheel is incompatible with Ubuntu 22.04 x86_64 / CPython 3.11: {path.name}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_entries = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and len(PurePosixPath(info.filename).parts) == 2
                and PurePosixPath(info.filename).parts[0].endswith(".dist-info")
                and PurePosixPath(info.filename).parts[1] == "METADATA"
            ]
            if len(metadata_entries) != 1:
                raise ValueError(f"wheel must contain exactly one .dist-info/METADATA: {path.name}")
            metadata_entry = metadata_entries[0]
            if metadata_entry.file_size > WHEEL_METADATA_MAX_BYTES:
                raise ValueError(f"wheel METADATA exceeds safety limit: {path.name}")
            metadata = BytesParser(policy=compat32).parsebytes(archive.read(metadata_entry))
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"invalid wheel ZIP payload: {path.name}: {exc}") from exc

    metadata_names = metadata.get_all("Name", [])
    metadata_versions = metadata.get_all("Version", [])
    if len(metadata_names) != 1 or len(metadata_versions) != 1:
        raise ValueError(f"wheel METADATA requires exactly one Name and Version: {path.name}")
    metadata_name = _normalized_distribution_name(str(metadata_names[0]))
    filename_name = str(distribution)
    if metadata_name != filename_name:
        raise ValueError(f"wheel filename/METADATA name mismatch: {path.name}: {metadata_name}")
    try:
        metadata_version = Version(str(metadata_versions[0]))
    except InvalidVersion as exc:
        raise ValueError(f"invalid METADATA version in {path.name}: {exc}") from exc
    if metadata_version != version:
        raise ValueError(
            f"wheel filename/METADATA version mismatch: {path.name}: {metadata_version}"
        )

    requires_python_values = metadata.get_all("Requires-Python", [])
    if len(requires_python_values) > 1:
        raise ValueError(f"wheel METADATA repeats Requires-Python: {path.name}")
    if requires_python_values:
        try:
            requires_python = SpecifierSet(str(requires_python_values[0]))
        except InvalidSpecifier as exc:
            raise ValueError(f"invalid Requires-Python in {path.name}: {exc}") from exc
        target_python = Version(WHEEL_CLOSURE_TARGET["python_full_version"])
        if target_python not in requires_python:
            raise ValueError(
                f"wheel Requires-Python excludes target {target_python}: "
                f"{path.name}: {requires_python}"
            )

    requirements: list[Requirement] = []
    for raw_requirement in metadata.get_all("Requires-Dist", []):
        try:
            requirements.append(Requirement(str(raw_requirement)))
        except InvalidRequirement as exc:
            raise ValueError(
                f"invalid Requires-Dist in {path.name}: {raw_requirement!r}: {exc}"
            ) from exc
    return {
        "name": filename_name,
        "version": str(version),
        "filename": path.name,
        "tags": tags,
        "requirements": tuple(requirements),
    }


def _marker_applies(requirement: Requirement, activated_extras: set[str]) -> bool:
    if requirement.marker is None:
        return True
    for extra in sorted({"", *activated_extras}):
        environment = dict(WHEEL_CLOSURE_TARGET)
        environment["extra"] = extra
        if requirement.marker.evaluate(environment=environment):
            return True
    return False


def _resolve_wheel_dependency_graph(
    by_name: Mapping[str, Sequence[Mapping[str, Any]]], roots: Iterable[str]
) -> dict[str, Any]:
    activated_extras: dict[str, set[str]] = {
        _normalized_distribution_name(name): set() for name in roots
    }
    pending = sorted(activated_extras)
    processed_extras: dict[str, frozenset[str]] = {}
    edges: dict[tuple[str, str], dict[str, str]] = {}
    errors: list[str] = []
    while pending:
        parent_name = pending.pop(0)
        parent_candidates = by_name.get(parent_name, ())
        if len(parent_candidates) != 1:
            errors.append(
                f"dependency closure distribution {parent_name} unresolved; "
                f"candidate_count={len(parent_candidates)}"
            )
            continue
        extras = frozenset(activated_extras[parent_name])
        if processed_extras.get(parent_name) == extras:
            continue
        processed_extras[parent_name] = extras
        parent = parent_candidates[0]
        for requirement in parent["requirements"]:
            if not _marker_applies(requirement, set(extras)):
                continue
            rendered = str(requirement)
            child_name = _normalized_distribution_name(requirement.name)
            edge_key = (parent_name, rendered)
            if requirement.url is not None:
                errors.append(
                    f"active direct-URL dependency is not allowed: {parent_name} -> {rendered}"
                )
                continue
            child_candidates = by_name.get(child_name, ())
            if len(child_candidates) != 1:
                errors.append(
                    f"active dependency missing or ambiguous: {parent_name} -> {rendered}; "
                    f"candidate_count={len(child_candidates)}"
                )
                continue
            child = child_candidates[0]
            child_version = Version(str(child["version"]))
            if requirement.specifier and child_version not in requirement.specifier:
                errors.append(
                    f"active dependency version mismatch: {parent_name} -> {rendered}; "
                    f"cached={child_version}"
                )
                continue
            edges[edge_key] = {
                "from": parent_name,
                "requires": rendered,
                "to": child_name,
                "version": str(child_version),
            }
            previous_extras = activated_extras.setdefault(child_name, set())
            requested_extras = {str(extra) for extra in requirement.extras}
            changed = not requested_extras.issubset(previous_extras)
            previous_extras.update(requested_extras)
            if child_name not in processed_extras or changed:
                pending.append(child_name)
                pending.sort()

    rendered_edges = [edges[key] for key in sorted(edges)]
    edge_payload = json.dumps(rendered_edges, sort_keys=True, separators=(",", ":")).encode()
    resolved: dict[str, dict[str, str]] = {}
    for name in sorted(activated_extras):
        candidates = by_name.get(name, ())
        if len(candidates) == 1:
            resolved[name] = {
                "filename": str(candidates[0]["filename"]),
                "version": str(candidates[0]["version"]),
            }
    return {
        "resolved": resolved,
        "resolved_distribution_count": len(resolved),
        "active_requires_dist_count": len(rendered_edges),
        "dependency_edges_sha256": sha256_bytes(edge_payload),
        "requested_extras": {
            name: sorted(extras) for name, extras in sorted(activated_extras.items()) if extras
        },
        "errors": errors,
    }


def wheel_cache_inventory(
    root: Path, contract: Mapping[str, Any], branch: Mapping[str, Any] | None
) -> dict[str, Any]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    root = root.resolve()
    if branch is None:
        return {"ok": False, "errors": ["CUDA branch is required for wheel inventory"]}
    if not root.is_dir():
        return {"ok": False, "errors": ["wheel cache root is missing or not a directory"]}
    try:
        pins = _exact_requirement_pins(contract, branch)
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            errors.append(f"symbolic link rejected: {path.relative_to(root).as_posix()}")
            continue
        if not path.is_file():
            continue
        try:
            record = _parse_wheel_filename(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        record["path"] = path.relative_to(root).as_posix()
        records.append(record)
        by_name.setdefault(record["name"], []).append(record)
    for name, candidates in sorted(by_name.items()):
        if len(candidates) != 1:
            versions = sorted({item["version"] for item in candidates})
            errors.append(f"ambiguous wheel distribution {name}: versions={versions}")
    selected: dict[str, dict[str, str]] = {}
    for name, expected in sorted(pins.items()):
        candidates = by_name.get(name, [])
        compatible = [
            item for item in candidates if pinned_version_matches(item["version"], expected)
        ]
        if len(compatible) != 1 or len(candidates) != 1:
            found = sorted({item["version"] for item in candidates})
            errors.append(f"top-level wheel {name}=={expected} unresolved; found={found}")
            continue
        selected[name] = {
            "filename": compatible[0]["filename"],
            "version": compatible[0]["version"],
        }
    for name, expected in sorted(contract["packages"].items()):
        normalized = _normalized_distribution_name(name)
        if pins.get(normalized) != expected:
            errors.append(f"contract pin is absent from requirements: {name}=={expected}")
    torch = selected.get("torch")
    expected_torch = pins.get("torch")
    if torch is not None and torch["version"] != f"{expected_torch}+{branch['name']}":
        errors.append(
            "torch wheel local version must exactly identify the selected branch: "
            f"expected={expected_torch}+{branch['name']}; found={torch['version']}"
        )

    deployment_closure = _resolve_wheel_dependency_graph(by_name, pins)
    cache_closure = _resolve_wheel_dependency_graph(by_name, by_name)
    errors.extend(deployment_closure.pop("errors"))
    errors.extend(cache_closure.pop("errors"))
    summary = {
        "schema_version": "wheel-cache-inventory-v2",
        "closure_target": {
            "marker_environment": dict(WHEEL_CLOSURE_TARGET),
            "wheel_tags": {
                "interpreter": "cp311",
                "abi": "cp311/abi3/none",
                "architecture": "x86_64",
                "maximum_manylinux_glibc": "2.35",
                "nominal_os": "Ubuntu 22.04",
            },
        },
        "wheel_file_count": len(records),
        "distribution_count": len(by_name),
        "top_level": selected,
        "deployment_closure": deployment_closure,
        "all_cached_distributions_closure": cache_closure,
    }
    return {"ok": not errors, "summary": summary, "errors": sorted(set(errors))[:100]}


def manifest_metadata(
    *, kind: str, contract: Mapping[str, Any], contract_path: Path, cuda_branch: str | None
) -> dict[str, Any]:
    if kind == "model-snapshot":
        return {
            "environment_id": contract["environment_id"],
            "contract_sha256": sha256_file(contract_path),
            "repo_id": contract["model"]["repo_id"],
            "revision": contract["model"]["revision"],
        }
    matches = [item for item in contract["cuda_branches"] if item["name"] == cuda_branch]
    if len(matches) != 1:
        raise ValueError("wheel-cache manifest requires a configured --cuda-branch")
    branch = matches[0]
    return {
        "environment_id": contract["environment_id"],
        "contract_sha256": sha256_file(contract_path),
        "cuda_branch": branch["name"],
        "requirements": _requirement_inventory(contract, branch),
    }


def cache_semantics_ok(
    result: Mapping[str, Any],
    *,
    kind: str,
    contract: Mapping[str, Any],
    contract_path: Path,
    branch: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if result.get("ok") is not True:
        errors.append("file manifest verification failed")
        return False, errors
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return False, ["manifest metadata missing"]
    if result.get("manifest_kind") != kind:
        errors.append("manifest kind mismatch")
    if metadata.get("environment_id") != contract["environment_id"]:
        errors.append("environment_id mismatch")
    if metadata.get("contract_sha256") != sha256_file(contract_path):
        errors.append("contract SHA256 mismatch")
    if kind == "model-snapshot":
        for key in ("repo_id", "revision"):
            if metadata.get(key) != contract["model"][key]:
                errors.append(f"model {key} mismatch")
    else:
        if branch is None:
            errors.append("CUDA branch was not selected")
        else:
            if metadata.get("cuda_branch") != branch["name"]:
                errors.append("CUDA branch mismatch")
            if metadata.get("requirements") != _requirement_inventory(contract, branch):
                errors.append("requirements inventory mismatch")
    return not errors, errors


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", value)
    if not match:
        raise ValueError(f"not a numeric version: {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def version_at_least(actual: str, minimum: str) -> bool:
    left = version_tuple(actual)
    right = version_tuple(minimum)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def pinned_version_matches(actual: str | None, expected: str) -> bool:
    return actual is not None and (actual == expected or actual.startswith(expected + "+"))


def choose_cuda_branch(
    driver_version: str, branches: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    for branch in sorted(branches, key=lambda item: int(item["priority"])):
        if version_at_least(driver_version, str(branch["minimum_linux_driver"])):
            return dict(branch)
    return None


def run_command(
    command: Sequence[str],
    timeout_s: float = 30.0,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=dict(environment) if environment is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _number(value: str, *, integer: bool = False) -> int | float | None:
    value = value.strip()
    if not value or value.casefold() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return int(float(value)) if integer else float(value)
    except ValueError:
        return None


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parsed in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not parsed or all(not item.strip() for item in parsed):
            continue
        if len(parsed) != len(GPU_QUERY_FIELDS):
            raise ValueError(
                f"nvidia-smi returned {len(parsed)} columns; expected {len(GPU_QUERY_FIELDS)}"
            )
        row = dict(zip(GPU_QUERY_FIELDS, (item.strip() for item in parsed), strict=True))
        rows.append(
            {
                "index": _number(row["index"], integer=True),
                "name": row["name"],
                "uuid": row["uuid"],
                "memory_total_mib": _number(row["memory.total"], integer=True),
                "memory_used_mib": _number(row["memory.used"], integer=True),
                "driver_version": row["driver_version"],
                "pci_bus_id": row["pci.bus_id"],
                "compute_capability": row["compute_cap"],
                "temperature_c": _number(row["temperature.gpu"], integer=True),
                "utilization_percent": _number(row["utilization.gpu"], integer=True),
                "pstate": row["pstate"],
                "power_draw_w": _number(row["power.draw"]),
                "power_limit_w": _number(row["power.limit"]),
            }
        )
    return rows


def _parse_pcie_csv(
    text: str, fields: Sequence[str], expected_identities: set[tuple[int, str]]
) -> dict[tuple[int, str], dict[str, int | None]]:
    values: dict[tuple[int, str], dict[str, int | None]] = {}
    for parsed in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not parsed or all(not item.strip() for item in parsed):
            continue
        if len(parsed) != len(fields) + 2:
            raise ValueError(
                f"PCIe query returned {len(parsed)} columns; expected {len(fields) + 2}"
            )
        index = _number(parsed[0], integer=True)
        raw_uuid = parsed[1].strip()
        if not isinstance(index, int):
            raise ValueError("PCIe query returned a non-integer GPU index")
        identity = (index, raw_uuid)
        if identity not in expected_identities or identity in values:
            raise ValueError("PCIe query GPU identities do not match the base query")
        values[identity] = {
            PCIE_QUERY_FIELDS[field]: _number(value, integer=True)
            for field, value in zip(fields, parsed[2:], strict=True)
        }
    if set(values) != expected_identities:
        raise ValueError("PCIe query did not return every base-query GPU identity")
    return values


def collect_pcie_links(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    identities = {
        (item.get("index"), item.get("uuid"))
        for item in gpus
        if isinstance(item.get("index"), int) and isinstance(item.get("uuid"), str)
    }
    records = {
        identity: {name: None for name in PCIE_QUERY_FIELDS.values()} for identity in identities
    }
    attempts: dict[str, dict[str, Any]] = {}
    fields = tuple(PCIE_QUERY_FIELDS)
    combined = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid," + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    attempts["combined"] = {
        "ok": combined["ok"],
        "returncode": combined["returncode"],
        "error": combined["stderr"].strip()[-2000:],
    }
    if combined["ok"]:
        try:
            for identity, values in _parse_pcie_csv(combined["stdout"], fields, identities).items():
                records[identity].update(values)
        except ValueError as exc:
            attempts["combined"]["ok"] = False
            attempts["combined"]["error"] = str(exc)

    missing_fields = [
        field
        for field, name in PCIE_QUERY_FIELDS.items()
        if not records or any(values[name] is None for values in records.values())
    ]
    for field in missing_fields:
        result = run_command(
            [
                "nvidia-smi",
                f"--query-gpu=index,uuid,{field}",
                "--format=csv,noheader,nounits",
            ]
        )
        attempt = {
            "ok": result["ok"],
            "returncode": result["returncode"],
            "error": result["stderr"].strip()[-2000:],
        }
        attempts[field] = attempt
        if not result["ok"]:
            continue
        try:
            parsed = _parse_pcie_csv(result["stdout"], (field,), identities)
        except ValueError as exc:
            attempt["ok"] = False
            attempt["error"] = str(exc)
            continue
        name = PCIE_QUERY_FIELDS[field]
        for identity, values in parsed.items():
            records[identity][name] = values[name]

    for gpu in gpus:
        identity = (gpu.get("index"), gpu.get("uuid"))
        gpu["pcie_link"] = records.get(
            identity, {name: None for name in PCIE_QUERY_FIELDS.values()}
        )
    available_fields = [
        field
        for field, name in PCIE_QUERY_FIELDS.items()
        if records and all(values[name] is not None for values in records.values())
    ]
    unavailable_fields = sorted(set(fields) - set(available_fields))
    return {
        "complete": len(identities) == len(gpus) > 0 and not unavailable_fields,
        "available_fields": available_fields,
        "unavailable_fields": unavailable_fields,
        "attempts": attempts,
    }


def collect_gpus() -> dict[str, Any]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ]
    )
    if not result["ok"]:
        return {
            "ok": False,
            "error": result["stderr"].strip(),
            "gpus": [],
            "pcie_query": {
                "complete": False,
                "available_fields": [],
                "unavailable_fields": sorted(PCIE_QUERY_FIELDS),
                "attempts": {},
            },
        }
    try:
        rows = parse_gpu_csv(result["stdout"])
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "gpus": [],
            "pcie_query": {
                "complete": False,
                "available_fields": [],
                "unavailable_fields": sorted(PCIE_QUERY_FIELDS),
                "attempts": {},
            },
        }
    pcie = collect_pcie_links(rows)
    return {"ok": True, "error": None, "gpus": rows, "pcie_query": pcie}


def collect_thermal_sample() -> list[dict[str, Any]]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,temperature.gpu,power.draw,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5,
    )
    if not result["ok"]:
        raise RuntimeError(result["stderr"].strip() or "nvidia-smi thermal query failed")
    rows = []
    for parsed in csv.reader(io.StringIO(result["stdout"]), skipinitialspace=True):
        if len(parsed) != 6:
            raise RuntimeError(f"thermal query returned {len(parsed)} columns")
        rows.append(
            {
                "index": _number(parsed[0], integer=True),
                "uuid": parsed[1].strip(),
                "temperature_c": _number(parsed[2], integer=True),
                "power_draw_w": _number(parsed[3]),
                "utilization_percent": _number(parsed[4], integer=True),
                "pstate": parsed[5].strip(),
            }
        )
    return rows


class ThermalSampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def target() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(
                        {"captured_at": utc_now(), "gpus": collect_thermal_sample()}
                    )
                except Exception as exc:  # best effort collector; gate fails later
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=target, name="thermal-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))

    def summary(self) -> dict[str, Any]:
        by_uuid: dict[str, list[dict[str, Any]]] = {}
        for sample in self.samples:
            for gpu in sample["gpus"]:
                by_uuid.setdefault(str(gpu["uuid"]), []).append(gpu)
        summaries = []
        for raw_uuid, values in sorted(by_uuid.items()):
            temperatures = [
                item["temperature_c"] for item in values if item["temperature_c"] is not None
            ]
            power = [item["power_draw_w"] for item in values if item["power_draw_w"] is not None]
            utilization = [
                item["utilization_percent"]
                for item in values
                if item["utilization_percent"] is not None
            ]
            summaries.append(
                {
                    "uuid": raw_uuid,
                    "sample_count": len(values),
                    "temperature_max_c": max(temperatures) if temperatures else None,
                    "temperature_start_c": temperatures[0] if temperatures else None,
                    "temperature_end_c": temperatures[-1] if temperatures else None,
                    "power_draw_max_w": max(power) if power else None,
                    "utilization_max_percent": max(utilization) if utilization else None,
                }
            )
        return {
            "sample_interval_s": self.interval_s,
            "sample_rounds": len(self.samples),
            "gpus": summaries,
            "errors": self.errors[-20:],
        }


def read_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def collect_cpu_topology() -> dict[str, Any]:
    command_environment = os.environ.copy()
    command_environment.update({"LC_ALL": "C", "LANG": "C"})
    result = run_command(["lscpu", "--json"], environment=command_environment)
    output_format = "json"
    if not result["ok"]:
        fallback = run_command(["lscpu"], environment=command_environment)
        if fallback["ok"]:
            result = fallback
            output_format = "text"
    numa_node_count: int | None = None
    if result["ok"]:
        if output_format == "json":
            try:
                payload = json.loads(result["stdout"])
                entries = payload.get("lscpu", []) if isinstance(payload, dict) else []
                for item in entries:
                    if str(item.get("field", "")).strip().casefold() == "numa node(s):":
                        value = _number(str(item.get("data", "")), integer=True)
                        numa_node_count = value if isinstance(value, int) else None
                        break
            except (AttributeError, json.JSONDecodeError):
                output_format = "invalid-json"
        else:
            match = re.search(r"^NUMA node\(s\):\s+(\d+)\s*$", result["stdout"], re.MULTILINE)
            numa_node_count = int(match.group(1)) if match else None
    return {
        "ok": result["ok"],
        "format": output_format,
        "numa_node_count": numa_node_count,
        "stdout": result["stdout"],
        "stderr": result["stderr"][-4000:],
    }


def system_ram_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    match = re.search(
        r"^MemTotal:\s+(\d+)\s+kB$",
        path.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    return int(match.group(1)) / (1024 * 1024) if match else None


def dev_shm_gib() -> float | None:
    path = Path("/dev/shm")
    return shutil.disk_usage(path).total / (1024**3) if path.exists() else None


def memlock_limit_bytes() -> int | str | None:
    if resource is None:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (AttributeError, OSError):
        return None
    return "unlimited" if soft == resource.RLIM_INFINITY else int(soft)


def package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in PACKAGE_DISTRIBUTIONS:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    result: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        bf16_supported = []
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                bf16_supported.append(bool(torch.cuda.is_bf16_supported()))
        result["bf16_supported"] = bf16_supported
        result["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    str(value) for value in torch.cuda.get_device_capability(index)
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    return result


def probe_url(url: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "faster-glm-asr-rtx3090-readiness/1.0"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            final_host = urllib.parse.urlparse(response.geturl()).hostname
        return {
            "url": url,
            "ok": 200 <= status < 400,
            "status": status,
            "final_host": final_host,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": None,
        }
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {
            "url": url,
            "ok": False,
            "status": None,
            "final_host": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_utc_timestamp(value: object, where: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty RFC3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{where} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{where} must carry an explicit UTC offset")
    return parsed.astimezone(UTC)


def _provider_health_evidence(
    *,
    evidence_path: str | None,
    checksum_path: str | None,
    window_start: datetime,
    window_end: datetime,
    policy: Mapping[str, Any],
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    requested = evidence_path is not None or checksum_path is not None
    result: dict[str, Any] = {
        "requested": requested,
        "available": False,
        "valid": None if not requested else False,
        "matches": [],
        "error": None,
    }
    if not requested:
        return result
    if evidence_path is None or checksum_path is None:
        result["error"] = "evidence JSON and detached SHA256 file are both required"
        return result
    try:
        raw_evidence_path = Path(evidence_path)
        raw_checksum_path = Path(checksum_path)
        if raw_evidence_path.is_symlink() or raw_checksum_path.is_symlink():
            raise ValueError("provider evidence and checksum must not be symbolic links")
        evidence = raw_evidence_path.resolve(strict=True)
        checksum = raw_checksum_path.resolve(strict=True)
        maximum_bytes = int(policy["max_file_bytes"])
        if evidence.stat().st_size > maximum_bytes or checksum.stat().st_size > 66:
            raise ValueError("provider evidence or checksum exceeds its strict size limit")
        expected = checksum.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("detached checksum must contain exactly one lowercase SHA-256")
        observed = sha256_file(evidence)
        if observed != expected:
            raise ValueError("provider evidence SHA-256 mismatch")
        document = load_json(evidence)
        required = {
            "schema_version",
            "provider",
            "instance_reference",
            "window_start",
            "window_end",
            "captured_at",
            "nvidia_xid_events",
        }
        if set(document) != required:
            raise ValueError("provider evidence has unexpected or missing fields")
        if document["schema_version"] != policy["provider_schema_version"]:
            raise ValueError("provider evidence schema_version mismatch")
        for key in ("provider", "instance_reference"):
            if not isinstance(document[key], str) or not document[key].strip():
                raise ValueError(f"provider evidence {key} must be non-empty")
        evidence_start = _parse_utc_timestamp(
            document["window_start"], "provider evidence window_start"
        )
        evidence_end = _parse_utc_timestamp(document["window_end"], "provider evidence window_end")
        captured_at = _parse_utc_timestamp(document["captured_at"], "provider evidence captured_at")
        now = datetime.now(UTC)
        future_skew = timedelta(seconds=int(policy["max_future_skew_seconds"]))
        reporting_delay = timedelta(seconds=int(policy["max_reporting_delay_seconds"]))
        finalization_lead = timedelta(seconds=int(policy["max_finalization_lead_seconds"]))
        if not evidence_start < evidence_end:
            raise ValueError("provider evidence window must have positive duration")
        if evidence_start > window_start or evidence_end < window_end:
            raise ValueError("provider evidence does not cover the requested Xid window")
        if evidence_end > now + future_skew or captured_at > now + future_skew:
            raise ValueError("provider evidence contains a future timestamp")
        if captured_at < evidence_end or captured_at - evidence_end > reporting_delay:
            raise ValueError("provider evidence captured_at is outside the reporting window")
        evidence_mtime = datetime.fromtimestamp(evidence.stat().st_mtime, UTC)
        if evidence_mtime + finalization_lead < window_end:
            raise ValueError("provider evidence was finalized before the requested window ended")
        events = document["nvidia_xid_events"]
        if not isinstance(events, list):
            raise ValueError("provider evidence nvidia_xid_events must be an array")
        parsed_events = []
        previous_timestamp: datetime | None = None
        for index, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {"timestamp", "code", "summary"}:
                raise ValueError(f"provider evidence event {index} has an invalid schema")
            timestamp = _parse_utc_timestamp(
                event["timestamp"], f"provider evidence event {index} timestamp"
            )
            if not evidence_start <= timestamp <= evidence_end:
                raise ValueError(f"provider evidence event {index} is outside its evidence window")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError("provider evidence events must be timestamp ordered")
            previous_timestamp = timestamp
            code = event["code"]
            summary = event["summary"]
            if not isinstance(code, int) or isinstance(code, bool) or code <= 0:
                raise ValueError(f"provider evidence event {index} code must be positive")
            if not isinstance(summary, str) or not summary.strip() or len(summary) > 500:
                raise ValueError(f"provider evidence event {index} summary is invalid")
            if window_start <= timestamp <= window_end:
                parsed_events.append(
                    {
                        "source": "provider",
                        "timestamp": timestamp.isoformat(),
                        "code": code,
                        "summary": summary,
                    }
                )
        result.update(
            {
                "available": True,
                "valid": True,
                "evidence_name": evidence.name,
                "checksum_name": checksum.name,
                "sha256": observed,
                "provider": document["provider"],
                "instance_reference": document["instance_reference"],
                "evidence_window_start": evidence_start.isoformat(),
                "evidence_window_end": evidence_end.isoformat(),
                "captured_at": captured_at.isoformat(),
                "matches": parsed_events,
            }
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if wait_seconds > 0:
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                retry = _provider_health_evidence(
                    evidence_path=evidence_path,
                    checksum_path=checksum_path,
                    window_start=window_start,
                    window_end=window_end,
                    policy=policy,
                    wait_seconds=0.0,
                )
                result = retry
                if retry.get("valid") is True:
                    break
    return result


def _local_xid_source(
    name: str,
    command: Sequence[str],
    *,
    readability_command: Sequence[str],
    readability_requires_records: bool,
    window_start: datetime,
    window_end: datetime,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command_result = run_command(command, timeout_s=15, environment=environment)
    stdout = command_result["stdout"]
    stderr = command_result["stderr"]
    combined = f"{stdout}\n{stderr}".casefold()
    unavailable_markers = (
        "no journal files were found",
        "permission denied",
        "operation not permitted",
        "read kernel buffer failed",
        "failed to open",
        "not seeing messages",
    )
    lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and line.strip().casefold() != "-- no entries --"
    ]
    query_readable = command_result["ok"] is True and not any(
        marker in combined for marker in unavailable_markers
    )
    probe: dict[str, Any] | None = None
    available = query_readable and bool(lines)
    if query_readable and not lines:
        probe_result = run_command(readability_command, timeout_s=15, environment=environment)
        probe_stdout = probe_result["stdout"]
        probe_stderr = probe_result["stderr"]
        probe_combined = f"{probe_stdout}\n{probe_stderr}".casefold()
        probe_lines = [
            line.strip()
            for line in probe_stdout.splitlines()
            if line.strip() and line.strip().casefold() != "-- no entries --"
        ]
        probe_available = probe_result["ok"] is True and not any(
            marker in probe_combined for marker in unavailable_markers
        )
        if readability_requires_records:
            probe_available = probe_available and bool(probe_lines)
        available = probe_available
        probe = {
            "available": probe_available,
            "returncode": probe_result["returncode"],
            "record_count": len(probe_lines),
            "stderr": probe_stderr.strip()[-2000:],
        }
    matches = (
        [
            {"source": name, "line": line}
            for line in lines
            if re.search(r"NVRM.*Xid|Xid.*NVRM", line, re.IGNORECASE)
        ]
        if available
        else []
    )
    reason = None
    if not available:
        if not command_result["ok"]:
            reason = f"command failed with returncode={command_result['returncode']}"
        elif not lines:
            reason = "empty window could not be paired with a successful readability probe"
        else:
            reason = "source reported unavailable journal or insufficient permission"
    return {
        "name": name,
        "available": available,
        "returncode": command_result["returncode"],
        "record_count": len(lines),
        "matches": matches[-20:],
        "reason": reason,
        "stderr": stderr.strip()[-2000:],
        "readability_probe": probe,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
    }


def collect_xid_events(
    *,
    window_start: datetime,
    window_end: datetime,
    policy: Mapping[str, Any],
    provider_evidence_path: str | None = None,
    provider_checksum_path: str | None = None,
    provider_wait_seconds: float = 0.0,
) -> dict[str, Any]:
    window_start = window_start.astimezone(UTC)
    window_end = window_end.astimezone(UTC)
    if not window_start < window_end:
        raise ValueError("Xid query window must have positive duration")
    timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    start_text = window_start.strftime(timestamp_format)
    end_text = window_end.strftime(timestamp_format)
    command_environment = os.environ.copy()
    command_environment.update({"LC_ALL": "C", "LANG": "C", "TZ": "UTC"})
    local_sources = [
        _local_xid_source(
            "journalctl",
            [
                "journalctl",
                "-k",
                "--since",
                start_text,
                "--until",
                end_text,
                "--no-pager",
                "-o",
                "short-iso-precise",
            ],
            readability_command=[
                "journalctl",
                "-k",
                "-n",
                "1",
                "--no-pager",
                "-o",
                "short-iso-precise",
            ],
            readability_requires_records=True,
            window_start=window_start,
            window_end=window_end,
            environment=command_environment,
        ),
        _local_xid_source(
            "dmesg",
            [
                "dmesg",
                "--color=never",
                "--since",
                start_text,
                "--until",
                end_text,
            ],
            readability_command=[
                "dmesg",
                "--color=never",
                "--since",
                (window_start - timedelta(seconds=1)).strftime(timestamp_format),
                "--until",
                window_start.strftime(timestamp_format),
            ],
            readability_requires_records=False,
            window_start=window_start,
            window_end=window_end,
            environment=command_environment,
        ),
    ]
    provider = _provider_health_evidence(
        evidence_path=provider_evidence_path,
        checksum_path=provider_checksum_path,
        window_start=window_start,
        window_end=window_end,
        policy=policy,
        wait_seconds=provider_wait_seconds,
    )
    matches = [match for source in local_sources for match in source["matches"]]
    matches.extend(provider["matches"])
    available = any(source["available"] for source in local_sources) or provider["available"]
    return {
        "available": available,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "local_sources": local_sources,
        "provider_evidence": provider,
        "matches": matches,
    }


def capture_pip_state(path: Path) -> dict[str, Any]:
    freeze = run_command([sys.executable, "-m", "pip", "freeze", "--all"], timeout_s=120)
    check = run_command([sys.executable, "-m", "pip", "check"], timeout_s=120)
    record: dict[str, Any] = {
        "freeze_ok": freeze["ok"],
        "pip_check_ok": check["ok"],
        "pip_check_stdout": check["stdout"][-4000:],
        "pip_check_stderr": check["stderr"][-4000:],
        "artifact_name": path.name,
        "sha256": None,
        "size_bytes": None,
        "line_count": None,
    }
    if freeze["ok"]:
        payload = freeze["stdout"].replace("\r\n", "\n").encode("utf-8")
        atomic_write_bytes(path, payload)
        record.update(
            {
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
                "line_count": len(payload.splitlines()),
            }
        )
    else:
        record["freeze_error"] = freeze["stderr"][-4000:]
    return record


def fresh_offline_install(
    *,
    artifact_directory: Path,
    wheel_root: Path,
    workdir: Path,
    contract: Mapping[str, Any],
    branch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prove that a clean target interpreter can install only from the wheel cache."""

    record: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "network_disabled": True,
        "venv_directory": "fresh-offline-venv",
        "freeze_artifact": "fresh-offline-freeze.txt",
        "freeze_sha256": None,
        "steps": [],
        "errors": [],
    }
    if platform.system().casefold() != "linux":
        record["errors"].append("fresh offline install must run on the target Linux host")
        return record
    if branch is None:
        record["errors"].append("CUDA branch is unavailable")
        return record
    wheel_root = wheel_root.resolve()
    if not wheel_root.is_dir():
        record["errors"].append("wheel cache root is unavailable")
        return record
    venv_root = artifact_directory / "fresh-offline-venv"
    freeze_path = artifact_directory / "fresh-offline-freeze.txt"
    if venv_root.exists() or freeze_path.exists():
        record["errors"].append("fresh offline install artifact already exists")
        return record
    record["attempted"] = True
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_FIND_LINKS": str(wheel_root),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    python_path = venv_root / "bin" / "python"
    requirements = [
        REPOSITORY_ROOT / safe_relative_path(str(branch["requirements_file"])),
        REPOSITORY_ROOT / safe_relative_path(str(contract["common_requirements_file"])),
    ]
    build_tool_requirements = [
        f"{name}=={contract['packages'][name]}" for name in ("setuptools", "wheel")
    ]
    commands = (
        ("create_venv", [sys.executable, "-m", "venv", str(venv_root)], 300.0),
        (
            "install_build_tools",
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--only-binary=:all:",
                "--find-links",
                str(wheel_root),
                *build_tool_requirements,
            ],
            300.0,
        ),
        (
            "install_requirements",
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--only-binary=:all:",
                "--find-links",
                str(wheel_root),
                "-r",
                str(requirements[0]),
                "-r",
                str(requirements[1]),
            ],
            1800.0,
        ),
        (
            "install_editable_repository",
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheel_root),
                "--no-build-isolation",
                "--no-deps",
                "--editable",
                str(workdir),
            ],
            600.0,
        ),
        ("pip_check", [str(python_path), "-m", "pip", "check"], 120.0),
        (
            "import_repository",
            [str(python_path), "-c", "import faster_glm_asr"],
            120.0,
        ),
    )
    for name, command, timeout_s in commands:
        result = run_command(command, timeout_s=timeout_s, environment=environment)
        record["steps"].append(
            {
                "name": name,
                "ok": result["ok"],
                "returncode": result["returncode"],
                "stderr": result["stderr"][-4000:],
            }
        )
        if result["ok"] is not True:
            record["errors"].append(f"{name} failed")
            return record
    freeze = run_command(
        [str(python_path), "-m", "pip", "freeze", "--all"],
        timeout_s=120,
        environment=environment,
    )
    if freeze["ok"] is not True:
        record["errors"].append("fresh environment freeze failed")
        return record
    payload = freeze["stdout"].replace("\r\n", "\n").encode("utf-8")
    atomic_write_bytes(freeze_path, payload)
    record["freeze_sha256"] = sha256_bytes(payload)
    record["freeze_line_count"] = len(payload.splitlines())
    record["ok"] = True
    return record


def gate(name: str, status: str, reason: str, evidence: Any = None) -> dict[str, Any]:
    if status not in {"pass", "warn", "fail"}:
        raise ValueError(f"invalid gate status: {status}")
    value = {"name": name, "status": status, "reason": reason}
    if evidence is not None:
        value["evidence"] = evidence
    return value


def _all_numbers(values: Iterable[Any], predicate: Callable[[Any], bool]) -> bool:
    collected = list(values)
    return bool(collected) and all(value is not None and predicate(value) for value in collected)


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()


def run_distributed_workers(
    *,
    world_size: int,
    stress_seconds: float,
    buffer_mib: int,
    warmup: int,
    iterations: int,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if world_size != 4:
        raise ValueError("full readiness requires exactly four ranks")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(Path(__file__).resolve()),
        "_worker",
        "--expected-world-size=4",
        f"--stress-seconds={stress_seconds}",
        f"--buffer-mib={buffer_mib}",
        f"--warmup={warmup}",
        f"--iterations={iterations}",
    ]
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NCCL_DEBUG": "WARN",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
    )
    sampler = ThermalSampler(interval_s=1.0)
    sampler.start()
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_environment,
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
        stdout, stderr = process.communicate()
    finally:
        sampler.stop()
    payload = None
    for line in stdout.splitlines():
        if line.startswith(WORKER_SENTINEL):
            try:
                payload = json.loads(line[len(WORKER_SENTINEL) :])
            except json.JSONDecodeError:
                payload = None
    result = {
        "ok": process.returncode == 0 and not timed_out and isinstance(payload, dict),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "payload": payload,
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-12000:],
    }
    return result, sampler.summary()


def _xid_gate(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    provider = result.get("provider_evidence", {})
    provider_invalid = provider.get("requested") is True and provider.get("valid") is not True
    status = (
        "fail"
        if result.get("matches")
        else "pass"
        if result.get("available") is True and not provider_invalid
        else "warn"
    )
    return gate(
        name,
        status,
        "Xid events in this explicit window fail; unavailable or invalid evidence warns",
        result,
    )


def evaluate_host_gates(
    report: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    target = contract["target"]
    gates: list[dict[str, Any]] = []
    os_release = report["host"]["os_release"]
    os_ok = (
        os_release.get("ID") == target["os_id"]
        and os_release.get("VERSION_ID") == target["os_version"]
    )
    gates.append(gate("os", "pass" if os_ok else "fail", "Ubuntu 22.04 exact target", os_release))
    architecture = report["host"]["architecture"]
    gates.append(
        gate(
            "architecture",
            "pass" if architecture == target["architecture"] else "fail",
            f"required {target['architecture']}",
            architecture,
        )
    )
    python_version = report["host"]["python"]
    try:
        python_ok = version_at_least(
            python_version, target["python_min_inclusive"]
        ) and not version_at_least(python_version, target["python_max_exclusive"])
    except ValueError:
        python_ok = False
    gates.append(
        gate("python", "pass" if python_ok else "fail", "required Python 3.11.x", python_version)
    )

    gpu_probe = report["hardware"]["gpu_probe_before"]
    gpus = gpu_probe.get("gpus", [])
    count_ok = gpu_probe.get("ok") is True and len(gpus) == int(target["gpu_count"])
    gates.append(
        gate("gpu_count", "pass" if count_ok else "fail", "four visible GPUs required", len(gpus))
    )
    names_ok = count_ok and all(target["gpu_name_contains"] in item["name"] for item in gpus)
    gates.append(
        gate(
            "gpu_model",
            "pass" if names_ok else "fail",
            "every GPU must be an RTX 3090",
            [item.get("name") for item in gpus],
        )
    )
    raw_uuids = [item.get("uuid") for item in gpus]
    uuid_ok = count_ok and all(raw_uuids) and len(set(raw_uuids)) == len(raw_uuids)
    gates.append(
        gate(
            "gpu_uuid",
            "pass" if uuid_ok else "fail",
            "four exact, non-empty, unique UUIDs are retained in private evidence",
            raw_uuids,
        )
    )
    memory_ok = count_ok and _all_numbers(
        (item.get("memory_total_mib") for item in gpus),
        lambda value: value >= target["gpu_memory_min_mib"],
    )
    gates.append(
        gate(
            "per_gpu_memory",
            "pass" if memory_ok else "fail",
            "each rank has its own >=23,000 MiB device; memory is not pooled",
            [item.get("memory_total_mib") for item in gpus],
        )
    )
    compute_ok = count_ok and all(
        item.get("compute_capability") == target["compute_capability"] for item in gpus
    )
    gates.append(
        gate(
            "compute_capability",
            "pass" if compute_ok else "fail",
            "Ampere sm_86 required",
            [item.get("compute_capability") for item in gpus],
        )
    )
    drivers = sorted({item.get("driver_version") for item in gpus if item.get("driver_version")})
    branch = report["environment"].get("selected_cuda_branch")
    gates.append(
        gate(
            "driver_cuda_branch",
            "pass" if len(drivers) == 1 and branch is not None else "fail",
            "one driver version and an eligible fixed CUDA wheel branch required",
            {"drivers": drivers, "selected": branch},
        )
    )
    idle_ok = (
        count_ok
        and _all_numbers(
            (item.get("memory_used_mib") for item in gpus),
            lambda value: value <= target["idle_memory_used_max_mib"],
        )
        and _all_numbers(
            (item.get("utilization_percent") for item in gpus),
            lambda value: value <= target["idle_utilization_max_percent"],
        )
        and _all_numbers(
            (item.get("temperature_c") for item in gpus),
            lambda value: value <= target["idle_temperature_max_c"],
        )
    )
    gates.append(
        gate(
            "gpu_idle",
            "pass" if idle_ok else "fail",
            "node must be exclusive and cool before acceptance",
            {
                "memory_used_mib": [item.get("memory_used_mib") for item in gpus],
                "utilization_percent": [item.get("utilization_percent") for item in gpus],
                "temperature_c": [item.get("temperature_c") for item in gpus],
            },
        )
    )
    power_limits = [item.get("power_limit_w") for item in gpus]
    power_ok = count_ok and _all_numbers(
        power_limits, lambda value: value >= target["preferred_power_limit_min_w"]
    )
    gates.append(
        gate(
            "power_limit",
            "pass" if power_ok else "warn",
            "a lower limit requires disclosure and manual review for throttling",
            power_limits,
        )
    )

    hardware = report["hardware"]
    pcie_query = gpu_probe.get("pcie_query", {})
    cpu_topology = hardware.get("cpu_topology", {})
    gpu_topology = hardware.get("topology", {})
    topology_complete = (
        pcie_query.get("complete") is True
        and cpu_topology.get("ok") is True
        and isinstance(cpu_topology.get("numa_node_count"), int)
        and gpu_topology.get("ok") is True
        and bool(str(gpu_topology.get("stdout", "")).strip())
    )
    gates.append(
        gate(
            "host_topology_observation",
            "pass" if topology_complete else "warn",
            "PCIe link values, lscpu NUMA count, and nvidia-smi topology matrix must be recorded",
            {
                "pcie_query": pcie_query,
                "cpu_topology_ok": cpu_topology.get("ok"),
                "numa_node_count": cpu_topology.get("numa_node_count"),
                "gpu_topology_ok": gpu_topology.get("ok"),
            },
        )
    )
    p2p_diagnostics = hardware.get("topology_p2p", {})
    p2p_recorded = set(p2p_diagnostics) == {"p", "r", "w", "n"} and all(
        (item.get("ok") is True and bool(str(item.get("stdout", "")).strip()))
        or (item.get("ok") is False and bool(str(item.get("stderr", "")).strip()))
        for item in p2p_diagnostics.values()
    )
    gates.append(
        gate(
            "topology_p2p_diagnostics",
            "pass" if p2p_recorded else "warn",
            "record nvidia-smi topo peer capability modes p/r/w/n or explicit unavailability",
            {
                mode: {"ok": value.get("ok"), "error": value.get("stderr", "")[-1000:]}
                for mode, value in p2p_diagnostics.items()
            },
        )
    )

    resources = report["host"]["resources"]
    for name, actual_key, minimum_key in (
        ("system_ram", "system_ram_gib", "system_ram_min_gib"),
        ("disk_free", "disk_free_gib", "disk_free_min_gib"),
        ("dev_shm", "dev_shm_gib", "dev_shm_min_gib"),
    ):
        actual = resources.get(actual_key)
        minimum = target[minimum_key]
        gates.append(
            gate(
                name,
                "pass" if actual is not None and actual >= minimum else "fail",
                f">={minimum} GiB required",
                actual,
            )
        )
    memlock = resources.get("memlock_soft_bytes")
    memlock_ok = memlock == "unlimited" or (
        isinstance(memlock, int) and memlock >= 64 * 1024 * 1024
    )
    gates.append(
        gate(
            "memlock",
            "pass" if memlock_ok else "warn",
            "unlimited or >=64 MiB preferred; full NCCL remains authoritative",
            memlock,
        )
    )

    probes = report["network"]["probes"]
    network_ok = bool(probes) and all(item.get("ok") is True for item in probes)
    cache_results = report["provenance"]["cache_verification"]
    both_caches_ok = all(
        cache_results.get(name, {}).get("ok") is True for name in ("wheels", "model")
    )
    gates.append(
        gate(
            "network_or_verified_caches",
            "pass" if network_ok or both_caches_ok else "fail",
            "all HEAD probes must pass, or both complete caches must verify",
            {"network_ok": network_ok, "both_caches_ok": both_caches_ok},
        )
    )
    gates.append(_xid_gate("preexisting_xid_window", report["hardware"]["xid_before"]))
    return gates


def evaluate_full_gates(
    report: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    gates = evaluate_host_gates(report, contract)
    target = contract["target"]
    branch = report["environment"].get("selected_cuda_branch")
    versions = report["environment"]["packages"]
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in contract["packages"].items()
        if not pinned_version_matches(versions.get(name), expected)
    }
    gates.append(
        gate(
            "package_pins",
            "pass" if not mismatches else "fail",
            "every fixed top-level package, including tokenizers, must match",
            mismatches,
        )
    )
    freeze = report["provenance"]["freeze"]
    freeze_ok = (
        freeze.get("before", {}).get("freeze_ok") is True
        and freeze.get("after", {}).get("freeze_ok") is True
        and freeze.get("before", {}).get("pip_check_ok") is True
        and freeze.get("after", {}).get("pip_check_ok") is True
        and freeze.get("before", {}).get("sha256") == freeze.get("after", {}).get("sha256")
        and freeze.get("before", {}).get("sha256") is not None
    )
    gates.append(
        gate(
            "pre_post_freeze",
            "pass" if freeze_ok else "fail",
            "pip check must pass and pip freeze --all must be byte-identical before/after",
            freeze,
        )
    )
    cache_results = report["provenance"]["cache_verification"]
    gates.append(
        gate(
            "wheel_cache_manifest",
            "pass" if cache_results.get("wheels", {}).get("ok") is True else "fail",
            "wheel cache files and contract/branch/requirements metadata must match",
            cache_results.get("wheels"),
        )
    )
    fresh_install = report["provenance"].get("fresh_offline_install") or {}
    gates.append(
        gate(
            "fresh_offline_install",
            "pass" if fresh_install.get("ok") is True else "fail",
            "a fresh target-host venv must install exact setuptools/wheel first, "
            "install both requirements files, install the editable repository with "
            "--no-index --no-build-isolation --no-deps, then pass pip check and import",
            fresh_install,
        )
    )
    gates.append(
        gate(
            "model_snapshot_manifest",
            "pass" if cache_results.get("model", {}).get("ok") is True else "fail",
            "model files and pinned repository/revision metadata must match",
            cache_results.get("model"),
        )
    )

    runtime = report["environment"]["torch_runtime"]
    expected_cuda = branch.get("torch_cuda_prefix") if isinstance(branch, dict) else None
    runtime_ok = (
        runtime.get("available") is True
        and runtime.get("cuda_available") is True
        and runtime.get("device_count") == 4
        and isinstance(runtime.get("cuda_runtime"), str)
        and runtime["cuda_runtime"].startswith(expected_cuda or "__missing__")
        and runtime.get("bf16_supported") == [True, True, True, True]
    )
    gates.append(
        gate(
            "torch_cuda_runtime",
            "pass" if runtime_ok else "fail",
            "selected CUDA runtime, four devices, and BF16 support are required",
            runtime,
        )
    )

    distributed = report.get("distributed") or {}
    worker = distributed.get("payload") or {}
    ranks = worker.get("ranks") if isinstance(worker, dict) else None
    bf16_ok = (
        distributed.get("ok") is True
        and worker.get("world_size") == 4
        and worker.get("backend") == "nccl"
        and isinstance(ranks, list)
        and len(ranks) == 4
        and all(item.get("bf16", {}).get("ok") is True for item in ranks)
    )
    gates.append(
        gate(
            "bf16_execution",
            "pass" if bf16_ok else "fail",
            "BF16 GEMM must match an FP32 oracle on every rank",
            [item.get("bf16") for item in ranks] if isinstance(ranks, list) else None,
        )
    )
    all_reduce = worker.get("all_reduce", {}) if isinstance(worker, dict) else {}
    nccl_ok = (
        distributed.get("ok") is True
        and all_reduce.get("correct") is True
        and isinstance(all_reduce.get("bus_bandwidth_min_gb_per_s"), (int, float))
        and all_reduce["bus_bandwidth_min_gb_per_s"] >= target["nccl_min_bus_bandwidth_gb_per_s"]
    )
    gates.append(
        gate(
            "nccl_all_reduce",
            "pass" if nccl_ok else "fail",
            "four-rank correctness plus minimum sanity bandwidth required",
            all_reduce,
        )
    )
    p2p = worker.get("p2p", {}) if isinstance(worker, dict) else {}
    query_complete = p2p.get("query_complete") is True
    gates.append(
        gate(
            "p2p_observation",
            "pass" if query_complete else "warn",
            "capture the complete directed matrix; false entries are observations, not warnings",
            p2p,
        )
    )

    thermal = report.get("thermal") or {}
    thermal_gpus = thermal.get("gpus", [])
    before_uuids = {
        item.get("uuid") for item in report["hardware"]["gpu_probe_before"].get("gpus", [])
    }
    thermal_ok = (
        not thermal.get("errors")
        and len(thermal_gpus) == 4
        and {item.get("uuid") for item in thermal_gpus} == before_uuids
        and all(
            item.get("sample_count", 0)
            >= contract["distributed"]["minimum_thermal_samples_per_gpu"]
            for item in thermal_gpus
        )
        and _all_numbers(
            (item.get("temperature_max_c") for item in thermal_gpus),
            lambda value: value <= target["stress_temperature_max_c"],
        )
        and isinstance(ranks, list)
        and all(
            item.get("stress", {}).get("requested_seconds", 0)
            >= contract["distributed"]["minimum_stress_seconds"]
            and item.get("stress", {}).get("iterations", 0) > 0
            for item in ranks
        )
    )
    gates.append(
        gate(
            "thermal_stability",
            "pass" if thermal_ok else "fail",
            "all exact UUIDs need sampled stress work below the thermal limit",
            thermal,
        )
    )

    post_probe = report["hardware"].get("gpu_probe_after") or {}
    post_gpus = post_probe.get("gpus", [])
    post_uuids = {item.get("uuid") for item in post_gpus}
    post_ok = (
        post_probe.get("ok") is True
        and len(post_gpus) == 4
        and post_uuids == before_uuids
        and _all_numbers(
            (item.get("temperature_c") for item in post_gpus),
            lambda value: value <= target["stress_temperature_max_c"],
        )
        and _all_numbers(
            (item.get("memory_used_mib") for item in post_gpus),
            lambda value: value <= target["idle_memory_used_max_mib"],
        )
    )
    gates.append(
        gate(
            "gpu_post_health",
            "pass" if post_ok else "fail",
            "the same four raw UUIDs must remain healthy with contexts released",
            post_probe,
        )
    )
    gates.append(_xid_gate("new_xid_during_preflight", report["hardware"]["xid_after"]))
    return gates


def _verify_cache(
    *,
    manifest_path: str | None,
    root: str | None,
    kind: str,
    contract: Mapping[str, Any],
    contract_path: Path,
    branch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not manifest_path or not root:
        return {
            "ok": False,
            "manifest_kind": kind,
            "errors": ["manifest and cache root are both required"],
        }
    try:
        path = Path(manifest_path).resolve(strict=True)
        result = verify_file_manifest(load_json(path), Path(root))
        semantic_ok, semantic_errors = cache_semantics_ok(
            result,
            kind=kind,
            contract=contract,
            contract_path=contract_path,
            branch=branch,
        )
        if kind == contract["provenance"]["wheel_manifest_kind"]:
            inventory = wheel_cache_inventory(Path(root), contract, branch)
            inventory_matches = inventory.get("ok") is True and result.get("metadata", {}).get(
                "wheel_inventory"
            ) == inventory.get("summary")
            result["wheel_inventory"] = inventory
            if not inventory_matches:
                semantic_errors.extend(inventory.get("errors", []))
                if result.get("metadata", {}).get("wheel_inventory") != inventory.get("summary"):
                    semantic_errors.append("wheel inventory metadata mismatch")
            semantic_ok = semantic_ok and inventory_matches
        result["file_manifest_ok"] = result["ok"]
        result["semantic_ok"] = semantic_ok
        result["errors"] = list(result.get("errors", [])) + semantic_errors
        result["ok"] = result["file_manifest_ok"] and semantic_ok
        result["manifest_sha256"] = sha256_file(path)
        result["manifest_name"] = path.name
        return result
    except Exception as exc:
        return {
            "ok": False,
            "manifest_kind": kind,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


def _artifact_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _report_outcome(gates: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    failures = [item["name"] for item in gates if item["status"] == "fail"]
    warnings = [item["name"] for item in gates if item["status"] == "warn"]
    status = "fail" if failures else "pass-with-warnings" if warnings else "pass"
    if phase == "host":
        decision = (
            "continue-hourly-to-full-validation"
            if status == "pass"
            else "manual-review-required-during-hourly-validation"
            if status == "pass-with-warnings"
            else "reject-instance"
        )
    else:
        decision = (
            "eligible-for-day-rental-and-ready-for-training"
            if status == "pass"
            else "manual-review-required-before-training"
            if status == "pass-with-warnings"
            else "not-ready-for-training"
        )
    return {
        "status": status,
        "ready": phase == "full" and status == "pass",
        "host_diagnostics_pass": phase == "host" and status == "pass",
        "day_rental_eligible": phase == "full" and status == "pass",
        "training_ready": phase == "full" and status == "pass",
        "failed_gates": failures,
        "warning_gates": warnings,
        "decision": decision,
    }


def run_preflight(args: argparse.Namespace) -> int:
    preflight_started = datetime.now(UTC)
    contract_path = Path(args.contract).resolve(strict=True)
    contract = load_contract(contract_path)
    for label, evidence, checksum in (
        (
            "before",
            args.provider_health_evidence_before,
            args.provider_health_evidence_before_sha256,
        ),
        (
            "after",
            args.provider_health_evidence_after,
            args.provider_health_evidence_after_sha256,
        ),
    ):
        if bool(evidence) != bool(checksum):
            raise ValueError(
                f"provider {label} evidence JSON and detached SHA256 path must be supplied together"
            )
    if args.phase == "host" and args.provider_health_evidence_after:
        raise ValueError("provider after evidence is valid only for --phase full")
    if args.phase == "full" and args.stress_seconds < float(
        contract["distributed"]["minimum_stress_seconds"]
    ):
        raise ValueError(
            "full phase stress duration is below the contract minimum of "
            f"{contract['distributed']['minimum_stress_seconds']} seconds"
        )
    if args.phase == "full" and args.distributed_timeout < max(600.0, args.stress_seconds + 120.0):
        raise ValueError(
            "full phase distributed timeout must be >=600 seconds and at least "
            "120 seconds longer than the stress interval"
        )
    output_path = Path(args.output).resolve() if args.output else default_output_path()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {output_path}")
    workdir = Path(args.workdir).resolve(strict=True)
    if not workdir.is_dir():
        raise ValueError("--workdir must be a directory")
    freeze_before_path = output_path.parent / "pip-freeze-before.txt"
    freeze_after_path = output_path.parent / "pip-freeze-after.txt"
    if args.phase == "full":
        for path in (freeze_before_path, freeze_after_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")

    gpu_before = collect_gpus()
    gpus = gpu_before.get("gpus", [])
    drivers = sorted({item["driver_version"] for item in gpus if item.get("driver_version")})
    branch = (
        choose_cuda_branch(drivers[0], contract["cuda_branches"]) if len(drivers) == 1 else None
    )
    selected_name = branch["name"] if branch else "unselected"
    cache_verification = {
        "wheels": _verify_cache(
            manifest_path=args.wheel_manifest,
            root=args.wheel_root,
            kind=contract["provenance"]["wheel_manifest_kind"],
            contract=contract,
            contract_path=contract_path,
            branch=branch,
        ),
        "model": _verify_cache(
            manifest_path=args.model_manifest,
            root=args.model_root,
            kind=contract["provenance"]["model_manifest_kind"],
            contract=contract,
            contract_path=contract_path,
            branch=branch,
        ),
    }
    fresh_install: dict[str, Any] | None = None
    if args.phase == "full":
        if cache_verification["wheels"].get("ok") is True and args.wheel_root:
            fresh_install = fresh_offline_install(
                artifact_directory=output_path.parent,
                wheel_root=Path(args.wheel_root),
                workdir=workdir,
                contract=contract,
                branch=branch,
            )
        else:
            fresh_install = {
                "attempted": False,
                "ok": False,
                "network_disabled": True,
                "errors": ["verified wheel cache is required before fresh offline install"],
            }
    endpoints = [
        str(url).format(cuda_branch=selected_name) for url in contract["network_endpoints"]
    ]
    network_probes = (
        [] if args.offline else [probe_url(url, args.network_timeout) for url in endpoints]
    )
    usage = shutil.disk_usage(workdir)
    cpu_topology = collect_cpu_topology()
    topology = run_command(["nvidia-smi", "topo", "-m"])
    topology_p2p = {
        mode: run_command(["nvidia-smi", "topo", "-p2p", mode]) for mode in ("p", "r", "w", "n")
    }
    nvlink = run_command(["nvidia-smi", "nvlink", "--status"])
    xid_policy = contract["xid_health_evidence"]
    xid_before = collect_xid_events(
        window_start=preflight_started - timedelta(seconds=int(xid_policy["lookback_seconds"])),
        window_end=preflight_started,
        policy=xid_policy,
        provider_evidence_path=args.provider_health_evidence_before,
        provider_checksum_path=args.provider_health_evidence_before_sha256,
        provider_wait_seconds=(
            args.provider_health_evidence_wait_seconds
            if args.provider_health_evidence_before
            else 0.0
        ),
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_id": contract["environment_id"],
        "phase": args.phase,
        "preflight_started_at": preflight_started.isoformat(),
        "captured_at": utc_now(),
        "privacy": {
            "artifact_class": "private-raw-readiness-evidence",
            "raw_gpu_uuid_policy": contract["provenance"]["raw_uuid_policy"],
            "public_release_allowed": False,
        },
        "contract": {
            "name": contract_path.name,
            "sha256": sha256_file(contract_path),
        },
        "tool": {
            "name": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "host": {
            "os_release": read_os_release(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "resources": {
                "logical_cpu_count": os.cpu_count(),
                "system_ram_gib": system_ram_gib(),
                "disk_total_gib": usage.total / (1024**3),
                "disk_free_gib": usage.free / (1024**3),
                "dev_shm_gib": dev_shm_gib(),
                "memlock_soft_bytes": memlock_limit_bytes(),
            },
            "binaries": {
                name: shutil.which(name) for name in ("nvidia-smi", "git", "ffmpeg", "nsys", "ncu")
            },
        },
        "hardware": {
            "gpu_probe_before": gpu_before,
            "gpu_probe_after": None,
            "cpu_topology": cpu_topology,
            "topology": {
                "ok": topology["ok"],
                "stdout": topology["stdout"],
                "stderr": topology["stderr"][-4000:],
            },
            "topology_p2p": {
                mode: {
                    "ok": result["ok"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"][-4000:],
                }
                for mode, result in topology_p2p.items()
            },
            "nvlink": {
                "ok": nvlink["ok"],
                "stdout": nvlink["stdout"],
                "stderr": nvlink["stderr"][-4000:],
            },
            "xid_before": xid_before,
            "xid_after": None,
            "xid_window_semantics": {
                "before": "pre-existing events in the fixed lookback ending at preflight start",
                "after": "only events from preflight start through the post-run capture",
            },
        },
        "environment": {
            "selected_cuda_branch": branch,
            "packages": package_versions(),
            "torch_runtime": torch_runtime(),
        },
        "network": {"offline_requested": args.offline, "probes": network_probes},
        "provenance": {
            "cache_verification": cache_verification,
            "fresh_offline_install": fresh_install,
            "freeze": {"before": None, "after": None, "exact_match": None},
        },
        "distributed": None,
        "thermal": None,
    }

    if args.phase == "full":
        before = capture_pip_state(freeze_before_path)
        report["provenance"]["freeze"]["before"] = before
        try:
            distributed, thermal = run_distributed_workers(
                world_size=int(contract["distributed"]["world_size"]),
                stress_seconds=args.stress_seconds,
                buffer_mib=int(contract["distributed"]["all_reduce_buffer_mib"]),
                warmup=int(contract["distributed"]["all_reduce_warmup_iterations"]),
                iterations=int(contract["distributed"]["all_reduce_measured_iterations"]),
                timeout_s=args.distributed_timeout,
            )
        except Exception as exc:
            distributed = {
                "ok": False,
                "returncode": None,
                "timed_out": False,
                "payload": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            thermal = {"sample_rounds": 0, "gpus": [], "errors": [str(exc)]}
        report["distributed"] = distributed
        report["thermal"] = thermal
        time.sleep(1.0)
        report["hardware"]["gpu_probe_after"] = collect_gpus()
        post_window_end = datetime.now(UTC)
        report["hardware"]["xid_after"] = collect_xid_events(
            window_start=preflight_started,
            window_end=post_window_end,
            policy=xid_policy,
            provider_evidence_path=args.provider_health_evidence_after,
            provider_checksum_path=args.provider_health_evidence_after_sha256,
            provider_wait_seconds=args.provider_health_evidence_wait_seconds,
        )
        after = capture_pip_state(freeze_after_path)
        report["provenance"]["freeze"]["after"] = after
        report["provenance"]["freeze"]["exact_match"] = before.get(
            "sha256"
        ) is not None and before.get("sha256") == after.get("sha256")
        report["gates"] = evaluate_full_gates(report, contract)
    else:
        report["gates"] = evaluate_host_gates(report, contract)

    report["overall"] = _report_outcome(report["gates"], args.phase)
    report["artifact"] = {
        "report": output_path.name,
        "directory_policy": "ignored artifacts/private tree",
        "freeze_before": freeze_before_path.name if args.phase == "full" else None,
        "freeze_after": freeze_after_path.name if args.phase == "full" else None,
    }
    atomic_write_json(output_path, report)
    summary = {
        "status": report["overall"]["status"],
        "decision": report["overall"]["decision"],
        "day_rental_eligible": report["overall"]["day_rental_eligible"],
        "training_ready": report["overall"]["training_ready"],
        "failed_gates": report["overall"]["failed_gates"],
        "warning_gates": report["overall"]["warning_gates"],
        "selected_cuda_branch": selected_name,
        "private_report": _artifact_label(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if report["overall"]["status"] == "pass":
        return 0
    return 3 if report["overall"]["status"] == "pass-with-warnings" else 2


def distributed_worker(args: argparse.Namespace) -> int:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        print(f"worker import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    required_environment = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required_environment):
        print("worker must be launched by torchrun", file=sys.stderr)
        return 2
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.expected_world_size or world_size != 4:
        print("worker requires exactly four ranks", file=sys.stderr)
        return 2
    if torch.cuda.device_count() != 4:
        print("each worker must see exactly four CUDA devices", file=sys.stderr)
        return 2
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    device = torch.device("cuda", local_rank)

    correctness = torch.tensor([float(rank + 1)], dtype=torch.float32, device=device)
    dist.all_reduce(correctness, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    scalar_ok = math.isclose(float(correctness.item()), float(expected), rel_tol=0, abs_tol=1e-6)

    try:
        torch.manual_seed(1000 + rank)
        left = torch.randn((512, 512), device=device, dtype=torch.bfloat16) * 0.01
        right = torch.randn((512, 512), device=device, dtype=torch.bfloat16) * 0.01
        output = left @ right
        reference = left.float() @ right.float()
        max_abs_error = float((output.float() - reference).abs().max().item())
        finite = bool(torch.isfinite(output).all().item())
        close = bool(torch.allclose(output.float(), reference, rtol=0.05, atol=0.02))
        supported = bool(torch.cuda.is_bf16_supported())
        bf16_result = {
            "supported": supported,
            "finite": finite,
            "close_to_fp32": close,
            "max_abs_error": max_abs_error,
            "ok": supported and finite and close,
        }
    except Exception as exc:
        bf16_result = {
            "supported": False,
            "finite": False,
            "close_to_fp32": False,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    peer_access: list[bool | None] = []
    for peer in range(world_size):
        if peer == local_rank:
            peer_access.append(True)
        else:
            try:
                peer_access.append(bool(torch.cuda.can_device_access_peer(local_rank, peer)))
            except Exception:
                peer_access.append(None)

    elements = args.buffer_mib * 1024 * 1024 // 4
    buffer = torch.full((elements,), float(rank + 1), dtype=torch.float32, device=device)
    for _ in range(args.warmup):
        buffer.fill_(float(rank + 1))
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    times_ms = []
    for _ in range(args.iterations):
        buffer.fill_(float(rank + 1))
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - started) * 1000)
    measured_correct = bool(torch.allclose(buffer[:1], torch.tensor([expected], device=device)))
    median_ms = statistics.median(times_ms)
    bytes_per_collective = args.buffer_mib * 1024 * 1024
    algorithmic_gb_per_s = bytes_per_collective / (median_ms / 1000) / 1e9
    # NCCL-tests bus bandwidth normalization for an all-reduce: algbw * 2*(n-1)/n.
    bus_gb_per_s = algorithmic_gb_per_s * 2 * (world_size - 1) / world_size

    stress_iterations = 0
    if args.stress_seconds > 0:
        side = 4096
        stress_left = torch.randn((side, side), device=device, dtype=torch.bfloat16) * 0.01
        stress_right = torch.randn((side, side), device=device, dtype=torch.bfloat16) * 0.01
        stress_output = torch.empty_like(stress_left)
        deadline = time.monotonic() + args.stress_seconds
        while time.monotonic() < deadline:
            torch.mm(stress_left, stress_right, out=stress_output)
            stress_iterations += 1
        torch.cuda.synchronize(device)

    rank_record = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": torch.cuda.get_device_name(local_rank),
        "capability": ".".join(str(item) for item in torch.cuda.get_device_capability(local_rank)),
        "bf16": bf16_result,
        "p2p_row": peer_access,
        "all_reduce": {
            "correct": scalar_ok and measured_correct,
            "median_ms": median_ms,
            "algorithmic_bandwidth_gb_per_s": algorithmic_gb_per_s,
            "bus_bandwidth_gb_per_s": bus_gb_per_s,
            "iterations": args.iterations,
            "buffer_mib": args.buffer_mib,
        },
        "stress": {
            "requested_seconds": args.stress_seconds,
            "iterations": stress_iterations,
        },
    }
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, rank_record)
    if rank == 0:
        rows = sorted(gathered, key=lambda item: item["rank"])
        matrix = [item["p2p_row"] for item in rows]
        query_complete = all(value is not None for row in matrix for value in row)
        full_mesh = query_complete and all(
            value
            for row_index, row in enumerate(matrix)
            for column_index, value in enumerate(row)
            if row_index != column_index
        )
        buses = [item["all_reduce"]["bus_bandwidth_gb_per_s"] for item in rows]
        payload = {
            "world_size": world_size,
            "backend": "nccl",
            "ranks": rows,
            "p2p": {
                "matrix": matrix,
                "query_complete": query_complete,
                "full_mesh": full_mesh,
            },
            "all_reduce": {
                "correct": all(item["all_reduce"]["correct"] for item in rows),
                "buffer_mib": args.buffer_mib,
                "iterations": args.iterations,
                "bus_bandwidth_min_gb_per_s": min(buses),
                "bus_bandwidth_median_gb_per_s": statistics.median(buses),
            },
        }
        print(
            WORKER_SENTINEL + json.dumps(payload, separators=(",", ":")),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


def manifest_command(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).resolve(strict=True)
    contract = load_contract(contract_path)
    root = Path(args.root)
    metadata = manifest_metadata(
        kind=args.kind,
        contract=contract,
        contract_path=contract_path,
        cuda_branch=args.cuda_branch,
    )
    if args.kind == "wheel-cache":
        branch = next(
            (item for item in contract["cuda_branches"] if item["name"] == args.cuda_branch),
            None,
        )
        inventory = wheel_cache_inventory(root, contract, branch)
        if inventory.get("ok") is not True:
            raise ValueError("wheel cache inventory failed: " + "; ".join(inventory["errors"]))
        metadata["wheel_inventory"] = inventory["summary"]
    manifest = build_file_manifest(root, kind=args.kind, metadata=metadata)
    atomic_write_json(Path(args.output), manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "output": Path(args.output).name,
                "kind": args.kind,
                "file_count": manifest["file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def verify_manifest_command(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).resolve(strict=True)
    contract = load_contract(contract_path)
    manifest_path = Path(args.manifest).resolve(strict=True)
    manifest = load_json(manifest_path)
    root = Path(args.root)
    result = verify_file_manifest(manifest, root)
    branch = next(
        (item for item in contract["cuda_branches"] if item["name"] == args.cuda_branch),
        None,
    )
    semantic_ok, semantic_errors = cache_semantics_ok(
        result,
        kind=str(manifest.get("kind")),
        contract=contract,
        contract_path=contract_path,
        branch=branch,
    )
    result["semantic_ok"] = semantic_ok
    result["errors"] = list(result["errors"]) + semantic_errors
    if manifest.get("kind") == "wheel-cache":
        inventory = wheel_cache_inventory(root, contract, branch)
        inventory_ok = inventory.get("ok") is True and manifest.get("metadata", {}).get(
            "wheel_inventory"
        ) == inventory.get("summary")
        result["wheel_inventory"] = inventory
        if not inventory_ok:
            result["errors"].extend(inventory.get("errors", []))
            if manifest.get("metadata", {}).get("wheel_inventory") != inventory.get("summary"):
                result["errors"].append("wheel inventory metadata mismatch")
        semantic_ok = semantic_ok and inventory_ok
        result["semantic_ok"] = semantic_ok
    result["ok"] = result["ok"] and semantic_ok
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run host-only or full readiness gates")
    run.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    run.add_argument("--phase", choices=("host", "full"), default="host")
    run.add_argument("--workdir", default=str(REPOSITORY_ROOT))
    run.add_argument(
        "--output",
        help="private JSON path; default is a unique artifacts/private run directory",
    )
    run.add_argument("--stress-seconds", type=float, default=300.0)
    run.add_argument("--distributed-timeout", type=float, default=900.0)
    run.add_argument("--network-timeout", type=float, default=10.0)
    run.add_argument("--offline", action="store_true")
    run.add_argument("--wheel-manifest")
    run.add_argument("--wheel-root")
    run.add_argument("--model-manifest")
    run.add_argument("--model-root")
    run.add_argument(
        "--provider-health-evidence-before",
        help="optional private provider evidence covering the preflight lookback",
    )
    run.add_argument(
        "--provider-health-evidence-before-sha256",
        help="detached SHA-256 file for the before evidence",
    )
    run.add_argument(
        "--provider-health-evidence-after",
        help="full phase evidence covering start through post-run capture; may appear during run",
    )
    run.add_argument(
        "--provider-health-evidence-after-sha256",
        help="detached SHA-256 file for the after evidence",
    )
    run.add_argument(
        "--provider-health-evidence-wait-seconds",
        type=float,
        default=30.0,
        help="wait for an atomically published after evidence pair (0 to 60 seconds)",
    )
    run.set_defaults(func=run_preflight)

    manifest = subparsers.add_parser(
        "manifest", help="create an exact, semantic cache SHA256 manifest"
    )
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--kind", choices=("wheel-cache", "model-snapshot"), required=True)
    manifest.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    manifest.add_argument("--cuda-branch", choices=("cu128", "cu126"))
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=manifest_command)

    verify = subparsers.add_parser(
        "verify-manifest", help="verify cache bytes and semantic provenance"
    )
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--root", required=True)
    verify.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    verify.add_argument("--cuda-branch", choices=("cu128", "cu126"))
    verify.set_defaults(func=verify_manifest_command)

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--expected-world-size", type=int, required=True)
    worker.add_argument("--stress-seconds", type=float, required=True)
    worker.add_argument("--buffer-mib", type=int, required=True)
    worker.add_argument("--warmup", type=int, required=True)
    worker.add_argument("--iterations", type=int, required=True)
    worker.set_defaults(func=distributed_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    for name in (
        "stress_seconds",
        "distributed_timeout",
        "network_timeout",
        "provider_health_evidence_wait_seconds",
    ):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and >=0")
    provider_wait = getattr(args, "provider_health_evidence_wait_seconds", 0.0)
    if provider_wait > 60:
        raise SystemExit("--provider-health-evidence-wait-seconds must be <=60")
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"preflight failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
