#!/usr/bin/env python3
"""Build and validate a data-minimized public ASR matrix summary.

The exporter accepts only a complete, validated plan/journal pair and the exact
18 artifacts produced by :mod:`matrix_aggregator`.  Private artifacts are used
only as inputs to validation and aggregation.  The public payload is rebuilt
from an explicit allowlist and contains no per-utterance identifiers, text,
tokens, audio fingerprints, prepared-input fingerprints, local paths, device
identifiers, Git branch, commands, or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.version import InvalidVersion, Version

from faster_glm_asr import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from faster_glm_asr.benchmarking import comparator, provenance
from faster_glm_asr.benchmarking import matrix_aggregator as aggregation
from faster_glm_asr.benchmarking import matrix_executor as executor
from faster_glm_asr.benchmarking import matrix_planner as planner
from faster_glm_asr.benchmarking import public_matrix_schema as public_schema
from faster_glm_asr.benchmarking import runner as benchmark_runner
from faster_glm_asr.data import librispeech

PUBLIC_SCHEMA = public_schema.PUBLIC_SCHEMA
EXPECTED_DATASET_LABEL = "librispeech-test-clean-3-v1"
PUBLIC_RESULTS_DIRECTORY = Path("benchmarks/results")
PRIVATE_ROOT = Path("artifacts/private")
MIB = 1024 * 1024

BUCKETS = public_schema.BUCKETS

# Ordered baseline -> candidate pairs.  The direction is the comparator's
# definition: speedup = baseline latency / candidate latency.  Values below one
# are valid evidence and are never clamped or relabelled.
COMPARISON_PAIRS = public_schema.COMPARISON_PAIRS
PACKAGE_KEYS = public_schema.PACKAGE_KEYS
FORBIDDEN_PUBLIC_KEYS = public_schema.FORBIDDEN_PUBLIC_KEYS
PREPARED_REQUEST_FAMILIES = {
    "hf": ("hf_cached", "hf_no_cache"),
    "custom": (
        "custom_full_prefix",
        "custom_greedy_full_prefix",
        "custom_tuple_cache",
        "custom_static_cache",
    ),
}
SAFE_PUBLIC_FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}\.json$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UPSTREAM_ID_RE = re.compile(r"^[0-9]+-[0-9]+-[0-9]+$")

MANIFEST_KEYS = {
    "sample_id",
    "audio_path",
    "audio_sha256",
    "duration_s",
    "language",
    "reference_text",
    "license_scope",
    "license_file_sha256",
    "upstream_url",
    "upstream_license_declared",
    "shareable",
    "duration_bucket",
    "direct_model_eligible",
    "manifest_parser_version",
    "dataset",
    "dataset_subset",
    "speaker_id",
    "chapter_id",
    "upstream_utterance_id",
    "transcript_file_sha256",
    "selection_version",
}
SELECTION_LOCK_KEYS = {
    "schema_version",
    "selection_version",
    "seed",
    "per_bucket",
    "duration_buckets",
    "candidate_counts",
    "candidate_inventory_sha256",
    "selected_upstream_ids",
    "selected_count",
    "license_file_name",
    "license_file_sha256",
    "upstream_url",
    "upstream_license_declared",
    "manifest_file",
    "manifest_sha256",
    "builder_source_sha256",
    "audio_copied",
    "publication_review_required",
}
DATA_CONFIG_KEYS = {
    "schema_version",
    "dataset",
    "subset",
    "upstream_page",
    "archive_url",
    "archive_file",
    "archive_size_bytes",
    "archive_md5",
    "archive_sha256",
    "upstream_checksum_url",
    "archive_member_count",
    "archive_root",
    "audio_format",
    "sample_rate_hz",
    "channels",
    "license",
    "license_file",
    "selection",
}
DATA_SELECTION_KEYS = {
    "version",
    "seed",
    "canonical_matrix_per_bucket",
    "quality_subset_per_bucket",
    "duration_buckets_seconds",
}
CANONICAL_DATA_CONFIG_PATH = Path("configs/data/librispeech-test-clean-v1.json")
LIBRISPEECH_BUILDER_PATH = Path("src/faster_glm_asr/data/librispeech.py")
DATASET_ARCHIVE_PATH = Path("artifacts/private/cache/datasets/librispeech/test-clean.tar.gz")
DATASET_EXTRACTED_ROOT = Path(
    "artifacts/private/cache/datasets/librispeech/extracted-v1/LibriSpeech"
)
DATASET_ARCHIVE_CONTRACT = {
    "size_bytes": 346663984,
    "sha256": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "member_count": 2840,
}
CORE_SOURCE_MODULES = (
    ("planner", planner, Path("src/faster_glm_asr/benchmarking/matrix_planner.py")),
    ("executor", executor, Path("src/faster_glm_asr/benchmarking/matrix_executor.py")),
    ("comparator", comparator, Path("src/faster_glm_asr/benchmarking/comparator.py")),
    ("aggregator", aggregation, Path("src/faster_glm_asr/benchmarking/matrix_aggregator.py")),
    ("runner", benchmark_runner, Path("src/faster_glm_asr/benchmarking/runner.py")),
    ("provenance", provenance, Path("src/faster_glm_asr/benchmarking/provenance.py")),
    ("librispeech", librispeech, LIBRISPEECH_BUILDER_PATH),
)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reparse_point(path: Path, *, follow_symlinks: bool = False) -> bool:
    try:
        attributes = path.stat(follow_symlinks=follow_symlinks).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _repo_path(path: Path, repository_root: Path, *, label: str) -> tuple[str, Path]:
    candidate = path if path.is_absolute() else repository_root / path
    for probe in (candidate, *candidate.parents):
        if probe == repository_root:
            break
        if probe.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository") from exc
    return relative.as_posix(), resolved


def _validate_executing_core_sources(evidence_root: Path) -> None:
    for label, module, relative in CORE_SOURCE_MODULES:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ValueError(f"executing {label} module has no source file")
        executing = Path(module_file).resolve()
        _evidence_relative, evidence = _repo_path(
            relative, evidence_root, label=f"evidence {label} source"
        )
        if not evidence.is_file():
            raise ValueError(f"evidence {label} source is missing or is a symbolic link")
        if _sha256_bytes(executing.read_bytes()) != _sha256_bytes(evidence.read_bytes()):
            raise ValueError(f"executing {label} source differs from the evidence checkout")


def _read_validated_configuration(
    plan_path: Path,
    *,
    evidence_root: Path,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    _relative, resolved_plan = _repo_path(plan_path, evidence_root, label="plan")
    plan, plan_raw = _load_json_object(resolved_plan, "benchmark plan")
    if _sha256_bytes(plan_raw) != expected_plan_sha256:
        raise ValueError("benchmark plan changed after matrix validation")
    matrix = plan.get("matrix_configuration")
    if not isinstance(matrix, dict) or set(matrix) != {"path", "file_sha256", "semantic_sha256"}:
        raise ValueError("benchmark plan matrix_configuration schema is invalid")
    relative, configuration_path = _repo_path(
        Path(str(matrix.get("path", ""))), evidence_root, label="matrix configuration"
    )
    if relative != matrix.get("path") or not configuration_path.is_file():
        raise ValueError("matrix configuration path binding is invalid")
    configuration, raw = _load_json_object(configuration_path, "matrix configuration")
    if _sha256_bytes(raw) != matrix.get("file_sha256"):
        raise ValueError("matrix configuration changed after matrix validation")
    validated = planner._validate_configuration(configuration)
    if planner._canonical_sha256(validated) != matrix.get("semantic_sha256"):
        raise ValueError("matrix configuration semantic binding is invalid")
    inputs = plan.get("inputs")
    manifest_binding = inputs.get("manifest") if isinstance(inputs, dict) else None
    if not isinstance(manifest_binding, dict) or set(manifest_binding) != {"path", "sha256"}:
        raise ValueError("benchmark plan manifest binding schema is invalid")
    manifest_relative, manifest_path = _repo_path(
        Path(validated["manifest"]), evidence_root, label="manifest"
    )
    if manifest_binding.get("path") != manifest_relative:
        raise ValueError("benchmark plan manifest path does not match matrix configuration")
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_binding.get("sha256") != _sha256_bytes(manifest_path.read_bytes())
    ):
        raise ValueError("benchmark plan manifest hash binding is invalid")
    return validated


def _load_complete_aggregate_set(
    aggregate_directory: Path,
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    evidence_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    relative, directory = _repo_path(
        aggregate_directory, evidence_root, label="aggregate directory"
    )
    relative_parts = Path(relative).parts
    if tuple(part.casefold() for part in relative_parts[:2]) != ("artifacts", "private"):
        raise ValueError("aggregate directory must be below artifacts/private")
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("aggregate directory must be a non-symlink directory")

    required_names = set(aggregation.FORMAL_OUTPUT_FILENAMES.values())
    entries = list(directory.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != required_names or any(
        not entry.is_file() or entry.is_symlink() for entry in entries
    ):
        raise ValueError(
            "aggregate directory must contain exactly the fixed 18 regular JSON artifacts; "
            f"missing={sorted(required_names - actual_names)}, "
            f"unexpected={sorted(actual_names - required_names)}"
        )

    admitted: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_bindings: list[dict[str, str]] = []
    for pair, filename in sorted(
        aggregation.FORMAL_OUTPUT_FILENAMES.items(), key=lambda item: item[1]
    ):
        payload, _raw = _load_json_object(directory / filename, f"aggregate {filename}")
        canonical_payload = _canonical_json_bytes(payload)
        if canonical_payload != _canonical_json_bytes(expected[pair]):
            raise ValueError(f"aggregate {filename} does not equal the validated rebuild")
        admitted[pair] = payload
        canonical_bindings.append(
            {"filename": filename, "canonical_sha256": _sha256_bytes(canonical_payload)}
        )
    aggregate_set_sha256 = _sha256_bytes(_canonical_json_bytes(canonical_bindings))
    return admitted, aggregate_set_sha256


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema is invalid")
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _strict_json_equal(left: Any, right: Any) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _load_manifest_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("LibriSpeech manifest must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("LibriSpeech manifest must be UTF-8 JSONL") from exc
    if len(lines) != len(BUCKETS) or any(not line.strip() for line in lines):
        raise ValueError("LibriSpeech manifest must contain exactly three nonblank rows")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_duplicate_rejecting_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {token}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"LibriSpeech manifest row {line_number} is invalid JSON") from exc
        rows.append(_require_exact_keys(value, MANIFEST_KEYS, f"manifest row {line_number}"))
    return rows, raw


def _validate_manifest_row(
    row: dict[str, Any],
    *,
    bucket: str,
    evidence_root: Path,
    manifest_path: Path,
    seen_ids: set[str],
    seen_audio: set[str],
) -> Path:
    upstream_id = row.get("upstream_utterance_id")
    if not isinstance(upstream_id, str) or UPSTREAM_ID_RE.fullmatch(upstream_id) is None:
        raise ValueError(f"LibriSpeech manifest {bucket} upstream ID is invalid")
    if row.get("sample_id") != f"librispeech-{upstream_id}" or upstream_id in seen_ids:
        raise ValueError(f"LibriSpeech manifest {bucket} sample identity is invalid")
    seen_ids.add(upstream_id)
    speaker_id, chapter_id, _utterance_id = upstream_id.split("-", 2)
    if row.get("speaker_id") != speaker_id or row.get("chapter_id") != chapter_id:
        raise ValueError(f"LibriSpeech manifest {bucket} speaker/chapter identity is invalid")

    duration = row.get("duration_s")
    if (
        type(duration) is not float
        or not math.isfinite(float(duration))
        or _bucket_for_duration(float(duration)) != bucket
        or row.get("duration_bucket") != bucket
    ):
        raise ValueError(f"LibriSpeech manifest {bucket} duration binding is invalid")
    expected_scalars = {
        "language": "en",
        "license_scope": (
            "LibriSpeech test-clean; CC BY 4.0; attribution/publication review required"
        ),
        "upstream_url": librispeech.UPSTREAM_URL,
        "upstream_license_declared": librispeech.UPSTREAM_LICENSE,
        "shareable": False,
        "direct_model_eligible": True,
        "manifest_parser_version": librispeech.MANIFEST_PARSER_VERSION,
        "dataset": "LibriSpeech",
        "dataset_subset": "test-clean",
        "selection_version": librispeech.SELECTION_VERSION,
    }
    for key, expected in expected_scalars.items():
        if not _strict_json_equal(row.get(key), expected):
            raise ValueError(f"LibriSpeech manifest {bucket} {key} is invalid")
    if not isinstance(row.get("reference_text"), str) or not row["reference_text"].strip():
        raise ValueError(f"LibriSpeech manifest {bucket} reference text is invalid")
    for key in ("audio_sha256", "license_file_sha256", "transcript_file_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"LibriSpeech manifest {bucket} {key} is invalid")

    audio_value = row.get("audio_path")
    if (
        not isinstance(audio_value, str)
        or not audio_value
        or "\x00" in audio_value
        or "\\" in audio_value
        or not audio_value.casefold().endswith(".flac")
        or public_schema.ABSOLUTE_PATH_RE.match(audio_value)
    ):
        raise ValueError(f"LibriSpeech manifest {bucket} audio path is invalid")
    _audio_relative, audio_path = _repo_path(
        manifest_path.parent / Path(audio_value),
        evidence_root,
        label=f"LibriSpeech manifest {bucket} audio",
    )
    if not audio_path.is_file():
        raise ValueError(f"LibriSpeech manifest {bucket} audio is not a regular file")
    with audio_path.open("rb") as stream:
        if stream.read(4) != b"fLaC":
            raise ValueError(f"LibriSpeech manifest {bucket} audio is not FLAC")
    actual_audio_sha256 = _sha256_bytes(audio_path.read_bytes())
    if actual_audio_sha256 != row["audio_sha256"] or actual_audio_sha256 in seen_audio:
        raise ValueError(f"LibriSpeech manifest {bucket} audio hash binding is invalid")
    seen_audio.add(actual_audio_sha256)
    return audio_path


def _expected_data_configuration() -> dict[str, Any]:
    contract = DATASET_ARCHIVE_CONTRACT
    if (
        not isinstance(contract, dict)
        or set(contract) != {"size_bytes", "sha256", "member_count"}
        or not _positive_int(contract.get("size_bytes"))
        or not _positive_int(contract.get("member_count"))
        or not isinstance(contract.get("sha256"), str)
        or SHA256_RE.fullmatch(contract["sha256"]) is None
    ):
        raise ValueError("LibriSpeech archive contract is invalid")
    return {
        "schema_version": "public-asr-source-lock-v0.1",
        "dataset": "LibriSpeech",
        "subset": "test-clean",
        "upstream_page": librispeech.UPSTREAM_URL,
        "archive_url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "archive_file": "test-clean.tar.gz",
        "archive_size_bytes": contract["size_bytes"],
        "archive_md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "archive_sha256": contract["sha256"],
        "upstream_checksum_url": "https://www.openslr.org/resources/12/md5sum.txt",
        "archive_member_count": contract["member_count"],
        "archive_root": "LibriSpeech",
        "audio_format": "FLAC",
        "sample_rate_hz": 16000,
        "channels": 1,
        "license": librispeech.UPSTREAM_LICENSE,
        "license_file": "LibriSpeech/LICENSE.TXT",
        "selection": {
            "version": librispeech.SELECTION_VERSION,
            "seed": librispeech.DEFAULT_SEED,
            "canonical_matrix_per_bucket": 1,
            "quality_subset_per_bucket": 8,
            "duration_buckets_seconds": [
                [0.0, 10.0],
                [10.0, 20.0],
                [20.0, 30.0],
            ],
        },
    }


def _validate_data_configuration(evidence_root: Path) -> dict[str, Any]:
    _relative, path = _repo_path(
        CANONICAL_DATA_CONFIG_PATH,
        evidence_root,
        label="canonical LibriSpeech data configuration",
    )
    configuration, _raw = _load_json_object(path, "canonical LibriSpeech data configuration")
    _require_exact_keys(configuration, DATA_CONFIG_KEYS, "canonical LibriSpeech configuration")
    _require_exact_keys(
        configuration.get("selection"),
        DATA_SELECTION_KEYS,
        "canonical LibriSpeech selection configuration",
    )
    if not _strict_json_equal(configuration, _expected_data_configuration()):
        raise ValueError("canonical LibriSpeech data configuration is not the admitted revision")
    return configuration


def _canonical_tar_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
    ):
        raise ValueError("LibriSpeech archive contains an unsafe member path")
    raw = value.rstrip("/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and re.fullmatch(r"[A-Za-z]:", path.parts[0]) is not None)
        or path.as_posix() != raw
    ):
        raise ValueError("LibriSpeech archive contains an unsafe member path")
    return path.as_posix()


def _tar_regular_payload(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[str, bytes]:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError("LibriSpeech archive regular member has no payload")
    digest = hashlib.sha256()
    prefix = b""
    size = 0
    with stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if len(prefix) < 4:
                prefix += block[: 4 - len(prefix)]
            size += len(block)
            digest.update(block)
    if size != member.size:
        raise ValueError("LibriSpeech archive member size is inconsistent")
    return digest.hexdigest(), prefix


def _walk_regular_tree(root: Path, *, label: str) -> tuple[set[str], set[str]]:
    if root.is_symlink() or _is_reparse_point(root) or not root.is_dir():
        raise ValueError(f"{label} must be a regular non-reparse directory")
    files: set[str] = set()
    directories: set[str] = set()
    pending: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
    while pending:
        directory, relative_parent = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = relative_parent / entry.name
                relative_name = relative.as_posix()
                if entry.is_symlink() or _is_reparse_point(path):
                    raise ValueError(f"{label} contains a symbolic link or reparse point")
                if entry.is_dir(follow_symlinks=False):
                    directories.add(relative_name)
                    pending.append((path, relative))
                elif entry.is_file(follow_symlinks=False):
                    files.add(relative_name)
                else:
                    raise ValueError(f"{label} contains a non-regular filesystem entry")
    return files, directories


def _validate_dataset_archive(
    evidence_root: Path,
    *,
    data_configuration: Mapping[str, Any],
    test_clean_root: Path,
) -> str:
    _archive_relative, archive_path = _repo_path(
        DATASET_ARCHIVE_PATH,
        evidence_root,
        label="LibriSpeech source archive",
    )
    if archive_path.is_symlink() or _is_reparse_point(archive_path) or not archive_path.is_file():
        raise ValueError("LibriSpeech source archive must be a regular non-reparse file")
    archive_size = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)
    if archive_size != data_configuration.get(
        "archive_size_bytes"
    ) or archive_sha256 != data_configuration.get("archive_sha256"):
        raise ValueError("LibriSpeech source archive size/SHA-256 binding is invalid")

    _dataset_relative, dataset_root = _repo_path(
        DATASET_EXTRACTED_ROOT,
        evidence_root,
        label="extracted LibriSpeech dataset root",
    )
    expected_test_clean_root = dataset_root / "test-clean"
    if test_clean_root.resolve() != expected_test_clean_root.resolve():
        raise ValueError("LibriSpeech manifest does not bind the fixed extracted dataset root")

    archive_files: dict[str, tuple[int, str]] = {}
    archive_directories: set[str] = set()
    license_record: tuple[int, str] | None = None
    seen_names: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) != data_configuration.get("archive_member_count"):
                raise ValueError("LibriSpeech source archive member count is invalid")
            for member in members:
                name = _canonical_tar_name(member.name)
                if name in seen_names:
                    raise ValueError("LibriSpeech source archive contains a duplicate member")
                seen_names.add(name)
                is_regular = member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                is_directory = member.type == tarfile.DIRTYPE
                if not is_regular and not is_directory:
                    raise ValueError("LibriSpeech source archive contains a link or special member")

                test_clean_prefix = "LibriSpeech/test-clean/"
                is_test_clean = name.startswith(test_clean_prefix)
                is_license = name == "LibriSpeech/LICENSE.TXT"
                if is_directory:
                    if is_test_clean:
                        archive_directories.add(name.removeprefix(test_clean_prefix))
                    continue
                digest, prefix = _tar_regular_payload(archive, member)
                if is_test_clean:
                    relative = name.removeprefix(test_clean_prefix)
                    if not relative:
                        raise ValueError("LibriSpeech archive test-clean member is invalid")
                    if name.endswith(".flac"):
                        if prefix != b"fLaC":
                            raise ValueError("LibriSpeech archive audio is not FLAC")
                    elif not name.endswith(".trans.txt"):
                        raise ValueError("LibriSpeech archive test-clean contains a non-FLAC asset")
                    archive_files[relative] = (member.size, digest)
                elif is_license:
                    license_record = (member.size, digest)
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise ValueError("LibriSpeech source archive is not a valid gzip tar archive") from exc

    if not archive_files or license_record is None:
        raise ValueError("LibriSpeech source archive lacks test-clean data or LICENSE.TXT")
    extracted_files, extracted_directories = _walk_regular_tree(
        expected_test_clean_root,
        label="extracted LibriSpeech test-clean tree",
    )
    expected_directories: set[str] = set()
    for relative in archive_files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    expected_directories.update(name for name in archive_directories if name)
    if extracted_files != set(archive_files) or extracted_directories != expected_directories:
        raise ValueError("extracted LibriSpeech test-clean tree has extra or missing entries")
    for relative, (expected_size, expected_sha256) in archive_files.items():
        extracted = expected_test_clean_root / Path(relative)
        if extracted.stat().st_size != expected_size or _sha256_file(extracted) != expected_sha256:
            raise ValueError("extracted LibriSpeech member differs from the source archive")

    license_path = dataset_root / "LICENSE.TXT"
    if (
        license_path.is_symlink()
        or _is_reparse_point(license_path)
        or not license_path.is_file()
        or license_path.stat().st_size != license_record[0]
        or _sha256_file(license_path) != license_record[1]
    ):
        raise ValueError("extracted LibriSpeech LICENSE.TXT differs from the source archive")
    return archive_sha256


def _validate_selection_lock(
    lock: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    manifest_raw: bytes,
    data_configuration: dict[str, Any],
    test_clean_root: Path,
) -> None:
    _require_exact_keys(lock, SELECTION_LOCK_KEYS, "LibriSpeech selection lock")
    bucket_definitions = [
        {
            "name": name,
            "lower_s_inclusive": lower,
            "upper_s": upper,
            "upper_inclusive": inclusive,
        }
        for name, lower, upper, inclusive in librispeech.BUCKETS
    ]
    license_path = test_clean_root.parent / Path(data_configuration["license_file"]).name
    if not license_path.is_file() or license_path.is_symlink():
        raise ValueError("LibriSpeech source license is missing from the admitted dataset root")
    license_sha256 = _sha256_bytes(license_path.read_bytes())
    candidates = librispeech.discover(test_clean_root)
    candidate_counts = {
        bucket: sum(item.duration_bucket == bucket for item in candidates)
        for bucket, _lower, _upper, _inclusive in librispeech.BUCKETS
    }
    candidate_inventory = [
        {
            "sample_id": item.sample_id,
            "audio_sha256": item.audio_sha256,
            "duration_s": round(item.duration_s, 9),
            "duration_bucket": item.duration_bucket,
            "transcript_file_sha256": item.transcript_file_sha256,
        }
        for item in sorted(candidates, key=lambda value: value.sample_id)
    ]
    selected = librispeech.select(
        candidates,
        per_bucket=data_configuration["selection"]["canonical_matrix_per_bucket"],
        seed=data_configuration["selection"]["seed"],
    )
    rebuilt_rows = librispeech._manifest_records(selected, manifest_path.parent, license_sha256)
    if not _strict_json_equal(rows, rebuilt_rows):
        raise ValueError("LibriSpeech manifest does not equal a fresh builder reconstruction")

    expected = {
        "schema_version": librispeech.SCHEMA_VERSION,
        "selection_version": librispeech.SELECTION_VERSION,
        "seed": data_configuration["selection"]["seed"],
        "per_bucket": data_configuration["selection"]["canonical_matrix_per_bucket"],
        "duration_buckets": bucket_definitions,
        "selected_upstream_ids": [row["upstream_utterance_id"] for row in rows],
        "selected_count": len(rows),
        "candidate_counts": candidate_counts,
        "candidate_inventory_sha256": librispeech._canonical_sha256(candidate_inventory),
        "license_file_name": license_path.name,
        "license_file_sha256": license_sha256,
        "upstream_url": librispeech.UPSTREAM_URL,
        "upstream_license_declared": librispeech.UPSTREAM_LICENSE,
        "manifest_file": "manifest.jsonl",
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "audio_copied": False,
        "publication_review_required": True,
    }
    for key, value in expected.items():
        if not _strict_json_equal(lock.get(key), value):
            raise ValueError(f"LibriSpeech selection lock {key} binding is invalid")
    # The builder digest is a generation-time claim whose historical bytes may
    # no longer exist.  Current executable semantics are established separately
    # by the plan-bound source map, byte-equality with the executing module, and
    # the fresh reconstruction above; do not substitute a current-source digest
    # for this otherwise unverifiable historical audit handle.
    for key in ("candidate_inventory_sha256", "builder_source_sha256"):
        value = lock.get(key)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"LibriSpeech selection lock {key} is invalid")
    if manifest_path.name != lock["manifest_file"]:
        raise ValueError("LibriSpeech selection lock does not name the plan-bound manifest")
    if any(row["license_file_sha256"] != lock["license_file_sha256"] for row in rows):
        raise ValueError("LibriSpeech manifest rows do not share the selection-lock license")


def _manifest_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "sample_id",
        "audio_path",
        "audio_sha256",
        "reference_text",
        "language",
        "duration_s",
        "shareable",
    }
    return {
        **{key: value for key, value in row.items() if key not in excluded},
        "shareable": row["shareable"],
    }


def _validate_librispeech_provenance(
    configuration: Mapping[str, Any],
    *,
    anchor: Mapping[str, Mapping[str, Any]],
    evidence_root: Path,
) -> str:
    _relative, manifest_path = _repo_path(
        Path(str(configuration["manifest"])), evidence_root, label="LibriSpeech manifest"
    )
    rows, manifest_raw = _load_manifest_rows(manifest_path)
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    audio_paths: list[Path] = []
    for row, (bucket, _lower, _upper, _inclusive) in zip(rows, BUCKETS, strict=True):
        audio_paths.append(
            _validate_manifest_row(
                row,
                bucket=bucket,
                evidence_root=evidence_root,
                manifest_path=manifest_path,
                seen_ids=seen_ids,
                seen_audio=seen_audio,
            )
        )

    test_clean_roots = set()
    for row, audio_path in zip(rows, audio_paths, strict=True):
        expected_parent = audio_path.parent
        if (
            expected_parent.name != row["chapter_id"]
            or expected_parent.parent.name != row["speaker_id"]
        ):
            raise ValueError("LibriSpeech audio path does not match its speaker/chapter identity")
        test_clean_roots.add(expected_parent.parent.parent)
    if len(test_clean_roots) != 1:
        raise ValueError("LibriSpeech manifest audio does not share one test-clean root")
    test_clean_root = test_clean_roots.pop()
    if test_clean_root.name != "test-clean" or not test_clean_root.is_dir():
        raise ValueError("LibriSpeech manifest audio root is not canonical test-clean")

    data_configuration = _validate_data_configuration(evidence_root)
    archive_sha256 = _validate_dataset_archive(
        evidence_root,
        data_configuration=data_configuration,
        test_clean_root=test_clean_root,
    )
    lock_path = manifest_path.parent / "selection-lock.json"
    lock, _raw = _load_json_object(lock_path, "LibriSpeech selection lock")
    _validate_selection_lock(
        lock,
        rows=rows,
        manifest_path=manifest_path,
        manifest_raw=manifest_raw,
        data_configuration=data_configuration,
        test_clean_root=test_clean_root,
    )

    for row, (bucket, _lower, _upper, _inclusive) in zip(rows, BUCKETS, strict=True):
        sample = anchor[bucket]
        expected_fields = {
            "sample_id": row["sample_id"],
            "audio_sha256": row["audio_sha256"],
            "language": row["language"],
            "duration_s": row["duration_s"],
            "metadata": _manifest_metadata(row),
        }
        actual_fields = {key: sample.get(key) for key in expected_fields}
        if not _strict_json_equal(actual_fields, expected_fields):
            raise ValueError(f"aggregate sample anchor does not match manifest row for {bucket}")
    return archive_sha256


def _bucket_for_duration(duration_s: float) -> str | None:
    for name, lower, upper, include_upper in BUCKETS:
        if duration_s >= lower and (duration_s <= upper if include_upper else duration_s < upper):
            return name
    return None


def _bucket_samples(payload: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != len(BUCKETS):
        raise ValueError(f"{label} must contain exactly one utterance per duration bucket")
    result: dict[str, Mapping[str, Any]] = {}
    for index, sample in enumerate(samples):
        location = f"{label}.samples[{index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{location} must be an object")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{location}.metadata must be an object")
        bucket = metadata.get("duration_bucket")
        if bucket not in {item[0] for item in BUCKETS}:
            raise ValueError(f"{location} has an unsupported duration bucket")
        duration = sample.get("duration_s")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or _bucket_for_duration(float(duration)) != bucket
        ):
            raise ValueError(f"{location} duration does not match its duration bucket")
        if bucket in result:
            raise ValueError(f"{label} contains a duplicate duration bucket {bucket!r}")
        result[str(bucket)] = sample
    expected_buckets = {item[0] for item in BUCKETS}
    if set(result) != expected_buckets:
        raise ValueError(f"{label} has missing duration buckets")
    return result


def _validate_matrix_identity(
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    anchor_payload = artifacts[(planner.IMPLEMENTATIONS[0], "canonical_latency")]
    anchor = _bucket_samples(anchor_payload, "matrix anchor")
    identity_fields = (
        "sample_id",
        "audio_sha256",
        "language",
        "duration_s",
        "metadata",
    )
    output_fields = (
        "generated_tokens",
        "generated_token_ids",
        "hypothesis",
    )
    implementation_families = {
        implementation: family
        for family, implementations in PREPARED_REQUEST_FAMILIES.items()
        for implementation in implementations
    }
    if set(implementation_families) != set(planner.IMPLEMENTATIONS) or len(
        implementation_families
    ) != sum(len(implementations) for implementations in PREPARED_REQUEST_FAMILIES.values()):
        raise RuntimeError("prepared-request families must partition the formal implementations")
    prepared_anchors = {
        family: _bucket_samples(
            artifacts[(implementations[0], "canonical_latency")],
            f"{family} prepared-request anchor",
        )
        for family, implementations in PREPARED_REQUEST_FAMILIES.items()
    }
    for pair, payload in artifacts.items():
        current = _bucket_samples(payload, f"aggregate {pair[0]}/{pair[1]}")
        family = implementation_families[pair[0]]
        for bucket, _lower, _upper, _inclusive in BUCKETS:
            for field in identity_fields:
                if not _strict_json_equal(current[bucket].get(field), anchor[bucket].get(field)):
                    raise ValueError(
                        f"matrix-wide sample identity differs for {pair[0]}/{pair[1]}/{bucket}"
                    )
            for field in output_fields:
                if not _strict_json_equal(current[bucket].get(field), anchor[bucket].get(field)):
                    raise ValueError(
                        f"matrix-wide token output differs for {pair[0]}/{pair[1]}/{bucket}"
                    )
            if not _strict_json_equal(
                current[bucket].get("prepared_request"),
                prepared_anchors[family][bucket].get("prepared_request"),
            ):
                raise ValueError(
                    "prepared request differs within "
                    f"{family} family for {pair[0]}/{pair[1]}/{bucket}"
                )
    return anchor


def _numeric_version(value: Any, label: str) -> str:
    if not isinstance(value, str) or public_schema.NUMERIC_VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a numeric dotted version")
    return value


def _normalized_package_version(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a PEP 440 version")
    try:
        normalized = str(Version(value))
    except InvalidVersion as exc:
        raise ValueError(f"{label} must be a PEP 440 version") from exc
    public = normalized.split("+", 1)[0]
    if public_schema.PUBLIC_PEP440_RE.fullmatch(public) is None:
        raise ValueError(f"{label} does not normalize to the strict public PEP 440 format")
    return public


def _public_environment(environment: Any) -> tuple[dict[str, Any], dict[str, Any], int]:
    if not isinstance(environment, dict):
        raise ValueError("benchmark environment must be an object")
    device_name = environment.get("device_name")
    if device_name not in {"NVIDIA GeForce RTX 3090", "NVIDIA RTX 3090"}:
        raise ValueError("public matrix requires an NVIDIA GeForce RTX 3090")
    total_memory = environment.get("total_memory_bytes")
    if (
        not isinstance(total_memory, int)
        or isinstance(total_memory, bool)
        or not 23_000 * MIB <= total_memory <= 25_000 * MIB
    ):
        raise ValueError("RTX 3090 observed memory is outside the 24 GiB-class gate")
    if environment.get("compute_capability") != [8, 6]:
        raise ValueError("RTX 3090 compute capability must be 8.6")

    package_versions = environment.get("package_versions")
    if not isinstance(package_versions, dict) or set(package_versions) != PACKAGE_KEYS:
        raise ValueError("software package version schema is incomplete")
    packages = {
        package: _normalized_package_version(package_versions[package], f"package {package}")
        for package in sorted(PACKAGE_KEYS)
    }
    drivers = environment.get("nvidia_driver_versions")
    if not isinstance(drivers, list) or len(drivers) != 1:
        raise ValueError("matrix must report exactly one common NVIDIA driver version")
    cudnn = environment.get("cudnn")
    if not isinstance(cudnn, int) or isinstance(cudnn, bool) or cudnn <= 0:
        raise ValueError("cuDNN version must be a positive integer")
    return (
        {
            "accelerator_model": "NVIDIA GeForce RTX 3090",
            "nominal_memory_gib": 24,
            "compute_capability": "8.6",
        },
        {
            "python": _numeric_version(environment.get("python"), "Python version"),
            "cuda_runtime": _numeric_version(environment.get("torch_cuda"), "CUDA runtime"),
            "cudnn_version": cudnn,
            "nvidia_driver_version": _numeric_version(drivers[0], "NVIDIA driver"),
            "packages": packages,
        },
        total_memory,
    )


def _mib(value: float) -> float:
    return float(value) / MIB


def _bounded_memory_int(value: Any, *, total_memory_bytes: int, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > total_memory_bytes
    ):
        raise ValueError(f"{label} is outside physical device memory")
    return value


def _validate_raw_memory_evidence(
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    total_memory_bytes: int,
) -> None:
    for pair, payload in artifacts.items():
        environment = payload.get("environment")
        if (
            not isinstance(environment, dict)
            or environment.get("total_memory_bytes") != total_memory_bytes
        ):
            raise ValueError(f"aggregate {pair[0]}/{pair[1]} physical memory identity differs")
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"aggregate {pair[0]}/{pair[1]} samples are invalid")
        for sample_index, sample in enumerate(samples):
            observations = sample.get("observations") if isinstance(sample, dict) else None
            if not isinstance(observations, list):
                raise ValueError(
                    f"aggregate {pair[0]}/{pair[1]} sample {sample_index} observations are invalid"
                )
            for observation_index, observation in enumerate(observations):
                label = (
                    f"aggregate {pair[0]}/{pair[1]} sample {sample_index} "
                    f"observation {observation_index}"
                )
                memory = observation.get("memory_bytes") if isinstance(observation, dict) else None
                if not isinstance(memory, dict):
                    raise ValueError(f"{label} allocator memory is invalid")
                snapshots: dict[str, tuple[int, int]] = {}
                for snapshot_name in ("before_prepare", "after_prepare", "after_generation"):
                    snapshot = memory.get(snapshot_name)
                    if not isinstance(snapshot, dict):
                        raise ValueError(f"{label} {snapshot_name} allocator snapshot is invalid")
                    allocated = _bounded_memory_int(
                        snapshot.get("allocated"),
                        total_memory_bytes=total_memory_bytes,
                        label=f"{label} {snapshot_name}.allocated",
                    )
                    reserved = _bounded_memory_int(
                        snapshot.get("reserved"),
                        total_memory_bytes=total_memory_bytes,
                        label=f"{label} {snapshot_name}.reserved",
                    )
                    if allocated > reserved:
                        raise ValueError(f"{label} {snapshot_name} allocated exceeds reserved")
                    snapshots[snapshot_name] = (allocated, reserved)
                allocated_peak = _bounded_memory_int(
                    memory.get("generation_allocated_peak"),
                    total_memory_bytes=total_memory_bytes,
                    label=f"{label} generation allocated peak",
                )
                reserved_peak = _bounded_memory_int(
                    memory.get("generation_reserved_peak"),
                    total_memory_bytes=total_memory_bytes,
                    label=f"{label} generation reserved peak",
                )
                allocated_delta = _bounded_memory_int(
                    memory.get("generation_allocated_peak_delta"),
                    total_memory_bytes=total_memory_bytes,
                    label=f"{label} generation allocated peak delta",
                )
                if allocated_peak > reserved_peak:
                    raise ValueError(f"{label} generation allocated peak exceeds reserved peak")
                if allocated_peak < max(
                    snapshots["after_prepare"][0], snapshots["after_generation"][0]
                ):
                    raise ValueError(f"{label} generation allocated peak is below a snapshot")
                if reserved_peak < max(
                    snapshots["after_prepare"][1], snapshots["after_generation"][1]
                ):
                    raise ValueError(f"{label} generation reserved peak is below a snapshot")
                expected_delta = max(0, allocated_peak - snapshots["after_prepare"][0])
                if allocated_delta != expected_delta:
                    raise ValueError(f"{label} generation allocated peak delta is inconsistent")

                nvml = observation.get("nvml_process_memory")
                if not isinstance(nvml, dict):
                    raise ValueError(f"{label} NVML process memory is invalid")
                if nvml.get("enabled") is not True:
                    continue
                samples_value = nvml.get("samples")
                sample_count = nvml.get("sample_count")
                if (
                    not isinstance(samples_value, list)
                    or not isinstance(sample_count, int)
                    or isinstance(sample_count, bool)
                    or sample_count != len(samples_value)
                    or sample_count <= 0
                ):
                    raise ValueError(f"{label} NVML process samples are inconsistent")
                used_values: list[int] = []
                for poll_index, poll in enumerate(samples_value):
                    used_values.append(
                        _bounded_memory_int(
                            poll.get("used_bytes") if isinstance(poll, dict) else None,
                            total_memory_bytes=total_memory_bytes,
                            label=f"{label} NVML sample {poll_index}.used_bytes",
                        )
                    )
                first = _bounded_memory_int(
                    nvml.get("first_used_bytes"),
                    total_memory_bytes=total_memory_bytes,
                    label=f"{label} NVML first used bytes",
                )
                last = _bounded_memory_int(
                    nvml.get("last_used_bytes"),
                    total_memory_bytes=total_memory_bytes,
                    label=f"{label} NVML last used bytes",
                )
                peak = _bounded_memory_int(
                    nvml.get("observed_peak_used_bytes"),
                    total_memory_bytes=total_memory_bytes,
                    label=f"{label} NVML observed peak used bytes",
                )
                if first != used_values[0] or last != used_values[-1] or peak != max(used_values):
                    raise ValueError(f"{label} NVML process memory summary is inconsistent")


def _request_memory_summary(
    sample: Mapping[str, Any], *, total_memory_bytes: int
) -> dict[str, Any]:
    observations = sample["observations"]
    allocator_values: list[float] = []
    process_values: list[float] = []
    poll_counts: list[int] = []
    for observation in observations:
        allocator_values.append(
            float(observation["memory_bytes"]["generation_allocated_peak_delta"])
        )
        nvml = observation["nvml_process_memory"]
        if nvml.get("scope") != "pre-prepare-sync-through-post-decode-output-hash":
            raise ValueError("request-memory NVML scope is unsupported")
        if nvml.get("peak_semantics") != (
            "polling-observed lower bound; not an allocator or exact peak"
        ):
            raise ValueError("request-memory NVML peak semantics are unsupported")
        process_peak = nvml.get("observed_peak_used_bytes")
        polls = nvml.get("sample_count")
        if (
            not isinstance(process_peak, (int, float))
            or isinstance(process_peak, bool)
            or not math.isfinite(float(process_peak))
            or float(process_peak) < 0
            or not isinstance(polls, int)
            or isinstance(polls, bool)
            or polls <= 0
        ):
            raise ValueError("request-memory observation is invalid")
        process_values.append(float(process_peak))
        poll_counts.append(polls)
    allocator = aggregation._summary(allocator_values)
    process = aggregation._summary(process_values)
    result = {
        "observations": len(observations),
        "allocator_peak_delta_p50_mib": _mib(float(allocator["p50"])),
        "allocator_peak_delta_p95_mib": _mib(float(allocator["p95"])),
        "process_peak_observed_p50_mib": _mib(float(process["p50"])),
        "process_peak_observed_p95_mib": _mib(float(process["p95"])),
        "minimum_poll_samples_per_request": min(poll_counts),
        "process_peak_observed_is_lower_bound": True,
    }
    physical_mib = _mib(total_memory_bytes)
    for key in (
        "allocator_peak_delta_p50_mib",
        "allocator_peak_delta_p95_mib",
        "process_peak_observed_p50_mib",
        "process_peak_observed_p95_mib",
    ):
        if result[key] > physical_mib:
            raise ValueError(f"public request-memory metric {key} exceeds physical memory")
    return result


def _result_rows(
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    total_memory_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for implementation in planner.IMPLEMENTATIONS:
        canonical = _bucket_samples(
            artifacts[(implementation, "canonical_latency")], f"canonical {implementation}"
        )
        memory = _bucket_samples(
            artifacts[(implementation, "request_memory")], f"memory {implementation}"
        )
        bucket_rows = []
        for bucket, _lower, _upper, _inclusive in BUCKETS:
            canonical_sample = canonical[bucket]
            generation = canonical_sample["summary"]["generation_ms"]
            rtf = canonical_sample["summary"]["rtf_generation"]
            bucket_rows.append(
                {
                    "name": bucket,
                    "canonical": {
                        "observations": int(generation["n"]),
                        "generation_p50_ms": float(generation["p50"]),
                        "generation_p95_ms": float(generation["p95"]),
                        "rtf_p50": float(rtf["p50"]),
                        "rtf_p95": float(rtf["p95"]),
                    },
                    "request_memory": _request_memory_summary(
                        memory[bucket], total_memory_bytes=total_memory_bytes
                    ),
                }
            )
        rows.append({"implementation": implementation, "duration_buckets": bucket_rows})
    return rows


def _comparison_rows(
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline_name, candidate_name in COMPARISON_PAIRS:
        baseline = artifacts[(baseline_name, "canonical_latency")]
        candidate = artifacts[(candidate_name, "canonical_latency")]
        comparison = comparator.compare(baseline, candidate)
        baseline_buckets = _bucket_samples(baseline, f"comparison baseline {baseline_name}")
        sample_to_bucket = {
            str(sample["sample_id"]): bucket for bucket, sample in baseline_buckets.items()
        }
        public_buckets: dict[str, dict[str, Any]] = {}
        for private_row in comparison["samples"]:
            bucket = sample_to_bucket.get(str(private_row.get("sample_id", "")))
            if bucket is None or bucket in public_buckets:
                raise ValueError("comparison samples do not map one-to-one onto duration buckets")
            public_buckets[bucket] = {
                "name": bucket,
                "generation_p50_speedup": float(private_row["p50_speedup"]),
                "generation_p95_speedup": float(private_row["p95_speedup"]),
            }
        if set(public_buckets) != {item[0] for item in BUCKETS}:
            raise ValueError("comparison is missing a duration bucket")
        rows.append(
            {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "exact_token_parity": True,
                "speedup_definition": "baseline latency / candidate latency",
                "duration_buckets": [public_buckets[item[0]] for item in BUCKETS],
            }
        )
    return rows


def _bucket_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "minimum_seconds": lower,
            "maximum_seconds": upper,
            "maximum_inclusive": include_upper,
        }
        for name, lower, upper, include_upper in BUCKETS
    ]


def _executing_source_sha256(module: Any, *, label: str) -> str:
    source_value = getattr(module, "__file__", None)
    if not isinstance(source_value, str):
        raise ValueError(f"executing {label} module has no source file")
    source = Path(source_value)
    if source.is_symlink() or _is_reparse_point(source) or not source.is_file():
        raise ValueError(f"executing {label} source is not a regular file")
    return _sha256_file(source)


def _build_public_evidence(
    *,
    plan_path: Path,
    journal_path: Path,
    evidence_root: Path,
    matrix_aggregation: Mapping[str, Any],
    aggregate_set_sha256: str,
    dataset_archive_sha256: str,
) -> dict[str, str]:
    plan, plan_raw = _load_json_object(plan_path, "benchmark plan")
    _journal, journal_raw = _load_json_object(journal_path, "run journal")
    plan_sha256 = _sha256_bytes(plan_raw)
    journal_sha256 = _sha256_bytes(journal_raw)
    if (
        matrix_aggregation.get("plan_sha256") != plan_sha256
        or matrix_aggregation.get("journal_sha256") != journal_sha256
    ):
        raise ValueError("matrix aggregation plan/journal evidence binding is invalid")

    repository = plan.get("repository")
    if (
        not isinstance(repository, dict)
        or set(repository) != {"commit", "dirty"}
        or not isinstance(repository.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", repository["commit"]) is None
        or repository.get("dirty") is not False
    ):
        raise ValueError("benchmark plan Git evidence is invalid")
    inputs = plan.get("inputs")
    sources = inputs.get("sources") if isinstance(inputs, dict) else None
    if not isinstance(sources, dict):
        raise ValueError("benchmark plan source-map evidence is invalid")
    rebuilt_sources = benchmark_runner._source_metadata(evidence_root)
    expected_source_lock = {
        **rebuilt_sources,
        "lock_sha256": planner._canonical_sha256(rebuilt_sources),
    }
    if not _strict_json_equal(sources, expected_source_lock):
        raise ValueError("benchmark plan source-map evidence changed after execution")
    source_map_sha256 = rebuilt_sources.get("aggregate_sha256")
    if not isinstance(source_map_sha256, str) or SHA256_RE.fullmatch(source_map_sha256) is None:
        raise ValueError("benchmark source-map digest is invalid")
    for label, value in (
        ("aggregate set", aggregate_set_sha256),
        ("dataset archive", dataset_archive_sha256),
    ):
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{label} digest is invalid")
    return {
        "git_commit": repository["commit"],
        "source_map_sha256": source_map_sha256,
        "plan_sha256": plan_sha256,
        "journal_sha256": journal_sha256,
        "aggregate_set_sha256": aggregate_set_sha256,
        "dataset_archive_sha256": dataset_archive_sha256,
        "exporter_source_sha256": _executing_source_sha256(sys.modules[__name__], label="exporter"),
        "schema_source_sha256": _executing_source_sha256(public_schema, label="public schema"),
    }


def build_public_summary(
    plan_path: Path,
    journal_path: Path,
    aggregate_directory: Path,
    *,
    evidence_root: Path,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all private evidence and rebuild one public allowlisted summary."""
    if evidence_root.is_symlink() or _is_reparse_point(evidence_root) or not evidence_root.is_dir():
        raise ValueError("evidence root must be an existing non-reparse directory")
    evidence_root = evidence_root.resolve()
    _validate_executing_core_sources(evidence_root)
    _plan_relative, resolved_plan = _repo_path(plan_path, evidence_root, label="plan")
    _journal_relative, resolved_journal = _repo_path(journal_path, evidence_root, label="journal")
    _aggregate_relative, resolved_aggregate_directory = _repo_path(
        aggregate_directory, evidence_root, label="aggregate directory"
    )
    expected = aggregation.build_aggregates(
        resolved_plan,
        resolved_journal,
        repository_root=evidence_root,
        repository_state=repository_state,
    )
    artifacts, aggregate_set_sha256 = _load_complete_aggregate_set(
        resolved_aggregate_directory, expected, evidence_root=evidence_root
    )
    first = artifacts[(planner.IMPLEMENTATIONS[0], "canonical_latency")]
    matrix_aggregation = first.get("matrix_aggregation")
    if not isinstance(matrix_aggregation, dict):
        raise ValueError("canonical aggregate lacks matrix aggregation provenance")
    configuration = _read_validated_configuration(
        resolved_plan,
        evidence_root=evidence_root,
        expected_plan_sha256=str(matrix_aggregation.get("plan_sha256", "")),
    )
    if configuration["dataset_label"] != EXPECTED_DATASET_LABEL:
        raise ValueError(f"public summary requires dataset_label={EXPECTED_DATASET_LABEL!r}")
    if configuration["model_name"] != DEFAULT_MODEL_ID:
        raise ValueError(f"public summary requires model_name={DEFAULT_MODEL_ID!r}")
    if configuration["model_revision"] != DEFAULT_MODEL_REVISION:
        raise ValueError("public summary requires the package-pinned model revision")

    anchor = _validate_matrix_identity(artifacts)
    dataset_archive_sha256 = _validate_librispeech_provenance(
        configuration,
        anchor=anchor,
        evidence_root=evidence_root,
    )
    hardware, software, total_memory_bytes = _public_environment(first.get("environment"))
    _validate_raw_memory_evidence(artifacts, total_memory_bytes=total_memory_bytes)
    public_evidence = _build_public_evidence(
        plan_path=resolved_plan,
        journal_path=resolved_journal,
        evidence_root=evidence_root,
        matrix_aggregation=matrix_aggregation,
        aggregate_set_sha256=aggregate_set_sha256,
        dataset_archive_sha256=dataset_archive_sha256,
    )
    total_repeats = configuration["outer_passes"] * configuration["repeats"]
    summary: dict[str, Any] = {
        "schema_version": PUBLIC_SCHEMA,
        "artifact_scope": "public",
        "evidence": public_evidence,
        "benchmark": {
            "model": {
                "repository": DEFAULT_MODEL_ID,
                "revision": DEFAULT_MODEL_REVISION,
            },
            "dataset": {
                "name": "LibriSpeech",
                "subset": "test-clean",
                "selection": "one duration-stratified utterance per bucket",
                "utterance_count": len(BUCKETS),
                "duration_buckets": _bucket_definitions(),
            },
        },
        "hardware": hardware,
        "software": software,
        "protocol": {
            "implementations": list(planner.IMPLEMENTATIONS),
            "measurement_profiles": list(planner.PROFILES),
            "outer_passes": configuration["outer_passes"],
            "warmup_iterations_per_task": configuration["warmup"],
            "measured_repeats_per_pass": configuration["repeats"],
            "measured_repeats_per_bucket": total_repeats,
            "max_new_tokens": configuration["max_new_tokens"],
            "storage_activation_dtype": "torch.float32",
            "canonical_latency": {
                "instrumentation": "NVML polling disabled; diagnostic phase timing disabled",
                "synchronization_boundary": (
                    "CUDA synchronize immediately before and after autoregressive generation"
                ),
                "statistics_scope": "one duration bucket across all outer-pass observations",
                "rtf_definition": "generation wall-clock seconds / input audio seconds",
            },
            "request_memory": {
                "instrumentation": "NVML process polling plus PyTorch CUDA allocator counters",
                "poll_interval_ms": float(configuration["nvml_sample_ms"]),
                "process_peak_semantics": "polling-observed lower bound",
                "allocator_peak_delta_semantics": (
                    "generation allocated peak minus after-prepare allocated baseline, floored at zero"
                ),
                "canonical_latency_eligible": False,
            },
            "diagnostic_phase": {
                "public_metrics_included": False,
                "canonical_latency_eligible": False,
            },
        },
        "results": _result_rows(artifacts, total_memory_bytes=total_memory_bytes),
        "parity": {
            "status": "pass",
            "criterion": "exact generated-token sequence equality",
            "profile_scope": "all 18 implementation/profile aggregates",
            "implementations_checked": len(planner.IMPLEMENTATIONS),
            "duration_buckets_checked": len(BUCKETS),
        },
        "comparisons": _comparison_rows(artifacts),
    }
    errors = validate_public_summary(summary)
    if errors:
        raise ValueError("rebuilt public summary failed its schema: " + "; ".join(errors))
    return summary


def validate_public_summary(payload: Any) -> list[str]:
    """Validate with the standalone schema shared by the release guard."""
    return public_schema.validate_public_summary(payload)


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite public summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _destination(
    path: Path,
    *,
    publication_root: Path,
    public: bool,
) -> Path:
    relative, resolved = _repo_path(path, publication_root, label="output")
    relative_path = Path(relative)
    expected_parent = PUBLIC_RESULTS_DIRECTORY if public else PRIVATE_ROOT
    if public:
        if relative_path.parent != expected_parent:
            raise ValueError(f"published summaries must be direct children of {expected_parent}")
    elif tuple(part.casefold() for part in relative_path.parts[:2]) != (
        "artifacts",
        "private",
    ):
        raise ValueError("export drafts must stay below artifacts/private")
    if SAFE_PUBLIC_FILENAME_RE.fullmatch(relative_path.name) is None:
        raise ValueError("output filename must be a conservative lowercase .json name")
    return resolved


def export_summary(
    plan_path: Path,
    journal_path: Path,
    aggregate_directory: Path,
    output: Path,
    *,
    evidence_root: Path,
    publication_root: Path,
    publish: bool = False,
    repository_state: Mapping[str, Any] | None = None,
) -> Path:
    if evidence_root.is_symlink() or _is_reparse_point(evidence_root):
        raise ValueError("evidence root must not be a symbolic link or reparse point")
    if publication_root.is_symlink() or _is_reparse_point(publication_root):
        raise ValueError("publication root must not be a symbolic link or reparse point")
    evidence_root = evidence_root.resolve()
    publication_root = publication_root.resolve()
    if not publication_root.is_dir():
        raise ValueError("publication root must be an existing non-symlink directory")
    if (
        evidence_root == publication_root
        or evidence_root in publication_root.parents
        or publication_root in evidence_root.parents
    ):
        raise ValueError("evidence and publication roots must be separate, non-overlapping trees")
    summary = build_public_summary(
        plan_path,
        journal_path,
        aggregate_directory,
        evidence_root=evidence_root,
        repository_state=repository_state,
    )
    destination = _destination(output, publication_root=publication_root, public=publish)
    _write_json_no_overwrite(destination, summary)
    return destination


def _add_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help=(
            "clean immutable checkout recorded by the plan; may be separate from "
            "the checkout containing this exporter"
        ),
    )
    parser.add_argument(
        "--publication-root",
        type=Path,
        required=True,
        help=(
            "checkout receiving the draft or published JSON; never used to resolve "
            "private evidence inputs"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser(
        "export", help="rebuild a checked draft below artifacts/private"
    )
    _add_evidence_arguments(export_parser)
    check_parser = subparsers.add_parser("check", help="validate one public summary")
    check_parser.add_argument("artifact", type=Path)
    publish_parser = subparsers.add_parser(
        "publish", help="rebuild and atomically publish below benchmarks/results"
    )
    _add_evidence_arguments(publish_parser)
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            if args.artifact.is_symlink() or not args.artifact.is_file():
                raise ValueError("public summary must be a regular non-symlink file")
            payload = public_schema.load_public_summary_bytes(args.artifact.read_bytes())
            errors = validate_public_summary(payload)
            if errors:
                for error in errors:
                    print(f"error: {error}", file=sys.stderr)
                return 1
            print(f"public matrix summary check passed: {args.artifact}")
            return 0
        destination = export_summary(
            args.plan,
            args.journal,
            args.aggregate_dir,
            args.output,
            evidence_root=args.evidence_root,
            publication_root=args.publication_root,
            publish=args.command == "publish",
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    action = "published" if args.command == "publish" else "exported"
    print(f"public matrix summary {action}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
