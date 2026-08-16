#!/usr/bin/env python3
"""Stitch private segment-level ASR results and score complete recordings.

This is a deterministic, offline evidence transform.  It accepts only a
private benchmark result produced from a locked materialized-segment bundle,
revalidates that bundle (including canonical PCM slices), binds every result
sample back to its manifest record, and evaluates quality only after grouping
all segments of a source recording.

The overlap rule is deliberately small and auditable: after applying the same
frozen Unicode normalization used by the benchmark, remove the longest suffix
of the accumulated word sequence that exactly equals a prefix of the next
segment.  The configured minimum and maximum overlap lengths are hashed into
the output.  This baseline is not a claim that lexical overlap stitching is
optimal; it is a reproducible control for later VAD/alignment experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from faster_glm_asr.benchmarking import runner as benchmark
from faster_glm_asr.data import long_audio

BENCHMARK_SCHEMA_VERSION = "0.3"
OUTPUT_SCHEMA_VERSION = "asr-source-stitch-v0.1"
ALGORITHM_VERSION = "longest-normalized-suffix-prefix-v0.1"
DEFAULT_MIN_OVERLAP_WORDS = 2
DEFAULT_MAX_OVERLAP_WORDS = 64
RUNNER_IMPLEMENTATIONS = {
    "hf_cached",
    "hf_no_cache",
    "custom_full_prefix",
    "custom_greedy_full_prefix",
    "custom_tuple_cache",
    "custom_static_cache",
}
NORMALIZATION = "NFKC + casefold + Unicode punctuation/symbol removal + whitespace collapse"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json(text: str, *, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = _decode_json(path.read_text(encoding="utf-8"), label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _load_jsonl_objects(path: Path, *, label: str, allow_empty: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = _decode_json(line, label=f"{label} line {line_number}")
        if not isinstance(item, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        records.append(item)
    if not records and not allow_empty:
        raise ValueError(f"{label} contains no records")
    return records


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a 64-character SHA256")
    return str(value).lower()


def _require_nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _require_plain_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_sidecar(
    records: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    manifest_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Validate private reference semantics, not just the lock-file hash."""

    plan_sources = plan.get("sources")
    if not isinstance(plan_sources, list) or not plan_sources:
        raise ValueError("segmentation plan sources must be a non-empty list")
    source_plan: dict[str, Mapping[str, Any]] = {}
    expected_reference_order: list[str] = []
    for index, source in enumerate(plan_sources):
        if not isinstance(source, dict):
            raise ValueError(f"segmentation plan source {index} must be an object")
        source_id = _require_nonempty_text(
            source.get("source_recording_id"),
            label=f"segmentation plan source {index}.source_recording_id",
        )
        if source_id in source_plan:
            raise ValueError(f"duplicate source ID in segmentation plan: {source_id!r}")
        source_plan[source_id] = source
        status = source.get("reference_status")
        if status not in {
            "unavailable",
            "file-level-only",
            "timed-cues-available",
        }:
            raise ValueError(f"segmentation plan source {source_id!r} has invalid reference_status")
        if status != "unavailable":
            expected_reference_order.append(source_id)

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for record in manifest_records:
        source_id = str(record.get("source_recording_id", ""))
        by_source.setdefault(source_id, []).append(record)
    if set(by_source) != set(source_plan):
        raise ValueError("manifest and segmentation-plan source sets differ")

    actual_reference_order: list[str] = []
    sidecar_index: dict[str, Mapping[str, Any]] = {}
    expected_keys = {
        "source_recording_id",
        "source_reference_sha256",
        "reference_status",
        "full_reference_text",
        "full_reference_text_sha256",
        "cues",
    }
    for record_index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(
                f"private reference record {record_index} has missing or unknown fields"
            )
        source_id = _require_nonempty_text(
            record.get("source_recording_id"),
            label=f"private reference record {record_index}.source_recording_id",
        )
        if source_id not in source_plan:
            raise ValueError(f"private reference has unknown source ID {source_id!r}")
        if source_id in sidecar_index:
            raise ValueError(f"private reference has duplicate source ID {source_id!r}")
        sidecar_index[source_id] = record
        actual_reference_order.append(source_id)

        status = record.get("reference_status")
        if status not in {"file-level-only", "timed-cues-available"}:
            raise ValueError(f"private reference source {source_id!r} has invalid reference_status")
        if status != source_plan[source_id].get("reference_status"):
            raise ValueError(f"private reference status disagrees for source {source_id!r}")
        _require_sha256(
            record.get("source_reference_sha256"),
            label=f"private reference source {source_id}.source_reference_sha256",
        )
        full_text = _require_nonempty_text(
            record.get("full_reference_text"),
            label=f"private reference source {source_id}.full_reference_text",
        )
        declared_text_hash = _require_sha256(
            record.get("full_reference_text_sha256"),
            label=f"private reference source {source_id}.full_reference_text_sha256",
        )
        actual_text_hash = _sha256_bytes(full_text.encode("utf-8"))
        if declared_text_hash != actual_text_hash:
            raise ValueError(f"private reference text SHA256 mismatch for source {source_id!r}")

        cues = record.get("cues")
        if not isinstance(cues, list):
            raise ValueError(f"private reference cues for {source_id!r} must be a list")
        if status == "timed-cues-available" and not cues:
            raise ValueError(f"timed reference for {source_id!r} contains no cues")
        if status == "file-level-only" and cues:
            raise ValueError(f"file-level reference for {source_id!r} must not have cues")

        canonical_samples = _require_plain_int(
            source_plan[source_id].get("canonical_samples"),
            label=f"segmentation plan source {source_id}.canonical_samples",
            minimum=1,
        )
        cue_ids: list[str] = []
        cue_texts: list[str] = []
        for cue_index, cue in enumerate(cues):
            if not isinstance(cue, dict) or set(cue) != {
                "cue_id",
                "order",
                "start_sample",
                "end_sample",
                "text",
            }:
                raise ValueError(
                    f"private reference cue {cue_index} for {source_id!r} has "
                    "missing or unknown fields"
                )
            expected_cue_id = f"cue-{cue_index:06d}"
            if cue.get("cue_id") != expected_cue_id or cue.get("order") != cue_index:
                raise ValueError(
                    f"private reference cues for {source_id!r} are not in canonical order"
                )
            start = _require_plain_int(
                cue.get("start_sample"),
                label=f"private reference cue {expected_cue_id}.start_sample",
            )
            end = _require_plain_int(
                cue.get("end_sample"),
                label=f"private reference cue {expected_cue_id}.end_sample",
                minimum=1,
            )
            if not 0 <= start < end <= canonical_samples:
                raise ValueError(
                    f"private reference cue {expected_cue_id} for {source_id!r} "
                    "falls outside canonical audio"
                )
            cue_texts.append(
                _require_nonempty_text(
                    cue.get("text"),
                    label=f"private reference cue {expected_cue_id}.text",
                )
            )
            cue_ids.append(expected_cue_id)
        if status == "timed-cues-available" and " ".join(cue_texts) != full_text:
            raise ValueError(
                f"private reference cue text disagrees with full text for {source_id!r}"
            )

        source_segments = sorted(by_source[source_id], key=lambda item: int(item["segment_index"]))
        expected_owned: list[str] = []
        for segment in source_segments:
            expected_intersecting = [
                str(cue["cue_id"])
                for cue in cues
                if int(cue["start_sample"]) < int(segment["end_sample"])
                and int(cue["end_sample"]) > int(segment["start_sample"])
            ]
            expected_segment_owned = [
                str(cue["cue_id"])
                for cue in cues
                if int(segment["ownership_start_sample"])
                <= (int(cue["start_sample"]) + int(cue["end_sample"])) // 2
                < int(segment["ownership_end_sample"])
            ]
            if segment.get("intersecting_cue_ids") != expected_intersecting:
                raise ValueError(
                    f"private cues disagree with intersecting_cue_ids for "
                    f"{segment.get('sample_id')!r}"
                )
            if segment.get("owned_cue_ids") != expected_segment_owned:
                raise ValueError(
                    f"private cues disagree with owned_cue_ids for {segment.get('sample_id')!r}"
                )
            expected_owned.extend(expected_segment_owned)
        if expected_owned != cue_ids:
            raise ValueError(f"private reference cues are not owned exactly once for {source_id!r}")

    if actual_reference_order != expected_reference_order:
        raise ValueError("private reference source IDs/order differ from the segmentation plan")

    for source_id, source in source_plan.items():
        if source.get("reference_status") == "unavailable":
            for segment in by_source[source_id]:
                if segment.get("intersecting_cue_ids") or segment.get("owned_cue_ids"):
                    raise ValueError(f"source {source_id!r} has cues despite unavailable reference")
    return sidecar_index


def _load_and_validate_bundle(
    bundle_dir: Path,
) -> tuple[
    Path,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Mapping[str, Any]],
]:
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"segment bundle directory not found: {bundle_dir}")
    manifest_path = bundle_dir / "segments.jsonl"
    plan_path = bundle_dir / "segmentation-plan.json"
    lock_path = bundle_dir / "bundle-lock.json"
    sidecar_path = bundle_dir / "source-references.private.jsonl"

    manifest_records = _load_jsonl_objects(
        manifest_path, label="segment manifest", allow_empty=False
    )
    plan = _load_json_object(plan_path, label="segmentation plan")
    lock = _load_json_object(lock_path, label="bundle lock")
    sidecar_records = _load_jsonl_objects(
        sidecar_path, label="private reference sidecar", allow_empty=True
    )

    # Reuse the benchmark's full schema/audio validator.  This recomputes the
    # plan, all file/PCM hashes, and every canonical-source slice relationship.
    loaded_samples, loaded_manifest_hash = benchmark._load_manifest(manifest_path)
    if loaded_manifest_hash != _sha256_file(manifest_path):
        raise RuntimeError("benchmark manifest validator returned an inconsistent hash")
    if [sample.sample_id for sample in loaded_samples] != [
        str(record.get("sample_id", "")) for record in manifest_records
    ]:
        raise ValueError("segment manifest parsing changed sample order")
    plan_sources = plan.get("sources")
    if not isinstance(plan_sources, list):
        raise ValueError("segmentation plan sources must be a list")
    expected_manifest_order: list[str] = []
    for source in plan_sources:
        if not isinstance(source, dict):
            raise ValueError("segmentation plan source must be an object")
        source_id = str(source.get("source_recording_id", ""))
        segment_count = _require_plain_int(
            source.get("segment_count"),
            label=f"segmentation plan source {source_id}.segment_count",
            minimum=1,
        )
        expected_manifest_order.extend(
            f"{source_id}__seg-{index:05d}" for index in range(segment_count)
        )
    actual_manifest_order = [str(record["sample_id"]) for record in manifest_records]
    if actual_manifest_order != expected_manifest_order:
        raise ValueError(
            "segment manifest source grouping/order differs from the segmentation plan"
        )

    sidecar_index = _validate_sidecar(sidecar_records, plan=plan, manifest_records=manifest_records)
    return manifest_path, manifest_records, plan, lock, sidecar_index


def _validate_benchmark_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("benchmark result.run must be an object")
    if run.get("artifact_scope") != "private":
        raise ValueError("source stitching requires a private benchmark artifact")
    implementation = _require_nonempty_text(
        run.get("implementation"), label="benchmark result.run.implementation"
    )
    if implementation not in RUNNER_IMPLEMENTATIONS:
        raise ValueError("benchmark result has unsupported implementation")
    model_name = _require_nonempty_text(
        run.get("model_name"), label="benchmark result.run.model_name"
    )
    model_revision = run.get("model_revision")
    if not isinstance(model_revision, str) or COMMIT_RE.fullmatch(model_revision) is None:
        raise ValueError("benchmark result.run.model_revision must be a 40-hex commit")
    measurement_profile = _require_nonempty_text(
        run.get("measurement_profile"),
        label="benchmark result.run.measurement_profile",
    )
    if measurement_profile != "canonical_latency":
        raise ValueError("source quality must be derived from the canonical_latency artifact")
    eligibility = run.get("metric_eligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("canonical_latency") is not True
        or run.get("phase_timing") is not False
        or not isinstance(run.get("nvml_sample_ms"), (int, float))
        or isinstance(run.get("nvml_sample_ms"), bool)
        or float(run["nvml_sample_ms"]) != 0.0
    ):
        raise ValueError("benchmark result is not an eligible canonical latency pass")
    manifest_hash = _require_sha256(
        run.get("manifest_sha256"), label="benchmark result.run.manifest_sha256"
    )

    git = payload.get("git")
    if not isinstance(git, dict):
        raise ValueError("benchmark result.git must be an object")
    commit = git.get("commit")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("benchmark result.git.commit must be a 40-hex commit")
    if not isinstance(git.get("dirty"), bool):
        raise ValueError("benchmark result.git.dirty must be boolean")

    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("benchmark result.sources must be an object")
    if set(sources) != {"schema_version", "files", "aggregate_sha256"}:
        raise ValueError("benchmark result.sources schema mismatch")
    if sources.get("schema_version") != benchmark.SOURCE_MAP_SCHEMA:
        raise ValueError("benchmark result.sources schema is unsupported")
    source_files = sources.get("files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("benchmark result.sources.files must be non-empty")
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(value, dict)
        or set(value) != {"size_bytes", "sha256"}
        or not isinstance(value.get("size_bytes"), int)
        or isinstance(value.get("size_bytes"), bool)
        or value["size_bytes"] < 0
        or not _is_sha256(value.get("sha256"))
        for path, value in source_files.items()
    ):
        raise ValueError("benchmark result source-map records are invalid")
    aggregate_hash = _require_sha256(
        sources.get("aggregate_sha256"),
        label="benchmark result.sources.aggregate_sha256",
    )
    canonical_sources = json.dumps(
        source_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical_sources).hexdigest() != aggregate_hash:
        raise ValueError("benchmark result source-map aggregate is inconsistent")

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("benchmark result.environment must be an object")
    environment_lock_hash = _require_sha256(
        environment.get("environment_lock_sha256"),
        label="benchmark result.environment.environment_lock_sha256",
    )
    return {
        "implementation": implementation,
        "model_name": model_name,
        "model_revision": model_revision.lower(),
        "measurement_profile": measurement_profile,
        "manifest_sha256": manifest_hash,
        "git_commit": commit.lower(),
        "git_dirty": git["dirty"],
        "environment_lock_sha256": environment_lock_hash,
        "benchmark_source_map": sources,
    }


def _validate_result_samples(
    payload: Mapping[str, Any], manifest_records: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("benchmark result.samples must be a non-empty list")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"benchmark result sample {index} must be an object")

    sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("benchmark result contains an empty sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("benchmark result contains duplicate segment sample IDs")
    manifest_ids = [str(record["sample_id"]) for record in manifest_records]
    if sample_ids != manifest_ids:
        missing = sorted(set(manifest_ids) - set(sample_ids))
        extra = sorted(set(sample_ids) - set(manifest_ids))
        if missing or extra:
            raise ValueError(
                f"benchmark result segment completeness mismatch: missing={missing}, extra={extra}"
            )
        raise ValueError("benchmark result segment order differs from the manifest")

    validated: list[Mapping[str, Any]] = []
    excluded_manifest_fields = {
        "sample_id",
        "audio_path",
        "audio_sha256",
        "language",
        "duration_s",
    }
    for index, (sample, manifest) in enumerate(zip(samples, manifest_records, strict=True)):
        location = f"benchmark result sample {index} ({manifest['sample_id']})"
        actual_audio_hash = _require_sha256(
            sample.get("audio_sha256"), label=f"{location}.audio_sha256"
        )
        if actual_audio_hash != str(manifest["audio_sha256"]).lower():
            raise ValueError(f"{location} audio SHA256 disagrees with the manifest")
        if sample.get("language") != manifest.get("language"):
            raise ValueError(f"{location} language disagrees with the manifest")
        duration = sample.get("duration_s")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or float(duration) <= 0
            or abs(float(duration) - float(manifest["duration_s"])) > 0.5 / 16000
        ):
            raise ValueError(f"{location} duration disagrees with the manifest")

        expected_metadata = {
            key: value for key, value in manifest.items() if key not in excluded_manifest_fields
        }
        if sample.get("metadata") != expected_metadata:
            raise ValueError(f"{location} metadata does not exactly bind the manifest")
        hypothesis = sample.get("hypothesis")
        if not isinstance(hypothesis, str):
            raise ValueError(f"{location}.hypothesis must be text")
        declared_hypothesis_hash = _require_sha256(
            sample.get("hypothesis_sha256"),
            label=f"{location}.hypothesis_sha256",
        )
        if declared_hypothesis_hash != _sha256_bytes(hypothesis.encode("utf-8")):
            raise ValueError(f"{location} hypothesis SHA256 mismatch")

        token_ids = sample.get("generated_token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
        ):
            raise ValueError(
                f"{location}.generated_token_ids must be a non-empty list of non-negative ints"
            )
        if sample.get("generated_tokens") != len(token_ids):
            raise ValueError(f"{location}.generated_tokens disagrees with token IDs")
        declared_token_hash = _require_sha256(
            sample.get("generated_token_ids_sha256"),
            label=f"{location}.generated_token_ids_sha256",
        )
        actual_token_hash = _sha256_bytes(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        )
        if declared_token_hash != actual_token_hash:
            raise ValueError(f"{location} generated-token SHA256 mismatch")
        if sample.get("quality") is not None:
            raise ValueError(
                f"{location} must have quality=null; segment references are private "
                "and source-scoped"
            )
        validated.append(sample)
    return validated


def _longest_suffix_prefix_overlap(
    accumulated: Sequence[str],
    incoming: Sequence[str],
    *,
    min_overlap_words: int,
    max_overlap_words: int,
) -> int:
    upper = min(len(accumulated), len(incoming), max_overlap_words)
    for width in range(upper, min_overlap_words - 1, -1):
        if list(accumulated[-width:]) == list(incoming[:width]):
            return width
    return 0


def _quality(reference: str | None, hypothesis: str) -> dict[str, Any] | None:
    if reference is None:
        return None
    quality = benchmark._quality(reference, hypothesis)
    if quality is None:
        raise RuntimeError("quality evaluator unexpectedly returned null")
    return {**quality, "normalization": NORMALIZATION}


def _aggregate_source_quality(qualities: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    quality_list = list(qualities)
    aggregate = benchmark._aggregate_quality({"quality": quality} for quality in quality_list)
    aggregate["sources_total"] = len(quality_list)
    aggregate["sources_with_reference"] = sum(quality is not None for quality in quality_list)
    aggregate["sources_evaluated"] = sum(
        quality is not None and quality.get("status") == "evaluated" for quality in quality_list
    )
    return aggregate


def _stitch_source(
    *,
    source_plan: Mapping[str, Any],
    manifest_records: Sequence[Mapping[str, Any]],
    result_samples: Sequence[Mapping[str, Any]],
    reference_record: Mapping[str, Any] | None,
    min_overlap_words: int,
    max_overlap_words: int,
) -> dict[str, Any]:
    source_id = str(source_plan["source_recording_id"])
    accumulated: list[str] = []
    decisions: list[dict[str, Any]] = []
    for manifest, sample in zip(manifest_records, result_samples, strict=True):
        incoming = benchmark._normalize_text(str(sample["hypothesis"])).split()
        overlap = _longest_suffix_prefix_overlap(
            accumulated,
            incoming,
            min_overlap_words=min_overlap_words,
            max_overlap_words=max_overlap_words,
        )
        appended = incoming[overlap:]
        accumulated.extend(appended)
        decisions.append(
            {
                "sample_id": manifest["sample_id"],
                "segment_index": manifest["segment_index"],
                "input_hypothesis_sha256": str(sample["hypothesis_sha256"]).lower(),
                "normalized_input_word_count": len(incoming),
                "matched_overlap_words": overlap,
                "appended_word_count": len(appended),
                "stitched_word_count_after": len(accumulated),
            }
        )

    hypothesis = " ".join(accumulated)
    reference_text = (
        str(reference_record["full_reference_text"]) if reference_record is not None else None
    )
    quality = _quality(reference_text, hypothesis)
    reference_provenance = {
        "status": source_plan["reference_status"],
        "source_reference_sha256": (
            str(reference_record["source_reference_sha256"]).lower()
            if reference_record is not None
            else None
        ),
        "full_reference_text_sha256": (
            str(reference_record["full_reference_text_sha256"]).lower()
            if reference_record is not None
            else None
        ),
    }
    return {
        "source_recording_id": source_id,
        "split_group_id": source_plan["split_group_id"],
        "split": source_plan["split"],
        "language": manifest_records[0]["language"],
        "source_audio_sha256": str(source_plan["source_audio_sha256"]).lower(),
        "canonical_source_audio_sha256": str(source_plan["canonical_source_audio_sha256"]).lower(),
        "canonical_pcm_sha256": str(source_plan["canonical_pcm_sha256"]).lower(),
        "canonical_samples": source_plan["canonical_samples"],
        "segment_count": len(manifest_records),
        "segment_sample_ids": [record["sample_id"] for record in manifest_records],
        "hypothesis": hypothesis,
        "hypothesis_sha256": _sha256_bytes(hypothesis.encode("utf-8")),
        "hypothesis_word_count": len(accumulated),
        "stitch_decisions": decisions,
        "reference": reference_provenance,
        "quality_status": (
            str(quality["status"]) if quality is not None else "unavailable_no_reference"
        ),
        "quality": quality,
    }


def create_stitch_payload(
    benchmark_result_path: Path,
    segment_bundle: Path,
    *,
    min_overlap_words: int = DEFAULT_MIN_OVERLAP_WORDS,
    max_overlap_words: int = DEFAULT_MAX_OVERLAP_WORDS,
) -> dict[str, Any]:
    min_overlap_words = _require_plain_int(min_overlap_words, label="min_overlap_words", minimum=1)
    max_overlap_words = _require_plain_int(max_overlap_words, label="max_overlap_words", minimum=1)
    if min_overlap_words > max_overlap_words:
        raise ValueError("min_overlap_words must be <= max_overlap_words")

    benchmark_result_path = benchmark_result_path.resolve()
    payload = _load_json_object(benchmark_result_path, label="private benchmark result")
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"private benchmark result schema must be {BENCHMARK_SCHEMA_VERSION!r}")
    run_provenance = _validate_benchmark_provenance(payload)
    (
        manifest_path,
        manifest_records,
        plan,
        lock,
        sidecar_index,
    ) = _load_and_validate_bundle(segment_bundle)
    actual_manifest_hash = _sha256_file(manifest_path)
    if run_provenance["manifest_sha256"] != actual_manifest_hash:
        raise ValueError("benchmark result manifest SHA256 does not match the segment bundle")
    result_samples = _validate_result_samples(payload, manifest_records)

    configuration = {
        "algorithm_version": ALGORITHM_VERSION,
        "input_order": "segments.jsonl exact order; segment_index ascending per source",
        "matching_unit": "normalized whitespace-delimited word",
        "normalization": NORMALIZATION,
        "output_hypothesis": "single-space join of normalized words",
        "min_overlap_words": min_overlap_words,
        "max_overlap_words": max_overlap_words,
        "tie_break": "largest exact suffix-prefix width",
    }
    configuration_hash = _sha256_bytes(_canonical_json_bytes(configuration))

    by_source_manifest: dict[str, list[Mapping[str, Any]]] = {}
    by_source_samples: dict[str, list[Mapping[str, Any]]] = {}
    for manifest, sample in zip(manifest_records, result_samples, strict=True):
        source_id = str(manifest["source_recording_id"])
        by_source_manifest.setdefault(source_id, []).append(manifest)
        by_source_samples.setdefault(source_id, []).append(sample)

    plan_sources = plan["sources"]
    source_groups: list[dict[str, Any]] = []
    for source_plan in plan_sources:
        source_id = str(source_plan["source_recording_id"])
        manifests = by_source_manifest.get(source_id, [])
        samples = by_source_samples.get(source_id, [])
        if len(manifests) != source_plan["segment_count"] or len(samples) != len(manifests):
            raise ValueError(f"source {source_id!r} has incomplete benchmark segments")
        indices = [int(record["segment_index"]) for record in manifests]
        if indices != list(range(len(manifests))):
            raise ValueError(f"source {source_id!r} segment order is incomplete")
        source_groups.append(
            _stitch_source(
                source_plan=source_plan,
                manifest_records=manifests,
                result_samples=samples,
                reference_record=sidecar_index.get(source_id),
                min_overlap_words=min_overlap_words,
                max_overlap_words=max_overlap_words,
            )
        )

    script_path = Path(__file__).resolve()
    transform_source_records = {
        "faster_glm_asr.data.segment_stitch": {
            "size_bytes": script_path.stat().st_size,
            "sha256": _sha256_file(script_path),
        },
        "faster_glm_asr.benchmarking.runner": {
            "size_bytes": Path(benchmark.__file__).resolve().stat().st_size,
            "sha256": _sha256_file(Path(benchmark.__file__).resolve()),
        },
        "faster_glm_asr.data.long_audio": {
            "size_bytes": Path(long_audio.__file__).resolve().stat().st_size,
            "sha256": _sha256_file(Path(long_audio.__file__).resolve()),
        },
    }
    bundle_dir = manifest_path.parent
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "artifact_scope": "private",
        "inputs": {
            "benchmark_result_sha256": _sha256_file(benchmark_result_path),
            "segment_manifest_sha256": actual_manifest_hash,
            "segmentation_plan_sha256": _sha256_file(bundle_dir / "segmentation-plan.json"),
            "bundle_lock_sha256": _sha256_file(bundle_dir / "bundle-lock.json"),
            "private_reference_sidecar_sha256": _sha256_file(
                bundle_dir / "source-references.private.jsonl"
            ),
            "source_manifest_sha256": str(lock["source_manifest_sha256"]).lower(),
        },
        "stitch_configuration": configuration,
        "stitch_configuration_sha256": configuration_hash,
        "provenance": {
            **run_provenance,
            "transform_sources": transform_source_records,
        },
        "aggregate_quality": _aggregate_source_quality(group["quality"] for group in source_groups),
        "source_groups": source_groups,
    }
    return result


def _write_json_atomically(output: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def build(
    benchmark_result_path: Path,
    segment_bundle: Path,
    output: Path,
    *,
    min_overlap_words: int = DEFAULT_MIN_OVERLAP_WORDS,
    max_overlap_words: int = DEFAULT_MAX_OVERLAP_WORDS,
    overwrite: bool = False,
) -> int:
    output = output.resolve()
    benchmark_result_path = benchmark_result_path.resolve()
    input_paths = {
        benchmark_result_path,
        (segment_bundle.resolve() / "segments.jsonl"),
        (segment_bundle.resolve() / "segmentation-plan.json"),
        (segment_bundle.resolve() / "bundle-lock.json"),
        (segment_bundle.resolve() / "source-references.private.jsonl"),
    }
    if output in input_paths:
        raise ValueError("output must not replace a benchmark or bundle input")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists; choose a versioned path or pass --overwrite: {output}"
        )
    result = create_stitch_payload(
        benchmark_result_path,
        segment_bundle,
        min_overlap_words=min_overlap_words,
        max_overlap_words=max_overlap_words,
    )
    _write_json_atomically(output, result, overwrite=overwrite)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch a private segment benchmark into source-level hypotheses and WER/CER evidence."
        )
    )
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--segment-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-overlap-words", type=int, default=DEFAULT_MIN_OVERLAP_WORDS)
    parser.add_argument("--max-overlap-words", type=int, default=DEFAULT_MAX_OVERLAP_WORDS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="intentionally replace an existing output (default: refuse)",
    )
    args = parser.parse_args()
    try:
        build(
            args.benchmark_result,
            args.segment_bundle,
            args.output,
            min_overlap_words=args.min_overlap_words,
            max_overlap_words=args.max_overlap_words,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"source-level result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
