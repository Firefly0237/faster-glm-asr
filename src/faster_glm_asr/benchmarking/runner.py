#!/usr/bin/env python3
"""Evidence-oriented GLM-ASR inference benchmark.

The script deliberately keeps the custom full-prefix, custom-tuple-cache, custom-static-cache,
and official Hugging Face paths as separate configuration IDs.  One invocation
benchmarks one configuration so model state and allocator history are not
silently shared.  Use the same manifest and alternate invocation order when
collecting paired baseline/optimized observations.

This module records measurements; it never fabricates or pre-populates results.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from faster_glm_asr import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from faster_glm_asr.benchmarking import provenance

DEFAULT_MODEL_NAME = DEFAULT_MODEL_ID
SOURCE_MANIFEST_PARSER_VERSION = "asr-inventory-v0.2"
SEGMENT_MANIFEST_PARSER_VERSION = "asr-segment-v0.1"
MANIFEST_PARSER_VERSION = SOURCE_MANIFEST_PARSER_VERSION
MANIFEST_PARSER_VERSIONS = {
    SOURCE_MANIFEST_PARSER_VERSION,
    SEGMENT_MANIFEST_PARSER_VERSION,
}
DIRECT_MODEL_MAX_AUDIO_S = 600.0
TEXT_MAX_POSITION_EMBEDDINGS = 8192
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SOURCE_MAP_SCHEMA = "faster-glm-asr-source-map-v1"
SOURCE_PACKAGE_DIRECTORIES = ("modeling", "kernels", "benchmarking", "data")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    audio_path: Path
    audio_sha256: str | None
    reference_text: str | None
    language: str
    declared_duration_s: float | None
    shareable: bool
    metadata: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomically(output: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    """Durably stage JSON, then atomically publish it without accidental overwrite."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            # os.link atomically fails when the destination appeared while this
            # process was running; os.replace would silently overwrite it.
            os.link(temporary, output)
            temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _validate_materialized_segment_audio(
    audio_path: Path, item: Mapping[str, Any], line_number: int
) -> None:
    """Bind segment schema claims to the actual canonical PCM16 WAV bytes."""
    try:
        with wave.open(str(audio_path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
            pcm_bytes = stream.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"manifest line {line_number}: segment audio is not a valid WAV") from exc
    expected_frames = int(item["end_sample"]) - int(item["start_sample"])
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != 16000
        or compression != "NONE"
        or frame_count != expected_frames
        or len(pcm_bytes) != expected_frames * 2
    ):
        raise ValueError(
            f"manifest line {line_number}: segment WAV must be uncompressed "
            "mono 16 kHz PCM16 with frame count equal to [start,end) length"
        )
    actual_pcm_hash = hashlib.sha256(pcm_bytes).hexdigest()
    if actual_pcm_hash.lower() != str(item["segment_pcm_sha256"]).lower():
        raise ValueError(f"manifest line {line_number}: segment_pcm_sha256 mismatch")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"segment bundle missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"segment bundle {label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"segment bundle {label} must be a JSON object")
    return value


def _validate_segment_bundle_files(
    manifest_path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    """Verify the sibling plan/lock/reference sidecar for a segment bundle."""
    if not records:
        return
    plan_path = manifest_path.parent / "segmentation-plan.json"
    lock_path = manifest_path.parent / "bundle-lock.json"
    references_path = manifest_path.parent / "source-references.private.jsonl"
    plan = _read_json_object(plan_path, "segmentation-plan.json")
    lock = _read_json_object(lock_path, "bundle-lock.json")

    expected_plan_keys = {
        "schema_version",
        "source_manifest_sha256",
        "strategy",
        "target_sample_rate_hz",
        "target_channels",
        "chunk_samples",
        "overlap_samples",
        "tail_policy",
        "interval_convention",
        "reference_policy",
        "derived_shareable_default",
        "source_code_sha256",
        "sources",
    }
    if set(plan) != expected_plan_keys:
        raise ValueError("segment bundle plan schema has missing or unknown fields")
    if plan.get("schema_version") != "asr-segmentation-plan-v0.1":
        raise ValueError("segment bundle has unsupported segmentation plan schema")
    if (
        plan.get("strategy") != "fixed-window-v1"
        or plan.get("target_sample_rate_hz") != 16000
        or plan.get("target_channels") != 1
        or plan.get("interval_convention") != "[start,end)"
        or plan.get("tail_policy") != "keep-short"
    ):
        raise ValueError("segment bundle plan contract is inconsistent")
    for key in ("chunk_samples", "overlap_samples"):
        if not isinstance(plan.get(key), int) or isinstance(plan.get(key), bool):
            raise ValueError(f"segment bundle plan {key} must be an integer")
    if not 0 <= int(plan["overlap_samples"]) < int(plan["chunk_samples"]):
        raise ValueError("segment bundle plan has invalid chunk/overlap samples")
    if int(plan["chunk_samples"]) > DIRECT_MODEL_MAX_AUDIO_S * 16000:
        raise ValueError("segment bundle plan exceeds the direct-model guardrail")
    if plan.get("reference_policy") != "source-level-after-stitch-v1":
        raise ValueError("segment bundle plan has unsupported reference policy")
    if plan.get("derived_shareable_default") is not False:
        raise ValueError("segment bundle plan must default derived audio to private")
    for key in ("source_manifest_sha256", "source_code_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(plan.get(key, ""))):
            raise ValueError(f"segment bundle plan {key} is not a SHA256")

    expected_lock_keys = {
        "schema_version",
        "source_manifest_sha256",
        "segmentation_plan_sha256",
        "segment_manifest_sha256",
        "private_reference_sidecar_sha256",
    }
    if set(lock) != expected_lock_keys:
        raise ValueError("segment bundle lock schema has missing or unknown fields")
    if lock.get("schema_version") != "asr-segment-bundle-lock-v0.1":
        raise ValueError("segment bundle has unsupported bundle-lock schema")
    for key in expected_lock_keys - {"schema_version"}:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(lock.get(key, ""))):
            raise ValueError(f"segment bundle lock {key} is not a SHA256")

    actual_plan_hash = _sha256(plan_path)
    actual_manifest_hash = _sha256(manifest_path)
    actual_references_hash = _sha256(references_path) if references_path.is_file() else None
    if str(lock["segmentation_plan_sha256"]).lower() != actual_plan_hash.lower():
        raise ValueError("segment bundle lock does not match segmentation plan")
    if str(lock["segment_manifest_sha256"]).lower() != actual_manifest_hash.lower():
        raise ValueError("segment bundle lock does not match segment manifest")
    if (
        actual_references_hash is None
        or str(lock["private_reference_sidecar_sha256"]).lower() != actual_references_hash.lower()
    ):
        raise ValueError("segment bundle lock does not match private reference sidecar")
    if (
        str(lock["source_manifest_sha256"]).lower()
        != str(plan.get("source_manifest_sha256", "")).lower()
    ):
        raise ValueError("segment bundle source-manifest hashes disagree")
    if any(
        str(record["segmentation_plan_sha256"]).lower() != actual_plan_hash.lower()
        for record in records
    ):
        raise ValueError("segment records do not bind to the sibling plan")

    plan_sources = plan.get("sources")
    if not isinstance(plan_sources, list) or not plan_sources:
        raise ValueError("segment bundle plan sources must be a non-empty list")
    source_index: dict[str, Mapping[str, Any]] = {}
    expected_source_keys = {
        "source_recording_id",
        "split_group_id",
        "split",
        "source_audio_sha256",
        "canonical_source_audio_sha256",
        "canonical_pcm_sha256",
        "canonical_samples",
        "canonicalizer",
        "segment_count",
        "reference_status",
    }
    expected_canonicalizer_keys = {
        "decoder",
        "source_sample_rate_hz",
        "target_sample_rate_hz",
        "channel_policy",
        "resampler",
        "pcm_quantization",
        "numpy_version",
        "soundfile_version",
        "scipy_version",
    }
    for source in plan_sources:
        if not isinstance(source, dict):
            raise ValueError("segment bundle plan source must be an object")
        if set(source) != expected_source_keys:
            raise ValueError("segment bundle plan source schema is inconsistent")
        source_id = source.get("source_recording_id")
        if not isinstance(source_id, str) or source_id in source_index:
            raise ValueError("segment bundle plan has invalid/duplicate source ID")
        canonicalizer = source.get("canonicalizer")
        if (
            not isinstance(canonicalizer, dict)
            or set(canonicalizer) != expected_canonicalizer_keys
            or not isinstance(canonicalizer.get("decoder"), str)
        ):
            raise ValueError("segment bundle plan lacks canonicalizer provenance")
        source_index[source_id] = source

    record_sources = {str(record["source_recording_id"]) for record in records}
    if set(source_index) != record_sources:
        raise ValueError("segment bundle plan and manifest source sets differ")
    for source_id in record_sources:
        source_records = [
            record for record in records if str(record["source_recording_id"]) == source_id
        ]
        plan_source = source_index[source_id]
        representative = source_records[0]
        canonical_samples = plan_source.get("canonical_samples")
        if (
            not isinstance(canonical_samples, int)
            or isinstance(canonical_samples, bool)
            or canonical_samples <= 0
        ):
            raise ValueError(f"segment bundle plan canonical_samples is invalid for {source_id!r}")
        expected = {
            "split_group_id": representative["split_group_id"],
            "split": representative["split"],
            "source_audio_sha256": representative["source_audio_sha256"],
            "canonical_source_audio_sha256": representative["canonical_source_audio_sha256"],
            "canonical_pcm_sha256": representative["canonical_pcm_sha256"],
            "segment_count": len(source_records),
            "reference_status": representative["reference_status"],
        }
        hash_fields = {
            "source_audio_sha256",
            "canonical_source_audio_sha256",
            "canonical_pcm_sha256",
        }
        if any(
            (
                str(plan_source.get(key, "")).lower() != str(value).lower()
                if key in hash_fields
                else plan_source.get(key) != value
            )
            for key, value in expected.items()
        ):
            raise ValueError(f"segment bundle plan metadata disagrees for source {source_id!r}")

        canonical_path = manifest_path.parent / "canonical" / f"{source_id}.wav"
        try:
            with wave.open(str(canonical_path), "rb") as stream:
                channels = stream.getnchannels()
                sample_width = stream.getsampwidth()
                sample_rate = stream.getframerate()
                frame_count = stream.getnframes()
                compression = stream.getcomptype()
                canonical_pcm = stream.readframes(frame_count)
        except (OSError, EOFError, wave.Error) as exc:
            raise ValueError(
                f"segment bundle canonical WAV is missing or invalid for {source_id!r}"
            ) from exc
        if (
            channels != 1
            or sample_width != 2
            or sample_rate != 16000
            or compression != "NONE"
            or frame_count != canonical_samples
            or len(canonical_pcm) != canonical_samples * 2
        ):
            raise ValueError(f"segment bundle canonical WAV contract differs for {source_id!r}")
        if (
            _sha256(canonical_path).lower()
            != str(plan_source["canonical_source_audio_sha256"]).lower()
            or hashlib.sha256(canonical_pcm).hexdigest().lower()
            != str(plan_source["canonical_pcm_sha256"]).lower()
        ):
            raise ValueError(f"segment bundle canonical WAV hashes differ for {source_id!r}")

        for record in source_records:
            expected_relative_path = (Path("segments") / f"{record['sample_id']}.wav").as_posix()
            if str(record.get("audio_path", "")).replace("\\", "/") != (expected_relative_path):
                raise ValueError(
                    f"segment bundle audio path is noncanonical for {record.get('sample_id')!r}"
                )
            segment_path = manifest_path.parent / expected_relative_path
            try:
                with wave.open(str(segment_path), "rb") as stream:
                    segment_frames = stream.getnframes()
                    segment_pcm = stream.readframes(segment_frames)
            except (OSError, EOFError, wave.Error) as exc:
                raise ValueError(
                    f"segment bundle WAV is missing or invalid for {record.get('sample_id')!r}"
                ) from exc
            start_byte = int(record["start_sample"]) * 2
            end_byte = int(record["end_sample"]) * 2
            if segment_pcm != canonical_pcm[start_byte:end_byte]:
                raise ValueError(
                    f"segment PCM does not equal the declared canonical source "
                    f"slice for {record.get('sample_id')!r}"
                )

        chunk_samples = int(plan["chunk_samples"])
        overlap_samples = int(plan["overlap_samples"])
        expected_intervals: list[tuple[int, int]] = []
        start = 0
        while True:
            end = min(start + chunk_samples, canonical_samples)
            expected_intervals.append((start, end))
            if end == canonical_samples:
                break
            start = end - overlap_samples
        ordered = sorted(source_records, key=lambda item: int(item["segment_index"]))
        actual_intervals = [
            (int(record["start_sample"]), int(record["end_sample"])) for record in ordered
        ]
        if actual_intervals != expected_intervals:
            raise ValueError(
                f"segment records do not implement the locked fixed plan for {source_id!r}"
            )
        for index, (record, (start, end)) in enumerate(
            zip(ordered, expected_intervals, strict=True)
        ):
            expected_left = expected_intervals[index - 1][1] - start if index > 0 else 0
            expected_right = (
                end - expected_intervals[index + 1][0] if index + 1 < len(expected_intervals) else 0
            )
            if (
                record["short_tail"] is not (end - start < chunk_samples)
                or int(record["left_overlap_samples"]) != expected_left
                or int(record["right_overlap_samples"]) != expected_right
            ):
                raise ValueError(
                    f"segment plan-derived fields disagree for {source_id!r} segment {index}"
                )
            for key, expected_seconds in (
                ("start_s", start / 16000),
                ("end_s", end / 16000),
                ("duration_s", (end - start) / 16000),
            ):
                value = record.get(key)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or abs(float(value) - expected_seconds) > 0.5 / 16000
                ):
                    raise ValueError(
                        f"segment {key} disagrees with locked sample boundaries for "
                        f"{source_id!r} segment {index}"
                    )


def _validate_segment_manifest_item(item: Mapping[str, Any], line_number: int) -> None:
    required = {
        "manifest_schema_version",
        "source_recording_id",
        "split_group_id",
        "segment_index",
        "segment_count",
        "source_audio_sha256",
        "canonical_source_audio_sha256",
        "canonical_pcm_sha256",
        "segment_pcm_sha256",
        "sample_rate_hz",
        "channels",
        "start_sample",
        "end_sample",
        "start_s",
        "end_s",
        "duration_s",
        "interval_convention",
        "left_overlap_samples",
        "right_overlap_samples",
        "ownership_start_sample",
        "ownership_end_sample",
        "short_tail",
        "planner_strategy",
        "segmenter_version",
        "segmentation_plan_sha256",
        "source_shareable",
        "reference_status",
        "intersecting_cue_ids",
        "owned_cue_ids",
        "reference_assignment_policy",
        "primary_quality_scope",
        "split",
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"manifest line {line_number}: segment record missing {missing}")
    if item["manifest_schema_version"] != "asr-segment-manifest-v0.1":
        raise ValueError(f"manifest line {line_number}: unsupported segment manifest schema")
    if item["segmenter_version"] != "asr-fixed-window-segmenter-v0.1":
        raise ValueError(f"manifest line {line_number}: unsupported segmenter_version")
    if item["planner_strategy"] != "fixed-window-v1":
        raise ValueError(f"manifest line {line_number}: unsupported planner_strategy")
    if item["interval_convention"] != "[start,end)":
        raise ValueError(f"manifest line {line_number}: invalid interval convention")
    if item["primary_quality_scope"] != "source-recording-after-stitch":
        raise ValueError(
            f"manifest line {line_number}: segment quality scope must be "
            "source-recording-after-stitch"
        )
    for key in (
        "source_audio_sha256",
        "canonical_source_audio_sha256",
        "canonical_pcm_sha256",
        "segment_pcm_sha256",
        "segmentation_plan_sha256",
    ):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(item[key])):
            raise ValueError(f"manifest line {line_number}: {key} must be 64 hex characters")
    integers = {
        key: item[key]
        for key in (
            "segment_index",
            "segment_count",
            "sample_rate_hz",
            "channels",
            "start_sample",
            "end_sample",
            "left_overlap_samples",
            "right_overlap_samples",
            "ownership_start_sample",
            "ownership_end_sample",
        )
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integers.values()):
        raise ValueError(f"manifest line {line_number}: segment indices must be integers")
    if not 0 <= integers["segment_index"] < integers["segment_count"]:
        raise ValueError(f"manifest line {line_number}: invalid segment index/count")
    if integers["sample_rate_hz"] != 16000 or integers["channels"] != 1:
        raise ValueError(f"manifest line {line_number}: segments must be canonical 16 kHz mono")
    if not 0 <= integers["start_sample"] < integers["end_sample"]:
        raise ValueError(f"manifest line {line_number}: invalid half-open sample interval")
    if (
        integers["left_overlap_samples"] < 0
        or integers["right_overlap_samples"] < 0
        or not integers["start_sample"]
        <= integers["ownership_start_sample"]
        < integers["ownership_end_sample"]
        <= integers["end_sample"]
    ):
        raise ValueError(f"manifest line {line_number}: invalid overlap/ownership interval")
    duration_from_samples = (integers["end_sample"] - integers["start_sample"]) / integers[
        "sample_rate_hz"
    ]
    try:
        duration_s = float(item["duration_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest line {line_number}: duration_s must be numeric") from exc
    if not math.isfinite(duration_s) or abs(duration_s - duration_from_samples) > (
        1.0 / integers["sample_rate_hz"]
    ):
        raise ValueError(f"manifest line {line_number}: duration_s disagrees with sample interval")
    if (
        not isinstance(item["source_recording_id"], str)
        or not isinstance(item["split_group_id"], str)
        or not OPAQUE_ID_RE.fullmatch(item["source_recording_id"])
        or not OPAQUE_ID_RE.fullmatch(item["split_group_id"])
    ):
        raise ValueError(
            f"manifest line {line_number}: source/group IDs must be opaque and filesystem-safe"
        )
    if not isinstance(item["source_shareable"], bool):
        raise ValueError(f"manifest line {line_number}: source_shareable must be boolean")
    if item.get("shareable") is not False:
        raise ValueError(
            f"manifest line {line_number}: derived segments default to shareable=false"
        )
    if not isinstance(item.get("short_tail"), bool):
        raise ValueError(f"manifest line {line_number}: short_tail must be boolean")
    if item["reference_status"] not in {
        "unavailable",
        "file-level-only",
        "timed-cues-available",
    }:
        raise ValueError(f"manifest line {line_number}: invalid reference_status")
    if item["reference_assignment_policy"] != "overlap-midpoint-owner-v1":
        raise ValueError(f"manifest line {line_number}: invalid reference assignment policy")
    for key in ("intersecting_cue_ids", "owned_cue_ids"):
        if not isinstance(item[key], list) or any(
            not isinstance(value, str) or not value for value in item[key]
        ):
            raise ValueError(f"manifest line {line_number}: {key} must be a list of strings")
        if len(set(item[key])) != len(item[key]):
            raise ValueError(f"manifest line {line_number}: {key} contains duplicate cue IDs")
    if not isinstance(item.get("language"), str) or not item["language"].strip():
        raise ValueError(f"manifest line {line_number}: language must be non-empty")
    if not isinstance(item["split"], str) or not item["split"].strip():
        raise ValueError(f"manifest line {line_number}: split must be non-empty")
    if "reference_text" in item:
        raise ValueError(
            f"manifest line {line_number}: segment records must not embed reference_text; "
            "score quality after source-level stitching"
        )


def _validate_segment_manifest_collection(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate source grouping, coverage, ownership, and split invariants."""
    if not records:
        return
    plan_hashes = {str(record["segmentation_plan_sha256"]) for record in records}
    if len(plan_hashes) != 1:
        raise ValueError("segment manifest mixes multiple segmentation plans")

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    group_splits: dict[str, str] = {}
    content_hash_splits: dict[tuple[str, str], str] = {}
    for record in records:
        source_id = str(record["source_recording_id"])
        split = str(record["split"])
        by_source.setdefault(source_id, []).append(record)
        for label, value, table in (
            ("split_group_id", str(record["split_group_id"]), group_splits),
            (
                "source_audio_sha256",
                ("source_audio_sha256", str(record["source_audio_sha256"]).lower()),
                content_hash_splits,
            ),
            (
                "canonical_source_audio_sha256",
                (
                    "canonical_source_audio_sha256",
                    str(record["canonical_source_audio_sha256"]).lower(),
                ),
                content_hash_splits,
            ),
            (
                "canonical_pcm_sha256",
                ("canonical_pcm_sha256", str(record["canonical_pcm_sha256"]).lower()),
                content_hash_splits,
            ),
        ):
            previous = table.get(value)
            if previous is not None and previous != split:
                raise ValueError(
                    f"segment manifest {label} {value!r} crosses splits {previous!r}/{split!r}"
                )
            table[value] = split

    constant_fields = (
        "segment_count",
        "split_group_id",
        "split",
        "language",
        "source_audio_sha256",
        "canonical_source_audio_sha256",
        "canonical_pcm_sha256",
        "segmentation_plan_sha256",
        "source_shareable",
        "reference_status",
    )
    for source_id, source_records in by_source.items():
        ordered = sorted(source_records, key=lambda item: int(item["segment_index"]))
        expected_count = int(ordered[0]["segment_count"])
        if len(ordered) != expected_count or [
            int(item["segment_index"]) for item in ordered
        ] != list(range(expected_count)):
            raise ValueError(f"segment manifest source {source_id!r} has incomplete indices")
        for field in constant_fields:
            if any(item[field] != ordered[0][field] for item in ordered[1:]):
                raise ValueError(f"segment manifest source {source_id!r} changes {field}")
        if int(ordered[0]["start_sample"]) != 0:
            raise ValueError(f"segment manifest source {source_id!r} does not start at sample 0")

        owned_cues: set[str] = set()
        for index, item in enumerate(ordered):
            if item["sample_id"] != f"{source_id}__seg-{index:05d}":
                raise ValueError(
                    f"segment manifest source {source_id!r} has noncanonical sample_id"
                )
            previous = ordered[index - 1] if index > 0 else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            expected_left = (
                int(previous["end_sample"]) - int(item["start_sample"])
                if previous is not None
                else 0
            )
            expected_right = (
                int(item["end_sample"]) - int(following["start_sample"])
                if following is not None
                else 0
            )
            if expected_left < 0 or expected_right < 0:
                raise ValueError(f"segment manifest source {source_id!r} contains an audio gap")
            if (
                int(item["left_overlap_samples"]) != expected_left
                or int(item["right_overlap_samples"]) != expected_right
            ):
                raise ValueError(f"segment manifest source {source_id!r} has inconsistent overlap")
            expected_owner_start = (
                0
                if previous is None
                else (int(item["start_sample"]) + int(previous["end_sample"])) // 2
            )
            expected_owner_end = (
                int(ordered[-1]["end_sample"])
                if following is None
                else (int(following["start_sample"]) + int(item["end_sample"])) // 2
            )
            if (
                int(item["ownership_start_sample"]) != expected_owner_start
                or int(item["ownership_end_sample"]) != expected_owner_end
            ):
                raise ValueError(
                    f"segment manifest source {source_id!r} has inconsistent ownership"
                )
            owned = set(item["owned_cue_ids"])
            if not owned.issubset(set(item["intersecting_cue_ids"])):
                raise ValueError(
                    f"segment manifest source {source_id!r} owns a nonintersecting cue"
                )
            if owned_cues.intersection(owned):
                raise ValueError(f"segment manifest source {source_id!r} owns a cue more than once")
            owned_cues.update(owned)


def _load_manifest(path: Path) -> tuple[list[Sample], str]:
    manifest_hash = _sha256(path)
    samples: list[Sample] = []
    segment_records: list[Mapping[str, Any]] = []
    parser_versions: set[str] = set()
    seen_sample_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = (
            "sample_id",
            "audio_path",
            "audio_sha256",
            "language",
            "duration_s",
            "direct_model_eligible",
            "manifest_parser_version",
            "shareable",
        )
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f"manifest line {line_number} missing {missing}")
        sample_id = str(item["sample_id"])
        if not sample_id:
            raise ValueError(f"manifest line {line_number}: sample_id is empty")
        if sample_id in seen_sample_ids:
            raise ValueError(f"manifest line {line_number}: duplicate sample_id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        if item["direct_model_eligible"] is not True:
            raise ValueError(
                f"manifest line {line_number} ({sample_id}) must have "
                "direct_model_eligible=true; source recordings requiring "
                "segmentation cannot enter this benchmark"
            )
        parser_version = item["manifest_parser_version"]
        if not isinstance(parser_version, str):
            raise ValueError(
                f"manifest line {line_number}: manifest_parser_version must be a string"
            )
        if parser_version not in MANIFEST_PARSER_VERSIONS:
            raise ValueError(
                f"manifest line {line_number}: unsupported manifest_parser_version "
                f"{parser_version!r}; expected one of "
                f"{sorted(MANIFEST_PARSER_VERSIONS)}"
            )
        parser_versions.add(parser_version)
        if parser_version == SEGMENT_MANIFEST_PARSER_VERSION:
            _validate_segment_manifest_item(item, line_number)
            segment_records.append(item)
        if not isinstance(item["shareable"], bool):
            raise ValueError(f"manifest line {line_number}: shareable must be a JSON boolean")
        if item["shareable"] is True:
            authorization_id = item.get("authorization_id")
            if not isinstance(authorization_id, str) or not OPAQUE_ID_RE.fullmatch(
                authorization_id
            ):
                raise ValueError(
                    f"manifest line {line_number}: shareable=true requires an "
                    "opaque authorization_id"
                )
        try:
            declared_duration_s = float(item["duration_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"manifest line {line_number}: duration_s must be numeric") from exc
        if not 0.0 < declared_duration_s <= DIRECT_MODEL_MAX_AUDIO_S:
            raise ValueError(
                f"manifest line {line_number} ({sample_id}) duration_s must be "
                f"in (0, {DIRECT_MODEL_MAX_AUDIO_S:g}] for direct-model benchmarking"
            )
        expected_hash = str(item["audio_sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError(f"manifest line {line_number}: audio_sha256 must be 64 hex characters")

        audio_path = Path(item["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio for {item['sample_id']} not found: {audio_path}")

        actual_hash = _sha256(audio_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                f"SHA256 mismatch for {sample_id}: expected {expected_hash}, got {actual_hash}"
            )
        if parser_version == SEGMENT_MANIFEST_PARSER_VERSION:
            _validate_materialized_segment_audio(audio_path, item, line_number)

        samples.append(
            Sample(
                sample_id=sample_id,
                audio_path=audio_path,
                audio_sha256=expected_hash,
                reference_text=item.get("reference_text"),
                language=str(item["language"]),
                declared_duration_s=declared_duration_s,
                shareable=item["shareable"],
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "sample_id",
                        "audio_path",
                        "audio_sha256",
                        "reference_text",
                        "language",
                        "duration_s",
                        "shareable",
                    }
                },
            )
        )
    if not samples:
        raise ValueError("manifest contains no samples")
    if len(parser_versions) != 1:
        raise ValueError("manifest must not mix source and segment parser versions")
    _validate_segment_manifest_collection(segment_records)
    _validate_segment_bundle_files(path, segment_records)
    return samples, manifest_hash


def _read_audio(path: Path, target_rate: int = 16000) -> tuple[np.ndarray, float]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                "soundfile is required for non-WAV inputs; install the frozen benchmark environment"
            ) from None
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {sample_width}") from None
        audio = audio.reshape(-1, channels).mean(axis=1)

    source_duration = len(audio) / float(sample_rate)
    if sample_rate != target_rate:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("scipy is required for deterministic resampling") from exc
        divisor = math.gcd(int(sample_rate), target_rate)
        audio = resample_poly(
            audio,
            up=target_rate // divisor,
            down=int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)

    return np.ascontiguousarray(audio, dtype=np.float32), source_duration


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    normalized: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if (
            category.startswith("P")
            or category.startswith("S")
            or category.startswith("Z")
            or char.isspace()
        ):
            normalized.append(" ")
        else:
            normalized.append(char)
    return " ".join("".join(normalized).split())


def _edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    # Each cell is (total_cost, substitutions, deletions, insertions).  Keep
    # only two rows: full O(R*H) Python-object storage is prohibitive for long
    # transcripts even though the dynamic-programming time is unchanged.
    cols = len(hypothesis) + 1
    previous: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(cols)]
    for i in range(1, len(reference) + 1):
        current: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            candidates = (
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            current.append(min(candidates))
        previous = current

    cost, substitutions, deletions, insertions = previous[-1]
    return {
        "errors": cost,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_units": len(reference),
    }


def _quality(reference: str | None, hypothesis: str) -> dict[str, Any] | None:
    if reference is None:
        return None
    reference_normalized = _normalize_text(reference)
    hypothesis_normalized = _normalize_text(hypothesis)
    word_counts = _edit_counts(reference_normalized.split(), hypothesis_normalized.split())
    char_counts = _edit_counts(
        list(reference_normalized.replace(" ", "")),
        list(hypothesis_normalized.replace(" ", "")),
    )
    word_denominator = word_counts["reference_units"]
    char_denominator = char_counts["reference_units"]
    return {
        "reference_normalized": reference_normalized,
        "hypothesis_normalized": hypothesis_normalized,
        "word": word_counts,
        "char": char_counts,
        "wer": (word_counts["errors"] / word_denominator if word_denominator else None),
        "cer": (char_counts["errors"] / char_denominator if char_denominator else None),
        "status": (
            "evaluated"
            if word_denominator and char_denominator
            else "not_evaluated_empty_after_normalization"
        ),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty input")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def discover_source_paths(repository_root: Path) -> tuple[str, ...]:
    """Return the complete sorted package source set without requiring Git.

    Uncommitted files are deliberately included: the byte-level map is the
    evidence binding, while Git metadata remains useful context rather than a
    prerequisite for running a local benchmark.
    """
    package_root = repository_root.resolve() / "src" / "faster_glm_asr"
    paths: list[str] = []
    for directory in SOURCE_PACKAGE_DIRECTORIES:
        root = package_root / directory
        if not root.is_dir():
            raise FileNotFoundError(f"source package directory is missing: {root}")
        paths.extend(
            path.relative_to(repository_root.resolve()).as_posix()
            for path in root.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        )
    result = tuple(sorted(set(paths)))
    if not result:
        raise ValueError("source map is empty")
    return result


def _source_metadata(repository_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in discover_source_paths(repository_root):
        path = repository_root / relative
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema_version": SOURCE_MAP_SCHEMA,
        "files": files,
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _resolve_repository_root(value: Path | None) -> Path:
    """Resolve a real source checkout instead of guessing from an installed wheel."""
    candidate = value.resolve() if value is not None else Path(__file__).resolve().parents[3]
    required = (
        candidate / "pyproject.toml",
        candidate / "src/faster_glm_asr/benchmarking/runner.py",
        candidate / provenance.READINESS_CONTRACT_PATH,
        candidate / provenance.READINESS_TOOL_PATH,
    )
    if not (candidate / ".git").exists() or any(not path.is_file() for path in required):
        raise ValueError(
            "benchmark provenance requires a clean source checkout; run from the "
            "repository or pass --repository-root pointing to that checkout"
        )
    return candidate


def _memory_snapshot(torch: Any) -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def _normalized_gpu_uuid(value: str) -> str:
    """Normalize CUDA/NVML UUID spellings for identity comparison."""
    normalized = str(value).strip().lower()
    for prefix in ("gpu-", "mig-"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _nvml_uuid_candidates(cuda_uuid: str) -> list[str]:
    """Expand PyTorch's bare CUDA UUID into NVML-compatible spellings."""
    value = str(cuda_uuid).strip()
    if not value:
        return []
    if value.upper().startswith(("GPU-", "MIG-")):
        return [value]
    return [f"GPU-{value}", f"MIG-{value}", value]


def _nvml_handle_by_uuid(pynvml: Any, identifier: str) -> Any:
    try:
        return pynvml.nvmlDeviceGetHandleByUUID(identifier)
    except TypeError:
        return pynvml.nvmlDeviceGetHandleByUUID(identifier.encode("ascii"))


class NvmlProcessSampler:
    """Optional per-process NVML sampler for a separate memory-focused run."""

    def __init__(
        self,
        device_index: int,
        interval_ms: float,
        device_uuid: str | None = None,
    ) -> None:
        self.interval_ms = interval_ms
        self.available = False
        self.reason: str | None = None
        self._samples: list[tuple[float, int]] = []
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._started: float | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        self._query: Any = None
        self._device_selection: str | None = None
        self._device_uuid: str | None = None
        if interval_ms <= 0:
            self.reason = "disabled (--nvml-sample-ms=0)"
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            normalized_uuid = str(device_uuid).strip() if device_uuid else ""
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            visible_identifier: str | None = None
            if visible:
                identifiers = [item.strip() for item in visible.split(",")]
                if device_index >= len(identifiers):
                    raise RuntimeError("CUDA logical index is outside CUDA_VISIBLE_DEVICES")
                visible_identifier = identifiers[device_index]

            if visible_identifier and visible_identifier.upper().startswith(("GPU-", "MIG-")):
                if normalized_uuid and _normalized_gpu_uuid(
                    visible_identifier
                ) != _normalized_gpu_uuid(normalized_uuid):
                    raise RuntimeError(
                        "CUDA_VISIBLE_DEVICES UUID does not match the active PyTorch CUDA UUID"
                    )
                self._handle = _nvml_handle_by_uuid(pynvml, visible_identifier)
                self._device_selection = "CUDA_VISIBLE_DEVICES UUID -> NVML UUID"
            elif normalized_uuid:
                failures = []
                for candidate in _nvml_uuid_candidates(normalized_uuid):
                    try:
                        self._handle = _nvml_handle_by_uuid(pynvml, candidate)
                        self._device_selection = "PyTorch CUDA UUID candidate -> NVML UUID"
                        break
                    except Exception as exc:
                        failures.append(f"{candidate}: {type(exc).__name__}")
                if self._handle is None:
                    raise RuntimeError(
                        "cannot resolve PyTorch CUDA UUID through NVML candidates: "
                        + ", ".join(failures)
                    )
                if visible_identifier and visible_identifier.isdigit():
                    self._device_selection += " (numeric CUDA mask verified by UUID)"
                elif visible_identifier:
                    raise RuntimeError("cannot map CUDA_VISIBLE_DEVICES identifier to NVML")
            elif visible_identifier and visible_identifier.isdigit():
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible_identifier))
                self._device_selection = "CUDA_VISIBLE_DEVICES physical index -> NVML index"
            elif visible_identifier:
                raise RuntimeError("cannot map CUDA_VISIBLE_DEVICES identifier to NVML")
            else:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._device_selection = "unmasked CUDA index -> NVML index"
            reported_uuid = pynvml.nvmlDeviceGetUUID(self._handle)
            if isinstance(reported_uuid, bytes):
                reported_uuid = reported_uuid.decode("ascii")
            self._device_uuid = str(reported_uuid)
            if normalized_uuid and _normalized_gpu_uuid(self._device_uuid) != _normalized_gpu_uuid(
                normalized_uuid
            ):
                raise RuntimeError(
                    "resolved NVML device UUID does not match active PyTorch CUDA device UUID"
                )
            for name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            ):
                query = getattr(pynvml, name, None)
                if query is not None:
                    self._query = query
                    break
            if self._query is None:
                raise RuntimeError("no compute-process query in pynvml")
            self.available = True
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    def _read_process_bytes(self) -> int | None:
        if not self.available:
            return None
        processes = self._query(self._handle)
        values = []
        unavailable = getattr(self._pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for process in processes:
            if int(process.pid) != os.getpid():
                continue
            used = getattr(process, "usedGpuMemory", None)
            if used is None or used == unavailable:
                continue
            values.append(int(used))
        return max(values) if values else None

    def _sample_once(self, started: float) -> None:
        try:
            used = self._read_process_bytes()
            if used is not None:
                self._samples.append(((time.perf_counter() - started) * 1000.0, used))
        except Exception as exc:
            self.available = False
            self.reason = f"sampling failed: {type(exc).__name__}: {exc}"

    def start(self) -> None:
        self._samples = []
        if not self.available:
            return
        started = time.perf_counter()
        self._started = started
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._sample_once(started)

        def loop() -> None:
            interval_s = self.interval_ms / 1000.0
            while not stop_event.wait(interval_s):
                self._sample_once(started)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_ms / 500.0))
            if self._thread.is_alive():
                self.available = False
                self.reason = "NVML sampler thread did not stop before timeout"
                raise RuntimeError(self.reason)
        if self.available and self._started is not None:
            # Include one final post-operation reading.
            self._sample_once(self._started)
        values = [used for _, used in self._samples]
        summary: dict[str, Any] = {
            "enabled": self.interval_ms > 0,
            "available": self.available,
            "interval_ms": self.interval_ms,
            "sample_count": len(values),
            "reason": self.reason,
            "device_selection": self._device_selection,
            "nvml_device_uuid": self._device_uuid,
            "scope": "pre-prepare-sync-through-post-decode-output-hash",
            "peak_semantics": ("polling-observed lower bound; not an allocator or exact peak"),
            "samples": [
                {"elapsed_ms": elapsed_ms, "used_bytes": used_bytes}
                for elapsed_ms, used_bytes in self._samples
            ],
        }
        if values:
            elapsed = [timestamp for timestamp, _ in self._samples]
            gaps = [right - left for left, right in pairwise(elapsed)]
            summary.update(
                {
                    "first_used_bytes": values[0],
                    "last_used_bytes": values[-1],
                    "observed_peak_used_bytes": max(values),
                    "coverage_ms": elapsed[-1],
                    "max_sample_gap_ms": max(gaps, default=0.0),
                }
            )
        elif self.interval_ms > 0 and summary["reason"] is None:
            summary["reason"] = (
                "current PID was not reported by the selected NVML device "
                "(check MIG/MPS/device mapping and permissions)"
            )
        self._stop_event = None
        self._thread = None
        self._started = None
        return summary

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_ms / 500.0))
            if self._thread.is_alive():
                # Avoid shutting NVML down while a blocked sampling query may
                # still be using it.  Process teardown will reclaim the handle.
                return
        self._stop_event = None
        self._thread = None
        self._started = None
        if self._pynvml is not None:
            with suppress(Exception):
                self._pynvml.nvmlShutdown()


class CudaPhaseRecorder:
    """Collect CUDA-event phase timings for a diagnostic instrumentation run.

    Event recording itself adds overhead, especially for one-token decode.  The
    canonical uninstrumented wall-clock result remains the performance metric;
    these phase values are for bottleneck attribution and are emitted only when
    ``--phase-timing`` is explicit.
    """

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self._active: dict[str, Any] = {}
        self._pairs: dict[str, list[tuple[Any, Any]]] = {}

    def callback(self, label: str, boundary: str) -> None:
        if boundary == "start":
            if label in self._active:
                raise RuntimeError(f"phase already active: {label}")
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self._active[label] = event
            return
        if boundary != "end":
            raise ValueError(f"unknown phase boundary: {boundary}")
        start = self._active.pop(label, None)
        if start is None:
            raise RuntimeError(f"phase ended without a matching start: {label}")
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        self._pairs.setdefault(label, []).append((start, end))

    def finalize(self) -> dict[str, Any]:
        if self._active:
            raise RuntimeError(f"unfinished CUDA phases: {sorted(self._active)}")
        result: dict[str, Any] = {}
        for label, pairs in sorted(self._pairs.items()):
            values = [float(start.elapsed_time(end)) for start, end in pairs]
            result[label] = {
                "events_ms": values,
                "summary_ms": _summary(values),
                "total_ms": float(sum(values)),
            }
        return result

    def discard_active(self, label: str) -> None:
        """Discard a trailing start marker that has no following milestone."""
        self._active.pop(label, None)


def _measurement_profile(nvml_sample_ms: float, phase_timing: bool) -> str:
    """Return a mutually exclusive profile with explicit claim eligibility."""
    if nvml_sample_ms > 0 and phase_timing:
        raise ValueError(
            "--nvml-sample-ms and --phase-timing are mutually exclusive; "
            "run separate request_memory and diagnostic_phase trials"
        )
    if nvml_sample_ms > 0:
        return "request_memory"
    if phase_timing:
        return "diagnostic_phase"
    return "canonical_latency"


def _public_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove private cleartext, paths, and stable content fingerprints.

    The result is data-minimized and pseudonymized, not guaranteed anonymous.
    Authorization and a manual disclosure review remain mandatory.
    """
    public = copy.deepcopy(dict(result))
    public["run"].pop("manifest_path", None)
    public["run"].pop("manifest_sha256", None)
    public["run"]["private_manifest_fingerprint_redacted"] = True
    public["run"]["artifact_scope"] = "public"
    environment = public.get("environment")
    if isinstance(environment, dict):
        environment.pop("device_uuid", None)
        environment.pop("cuda_visible_devices", None)
        environment["device_identity_redacted"] = True
    git = public.get("git")
    if isinstance(git, dict):
        git.pop("branch", None)
        git["branch_redacted"] = True

    private_safe_metadata = {
        "shareable",
        "duration_bucket",
        "direct_model_eligible",
        "manifest_parser_version",
    }
    public_safe_metadata = private_safe_metadata | {
        "authorization_id",
        "collection",
        "speaker_count",
        "sample_rate_hz",
        "channels",
        "license_scope",
        "split",
    }

    private_counter = 0
    summary_metrics = (
        "generation_ms",
        "processor_plus_generation_ms",
        "request_no_file_io_ms",
        "rtf_generation",
    )
    summary_statistics = ("n", "mean", "std", "min", "p50", "p95", "max")
    for sample_index, sample in enumerate(public["samples"]):
        metadata = sample.get("metadata", {})
        shareable = metadata.get("shareable") is True
        metadata.pop("reference_path", None)
        quality = sample.get("quality")
        if quality is not None:
            quality.pop("reference_normalized", None)
            quality.pop("hypothesis_normalized", None)
        if shareable:
            safe_quality = None
            if quality is not None:
                safe_quality = {
                    "wer": quality["wer"],
                    "cer": quality["cer"],
                    "status": quality["status"],
                }
            public["samples"][sample_index] = {
                "sample_id": sample["sample_id"],
                "language": sample["language"],
                "duration_s": sample["duration_s"],
                "generated_tokens": sample["generated_tokens"],
                "hypothesis": sample["hypothesis"],
                "quality": safe_quality,
                "summary": {
                    metric: {
                        statistic: sample["summary"][metric][statistic]
                        for statistic in summary_statistics
                    }
                    for metric in summary_metrics
                },
                "metadata": {
                    key: value for key, value in metadata.items() if key in public_safe_metadata
                },
            }
            continue
        private_counter += 1
        safe_quality = None
        if quality is not None:
            safe_quality = {
                "wer": quality["wer"],
                "cer": quality["cer"],
                "status": quality["status"],
            }
        # Rebuild the complete private sample from an allowlist.  Deleting
        # known sensitive fields from a copied object is fail-open: any future
        # free-form field could otherwise pass through unnoticed.
        public["samples"][sample_index] = {
            "sample_id": f"private-{private_counter:04d}",
            "language": "redacted",
            "duration_s": sample["duration_s"],
            "generated_tokens": sample["generated_tokens"],
            "quality": safe_quality,
            "summary": {
                metric: {
                    statistic: sample["summary"][metric][statistic]
                    for statistic in summary_statistics
                }
                for metric in summary_metrics
            },
            "metadata": {
                "shareable": False,
                "duration_bucket": (
                    "materialized-segment"
                    if metadata.get("manifest_parser_version") == SEGMENT_MANIFEST_PARSER_VERSION
                    else "direct-model"
                ),
                "direct_model_eligible": True,
                "manifest_parser_version": metadata.get(
                    "manifest_parser_version", SOURCE_MANIFEST_PARSER_VERSION
                ),
            },
        }
    return public


class Runner:
    def configuration(self) -> dict[str, Any]:
        raise NotImplementedError

    def prepare(self, audio: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def generate(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        phase_recorder: CudaPhaseRecorder | None = None,
    ) -> Any:
        raise NotImplementedError

    def decode(self, output: Any, inputs: dict[str, Any]) -> tuple[str, list[int]]:
        raise NotImplementedError


def _load_custom_snapshot(model_snapshot_root: Path, revision: str) -> tuple[Any, Any]:
    from faster_glm_asr.modeling.weight_loading import load_model_from_hf

    return load_model_from_hf(
        model_name=str(model_snapshot_root),
        revision=revision,
        local_files_only=True,
    )


def _load_hf_snapshot(model_snapshot_root: Path, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoProcessor, GlmAsrForConditionalGeneration

    local_root = str(model_snapshot_root)
    processor = AutoProcessor.from_pretrained(
        local_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = GlmAsrForConditionalGeneration.from_pretrained(
        local_root,
        local_files_only=True,
        dtype=torch.float32,
        use_safetensors=True,
        trust_remote_code=False,
    )
    return model, processor


class CustomRunner(Runner):
    def __init__(
        self,
        implementation: str,
        model_snapshot_root: Path,
        revision: str,
    ) -> None:
        import torch

        from faster_glm_asr.kernels import layers

        if not torch.cuda.is_available():
            raise RuntimeError("the full custom model benchmark requires CUDA")
        self.torch = torch
        self.implementation = implementation
        layers.Linear.BACKEND = "cublas"
        layers.MLP.FUSED = False
        if hasattr(layers, "EncoderMLP"):
            layers.EncoderMLP.FUSED = False

        self.model, self.processor = _load_custom_snapshot(model_snapshot_root, revision)
        self.device = torch.device("cuda")

    def configuration(self) -> dict[str, Any]:
        cache_mode = {
            "custom_tuple_cache": "tuple-torch-cat",
            "custom_static_cache": "static-preallocated",
        }.get(self.implementation, "none")
        return {
            "system": "custom-torch-triton-hybrid",
            "model_source": "validated-local-snapshot",
            "network_fallback": False,
            "storage_activation_dtype": "torch.float32",
            "linear_backend": "cublas",
            "mlp_fused": False,
            "kernel_dispatch_policy_version": "hybrid-auto-v1",
            "component_dispatch_policy": (
                "supported-norm-rope-embedding-conv-auto-triton-else-torch"
            ),
            "attention_dispatch_policy": (
                "dense-triton-if-padded-and-head-dim-lte-256-else-torch-dense"
            ),
            "gqa_kv_expansion": "explicit-expand",
            "long_form_attention_expectation": ("torch-dense-likely-for-30s-input"),
            "backend_trace_status": (
                "policy-inferred-diagnostic-trace-required-for-observed-backend"
            ),
            "performance_attribution": ("cache-strategy-only-no-triton-kernel-attribution"),
            "cache_mode": cache_mode,
            "greedy_fast_path": self.implementation
            in {
                "custom_greedy_full_prefix",
                "custom_tuple_cache",
                "custom_static_cache",
            },
            "token_output_mode": (
                "preallocated"
                if self.implementation in {"custom_tuple_cache", "custom_static_cache"}
                else "torch-cat"
            ),
        }

    def prepare(self, audio: np.ndarray) -> dict[str, Any]:
        inputs = self.processor.apply_transcription_request(audio)
        prepared: dict[str, Any] = {
            "input_features": inputs.input_features.to(
                device=self.device, dtype=self.torch.float32
            ),
            "input_ids": inputs.input_ids.to(device=self.device, dtype=self.torch.int64),
        }
        feature_mask = getattr(inputs, "input_features_mask", None)
        if feature_mask is not None:
            prepared["input_features_mask"] = feature_mask.to(
                device=self.device, dtype=self.torch.float32
            )
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            prepared["attention_mask"] = attention_mask.to(device=self.device)
        return prepared

    def generate(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        phase_recorder: CudaPhaseRecorder | None = None,
    ) -> Any:
        kwargs = {
            "input_ids": inputs["input_ids"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "phase_callback": (phase_recorder.callback if phase_recorder is not None else None),
        }
        if self.implementation == "custom_static_cache":
            return self.model.generate_static_cache(
                inputs["input_features"], do_sample=False, **kwargs
            )
        if self.implementation == "custom_tuple_cache":
            return self.model.generate_tuple_cache(
                inputs["input_features"], do_sample=False, **kwargs
            )
        top_k = 0 if self.implementation == "custom_greedy_full_prefix" else 1
        return self.model.generate(inputs["input_features"], temperature=1.0, top_k=top_k, **kwargs)

    def decode(self, output: Any, inputs: dict[str, Any]) -> tuple[str, list[int]]:
        input_length = int(inputs["input_ids"].shape[1])
        generated = output[:, input_length:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        token_ids = [int(token) for token in generated[0].detach().cpu().tolist()]
        return str(texts[0]).strip(), token_ids


class HuggingFaceRunner(Runner):
    def __init__(self, model_snapshot_root: Path, use_cache: bool) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the full Hugging Face benchmark requires CUDA")
        self.torch = torch
        self.device = torch.device("cuda")
        # The formal comparison matrix is deliberately single-precision.  The
        # custom implementation currently has a validated FP32 path only, so
        # running the reference at the same precision keeps cache strategy as
        # the variable under test.
        self.dtype = torch.float32
        self.use_cache = use_cache
        self.model, self.processor = _load_hf_snapshot(model_snapshot_root, torch)
        self.model = self.model.to(self.device)
        self.model.eval()

    def configuration(self) -> dict[str, Any]:
        return {
            "system": "huggingface-transformers",
            "model_source": "validated-local-snapshot",
            "network_fallback": False,
            "storage_activation_dtype": str(self.dtype),
            "cache_mode": "dynamic" if self.use_cache else "none",
            "do_sample": False,
        }

    def prepare(self, audio: np.ndarray) -> dict[str, Any]:
        batch = self.processor.apply_transcription_request(audio)
        prepared: dict[str, Any] = {}
        for name, value in dict(batch).items():
            if not hasattr(value, "to"):
                prepared[name] = value
            elif value.is_floating_point():
                prepared[name] = value.to(device=self.device, dtype=self.dtype)
            else:
                prepared[name] = value.to(device=self.device)
        return prepared

    def generate(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        phase_recorder: CudaPhaseRecorder | None = None,
    ) -> Any:
        if phase_recorder is None:
            return self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=self.use_cache,
                max_new_tokens=max_new_tokens,
            )

        phase_recorder.callback("time_to_first_token", "start")
        pre_handle = self.model.register_forward_pre_hook(
            lambda _module, _args: phase_recorder.callback("hf_model_forward", "start")
        )
        post_handle = self.model.register_forward_hook(
            lambda _module, _args, _output: phase_recorder.callback("hf_model_forward", "end")
        )

        torch_module = self.torch

        class TokenReadyMarker:
            def __init__(self) -> None:
                self.marked = False

            def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> Any:
                # Transformers evaluates stopping criteria only after selecting
                # and appending the next token.  This therefore measures the
                # documented token-ready milestone rather than the earlier
                # logits-processor boundary.
                if not self.marked:
                    phase_recorder.callback("time_to_first_token", "end")
                    self.marked = True
                else:
                    phase_recorder.callback("inter_token_interval", "end")
                phase_recorder.callback("inter_token_interval", "start")
                return torch_module.zeros(
                    input_ids.shape[0],
                    dtype=torch_module.bool,
                    device=input_ids.device,
                )

        token_ready_marker = TokenReadyMarker()

        try:
            return self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=self.use_cache,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[token_ready_marker],
            )
        finally:
            # The final token has no following token-ready milestone.
            phase_recorder.discard_active("time_to_first_token")
            phase_recorder.discard_active("inter_token_interval")
            pre_handle.remove()
            post_handle.remove()

    def decode(self, output: Any, inputs: dict[str, Any]) -> tuple[str, list[int]]:
        input_length = int(inputs["input_ids"].shape[1])
        generated = output[:, input_length:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        token_ids = [int(token) for token in generated[0].detach().cpu().tolist()]
        return str(texts[0]).strip(), token_ids


def _torch_environment(torch: Any) -> dict[str, Any]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, "uuid", None)
    if isinstance(device_uuid, bytes):
        device_uuid = device_uuid.decode("ascii")
    distributions = (
        "torch",
        "triton",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "numpy",
        "scipy",
        "soundfile",
        "nvidia-ml-py",
    )
    package_versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    try:
        driver_lines = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).splitlines()
        driver_versions = sorted({line.strip() for line in driver_lines if line.strip()})
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        driver_versions = []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_index": device,
        "device_name": properties.name,
        "device_uuid": str(device_uuid) if device_uuid is not None else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "nvidia_driver_versions": driver_versions,
        "package_versions": package_versions,
    }


def _prepared_metadata(prepared: Mapping[str, Any]) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for name, value in prepared.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            tensor_metadata = {
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            if name == "input_ids" or name.endswith("mask"):
                canonical = value.detach().cpu()
                canonical = canonical.ne(0) if name.endswith("mask") else canonical.long()
                content = json.dumps(
                    {
                        "shape": tensor_metadata["shape"],
                        "values": canonical.reshape(-1).tolist(),
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                tensor_metadata["content_sha256"] = hashlib.sha256(content).hexdigest()
            elif name == "input_features" and value.is_floating_point():
                # Formal runners both execute these exact FP32 values.  Hash the
                # stored bytes without quantizing across dtypes so paired-run
                # equality cannot hide a precision mismatch.
                import torch

                content = (
                    value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
                )
                tensor_metadata["content_sha256"] = hashlib.sha256(content).hexdigest()
            tensors[name] = tensor_metadata
    input_ids = prepared.get("input_ids")
    feature_mask = prepared.get("input_features_mask")
    return {
        "tensors": tensors,
        "request_batch": int(input_ids.shape[0]) if input_ids is not None else None,
        "prompt_tokens": int(input_ids.shape[1]) if input_ids is not None else None,
        "audio_placeholder_tokens": (
            int((input_ids == 59260).sum().item()) if input_ids is not None else None
        ),
        "audio_windows": (
            int(prepared["input_features"].shape[0]) if "input_features" in prepared else None
        ),
        "valid_feature_frames": (
            int(feature_mask.sum().item()) if feature_mask is not None else None
        ),
    }


def _validate_context_budget(
    prepared: Mapping[str, Any], max_new_tokens: int, sample_id: str
) -> None:
    input_ids = prepared.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError(f"prepared input_ids for {sample_id} must have shape [B,S]")
    if int(input_ids.shape[0]) != 1:
        raise ValueError(f"benchmark currently requires request batch_size=1: {sample_id}")
    requested_tokens = int(input_ids.shape[1]) + max_new_tokens
    if requested_tokens > TEXT_MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"prepared prompt plus max_new_tokens for {sample_id} requests "
            f"{requested_tokens} positions, exceeding fixed model limit "
            f"{TEXT_MAX_POSITION_EMBEDDINGS}; segment the audio or reduce generation"
        )


def _aggregate_quality(sample_results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    word = {
        key: 0 for key in ("errors", "substitutions", "deletions", "insertions", "reference_units")
    }
    char = dict(word)
    evaluated_samples = 0
    for result in sample_results:
        quality = result.get("quality")
        if quality is None:
            continue
        evaluated_samples += 1
        for key in word:
            word[key] += int(quality["word"][key])
            char[key] += int(quality["char"][key])
    word_evaluated = word["reference_units"] > 0
    char_evaluated = char["reference_units"] > 0
    return {
        "evaluated_samples": evaluated_samples,
        "word": word,
        "char": char,
        "wer": word["errors"] / word["reference_units"] if word_evaluated else None,
        "cer": char["errors"] / char["reference_units"] if char_evaluated else None,
        "status": (
            "evaluated"
            if word_evaluated and char_evaluated
            else "partially_evaluated"
            if word_evaluated or char_evaluated
            else "not_evaluated"
        ),
        "normalization": "NFKC + casefold + Unicode punctuation/symbol removal + whitespace collapse",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        help=(
            "clean faster-glm-asr source checkout used for provenance; required "
            "when the installed module cannot identify one"
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=(
            "hf_cached",
            "hf_no_cache",
            "custom_full_prefix",
            "custom_greedy_full_prefix",
            "custom_tuple_cache",
            "custom_static_cache",
        ),
        required=True,
    )
    parser.add_argument(
        "--artifact-scope",
        choices=("private", "public"),
        default="private",
        help="public redacts text/paths for samples not marked shareable=true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output intentionally",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-snapshot-root", type=Path, required=True)
    parser.add_argument("--model-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="enforce clean-Git and warmup gates for comparable matrix evidence",
    )
    parser.add_argument(
        "--environment-lock",
        type=Path,
        required=True,
        help=(
            "full pip-freeze/lock file whose SHA256 is embedded in the artifact; "
            "the strict comparator rejects artifacts without this fingerprint"
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--nvml-sample-ms",
        type=float,
        default=0.0,
        help="sample process memory in a separate memory run; 0 disables it",
    )
    parser.add_argument(
        "--phase-timing",
        action="store_true",
        help=(
            "record diagnostic CUDA-event phases; event instrumentation adds "
            "overhead, so run it separately from canonical latency trials"
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.max_new_tokens <= 0 or args.nvml_sample_ms < 0:
        parser.error("warmup and nvml-sample-ms must be >=0; repeats and max-new-tokens must be >0")
    try:
        measurement_profile = _measurement_profile(args.nvml_sample_ms, args.phase_timing)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists (use --overwrite intentionally): {args.output}")
    if not args.environment_lock.is_file():
        parser.error(f"environment lock not found: {args.environment_lock}")
    try:
        repository_root = _resolve_repository_root(args.repository_root)
    except ValueError as exc:
        parser.error(str(exc))
    git_metadata = _git_metadata(repository_root)
    git_is_clean = (
        isinstance(git_metadata.get("commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", git_metadata["commit"]) is not None
        and git_metadata.get("dirty") is False
    )
    if args.formal and not git_is_clean:
        parser.error("formal benchmark runs require a clean Git checkout")
    if args.formal and args.warmup < 1:
        parser.error("formal benchmark runs require warmup >= 1")
    comparability_reasons = []
    if not args.formal:
        comparability_reasons.append("runner-not-in-formal-mode")
    if not git_is_clean:
        comparability_reasons.append("git-dirty-or-unavailable")
    if args.warmup < 1:
        comparability_reasons.append("warmup-below-one")
    comparable = not comparability_reasons
    if args.artifact_scope == "public" and not comparable:
        parser.error("public artifacts require a clean formal run with warmup >= 1")

    model_snapshot_root = args.model_snapshot_root.resolve()
    try:
        model_snapshot = provenance.validate_model_snapshot(
            model_snapshot_root,
            args.model_snapshot_manifest,
            expected_repo_id=args.model_name,
            expected_revision=args.model_revision,
        )
        readiness = provenance.validate_readiness_report(
            args.readiness_report,
            repository_root=repository_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if model_snapshot["contract_sha256"] != readiness["contract_sha256"]:
        parser.error("model snapshot and readiness report contract hashes differ")
    manifest_path = args.manifest.resolve()
    samples, manifest_hash = _load_manifest(manifest_path)

    load_started = time.perf_counter()
    if args.implementation in {"hf_cached", "hf_no_cache"}:
        runner: Runner = HuggingFaceRunner(
            model_snapshot_root,
            use_cache=args.implementation == "hf_cached",
        )
    else:
        runner = CustomRunner(
            args.implementation,
            model_snapshot_root,
            args.model_revision,
        )
    runner_construction_envelope_ms = (time.perf_counter() - load_started) * 1000.0

    import torch

    environment = _torch_environment(torch)
    environment["environment_lock_sha256"] = _sha256(args.environment_lock.resolve())
    nvml_sampler = NvmlProcessSampler(
        device_index=torch.cuda.current_device(),
        interval_ms=args.nvml_sample_ms,
        device_uuid=environment.get("device_uuid"),
    )
    atexit.register(nvml_sampler.close)
    if measurement_profile == "request_memory" and not nvml_sampler.available:
        raise RuntimeError(
            "request_memory profile requires a working per-process NVML sampler: "
            f"{nvml_sampler.reason}"
        )
    sample_results: list[dict[str, Any]] = []

    for sample in samples:
        load_started = time.perf_counter()
        audio, measured_duration_s = _read_audio(sample.audio_path)
        audio_load_ms = (time.perf_counter() - load_started) * 1000.0
        if sample.declared_duration_s is not None and not math.isclose(
            measured_duration_s, sample.declared_duration_s, rel_tol=0.0, abs_tol=0.02
        ):
            raise ValueError(
                f"duration mismatch for {sample.sample_id}: manifest "
                f"{sample.declared_duration_s}, decoded {measured_duration_s}"
            )
        if measured_duration_s > DIRECT_MODEL_MAX_AUDIO_S:
            raise ValueError(
                f"decoded duration for {sample.sample_id} is "
                f"{measured_duration_s:.3f}s, exceeding the "
                f"{DIRECT_MODEL_MAX_AUDIO_S:g}s direct-model guardrail; "
                "build an explicit segment-level manifest"
            )

        for _ in range(args.warmup):
            warmup_inputs = runner.prepare(audio)
            _validate_context_budget(warmup_inputs, args.max_new_tokens, sample.sample_id)
            with torch.inference_mode():
                warmup_output = runner.generate(warmup_inputs, args.max_new_tokens)
            torch.cuda.synchronize()
            del warmup_output, warmup_inputs

        observations: list[dict[str, Any]] = []
        final_text = ""
        final_token_count = 0
        final_token_ids: list[int] = []
        prepared_metadata: dict[str, Any] | None = None
        for repeat_index in range(args.repeats):
            nvml_sampler.start()
            torch.cuda.synchronize()
            memory_before_prepare = _memory_snapshot(torch)
            preprocess_started = time.perf_counter()
            prepared = runner.prepare(audio)
            _validate_context_budget(prepared, args.max_new_tokens, sample.sample_id)
            torch.cuda.synchronize()
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
            current_prepared_metadata = _prepared_metadata(prepared)
            if prepared_metadata is None:
                prepared_metadata = current_prepared_metadata
            elif current_prepared_metadata != prepared_metadata:
                raise RuntimeError(
                    f"processor output changed across repeats for {sample.sample_id}"
                )
            memory_after_prepare = _memory_snapshot(torch)

            torch.cuda.reset_peak_memory_stats()
            phase_recorder = CudaPhaseRecorder(torch) if args.phase_timing else None
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = runner.generate(
                    prepared,
                    args.max_new_tokens,
                    phase_recorder=phase_recorder,
                )
            torch.cuda.synchronize()
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            phase_timings = phase_recorder.finalize() if phase_recorder is not None else None

            decode_started = time.perf_counter()
            current_text, current_token_ids = runner.decode(output, prepared)
            current_token_count = len(current_token_ids)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            output_digest = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
            token_ids_digest = hashlib.sha256(
                json.dumps(current_token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if repeat_index == 0:
                final_text = current_text
                final_token_count = current_token_count
                final_token_ids = current_token_ids
            elif current_text != final_text or current_token_ids != final_token_ids:
                raise RuntimeError(
                    f"greedy token sequence changed across repeats for "
                    f"{sample.sample_id}: repeat 0 tokens={final_token_count}, "
                    f"repeat {repeat_index} tokens={current_token_count}"
                )
            memory_after_generation = _memory_snapshot(torch)
            request_no_file_io_ms = preprocess_ms + generation_ms + decode_ms
            nvml_memory = nvml_sampler.stop()
            if measurement_profile == "request_memory" and (
                nvml_memory.get("available") is not True
                or nvml_memory.get("sample_count", 0) <= 0
                or "observed_peak_used_bytes" not in nvml_memory
            ):
                raise RuntimeError(
                    "request_memory profile did not produce a valid observed "
                    f"per-process peak: {nvml_memory.get('reason')}"
                )
            observations.append(
                {
                    "repeat": repeat_index,
                    "preprocess_ms": preprocess_ms,
                    "generation_ms": generation_ms,
                    "decode_ms": decode_ms,
                    "processor_plus_generation_ms": preprocess_ms + generation_ms,
                    "request_no_file_io_ms": request_no_file_io_ms,
                    "generated_tokens": current_token_count,
                    "hypothesis_sha256": output_digest,
                    "generated_token_ids_sha256": token_ids_digest,
                    "rtf_generation": generation_ms / 1000.0 / measured_duration_s,
                    "rtf_request_no_file_io": (
                        request_no_file_io_ms / 1000.0 / measured_duration_s
                    ),
                    "memory_bytes": {
                        "before_prepare": memory_before_prepare,
                        "after_prepare": memory_after_prepare,
                        "after_generation": memory_after_generation,
                        "generation_allocated_peak": int(torch.cuda.max_memory_allocated()),
                        "generation_reserved_peak": int(torch.cuda.max_memory_reserved()),
                        "generation_allocated_peak_delta": max(
                            0,
                            int(torch.cuda.max_memory_allocated())
                            - memory_after_prepare["allocated"],
                        ),
                    },
                    "nvml_process_memory": nvml_memory,
                    "diagnostic_cuda_phases": phase_timings,
                }
            )
            # Absolute allocator baselines for the next request must not retain
            # this request's prepared tensors or generated output.
            del output, prepared

        generation_values = [item["generation_ms"] for item in observations]
        e2e_values = [item["processor_plus_generation_ms"] for item in observations]
        rtf_values = [item["rtf_generation"] for item in observations]
        request_values = [item["request_no_file_io_ms"] for item in observations]
        quality = _quality(sample.reference_text, final_text)
        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "audio_sha256": sample.audio_sha256 or _sha256(sample.audio_path),
                "language": sample.language,
                "duration_s": measured_duration_s,
                "audio_load_ms": audio_load_ms,
                "prepared_request": prepared_metadata,
                "metadata": {**dict(sample.metadata), "shareable": sample.shareable},
                "hypothesis": final_text,
                "hypothesis_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
                "generated_tokens": final_token_count,
                "generated_token_ids": final_token_ids,
                "generated_token_ids_sha256": hashlib.sha256(
                    json.dumps(final_token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "quality": quality,
                "summary": {
                    "generation_ms": _summary(generation_values),
                    "processor_plus_generation_ms": _summary(e2e_values),
                    "request_no_file_io_ms": _summary(request_values),
                    "rtf_generation": _summary(rtf_values),
                },
                "observations": observations,
            }
        )

    all_generation = [
        observation["generation_ms"]
        for sample_result in sample_results
        for observation in sample_result["observations"]
    ]
    all_e2e = [
        observation["processor_plus_generation_ms"]
        for sample_result in sample_results
        for observation in sample_result["observations"]
    ]
    all_requests = [
        observation["request_no_file_io_ms"]
        for sample_result in sample_results
        for observation in sample_result["observations"]
    ]
    all_observations = [
        observation
        for sample_result in sample_results
        for observation in sample_result["observations"]
    ]
    request_memory_eligible = measurement_profile == "request_memory" and all(
        observation["nvml_process_memory"].get("available") is True
        and observation["nvml_process_memory"].get("sample_count", 0) > 0
        and "observed_peak_used_bytes" in observation["nvml_process_memory"]
        for observation in all_observations
    )
    diagnostic_phase_eligible = measurement_profile == "diagnostic_phase" and all(
        isinstance(observation.get("diagnostic_cuda_phases"), dict)
        and bool(observation["diagnostic_cuda_phases"])
        for observation in all_observations
    )
    result = {
        "schema_version": "0.3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run": {
            "implementation": args.implementation,
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "formal_run": args.formal,
            "comparability": {
                "status": "formal-comparable" if comparable else "ad-hoc-not-comparable",
                "reasons": comparability_reasons,
            },
            "warmup_gate": {
                "status": "pass" if args.warmup >= 1 else "fail",
                "minimum_iterations": 1,
                "observed_iterations": args.warmup,
                "custom_lazy_h2d_applicable": args.implementation.startswith("custom_"),
                "custom_lazy_h2d_covered": (
                    args.warmup >= 1 if args.implementation.startswith("custom_") else None
                ),
            },
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
            "runner_construction_envelope_ms": runner_construction_envelope_ms,
            "runner_construction_scope": (
                "implementation-specific host construction envelope; not a "
                "comparable model-load or H2D metric"
            ),
            "artifact_scope": args.artifact_scope,
            "measurement_profile": measurement_profile,
            "metric_eligibility": {
                "canonical_latency": (measurement_profile == "canonical_latency" and comparable),
                "request_memory_observed_peak": request_memory_eligible,
                "diagnostic_cuda_phases": diagnostic_phase_eligible,
                "cold_start": False,
                "model_load": False,
            },
            "nvml_sample_ms": args.nvml_sample_ms,
            "phase_timing": args.phase_timing,
            "phase_timing_warning": (
                "CUDA-event instrumentation adds per-step overhead; use the "
                "uninstrumented run for canonical latency comparisons"
                if args.phase_timing
                else None
            ),
            "runner_configuration": runner.configuration(),
        },
        "git": git_metadata,
        "sources": _source_metadata(repository_root),
        "model_snapshot": model_snapshot,
        "readiness": readiness,
        "environment": environment,
        "aggregate": {
            "generation_ms": _summary(all_generation),
            "processor_plus_generation_ms": _summary(all_e2e),
            "request_no_file_io_ms": _summary(all_requests),
            "quality": _aggregate_quality(sample_results),
        },
        "samples": sample_results,
    }

    if args.artifact_scope == "public":
        result = _public_artifact(result)
        from faster_glm_asr.benchmarking.public_artifact import validate_public_artifact

        public_errors = validate_public_artifact(result)
        if public_errors:
            raise RuntimeError(
                "refusing to write unsafe public artifact: " + "; ".join(public_errors)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json_atomically(args.output, result, overwrite=args.overwrite)
    except FileExistsError:
        print(
            f"error: output already exists (use --overwrite intentionally): {args.output}",
            file=sys.stderr,
        )
        nvml_sampler.close()
        atexit.unregister(nvml_sampler.close)
        return 2
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"raw result: {args.output.resolve()}")
    nvml_sampler.close()
    atexit.unregister(nvml_sampler.close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
