#!/usr/bin/env python3
"""Materialize deterministic long-audio segments for the ASR benchmark.

The source inventory remains immutable.  This builder decodes each source,
canonicalizes it to mono 16 kHz PCM16, plans fixed half-open sample intervals,
and writes a private bundle containing canonical WAVs, segment WAVs, a strict
segment manifest, a plan lock, and timestamp-preserving reference metadata.

The default 28 s / 2 s overlap is an unoptimized baseline, not a quality claim.
External VAD can later provide versioned cut candidates; it must not silently
delete audio or change the source-level test split.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
import wave
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_PARSER_VERSION = "asr-inventory-v0.2"
SEGMENT_PARSER_VERSION = "asr-segment-v0.1"
SEGMENT_MANIFEST_SCHEMA = "asr-segment-manifest-v0.1"
SEGMENTER_VERSION = "asr-fixed-window-segmenter-v0.1"
PLAN_SCHEMA = "asr-segmentation-plan-v0.1"
BUNDLE_LOCK_SCHEMA = "asr-segment-bundle-lock-v0.1"
TARGET_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = "28"
DEFAULT_OVERLAP_SECONDS = "2"
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TAG_RE = re.compile(r"<[^>]*>")
TIMESTAMP_TOKEN_RE = re.compile(
    r"^(?:(?P<hours>\d{1,3}):)?(?P<minutes>\d{1,2}):"
    r"(?P<seconds>\d{2})[.,](?P<milliseconds>\d{3})$"
)


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


def _seconds_to_samples(value: str, *, name: str) -> int:
    try:
        seconds = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite decimal number") from exc
    if not seconds.is_finite() or seconds < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    samples = seconds * TARGET_SAMPLE_RATE
    if samples != samples.to_integral_value():
        raise ValueError(
            f"{name}={value!r} does not map to an integer number of {TARGET_SAMPLE_RATE} Hz samples"
        )
    return int(samples)


def plan_fixed_segments(
    total_samples: int, chunk_samples: int, overlap_samples: int
) -> list[tuple[int, int]]:
    """Return complete, gap-free fixed windows as integer [start,end) pairs."""
    if not isinstance(total_samples, int) or isinstance(total_samples, bool):
        raise ValueError("total_samples must be an integer")
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must satisfy 0 <= overlap < chunk")
    result: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + chunk_samples, total_samples)
        result.append((start, end))
        if end == total_samples:
            break
        next_start = end - overlap_samples
        if next_start <= start:
            raise RuntimeError("segment planner failed to make forward progress")
        start = next_start
    return result


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_audio_float(path: Path) -> tuple[np.ndarray, int, str]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return (
            np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32),
            int(sample_rate),
            "soundfile",
        )
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(f"soundfile is required to decode non-WAV source {path}") from None

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"unsupported WAV sample width {sample_width}; use PCM16/PCM32 or install soundfile"
        )
    return (
        np.ascontiguousarray(audio.reshape(-1, channels).mean(axis=1), dtype=np.float32),
        int(sample_rate),
        "python-wave",
    )


def _canonical_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    audio, sample_rate, decoder = _read_audio_float(path)
    resampler = "identity"
    if sample_rate != TARGET_SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("scipy is required to canonicalize non-16 kHz audio") from exc
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            up=TARGET_SAMPLE_RATE // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32, copy=False)
        resampler = "scipy.signal.resample_poly"
    clipped = np.clip(audio, -1.0, 32767.0 / 32768.0)
    pcm = np.rint(clipped * 32768.0).astype("<i2", copy=False)
    return pcm, {
        "decoder": decoder,
        "source_sample_rate_hz": sample_rate,
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "channel_policy": "arithmetic-mean-to-mono",
        "resampler": resampler,
        "pcm_quantization": "clip-then-numpy-rint-to-little-endian-pcm16",
        "numpy_version": np.__version__,
        "soundfile_version": _package_version("soundfile"),
        "scipy_version": _package_version("scipy"),
    }


def _write_pcm16_wav(path: Path, pcm: np.ndarray) -> None:
    if pcm.ndim != 1 or pcm.dtype != np.dtype("<i2"):
        raise ValueError("canonical PCM must be one-dimensional little-endian int16")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(TARGET_SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())


def _timestamp_to_decimal(token: str) -> Decimal:
    match = TIMESTAMP_TOKEN_RE.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"invalid subtitle timestamp {token!r}")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid subtitle timestamp {token!r}")
    return Decimal(hours * 3600 + minutes * 60 + seconds) + Decimal(milliseconds) / Decimal(1000)


def _time_to_sample(value: Decimal, rounding: str) -> int:
    mode = ROUND_FLOOR if rounding == "floor" else ROUND_CEILING
    return int((value * TARGET_SAMPLE_RATE).to_integral_value(rounding=mode))


def parse_timed_reference(path: Path, total_samples: int) -> list[dict[str, Any]]:
    """Parse VTT/SRT cues without rolling-caption de-duplication."""
    if path.suffix.lower() not in {".vtt", ".srt"}:
        raise ValueError("timed reference must be .vtt or .srt")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    cues: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped == "WEBVTT":
            index += 1
            continue
        if stripped.startswith(("NOTE", "STYLE", "REGION")):
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            continue
        timestamp_line = stripped
        if "-->" not in timestamp_line:
            index += 1
            if index >= len(lines):
                break
            timestamp_line = lines[index].strip()
        if "-->" not in timestamp_line:
            continue
        left, right = timestamp_line.split("-->", 1)
        start_decimal = _timestamp_to_decimal(left.strip().split()[0])
        end_decimal = _timestamp_to_decimal(right.strip().split()[0])
        start_sample = _time_to_sample(start_decimal, "floor")
        end_sample = _time_to_sample(end_decimal, "ceil")
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError(f"invalid cue interval at subtitle line {index + 1}")
        if start_sample >= total_samples or end_sample > total_samples:
            raise ValueError(f"subtitle cue at line {index + 1} exceeds canonical audio length")
        index += 1
        text_lines = []
        while index < len(lines) and lines[index].strip():
            cleaned = html.unescape(TAG_RE.sub("", lines[index])).strip()
            if cleaned:
                text_lines.append(cleaned)
            index += 1
        text = " ".join(text_lines)
        if text:
            cues.append(
                {
                    "cue_id": f"cue-{len(cues):06d}",
                    "order": len(cues),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "text": text,
                }
            )
    if not cues:
        raise ValueError(f"no timed subtitle cues extracted from {path}")
    return cues


def _ownership_intervals(
    segments: Sequence[tuple[int, int]], total_samples: int
) -> list[tuple[int, int]]:
    seams = [0]
    for left, right in pairwise(segments):
        overlap_start = right[0]
        overlap_end = left[1]
        if overlap_start > overlap_end:
            raise RuntimeError("planned segments contain a gap")
        seams.append((overlap_start + overlap_end) // 2)
    seams.append(total_samples)
    return list(pairwise(seams))


def _resolve_source_path(value: str, source_manifest: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source_manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_source_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"source manifest line {line_number} must be an object")
        required = {
            "sample_id",
            "audio_path",
            "audio_sha256",
            "duration_s",
            "language",
            "split",
            "shareable",
            "manifest_parser_version",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"source manifest line {line_number} missing {missing}")
        source_id = str(item["sample_id"])
        if not OPAQUE_ID_RE.fullmatch(source_id):
            raise ValueError(
                f"source manifest line {line_number}: sample_id must be an opaque "
                "filesystem-safe identifier"
            )
        if source_id in seen_ids:
            raise ValueError(f"duplicate source sample_id {source_id!r}")
        seen_ids.add(source_id)
        if item["manifest_parser_version"] != SOURCE_PARSER_VERSION:
            raise ValueError(
                f"source manifest line {line_number}: expected {SOURCE_PARSER_VERSION}"
            )
        if not isinstance(item["shareable"], bool):
            raise ValueError(f"source manifest line {line_number}: shareable must be boolean")
        records.append(item)
    if not records:
        raise ValueError("source manifest contains no records")
    return records


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def build(
    source_manifest: Path,
    output_dir: Path,
    *,
    chunk_seconds: str = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: str = DEFAULT_OVERLAP_SECONDS,
) -> int:
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if output_dir.exists():
        raise FileExistsError(
            f"output bundle already exists; choose a new versioned path: {output_dir}"
        )
    chunk_samples = _seconds_to_samples(chunk_seconds, name="chunk_seconds")
    overlap_samples = _seconds_to_samples(overlap_seconds, name="overlap_seconds")
    if chunk_samples <= 0 or overlap_samples >= chunk_samples:
        raise ValueError("require chunk_seconds > overlap_seconds >= 0")
    if chunk_samples > 600 * TARGET_SAMPLE_RATE:
        raise ValueError("chunk_seconds exceeds the 600 s direct-model guardrail")

    source_records = _load_source_records(source_manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent)
    )
    manifest_records: list[dict[str, Any]] = []
    reference_records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    group_splits: dict[str, str] = {}
    hash_splits: dict[str, str] = {}
    canonical_hash_splits: dict[str, str] = {}
    try:
        for source in source_records:
            source_id = str(source["sample_id"])
            split = str(source["split"])
            split_group_id = str(source.get("split_group_id") or source_id)
            for key, value, table in (
                ("split_group_id", split_group_id, group_splits),
                ("audio_sha256", str(source["audio_sha256"]).lower(), hash_splits),
            ):
                previous = table.get(value)
                if previous is not None and previous != split:
                    raise ValueError(f"{key} {value!r} crosses splits {previous!r}/{split!r}")
                table[value] = split

            source_audio = _resolve_source_path(str(source["audio_path"]), source_manifest)
            source_hash = _sha256_file(source_audio)
            if source_hash.lower() != str(source["audio_sha256"]).lower():
                raise ValueError(f"source audio SHA256 mismatch for {source_id}")
            pcm, canonicalizer = _canonical_pcm16(source_audio)
            total_samples = int(pcm.shape[0])
            if total_samples <= 0:
                raise ValueError(f"source {source_id} decodes to empty audio")
            declared_duration = float(source["duration_s"])
            if not math.isfinite(declared_duration) or declared_duration <= 0:
                raise ValueError(f"source {source_id} duration_s must be finite and positive")
            canonical_duration = total_samples / TARGET_SAMPLE_RATE
            if abs(declared_duration - canonical_duration) > 0.05:
                raise ValueError(
                    f"source {source_id} duration changed by more than 50 ms during "
                    "canonicalization"
                )
            canonical_path = stage / "canonical" / f"{source_id}.wav"
            _write_pcm16_wav(canonical_path, pcm)
            canonical_file_hash = _sha256_file(canonical_path)
            canonical_pcm_hash = _sha256_bytes(pcm.tobytes())
            for label, value in (
                ("canonical_source_audio_sha256", canonical_file_hash),
                ("canonical_pcm_sha256", canonical_pcm_hash),
            ):
                previous = canonical_hash_splits.get(value.lower())
                if previous is not None and previous != split:
                    raise ValueError(f"{label} {value!r} crosses splits {previous!r}/{split!r}")
                canonical_hash_splits[value.lower()] = split
            segments = plan_fixed_segments(total_samples, chunk_samples, overlap_samples)
            ownership = _ownership_intervals(segments, total_samples)

            cues: list[dict[str, Any]] = []
            reference_status = "unavailable"
            reference_path_value = source.get("reference_path")
            if reference_path_value:
                reference_path = _resolve_source_path(str(reference_path_value), source_manifest)
                declared_reference_hash = source.get("reference_sha256")
                if not isinstance(declared_reference_hash, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", declared_reference_hash
                ):
                    raise ValueError(f"source {source_id} reference_path requires reference_sha256")
                actual_reference_hash = _sha256_file(reference_path)
                if actual_reference_hash.lower() != declared_reference_hash.lower():
                    raise ValueError(f"reference SHA256 mismatch for {source_id}")
                if reference_path.suffix.lower() in {".vtt", ".srt"}:
                    cues = parse_timed_reference(reference_path, total_samples)
                    reference_status = "timed-cues-available"
                    full_reference_text = " ".join(cue["text"] for cue in cues)
                elif reference_path.suffix.lower() == ".txt":
                    reference_status = "file-level-only"
                    full_reference_text = " ".join(
                        reference_path.read_text(encoding="utf-8-sig").split()
                    )
                else:
                    raise ValueError(
                        f"unsupported reference extension for {source_id}: {reference_path.suffix}"
                    )
                if not full_reference_text:
                    raise ValueError(f"reference text is empty for {source_id}")
                declared_reference_text = source.get("reference_text")
                if (
                    declared_reference_text is not None
                    and declared_reference_text != full_reference_text
                ):
                    raise ValueError(f"source manifest reference_text mismatch for {source_id}")
                actual_reference_text_hash = _sha256_bytes(full_reference_text.encode("utf-8"))
                declared_reference_text_hash = source.get("reference_text_sha256")
                if declared_reference_text_hash is not None and (
                    not isinstance(declared_reference_text_hash, str)
                    or actual_reference_text_hash.lower() != declared_reference_text_hash.lower()
                ):
                    raise ValueError(
                        f"source manifest reference_text_sha256 mismatch for {source_id}"
                    )
                reference_records.append(
                    {
                        "source_recording_id": source_id,
                        "source_reference_sha256": actual_reference_hash,
                        "reference_status": reference_status,
                        "full_reference_text": full_reference_text,
                        "full_reference_text_sha256": actual_reference_text_hash,
                        "cues": cues,
                    }
                )

            segment_count = len(segments)
            for segment_index, ((start, end), (owner_start, owner_end)) in enumerate(
                zip(segments, ownership, strict=True)
            ):
                segment_id = f"{source_id}__seg-{segment_index:05d}"
                segment_pcm = np.ascontiguousarray(pcm[start:end], dtype="<i2")
                segment_path = stage / "segments" / f"{segment_id}.wav"
                _write_pcm16_wav(segment_path, segment_pcm)
                left_overlap = (
                    max(0, segments[segment_index - 1][1] - start) if segment_index > 0 else 0
                )
                right_overlap = (
                    max(0, end - segments[segment_index + 1][0])
                    if segment_index + 1 < segment_count
                    else 0
                )
                intersecting = [
                    cue["cue_id"]
                    for cue in cues
                    if cue["start_sample"] < end and cue["end_sample"] > start
                ]
                owned = [
                    cue["cue_id"]
                    for cue in cues
                    if owner_start <= (cue["start_sample"] + cue["end_sample"]) // 2 < owner_end
                ]
                manifest_records.append(
                    {
                        "manifest_schema_version": SEGMENT_MANIFEST_SCHEMA,
                        "manifest_parser_version": SEGMENT_PARSER_VERSION,
                        "segmenter_version": SEGMENTER_VERSION,
                        "sample_id": segment_id,
                        "source_recording_id": source_id,
                        "split_group_id": split_group_id,
                        "segment_index": segment_index,
                        "segment_count": segment_count,
                        "audio_path": Path("segments", segment_path.name).as_posix(),
                        "audio_sha256": _sha256_file(segment_path),
                        "segment_pcm_sha256": _sha256_bytes(segment_pcm.tobytes()),
                        "source_audio_sha256": source_hash,
                        "canonical_source_audio_sha256": canonical_file_hash,
                        "canonical_pcm_sha256": canonical_pcm_hash,
                        "sample_rate_hz": TARGET_SAMPLE_RATE,
                        "channels": 1,
                        "start_sample": start,
                        "end_sample": end,
                        "start_s": start / TARGET_SAMPLE_RATE,
                        "end_s": end / TARGET_SAMPLE_RATE,
                        "duration_s": (end - start) / TARGET_SAMPLE_RATE,
                        "interval_convention": "[start,end)",
                        "left_overlap_samples": left_overlap,
                        "right_overlap_samples": right_overlap,
                        "ownership_start_sample": owner_start,
                        "ownership_end_sample": owner_end,
                        "short_tail": end - start < chunk_samples,
                        "planner_strategy": "fixed-window-v1",
                        "duration_bucket": "materialized-segment",
                        "direct_model_eligible": True,
                        "language": str(source["language"]),
                        "split": split,
                        "shareable": False,
                        "source_shareable": bool(source["shareable"]),
                        "reference_status": reference_status,
                        "intersecting_cue_ids": intersecting,
                        "owned_cue_ids": owned,
                        "reference_assignment_policy": "overlap-midpoint-owner-v1",
                        "primary_quality_scope": "source-recording-after-stitch",
                    }
                )
            source_summaries.append(
                {
                    "source_recording_id": source_id,
                    "split_group_id": split_group_id,
                    "split": split,
                    "source_audio_sha256": source_hash,
                    "canonical_source_audio_sha256": canonical_file_hash,
                    "canonical_pcm_sha256": canonical_pcm_hash,
                    "canonical_samples": total_samples,
                    "canonicalizer": canonicalizer,
                    "segment_count": segment_count,
                    "reference_status": reference_status,
                }
            )

        plan = {
            "schema_version": PLAN_SCHEMA,
            "source_manifest_sha256": _sha256_file(source_manifest),
            "strategy": "fixed-window-v1",
            "target_sample_rate_hz": TARGET_SAMPLE_RATE,
            "target_channels": 1,
            "chunk_samples": chunk_samples,
            "overlap_samples": overlap_samples,
            "tail_policy": "keep-short",
            "interval_convention": "[start,end)",
            "reference_policy": "source-level-after-stitch-v1",
            "derived_shareable_default": False,
            "source_code_sha256": _sha256_file(Path(__file__).resolve()),
            "sources": source_summaries,
        }
        plan_bytes = _canonical_json_bytes(plan)
        plan_path = stage / "segmentation-plan.json"
        plan_path.write_bytes(plan_bytes)
        plan_hash = _sha256_bytes(plan_bytes)
        for record in manifest_records:
            record["segmentation_plan_sha256"] = plan_hash
        manifest_path = stage / "segments.jsonl"
        _write_jsonl(manifest_path, manifest_records)
        references_path = stage / "source-references.private.jsonl"
        _write_jsonl(references_path, reference_records)
        bundle_lock = {
            "schema_version": BUNDLE_LOCK_SCHEMA,
            "source_manifest_sha256": _sha256_file(source_manifest),
            "segmentation_plan_sha256": plan_hash,
            "segment_manifest_sha256": _sha256_file(manifest_path),
            "private_reference_sidecar_sha256": _sha256_file(references_path),
        }
        (stage / "bundle-lock.json").write_bytes(_canonical_json_bytes(bundle_lock))
        os.replace(stage, output_dir)
        stage = None
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)

    print(f"wrote {len(manifest_records)} segments to {output_dir}")
    print(f"manifest_sha256={_sha256_file(output_dir / 'segments.jsonl')}")
    print(f"segmentation_plan_sha256={plan_hash}")
    print(f"bundle_lock_sha256={_sha256_file(output_dir / 'bundle-lock.json')}")
    print("quality_scope=source-recording-after-stitch")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-seconds", default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", default=DEFAULT_OVERLAP_SECONDS)
    args = parser.parse_args()
    try:
        return build(
            args.source_manifest,
            args.output_dir,
            chunk_seconds=str(args.chunk_seconds),
            overlap_seconds=str(args.overlap_seconds),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
