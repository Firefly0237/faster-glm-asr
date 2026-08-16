#!/usr/bin/env python3
"""Build an auditable ASR benchmark manifest from a small CSV inventory.

The script reads metadata and references but never copies audio.  Paths written
to the JSONL are relative to the output manifest where possible, so a private
dataset directory can be moved together without publishing its contents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import wave
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("id", "audio_path", "language")
KNOWN_COLUMNS = {
    "id",
    "audio_path",
    "reference_path",
    "language",
    "collection",
    "speaker_count",
    "split",
    "split_group_id",
    "shareable",
    "authorization_id",
    "reference_quality",
    "notes",
}
TIMESTAMP_RE = re.compile(
    r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}\s*-->\s*"
    r"(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}(?:\s+.*)?$"
)
TAG_RE = re.compile(r"<[^>]*>")
PARSER_VERSION = "asr-inventory-v0.2"
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
REFERENCE_QUALITY_VALUES = {
    "unknown",
    "raw-auto",
    "human-reviewed",
    "human-corrected",
}
# The pinned processor can form up to roughly 21 x 30 s windows, but the text
# context must also leave room for prompt and generated tokens.  Treat 600 s as
# the conservative direct-model boundary; longer source recordings require an
# explicit VAD/chunk/stitch manifest rather than silent truncation.
DIRECT_MODEL_MAX_AUDIO_S = 600.0


@dataclass(frozen=True)
class AudioInfo:
    sample_rate_hz: int
    channels: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate_hz


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_info(path: Path) -> AudioInfo:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return AudioInfo(
            sample_rate_hz=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
        )
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(f"soundfile is required to inspect non-WAV input: {path}") from None
        with wave.open(str(path), "rb") as wav_file:
            return AudioInfo(
                sample_rate_hz=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                frames=wav_file.getnframes(),
            )


def parse_bool(value: str, *, row_number: int) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"row {row_number}: shareable must be true/false, got {value!r}")


def parse_optional_int(value: str, *, row_number: int) -> int | None:
    if not value.strip():
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"row {row_number}: speaker_count must be positive")
    return parsed


def reference_text(path: Path) -> str:
    """Extract plain cue text without silently normalizing ASR content."""
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".txt":
        return " ".join(raw.split())
    if path.suffix.lower() not in {".vtt", ".srt"}:
        raise ValueError(f"unsupported reference extension {path.suffix!r}; use .txt/.vtt/.srt")

    kept = []
    in_cue = False
    skip_block = False
    for original_line in raw.splitlines():
        stripped = original_line.strip()
        if not stripped:
            in_cue = False
            skip_block = False
            continue
        if stripped.startswith(("NOTE", "STYLE", "REGION")):
            skip_block = True
            in_cue = False
            continue
        if skip_block or stripped == "WEBVTT":
            continue
        if TIMESTAMP_RE.match(stripped):
            in_cue = True
            continue
        if not in_cue:
            # SRT sequence numbers and VTT cue identifiers both occur before
            # the timestamp and are metadata, not transcript text.
            continue
        cleaned = html.unescape(TAG_RE.sub("", stripped)).strip()
        # Preserve repeated cue text: identical adjacent utterances can be real
        # speech.  Rolling-caption de-duplication needs timestamps and a named,
        # separately validated policy; silently dropping text corrupts WER.
        if cleaned:
            kept.append(cleaned)
    return " ".join(kept)


def resolve_input_path(value: str, inventory_dir: Path, *, field: str, row: int) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = inventory_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"row {row}: {field} does not exist: {path}")
    return path


def manifest_path(path: Path, output_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path, start=output_dir)).as_posix()
    except ValueError:
        # Windows paths on different drives cannot be represented relatively.
        return path.as_posix()


def rows_from_csv(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("inventory has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"inventory missing required columns: {missing}")
        unknown = sorted(set(reader.fieldnames) - KNOWN_COLUMNS)
        if unknown:
            raise ValueError(f"inventory contains unknown columns: {unknown}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"row {row_number}: too many CSV cells for the declared header")
            if not any((value or "").strip() for value in row.values()):
                continue
            yield row_number, {key: value or "" for key, value in row.items()}


def _write_jsonl_atomically(
    output: Path, records: Iterable[dict[str, object]], *, overwrite: bool
) -> None:
    """Publish JSONL without a check-then-replace race at ``output``."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            # link() creates the destination only if it is absent, unlike
            # os.replace() after an exists() check.  It is atomic on NTFS.
            os.link(temporary, output)
            temporary.unlink()
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def build(inventory: Path, output: Path, *, overwrite: bool = False) -> int:
    inventory = inventory.resolve()
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite intentionally): {output}")
    inventory_dir = inventory.parent
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    audio_hash_splits: dict[str, str] = {}
    records = []
    for row_number, row in rows_from_csv(inventory):
        sample_id = row["id"].strip()
        language = row["language"].strip()
        if not sample_id or not language:
            raise ValueError(f"row {row_number}: id and language must be non-empty")
        if not OPAQUE_ID_RE.fullmatch(sample_id):
            raise ValueError(f"row {row_number}: id must be an opaque filesystem-safe identifier")
        if not LANGUAGE_RE.fullmatch(language):
            raise ValueError(
                f"row {row_number}: language must be a BCP-47-like tag such as en, en-GB, or zh-CN"
            )
        if sample_id in sample_ids:
            raise ValueError(f"row {row_number}: duplicate id {sample_id!r}")
        sample_ids.add(sample_id)

        audio_path = resolve_input_path(
            row["audio_path"], inventory_dir, field="audio_path", row=row_number
        )
        info = audio_info(audio_path)
        split = row.get("split", "").strip() or "domain-test"
        split_group_id = row.get("split_group_id", "").strip() or sample_id
        authorization_id = row.get("authorization_id", "").strip() or None
        reference_quality = row.get("reference_quality", "").strip() or None
        shareable = parse_bool(row.get("shareable", ""), row_number=row_number)
        if not OPAQUE_ID_RE.fullmatch(split_group_id):
            raise ValueError(f"row {row_number}: split_group_id must be an opaque identifier")
        if authorization_id is not None and not OPAQUE_ID_RE.fullmatch(authorization_id):
            raise ValueError(f"row {row_number}: authorization_id must be an opaque identifier")
        if reference_quality not in REFERENCE_QUALITY_VALUES | {None}:
            raise ValueError(
                f"row {row_number}: reference_quality must be one of "
                f"{sorted(REFERENCE_QUALITY_VALUES)}"
            )
        if shareable and authorization_id is None:
            raise ValueError(f"row {row_number}: shareable=true requires authorization_id")
        audio_hash = sha256(audio_path)
        for label, value, table in (
            ("split_group_id", split_group_id, group_splits),
            ("audio_sha256", audio_hash, audio_hash_splits),
        ):
            previous = table.get(value)
            if previous is not None and previous != split:
                raise ValueError(
                    f"row {row_number}: {label} {value!r} crosses splits {previous!r}/{split!r}"
                )
            table[value] = split
        record = {
            "sample_id": sample_id,
            "audio_path": manifest_path(audio_path, output.parent),
            "audio_sha256": audio_hash,
            "language": language,
            "duration_s": round(info.duration_s, 9),
            "sample_rate_hz": info.sample_rate_hz,
            "channels": info.channels,
            "duration_bucket": (
                "direct-model"
                if info.duration_s <= DIRECT_MODEL_MAX_AUDIO_S
                else "requires-segmentation"
            ),
            "direct_model_eligible": info.duration_s <= DIRECT_MODEL_MAX_AUDIO_S,
            "manifest_parser_version": PARSER_VERSION,
            "collection": row.get("collection", "").strip() or None,
            "speaker_count": parse_optional_int(
                row.get("speaker_count", ""), row_number=row_number
            ),
            "split": split,
            "split_group_id": split_group_id,
            "shareable": shareable,
            "authorization_id": authorization_id,
            "reference_quality": reference_quality,
            "notes": row.get("notes", "").strip() or None,
        }

        reference_value = row.get("reference_path", "").strip()
        if reference_quality is not None and not reference_value:
            raise ValueError(f"row {row_number}: reference_quality requires reference_path")
        if reference_value:
            reference_path = resolve_input_path(
                reference_value,
                inventory_dir,
                field="reference_path",
                row=row_number,
            )
            extracted = reference_text(reference_path)
            if not extracted:
                raise ValueError(f"row {row_number}: extracted reference text is empty")
            record.update(
                {
                    "reference_path": manifest_path(reference_path, output.parent),
                    "reference_sha256": sha256(reference_path),
                    "reference_text": extracted,
                    "reference_text_sha256": hashlib.sha256(extracted.encode("utf-8")).hexdigest(),
                }
            )

        records.append({key: value for key, value in record.items() if value is not None})

    if not records:
        raise ValueError("inventory contains no samples")

    _write_jsonl_atomically(output, records, overwrite=overwrite)
    print(f"wrote {len(records)} samples to {output}")
    print(f"manifest_sha256={sha256(output)}")
    overlong = sum(not item["direct_model_eligible"] for item in records)
    if overlong:
        print(
            f"warning: {overlong} source recording(s) exceed "
            f"{DIRECT_MODEL_MAX_AUDIO_S:g}s and require a segment-level "
            "VAD/chunk/stitch manifest before model benchmarking",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.inventory.is_file():
        parser.error(f"inventory not found: {args.inventory}")
    try:
        return build(args.inventory, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
