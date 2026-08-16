#!/usr/bin/env python3
"""Build a deterministic, hash-locked LibriSpeech test-clean subset.

The builder does not download or copy audio.  It expects an extracted
``LibriSpeech/test-clean`` tree that the user obtained from an authorized
source, selects a balanced duration subset, and writes a private benchmark
manifest plus a provenance lock into a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "librispeech-subset-lock-v0.1"
MANIFEST_PARSER_VERSION = "asr-inventory-v0.2"
SELECTION_VERSION = "librispeech-duration-stratified-v0.1"
DEFAULT_SEED = "faster-glm-asr-public-quality-v1"
UPSTREAM_URL = "https://www.openslr.org/12"
UPSTREAM_LICENSE = "CC-BY-4.0"
SAMPLE_ID_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")
BUCKETS: tuple[tuple[str, float, float, bool], ...] = (
    ("short", 0.0, 10.0, False),
    ("medium", 10.0, 20.0, False),
    ("near-30s", 20.0, 30.0, True),
)


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    speaker_id: str
    chapter_id: str
    audio_path: Path
    audio_sha256: str
    duration_s: float
    reference_text: str
    transcript_file: Path
    transcript_file_sha256: str
    duration_bucket: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
    except ImportError:
        if path.suffix.casefold() != ".wav":
            raise RuntimeError(
                "soundfile is required to inspect LibriSpeech FLAC files; "
                "install the frozen benchmark environment"
            ) from None
        try:
            with wave.open(str(path), "rb") as stream:
                frames = stream.getnframes()
                sample_rate = stream.getframerate()
                channels = stream.getnchannels()
        except (OSError, EOFError, wave.Error) as exc:
            raise ValueError(f"invalid WAV file: {path}") from exc
    if frames <= 0 or sample_rate <= 0:
        raise ValueError(f"audio has invalid frames/sample rate: {path}")
    if sample_rate != 16000 or channels != 1:
        raise ValueError(
            f"LibriSpeech test-clean audio must be 16 kHz mono: {path} "
            f"has sample_rate={sample_rate}, channels={channels}"
        )
    duration = frames / sample_rate
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"audio has invalid duration: {path}")
    return duration


def _bucket_for(duration_s: float) -> str | None:
    for name, lower, upper, include_upper in BUCKETS:
        if duration_s >= lower and (duration_s <= upper if include_upper else duration_s < upper):
            return name
    return None


def _parse_transcript(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[1].strip():
            raise ValueError(f"{path}:{line_number}: expected '<utterance-id> <transcript>'")
        sample_id, text = fields[0], " ".join(fields[1].split())
        if SAMPLE_ID_RE.fullmatch(sample_id) is None:
            raise ValueError(f"{path}:{line_number}: invalid utterance ID")
        if sample_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate utterance ID")
        seen.add(sample_id)
        records.append((sample_id, text))
    if not records:
        raise ValueError(f"transcript file is empty: {path}")
    return records


def _find_audio(transcript_file: Path, sample_id: str) -> Path:
    matches = [
        path
        for suffix in (".flac", ".wav")
        if (path := transcript_file.parent / f"{sample_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{sample_id}: expected exactly one matching .flac/.wav file, found {len(matches)}"
        )
    return matches[0].resolve()


def discover(test_clean_root: Path) -> list[Candidate]:
    root = test_clean_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"test-clean directory does not exist: {root}")
    transcript_files = sorted(root.rglob("*.trans.txt"))
    if not transcript_files:
        raise ValueError("test-clean contains no *.trans.txt files")

    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    for transcript_file in transcript_files:
        transcript_hash = _sha256_file(transcript_file)
        for sample_id, reference_text in _parse_transcript(transcript_file):
            if sample_id in seen_ids:
                raise ValueError(f"duplicate utterance ID across transcripts: {sample_id}")
            seen_ids.add(sample_id)
            match = SAMPLE_ID_RE.fullmatch(sample_id)
            assert match is not None
            expected_transcript_name = f"{match.group(1)}-{match.group(2)}.trans.txt"
            if (
                transcript_file.name != expected_transcript_name
                or transcript_file.parent.name != match.group(2)
                or transcript_file.parent.parent.name != match.group(1)
            ):
                raise ValueError(
                    f"{sample_id}: transcript path does not match the canonical "
                    "LibriSpeech speaker/chapter layout"
                )
            audio_path = _find_audio(transcript_file, sample_id)
            duration_s = _audio_duration(audio_path)
            bucket = _bucket_for(duration_s)
            if bucket is None:
                continue
            candidates.append(
                Candidate(
                    sample_id=sample_id,
                    speaker_id=match.group(1),
                    chapter_id=match.group(2),
                    audio_path=audio_path,
                    audio_sha256=_sha256_file(audio_path),
                    duration_s=duration_s,
                    reference_text=reference_text,
                    transcript_file=transcript_file.resolve(),
                    transcript_file_sha256=transcript_hash,
                    duration_bucket=bucket,
                )
            )
    return candidates


def select(candidates: Sequence[Candidate], *, per_bucket: int, seed: str) -> list[Candidate]:
    if isinstance(per_bucket, bool) or per_bucket <= 0:
        raise ValueError("per_bucket must be a positive integer")
    if not seed:
        raise ValueError("seed must be non-empty")
    selected: list[Candidate] = []
    for bucket, _, _, _ in BUCKETS:
        eligible = [item for item in candidates if item.duration_bucket == bucket]
        if len(eligible) < per_bucket:
            raise ValueError(
                f"bucket {bucket!r} has {len(eligible)} candidates, requires {per_bucket}"
            )
        ranked = sorted(
            eligible,
            key=lambda item: (
                hashlib.sha256(
                    f"{SELECTION_VERSION}\0{seed}\0{item.sample_id}\0{item.audio_sha256}".encode()
                ).hexdigest(),
                item.sample_id,
            ),
        )
        selected.extend(ranked[:per_bucket])
    return sorted(
        selected,
        key=lambda item: (
            next(index for index, value in enumerate(BUCKETS) if value[0] == item.duration_bucket),
            item.sample_id,
        ),
    )


def _relative(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, start=base)).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest_records(
    selected: Sequence[Candidate], output_dir: Path, license_hash: str
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"librispeech-{item.sample_id}",
            "audio_path": _relative(item.audio_path, output_dir),
            "audio_sha256": item.audio_sha256,
            "duration_s": round(item.duration_s, 9),
            "language": "en",
            "reference_text": item.reference_text,
            "license_scope": "LibriSpeech test-clean; CC BY 4.0; attribution/publication review required",
            "license_file_sha256": license_hash,
            "upstream_url": UPSTREAM_URL,
            "upstream_license_declared": UPSTREAM_LICENSE,
            "shareable": False,
            "duration_bucket": item.duration_bucket,
            "direct_model_eligible": True,
            "manifest_parser_version": MANIFEST_PARSER_VERSION,
            "dataset": "LibriSpeech",
            "dataset_subset": "test-clean",
            "speaker_id": item.speaker_id,
            "chapter_id": item.chapter_id,
            "upstream_utterance_id": item.sample_id,
            "transcript_file_sha256": item.transcript_file_sha256,
            "selection_version": SELECTION_VERSION,
        }
        for item in selected
    ]


def _write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def build(
    test_clean_root: Path,
    output_dir: Path,
    *,
    license_file: Path,
    per_bucket: int,
    seed: str,
) -> Mapping[str, object]:
    test_clean_root = test_clean_root.resolve()
    output_dir = output_dir.resolve()
    license_file = license_file.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new versioned path: {output_dir}"
        )
    if not license_file.is_file() or license_file.stat().st_size <= 0:
        raise FileNotFoundError(f"license file is missing or empty: {license_file}")
    candidates = discover(test_clean_root)
    selected = select(candidates, per_bucket=per_bucket, seed=seed)
    license_hash = _sha256_file(license_file)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        manifest_records = _manifest_records(selected, output_dir, license_hash)
        manifest_text = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in manifest_records
        )
        manifest_path = stage / "manifest.jsonl"
        _write_text(manifest_path, manifest_text)

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
        bucket_counts = {
            bucket: sum(item.duration_bucket == bucket for item in candidates)
            for bucket, _, _, _ in BUCKETS
        }
        lock: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "selection_version": SELECTION_VERSION,
            "seed": seed,
            "per_bucket": per_bucket,
            "duration_buckets": [
                {
                    "name": name,
                    "lower_s_inclusive": lower,
                    "upper_s": upper,
                    "upper_inclusive": include_upper,
                }
                for name, lower, upper, include_upper in BUCKETS
            ],
            "candidate_counts": bucket_counts,
            "candidate_inventory_sha256": _canonical_sha256(candidate_inventory),
            "selected_upstream_ids": [item.sample_id for item in selected],
            "selected_count": len(selected),
            "license_file_name": license_file.name,
            "license_file_sha256": license_hash,
            "upstream_url": UPSTREAM_URL,
            "upstream_license_declared": UPSTREAM_LICENSE,
            "manifest_file": "manifest.jsonl",
            "manifest_sha256": _sha256_file(manifest_path),
            "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
            "audio_copied": False,
            "publication_review_required": True,
        }
        _write_text(
            stage / "selection-lock.json",
            json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        # Reserve the final directory with mkdir's create-if-absent semantics.
        # Publishing a directory with os.rename can replace an empty target on
        # POSIX, which is not an acceptable no-overwrite evidence boundary.
        output_dir.mkdir()
        try:
            os.replace(manifest_path, output_dir / "manifest.jsonl")
            # The lock is the commit marker and is deliberately published last.
            os.replace(stage / "selection-lock.json", output_dir / "selection-lock.json")
        except BaseException:
            shutil.rmtree(output_dir)
            raise
        return lock
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-clean-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--license-file",
        type=Path,
        help="local LibriSpeech LICENSE.TXT (defaults to test-clean/../LICENSE.TXT)",
    )
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    license_file = args.license_file or args.test_clean_root.parent / "LICENSE.TXT"
    lock = build(
        args.test_clean_root,
        args.output_dir,
        license_file=license_file,
        per_bucket=args.per_bucket,
        seed=args.seed,
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
