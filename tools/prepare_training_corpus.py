#!/usr/bin/env python3
"""Download, verify, and derive a small reproducible byte-level corpus.

The repository configuration intentionally distinguishes an upstream publisher's
advertised digest from a digest verified on this machine.  A completed manifest
is written only after every source file has been read locally and hash checked.

The derivation is streaming: complete documents are framed with a fixed delimiter
and written until each byte budget is exhausted.  Exact duplicate documents are
skipped within and across splits.  Any final gap too small for another complete
document is filled with LF bytes and reported explicitly in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CONFIG_SCHEMA = "training-corpus-source-v0.1"
MANIFEST_SCHEMA = "training-corpus-manifest-v0.1"
DERIVATION_VERSION = "complete-documents-byte-budget-v0.1"
DEFAULT_CHUNK_BYTES = 1024 * 1024
ALLOWED_SOURCE_ROLES = {"train", "validation", "license_evidence"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/training/tinystories-source.json"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts/private/training-data"


def default_source_dir() -> Path:
    return DEFAULT_ARTIFACT_ROOT / "sources/tinystories-f54c09fd"


def default_output_dir() -> Path:
    return DEFAULT_ARTIFACT_ROOT / "derived/tinystories-64mib-4mib-v1"


def _loader_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class SourceSpec:
    role: str
    local_filename: str
    url: str
    publisher_advertised_size_bytes: int | None
    publisher_advertised_sha256: str | None
    local_verification_status: str


@dataclass(frozen=True)
class DerivedSplit:
    split: str
    source_role: str
    filename: str
    target_bytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(DEFAULT_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a JSON object")
    return value


def _require_keys(
    value: Mapping[str, object], *, allowed: set[str], required: set[str], where: str
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{where}: unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{where}: missing keys: {sorted(missing)}")


def _positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _optional_sha256(value: object, where: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be null or a lowercase SHA-256 hex digest")
    return value


def _parse_config(
    raw: Mapping[str, object],
) -> tuple[list[SourceSpec], list[DerivedSplit], bytes, int, float]:
    _require_keys(
        raw,
        allowed={
            "schema_version",
            "dataset",
            "license",
            "source_files",
            "document_format",
            "derivation",
        },
        required={
            "schema_version",
            "dataset",
            "license",
            "source_files",
            "document_format",
            "derivation",
        },
        where="configuration",
    )
    if raw["schema_version"] != CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported schema_version {raw['schema_version']!r}; expected {CONFIG_SCHEMA!r}"
        )
    for key in ("dataset", "license"):
        if not isinstance(raw[key], dict):
            raise ValueError(f"{key} must be a JSON object")
    dataset = raw["dataset"]
    assert isinstance(dataset, dict)
    _require_keys(
        dataset,
        allowed={
            "name",
            "variant",
            "repository",
            "repository_url",
            "revision",
            "paper_url",
            "content_note",
        },
        required={"name", "revision"},
        where="dataset",
    )
    if not isinstance(dataset["name"], str) or not dataset["name"].strip():
        raise ValueError("dataset.name must be a non-empty string")
    revision = dataset["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("dataset.revision must be a full lowercase commit hash")
    for key in ("repository_url", "paper_url"):
        value = dataset.get(key)
        if value is not None and (
            not isinstance(value, str) or urllib.parse.urlparse(value).scheme != "https"
        ):
            raise ValueError(f"dataset.{key} must use HTTPS when present")

    license_record = raw["license"]
    assert isinstance(license_record, dict)
    _require_keys(
        license_record,
        allowed={
            "publisher_declared_identifier",
            "declaration_url",
            "terms_url",
            "evidence_source_role",
            "scope_note",
            "legal_review",
        },
        required={"publisher_declared_identifier", "legal_review"},
        where="license",
    )
    if (
        not isinstance(license_record["publisher_declared_identifier"], str)
        or not license_record["publisher_declared_identifier"].strip()
    ):
        raise ValueError("license.publisher_declared_identifier must be non-empty")
    if license_record["legal_review"] != "not_legal_advice":
        raise ValueError("license.legal_review must be 'not_legal_advice'")
    if license_record.get("evidence_source_role") not in {None, "license_evidence"}:
        raise ValueError("license.evidence_source_role must be 'license_evidence' when present")
    for key in ("declaration_url", "terms_url"):
        value = license_record.get(key)
        if value is not None and (
            not isinstance(value, str) or urllib.parse.urlparse(value).scheme != "https"
        ):
            raise ValueError(f"license.{key} must use HTTPS when present")

    source_values = raw["source_files"]
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("source_files must be a non-empty array")
    sources: list[SourceSpec] = []
    for index, value in enumerate(source_values):
        where = f"source_files[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a JSON object")
        _require_keys(
            value,
            allowed={
                "role",
                "local_filename",
                "url",
                "publisher_advertised_size_bytes",
                "publisher_advertised_sha256",
                "local_verification_status",
            },
            required={
                "role",
                "local_filename",
                "url",
                "publisher_advertised_size_bytes",
                "publisher_advertised_sha256",
                "local_verification_status",
            },
            where=where,
        )
        role = value["role"]
        filename = value["local_filename"]
        url = value["url"]
        status = value["local_verification_status"]
        if role not in ALLOWED_SOURCE_ROLES:
            raise ValueError(f"{where}.role must be one of {sorted(ALLOWED_SOURCE_ROLES)}")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise ValueError(f"{where}.local_filename must be a simple filename")
        if not isinstance(url, str) or urllib.parse.urlparse(url).scheme != "https":
            raise ValueError(f"{where}.url must use HTTPS")
        if status != "pending_local_download":
            raise ValueError(
                f"{where}.local_verification_status must remain "
                "'pending_local_download' in the source configuration; local "
                "verification belongs in the generated manifest"
            )
        size_value = value["publisher_advertised_size_bytes"]
        if size_value is not None:
            size_value = _positive_int(size_value, f"{where}.publisher_advertised_size_bytes")
        sources.append(
            SourceSpec(
                role=str(role),
                local_filename=filename,
                url=url,
                publisher_advertised_size_bytes=size_value,
                publisher_advertised_sha256=_optional_sha256(
                    value["publisher_advertised_sha256"],
                    f"{where}.publisher_advertised_sha256",
                ),
                local_verification_status=status,
            )
        )
    roles = [item.role for item in sources]
    if roles.count("train") != 1 or roles.count("validation") != 1:
        raise ValueError("source_files must contain exactly one train and one validation role")
    if roles.count("license_evidence") > 1:
        raise ValueError("source_files may contain at most one license_evidence role")
    if len({item.local_filename for item in sources}) != len(sources):
        raise ValueError("source local_filename values must be unique")
    if len({item.url for item in sources}) != len(sources):
        raise ValueError("source URLs must be unique")

    document_format = raw["document_format"]
    if not isinstance(document_format, dict):
        raise ValueError("document_format must be a JSON object")
    _require_keys(
        document_format,
        allowed={"encoding", "delimiter", "max_document_bytes", "canonical_framing"},
        required={"encoding", "delimiter", "max_document_bytes", "canonical_framing"},
        where="document_format",
    )
    if document_format["encoding"] != "utf-8":
        raise ValueError("document_format.encoding must be 'utf-8'")
    delimiter_text = document_format["delimiter"]
    if not isinstance(delimiter_text, str) or not delimiter_text:
        raise ValueError("document_format.delimiter must be a non-empty string")
    delimiter = delimiter_text.encode("utf-8")
    max_document_bytes = _positive_int(
        document_format["max_document_bytes"], "document_format.max_document_bytes"
    )
    if document_format["canonical_framing"] != "strip+LF+delimiter+LF":
        raise ValueError("document_format.canonical_framing must be 'strip+LF+delimiter+LF'")

    derivation = raw["derivation"]
    if not isinstance(derivation, dict):
        raise ValueError("derivation must be a JSON object")
    _require_keys(
        derivation,
        allowed={"version", "splits", "max_padding_fraction"},
        required={"version", "splits", "max_padding_fraction"},
        where="derivation",
    )
    if derivation["version"] != DERIVATION_VERSION:
        raise ValueError(
            f"unsupported derivation.version {derivation['version']!r}; "
            f"expected {DERIVATION_VERSION!r}"
        )
    padding_fraction = derivation["max_padding_fraction"]
    if (
        not isinstance(padding_fraction, (int, float))
        or isinstance(padding_fraction, bool)
        or not math.isfinite(float(padding_fraction))
        or not 0.0 <= float(padding_fraction) <= 0.05
    ):
        raise ValueError("derivation.max_padding_fraction must be in [0, 0.05]")
    split_values = derivation["splits"]
    if not isinstance(split_values, list) or len(split_values) != 2:
        raise ValueError("derivation.splits must contain train and validation")
    splits: list[DerivedSplit] = []
    for index, value in enumerate(split_values):
        where = f"derivation.splits[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a JSON object")
        _require_keys(
            value,
            allowed={"split", "source_role", "filename", "target_bytes"},
            required={"split", "source_role", "filename", "target_bytes"},
            where=where,
        )
        split = value["split"]
        source_role = value["source_role"]
        filename = value["filename"]
        if split not in {"train", "validation"} or source_role != split:
            raise ValueError(f"{where} must map each split to its same-named source role")
        if not isinstance(filename, str) or Path(filename).name != filename or not filename:
            raise ValueError(f"{where}.filename must be a simple filename")
        splits.append(
            DerivedSplit(
                split=split,
                source_role=source_role,
                filename=filename,
                target_bytes=_positive_int(value["target_bytes"], f"{where}.target_bytes"),
            )
        )
    if {item.split for item in splits} != {"train", "validation"}:
        raise ValueError("derivation.splits must contain train and validation exactly once")
    if len({item.filename for item in splits}) != 2:
        raise ValueError("derived split filenames must be unique")
    return sources, splits, delimiter, max_document_bytes, float(padding_fraction)


def _download_url(url: str, output: BinaryIO) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "faster-glm-asr-training-corpus/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        while True:
            block = response.read(DEFAULT_CHUNK_BYTES)
            if not block:
                break
            output.write(block)


def _verify_source(path: Path, spec: SourceSpec) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.is_symlink():
        raise ValueError(f"symbolic-link sources are not accepted: {path}")
    size = path.stat().st_size
    digest = _sha256_file(path)
    if (
        spec.publisher_advertised_size_bytes is not None
        and size != spec.publisher_advertised_size_bytes
    ):
        raise ValueError(
            f"{path}: size {size} does not match publisher-advertised "
            f"size {spec.publisher_advertised_size_bytes}"
        )
    if spec.publisher_advertised_sha256 is not None and digest != spec.publisher_advertised_sha256:
        raise ValueError(
            f"{path}: SHA-256 {digest} does not match publisher-advertised "
            f"SHA-256 {spec.publisher_advertised_sha256}"
        )
    return {
        "role": spec.role,
        "local_filename": spec.local_filename,
        "configured_url": spec.url,
        "size_bytes": size,
        "sha256": digest,
        "local_verification_status": "verified_complete_file",
        "publisher_advertised_size_bytes": spec.publisher_advertised_size_bytes,
        "publisher_advertised_sha256": spec.publisher_advertised_sha256,
        "publisher_claim_match": (
            True
            if spec.publisher_advertised_size_bytes is not None
            or spec.publisher_advertised_sha256 is not None
            else None
        ),
    }


def _ensure_sources(
    specs: Sequence[SourceSpec], source_dir: Path, *, download: bool
) -> list[dict[str, object]]:
    source_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    missing: list[SourceSpec] = []
    for spec in specs:
        path = source_dir / spec.local_filename
        if path.exists():
            records.append(_verify_source(path, spec))
        else:
            missing.append(spec)
    if missing and not download:
        names = ", ".join(item.local_filename for item in missing)
        raise FileNotFoundError(
            "source verification is pending; missing "
            f"{names}. Re-run with --download or place the complete pinned files "
            "in --source-dir. No completed evidence was written."
        )
    for spec in missing:
        final_path = source_dir / spec.local_filename
        temporary = source_dir / f".{spec.local_filename}.part-{uuid.uuid4().hex}"
        try:
            with temporary.open("xb") as stream:
                _download_url(spec.url, stream)
                stream.flush()
                os.fsync(stream.fileno())
            record = _verify_source(temporary, spec)
            # Atomic publication without replacing a file that appeared during
            # the download.  Hard-linking gives create-if-absent semantics.
            try:
                os.link(temporary, final_path)
            except FileExistsError:
                record = _verify_source(final_path, spec)
            records.append(record)
        finally:
            if temporary.exists():
                temporary.unlink()
    records_by_role = {str(item["role"]): item for item in records}
    return [records_by_role[item.role] for item in specs]


def _iter_documents(path: Path, delimiter: bytes, max_document_bytes: int) -> Iterable[bytes]:
    buffer = bytearray()
    with path.open("rb") as stream:
        while True:
            block = stream.read(DEFAULT_CHUNK_BYTES)
            if block:
                buffer.extend(block)
            while True:
                position = buffer.find(delimiter)
                if position < 0:
                    break
                if position > max_document_bytes:
                    raise ValueError(
                        f"{path}: document exceeds max_document_bytes={max_document_bytes}"
                    )
                yield bytes(buffer[:position])
                del buffer[: position + len(delimiter)]
            if len(buffer) > max_document_bytes:
                raise ValueError(
                    f"{path}: document exceeds max_document_bytes={max_document_bytes}"
                )
            if not block:
                break
    if len(buffer) > max_document_bytes:
        raise ValueError(f"{path}: document exceeds max_document_bytes={max_document_bytes}")
    if buffer.strip():
        yield bytes(buffer)


def _document_inventory_sha256(hashes: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(hashes):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _derive_split(
    source_path: Path,
    output_path: Path,
    spec: DerivedSplit,
    *,
    delimiter: bytes,
    max_document_bytes: int,
    already_selected_hashes: set[str],
    max_padding_fraction: float,
) -> tuple[dict[str, object], set[str]]:
    selected: set[str] = set()
    selected_bytes = 0
    selected_documents = 0
    duplicate_documents_skipped = 0
    documents_too_large_for_remaining_budget = 0
    output_digest = hashlib.sha256()
    with output_path.open("xb") as output:
        for raw_document in _iter_documents(source_path, delimiter, max_document_bytes):
            document = raw_document.strip()
            if not document:
                continue
            try:
                document.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{source_path}: document is not valid UTF-8") from exc
            document_hash = hashlib.sha256(document).hexdigest()
            if document_hash in already_selected_hashes or document_hash in selected:
                duplicate_documents_skipped += 1
                continue
            payload = document + b"\n" + delimiter + b"\n"
            remaining = spec.target_bytes - selected_bytes
            if len(payload) > remaining:
                documents_too_large_for_remaining_budget += 1
                continue
            output.write(payload)
            output_digest.update(payload)
            selected.add(document_hash)
            selected_bytes += len(payload)
            selected_documents += 1
            if selected_bytes == spec.target_bytes:
                break
        if selected_documents == 0:
            raise ValueError(f"{source_path}: no eligible non-empty documents")
        padding_bytes = spec.target_bytes - selected_bytes
        if padding_bytes / spec.target_bytes > max_padding_fraction:
            raise ValueError(
                f"{spec.split}: requires {padding_bytes} padding bytes "
                f"({padding_bytes / spec.target_bytes:.6%}), exceeding "
                f"max_padding_fraction={max_padding_fraction:.6%}"
            )
        padding = b"\n" * padding_bytes
        output.write(padding)
        output_digest.update(padding)
        output.flush()
        os.fsync(output.fileno())
    if output_path.stat().st_size != spec.target_bytes:
        raise AssertionError("derived output size does not equal its fixed byte budget")
    return (
        {
            "split": spec.split,
            "source_role": spec.source_role,
            "filename": spec.filename,
            "target_bytes": spec.target_bytes,
            "size_bytes": output_path.stat().st_size,
            "sha256": output_digest.hexdigest(),
            "selected_document_count": selected_documents,
            "selected_document_payload_bytes": selected_bytes,
            "selected_document_inventory_sha256": _document_inventory_sha256(selected),
            "duplicate_documents_skipped": duplicate_documents_skipped,
            "documents_too_large_for_remaining_budget": (documents_too_large_for_remaining_budget),
            "padding_byte": "0a",
            "padding_bytes": padding_bytes,
            "padding_fraction": padding_bytes / spec.target_bytes,
        },
        selected,
    )


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(
    config_path: Path,
    source_dir: Path,
    output_dir: Path,
    *,
    download: bool = False,
    artifact_root: Path | None = None,
) -> Mapping[str, object]:
    artifact_root = (artifact_root or DEFAULT_ARTIFACT_ROOT).resolve()
    config_path = config_path.resolve()
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    for name, path in (("source_dir", source_dir), ("output_dir", output_dir)):
        if path == artifact_root or not path.is_relative_to(artifact_root):
            raise ValueError(
                f"{name} must be a strict descendant of the private artifact root: {artifact_root}"
            )
    if (
        source_dir == output_dir
        or source_dir in output_dir.parents
        or output_dir in source_dir.parents
    ):
        raise ValueError("source_dir and output_dir must be disjoint private directories")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new versioned path: {output_dir}"
        )
    raw = _read_json(config_path)
    sources, splits, delimiter, max_document_bytes, max_padding_fraction = _parse_config(raw)
    source_records = _ensure_sources(sources, source_dir, download=download)
    source_by_role = {item.role: source_dir / item.local_filename for item in sources}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        selected_hashes: set[str] = set()
        output_records: list[dict[str, object]] = []
        for split in sorted(splits, key=lambda item: item.split != "train"):
            record, newly_selected = _derive_split(
                source_by_role[split.source_role],
                stage / split.filename,
                split,
                delimiter=delimiter,
                max_document_bytes=max_document_bytes,
                already_selected_hashes=selected_hashes,
                max_padding_fraction=max_padding_fraction,
            )
            if selected_hashes & newly_selected:
                raise AssertionError("cross-split selected document hash overlap")
            selected_hashes.update(newly_selected)
            output_records.append(record)

        training_data = {
            "mode": "byte_files",
            "train_files": [
                _loader_path(
                    output_dir / next(item.filename for item in splits if item.split == "train")
                )
            ],
            "validation_files": [
                _loader_path(
                    output_dir
                    / next(item.filename for item in splits if item.split == "validation")
                )
            ],
        }
        _write_text_exclusive(
            stage / "training-data.json",
            json.dumps(training_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        config_snapshot = json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _write_text_exclusive(stage / "source-config.json", config_snapshot)

        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete_local_verification",
            "dataset": raw["dataset"],
            "license": raw["license"],
            "source_config": {
                "filename": "source-config.json",
                "sha256": _sha256_file(stage / "source-config.json"),
            },
            "sources": source_records,
            "derivation": {
                "version": DERIVATION_VERSION,
                "document_delimiter_utf8": delimiter.decode("utf-8"),
                "canonical_framing": "strip+LF+delimiter+LF",
                "selection_order": "source order; complete documents that fit remaining budget",
                "exact_duplicate_policy": (
                    "SHA-256 of stripped UTF-8 document; skip within and across splits"
                ),
                "padding_policy": ("LF-only tail padding to exact byte budget; recorded per split"),
                "split_isolation": (
                    "distinct pinned upstream train/validation files plus zero selected "
                    "document SHA-256 overlap"
                ),
                "selected_document_sha256_overlap_count": 0,
                "max_document_bytes": max_document_bytes,
                "max_padding_fraction": max_padding_fraction,
            },
            "outputs": sorted(output_records, key=lambda item: item["split"] != "train"),
            "training_runtime": {
                "data_config_filename": "training-data.json",
                "data_config_sha256": _sha256_file(stage / "training-data.json"),
                "path_policy": (
                    "repository-relative paths when inside the repository; "
                    "absolute paths only for external fixture roots"
                ),
                "compatibility": ("experiments.distributed_training.train data.mode=byte_files"),
            },
            "builder": {
                "source_file": Path(__file__).name,
                "source_sha256": _sha256_file(Path(__file__).resolve()),
            },
        }
        manifest_path = stage / "corpus-manifest.json"
        _write_text_exclusive(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        manifest_hash = _sha256_file(manifest_path)
        _write_text_exclusive(
            stage / "corpus-manifest.sha256",
            f"{manifest_hash}  corpus-manifest.json\n",
        )

        output_dir.mkdir()
        try:
            # The checksum is the commit marker and is published last.
            for path in sorted(stage.iterdir()):
                if path.name != "corpus-manifest.sha256":
                    os.replace(path, output_dir / path.name)
            os.replace(
                stage / "corpus-manifest.sha256",
                output_dir / "corpus-manifest.sha256",
            )
        except BaseException:
            shutil.rmtree(output_dir)
            raise
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a hash-locked fixed-size byte training corpus."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing pinned sources; existing files are never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        manifest = prepare(
            args.config,
            args.source_dir,
            args.output_dir,
            download=args.download,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"corpus preparation failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
