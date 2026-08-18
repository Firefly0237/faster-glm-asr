#!/usr/bin/env python3
"""Fail-closed safety checks for the public source release.

The default inventory is equivalent to the tracked files plus unignored
untracked files that could be added to Git.  ``--include-history`` also scans
every blob reachable from every local ref, catching material deleted from the
current tree.  Findings never echo matched content.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

MAX_TEXT_BYTES = 2 * 1024 * 1024
INDEX_OBJECT_BATCH_SIZE = 32

PUBLIC_MATRIX_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "faster_glm_asr"
    / "benchmarking"
    / "public_matrix_schema.py"
)
PUBLIC_MATRIX_SCHEMA_RELATIVE = "src/faster_glm_asr/benchmarking/public_matrix_schema.py"

ALLOWED_TOP_LEVEL_DIRECTORIES = {
    ".github",
    "benchmarks",
    "configs",
    "containers",
    "docs",
    "examples",
    "experiments",
    "requirements",
    "scripts",
    "src",
    "tests",
    "tools",
}

ALLOWED_ROOT_EXACT = {
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "CMakeLists.txt",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "MANIFEST.in",
    "Makefile",
    "NOTICE",
    "README.draft.md",
    "README.md",
    "SECURITY.md",
    "environment.yml",
    "mkdocs.yml",
    "pyproject.toml",
    "tox.ini",
    "uv.lock",
}

ALLOWED_ROOT_PREFIXES = (
    "requirements-",
    "requirements.",
    "constraints-",
    "constraints.",
)

ALLOWED_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cff",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".csv",
    ".h",
    ".hpp",
    ".ini",
    ".in",
    ".j2",
    ".jinja",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".aac",
    ".bin",
    ".ckpt",
    ".doc",
    ".docx",
    ".flac",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".mp3",
    ".ncu-rep",
    ".nsys-rep",
    ".onnx",
    ".ogg",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".ppt",
    ".pptx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".wav",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}

FORBIDDEN_PATH_SEGMENTS = {
    ".cache",
    ".env",
    ".idea",
    ".pytest_cache",
    ".venv",
    ".vscode",
    "credentials",
    "interviews",
    "job" + "-packaging",
    "private",
    "report",
    "reports",
    "re" + "sumes",
    "secrets",
}

FORBIDDEN_TOP_LEVEL_DIRECTORIES = {
    "artifacts",
    "checkpoints",
    "data",
    "datasets",
    "inputs",
    "logs",
    "models",
    "outputs",
    "runs",
    "wandb",
}

FORBIDDEN_BASENAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
}

CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key-block",
        re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----"),
    ),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github-fine-grained-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("hugging-face-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{16,}\b")),
    (
        "authorization-header",
        re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    ),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|passwd|"
            r"access[_-]?token)\b\s*[:=]\s*['\"]?"
            r"(?!example\b|placeholder\b|redacted\b|none\b|null\b|<)"
            r"[A-Za-z0-9._~+/=-]{8,}"
        ),
    ),
    (
        "credential-in-url",
        re.compile(r"https?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    ),
    (
        "windows-user-path",
        re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:\\Users\\[^\\\s]+\\"),
    ),
    (
        "linux-home-path",
        re.compile(r"(?<![A-Za-z0-9])/" + "ho" + r"me/[^/\s]+/"),
    ),
    (
        "macos-home-path",
        re.compile(r"(?<![A-Za-z0-9])/" + "Us" + r"ers/[^/\s]+/"),
    ),
    ("root-home-path", re.compile(r"(?<![A-Za-z0-9])" + r"/" + r"root/")),
    ("career-material-zh-1", re.compile(r"\u6c42\u804c")),
    ("career-material-zh-2", re.compile(r"\u7b80\u5386")),
    ("career-material-zh-3", re.compile(r"\u9762\u7ecf")),
    ("career-material-zh-4", re.compile(r"\u9762\u8bd5\u51c6\u5907")),
    ("career-material-zh-5", re.compile(r"\u5927\u5382")),
    (
        "career-material-en",
        re.compile(r"career" + r"[- ]preparation", re.IGNORECASE),
    ),
    ("non-public-project-path", re.compile(r"docs/" + r"job" + r"-packaging")),
)


@dataclass(frozen=True, order=True)
class Finding:
    code: str
    path: str
    message: str
    origin: str = "candidate"


@dataclass(frozen=True)
class GitIndexEntry:
    path: str
    object_id: str
    mode: str
    stage: int


_public_matrix_schema_module: ModuleType | None = None


def _validate_public_matrix_schema_interface(module: ModuleType) -> ModuleType:
    if not callable(getattr(module, "load_public_summary_bytes", None)) or not callable(
        getattr(module, "validate_public_summary", None)
    ):
        raise RuntimeError("public matrix schema interface is incomplete")
    return module


def _load_public_matrix_schema_bytes(payload: bytes, *, origin: str) -> ModuleType:
    """Load the verified current-candidate schema without rereading the worktree."""
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("public matrix schema is not UTF-8") from exc
    module = ModuleType("_faster_glm_asr_public_matrix_schema")
    module.__file__ = origin
    exec(compile(source, origin, "exec"), module.__dict__)
    return _validate_public_matrix_schema_interface(module)


def _load_public_matrix_schema() -> ModuleType:
    """Load the standalone public-result validator without importing the package."""
    global _public_matrix_schema_module
    if _public_matrix_schema_module is not None:
        return _public_matrix_schema_module
    spec = importlib.util.spec_from_file_location(
        "_faster_glm_asr_public_matrix_schema",
        PUBLIC_MATRIX_SCHEMA_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("public matrix schema loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _public_matrix_schema_module = _validate_public_matrix_schema_interface(module)
    return module


def _is_public_matrix_summary(path_text: str) -> bool:
    path = PurePosixPath(path_text.replace("\\", "/"))
    return (
        len(path.parts) == 3
        and path.parts[:2] == ("benchmarks", "results")
        and path.suffix.casefold() == ".json"
    )


def _public_matrix_findings(path_text: str, payload: bytes, *, origin: str) -> list[Finding]:
    if not _is_public_matrix_summary(path_text):
        return []
    message = "public matrix summary failed strict parsing or schema validation"
    try:
        schema = _load_public_matrix_schema()
        parsed = schema.load_public_summary_bytes(payload)
        if not isinstance(parsed, dict):
            raise TypeError("public matrix loader returned a non-object")
        errors = schema.validate_public_summary(parsed)
        if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
            raise TypeError("public matrix validator returned an invalid result")
    except Exception:
        return [Finding("public-matrix-invalid", path_text, message, origin)]
    if errors:
        return [Finding("public-matrix-invalid", path_text, message, origin)]
    return []


def _run_git(root: Path, arguments: Sequence[str], *, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return result.stdout


def candidate_paths(root: Path) -> list[str]:
    """Return tracked plus unignored untracked paths, as Git would see them."""
    output = _run_git(
        root,
        ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    )
    assert isinstance(output, bytes)
    decoded = [os.fsdecode(item) for item in output.split(b"\0") if item]
    return sorted(set(path.replace("\\", "/") for path in decoded))


def index_entries(root: Path) -> list[GitIndexEntry]:
    """Return every staged Git index entry without reading the worktree."""
    output = _run_git(root, ["ls-files", "--stage", "-z"])
    assert isinstance(output, bytes)
    entries: list[GitIndexEntry] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path_bytes = record.split(b"\t", maxsplit=1)
            mode_bytes, object_id_bytes, stage_bytes = metadata.split(b" ")
            stage = int(stage_bytes.decode("ascii"))
            mode = mode_bytes.decode("ascii")
            object_id = object_id_bytes.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("git ls-files returned an invalid staged entry") from exc
        entries.append(
            GitIndexEntry(
                path=os.fsdecode(path_bytes).replace("\\", "/"),
                object_id=object_id,
                mode=mode,
                stage=stage,
            )
        )
    return entries


def _index_objects(
    root: Path, object_ids: Sequence[str]
) -> dict[str, tuple[str, int, bytes | None]]:
    """Read index objects in one bounded streaming ``git cat-file`` session."""
    ordered = list(dict.fromkeys(object_ids))
    if not ordered:
        return {}
    process = subprocess.Popen(
        ["git", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        process.stdin.write("".join(f"{object_id}\n" for object_id in ordered).encode("ascii"))
        process.stdin.close()
        objects: dict[str, tuple[str, int, bytes | None]] = {}
        for requested in ordered:
            header = process.stdout.readline()
            if not header:
                raise RuntimeError("git cat-file ended before every index object was read")
            fields = header.rstrip(b"\n").split(b" ")
            if len(fields) == 2 and fields[1] == b"missing":
                objects[requested] = ("missing", 0, None)
                continue
            if len(fields) != 3:
                raise RuntimeError("git cat-file returned an invalid object header")
            returned_id, type_bytes, size_bytes = fields
            try:
                returned = returned_id.decode("ascii")
                object_type = type_bytes.decode("ascii")
                size = int(size_bytes.decode("ascii"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError("git cat-file returned invalid object metadata") from exc
            if returned != requested or size < 0:
                raise RuntimeError("git cat-file returned an unexpected index object")

            remaining = size
            payload = bytearray() if size <= MAX_TEXT_BYTES else None
            while remaining:
                chunk = process.stdout.read(min(remaining, 64 * 1024))
                if not chunk:
                    raise RuntimeError("git cat-file returned a truncated index object")
                if payload is not None:
                    payload.extend(chunk)
                remaining -= len(chunk)
            if process.stdout.read(1) != b"\n":
                raise RuntimeError("git cat-file omitted an index object terminator")
            objects[requested] = (
                object_type,
                size,
                bytes(payload) if payload is not None else None,
            )
        stderr = process.stderr.read()
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(
                "git cat-file --batch failed: " + stderr.decode("utf-8", errors="replace").strip()
            )
        return objects
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise


def _root_name_allowed(name: str) -> bool:
    return name in ALLOWED_ROOT_EXACT or any(
        name.startswith(prefix) and name.endswith((".txt", ".lock"))
        for prefix in ALLOWED_ROOT_PREFIXES
    )


def validate_path(path_text: str) -> list[Finding]:
    findings: list[Finding] = []
    normalized = path_text.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        return [Finding("unsafe-path", path_text, "path must be relative and traversal-free")]
    if unicodedata.normalize("NFC", normalized) != normalized:
        findings.append(
            Finding("noncanonical-unicode-path", normalized, "path must use NFC normalization")
        )
    folded_parts = [part.casefold() for part in path.parts]
    if any(part in FORBIDDEN_PATH_SEGMENTS for part in folded_parts):
        findings.append(
            Finding("forbidden-path", normalized, "path belongs to a non-public artifact class")
        )
    if folded_parts[0] in FORBIDDEN_TOP_LEVEL_DIRECTORIES:
        findings.append(
            Finding(
                "forbidden-top-level",
                normalized,
                "top-level path belongs to a local artifact class",
            )
        )
    if path.name.casefold() in FORBIDDEN_BASENAMES:
        findings.append(
            Finding("forbidden-filename", normalized, "credential-bearing filename is forbidden")
        )
    suffix = path.suffix.casefold()
    if suffix in FORBIDDEN_SUFFIXES:
        findings.append(
            Finding("forbidden-extension", normalized, "binary or generated artifact is forbidden")
        )

    if len(path.parts) == 1:
        if not _root_name_allowed(path.name):
            findings.append(
                Finding("root-not-allowlisted", normalized, "root filename is not allowlisted")
            )
    elif path.parts[0] not in ALLOWED_TOP_LEVEL_DIRECTORIES:
        findings.append(
            Finding(
                "top-level-not-allowlisted", normalized, "top-level directory is not allowlisted"
            )
        )
    elif path.parts[0] == "benchmarks" and not _is_public_matrix_summary(normalized):
        findings.append(
            Finding(
                "benchmark-path-not-allowlisted",
                normalized,
                "benchmarks only admits direct public summary JSON under benchmarks/results",
            )
        )

    root_special = len(path.parts) == 1 and _root_name_allowed(path.name)
    github_special = path.parts[0] == ".github" and path.name == "CODEOWNERS"
    if not root_special and not github_special and suffix not in ALLOWED_TEXT_SUFFIXES:
        findings.append(
            Finding(
                "text-type-not-allowlisted",
                normalized,
                "file type is not approved UTF-8 source text",
            )
        )
    return findings


def validate_content(path_text: str, payload: bytes, *, origin: str = "candidate") -> list[Finding]:
    findings: list[Finding] = []
    if len(payload) > MAX_TEXT_BYTES:
        findings.append(
            Finding(
                "file-too-large",
                path_text,
                f"source text exceeds {MAX_TEXT_BYTES} bytes",
                origin,
            )
        )
        return findings
    findings.extend(_public_matrix_findings(path_text, payload, origin=origin))
    if b"\0" in payload:
        findings.append(Finding("binary-content", path_text, "NUL byte detected", origin))
        return findings
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        findings.append(Finding("invalid-utf8", path_text, "file is not strict UTF-8 text", origin))
        return findings
    lfs_marker = "version https://git-" + "lfs.github.com/spec/"
    if text.startswith(lfs_marker):
        findings.append(
            Finding("git-lfs-pointer", path_text, "Git LFS pointers are not allowed", origin)
        )
    for label, pattern in CONTENT_PATTERNS:
        if pattern.search(text):
            findings.append(
                Finding("sensitive-content", path_text, f"matched rule {label}", origin)
            )
    return findings


def _metadata_content_findings(
    path_text: str, payload: bytes, *, origin: str = "candidate"
) -> list[Finding]:
    findings: list[Finding] = []
    text = payload.decode("utf-8", errors="replace")
    if path_text == "LICENSE":
        required_phrases = (
            "Apache License",
            "Version 2.0, January 2004",
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
            "END OF TERMS AND CONDITIONS",
        )
        if not all(phrase in text for phrase in required_phrases):
            findings.append(
                Finding(
                    "license-invalid",
                    "LICENSE",
                    "Apache-2.0 license text is incomplete",
                    origin,
                )
            )
    elif path_text == "NOTICE":
        required_phrases = (
            "edin-mls-26-spring",
            "CC0 1.0 Universal",
            "https://github.com/zai-org/GLM-ASR",
            "Hugging Face Transformers",
            "model artifacts are not bundled here",
        )
        if not all(phrase in text for phrase in required_phrases):
            findings.append(
                Finding(
                    "notice-invalid",
                    "NOTICE",
                    "required upstream boundaries are missing",
                    origin,
                )
            )
    return findings


def _metadata_findings(
    candidate_set: set[str], payloads: dict[str, bytes], *, strict: bool
) -> list[Finding]:
    findings: list[Finding] = []
    required = {"LICENSE", "NOTICE"}
    if strict:
        required.add("README.md")
    for path in sorted(required - candidate_set):
        findings.append(Finding("missing-metadata", path, "required release metadata is missing"))
    if not strict and not {"README.md", "README.draft.md"} & candidate_set:
        findings.append(
            Finding(
                "missing-metadata",
                "README.md",
                "README.md or README.draft.md is required",
            )
        )
    if strict and "README.draft.md" in candidate_set:
        findings.append(
            Finding(
                "draft-present", "README.draft.md", "remove the draft after publishing README.md"
            )
        )

    for path_text in ("LICENSE", "NOTICE"):
        payload = payloads.get(path_text)
        if payload is not None:
            findings.extend(_metadata_content_findings(path_text, payload))
    return findings


def _is_link_or_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _safe_worktree_path(root: Path, path_text: str) -> tuple[Path | None, list[Finding]]:
    local_path = root.joinpath(*PurePosixPath(path_text).parts)
    probe = root
    for part in PurePosixPath(path_text).parts:
        probe /= part
        if _is_link_or_reparse_point(probe):
            return None, [
                Finding(
                    "symbolic-link",
                    path_text,
                    "symbolic links, junctions, and reparse points are not allowed",
                )
            ]
    resolved = local_path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, [
            Finding("path-escape", path_text, "candidate resolves outside the repository")
        ]
    return local_path, []


def _schema_origin_drift_finding() -> Finding:
    return Finding(
        "schema-origin-drift",
        PUBLIC_MATRIX_SCHEMA_RELATIVE,
        (
            "strict release requires one ordinary stage-0 schema blob whose bytes "
            "exactly equal the non-reparse worktree schema"
        ),
        "index",
    )


def _strict_public_matrix_schema(
    root: Path, entries: Sequence[GitIndexEntry]
) -> tuple[ModuleType | None, list[Finding]]:
    """Bind strict validation to the exact stage-0 schema selected for commit."""
    matches = [entry for entry in entries if entry.path == PUBLIC_MATRIX_SCHEMA_RELATIVE]
    stage_zero = [entry for entry in matches if entry.stage == 0]
    if len(matches) != 1 or len(stage_zero) != 1 or stage_zero[0].mode not in {"100644", "100755"}:
        return None, [_schema_origin_drift_finding()]

    entry = stage_zero[0]
    try:
        object_type, size, payload = _index_objects(root, [entry.object_id])[entry.object_id]
    except (KeyError, RuntimeError):
        return None, [_schema_origin_drift_finding()]
    if object_type != "blob" or size > MAX_TEXT_BYTES or payload is None:
        return None, [_schema_origin_drift_finding()]

    worktree_path, path_findings = _safe_worktree_path(root, PUBLIC_MATRIX_SCHEMA_RELATIVE)
    if worktree_path is None or path_findings or not worktree_path.is_file():
        return None, [_schema_origin_drift_finding()]
    try:
        worktree_payload = worktree_path.read_bytes()
    except OSError:
        return None, [_schema_origin_drift_finding()]
    if worktree_payload != payload:
        return None, [_schema_origin_drift_finding()]

    try:
        module = _load_public_matrix_schema_bytes(
            payload,
            origin=f"index:{PUBLIC_MATRIX_SCHEMA_RELATIVE}@{entry.object_id}",
        )
    except Exception:
        return None, [_schema_origin_drift_finding()]
    return module, []


def _index_findings(root: Path, entries: Sequence[GitIndexEntry]) -> list[Finding]:
    findings: list[Finding] = []
    for offset in range(0, len(entries), INDEX_OBJECT_BATCH_SIZE):
        batch = entries[offset : offset + INDEX_OBJECT_BATCH_SIZE]
        objects = _index_objects(root, [entry.object_id for entry in batch])
        for entry in batch:
            origin = f"index:{entry.object_id[:12]}"
            findings.extend(
                Finding(item.code, item.path, item.message, origin)
                for item in validate_path(entry.path)
            )
            if entry.stage != 0:
                findings.append(
                    Finding(
                        "unmerged-index",
                        entry.path,
                        "unmerged index stages are not publishable",
                        origin,
                    )
                )
            if entry.mode == "120000":
                findings.append(
                    Finding(
                        "symbolic-link",
                        entry.path,
                        "staged symbolic links are not allowed",
                        origin,
                    )
                )
            elif entry.mode == "160000":
                findings.append(
                    Finding("gitlink", entry.path, "staged Git links are not allowed", origin)
                )
            elif entry.mode not in {"100644", "100755"}:
                findings.append(
                    Finding(
                        "unsupported-git-mode",
                        entry.path,
                        f"staged Git mode {entry.mode} is not an ordinary file",
                        origin,
                    )
                )

            object_type, size, payload = objects[entry.object_id]
            if object_type == "missing":
                findings.append(
                    Finding(
                        "missing-index-object",
                        entry.path,
                        "staged object is unavailable",
                        origin,
                    )
                )
                continue
            if entry.mode != "160000" and object_type != "blob":
                findings.append(
                    Finding(
                        "invalid-index-object",
                        entry.path,
                        "staged ordinary path does not reference a blob",
                        origin,
                    )
                )
                continue
            if size > MAX_TEXT_BYTES:
                findings.append(
                    Finding(
                        "file-too-large",
                        entry.path,
                        f"staged source exceeds {MAX_TEXT_BYTES} bytes",
                        origin,
                    )
                )
                continue
            if payload is not None and object_type == "blob":
                findings.extend(validate_content(entry.path, payload, origin=origin))
                findings.extend(_metadata_content_findings(entry.path, payload, origin=origin))
    return findings


def _candidate_findings(root: Path, paths: Sequence[str], *, strict: bool) -> list[Finding]:
    findings: list[Finding] = []
    payloads: dict[str, bytes] = {}
    casefolded: dict[str, str] = {}
    for path_text in paths:
        folded = unicodedata.normalize("NFC", path_text).casefold()
        previous = casefolded.get(folded)
        if previous is not None and previous != path_text:
            findings.append(
                Finding("case-collision", path_text, f"case-fold collision with {previous}")
            )
        casefolded[folded] = path_text
        path_findings = validate_path(path_text)
        findings.extend(path_findings)
        if any(item.code == "unsafe-path" for item in path_findings):
            continue
        local_path, link_findings = _safe_worktree_path(root, path_text)
        findings.extend(link_findings)
        if local_path is None:
            continue
        if not local_path.is_file():
            findings.append(
                Finding("missing-candidate", path_text, "candidate path is not a regular file")
            )
            continue
        if local_path.stat().st_size > MAX_TEXT_BYTES:
            findings.append(
                Finding(
                    "file-too-large",
                    path_text,
                    f"source text exceeds {MAX_TEXT_BYTES} bytes",
                )
            )
            continue
        payload = local_path.read_bytes()
        payloads[path_text] = payload
        findings.extend(validate_content(path_text, payload))
    findings.extend(_metadata_findings(set(paths), payloads, strict=strict))
    return findings


def history_blobs(
    root: Path,
) -> list[tuple[str, str, str, int, bytes | None]]:
    """Return unique path/object records from every reachable tree.

    Payload is ``None`` for oversized objects so a release check cannot exhaust
    memory merely by encountering a large historical blob.
    """
    commits_output = _run_git(root, ["rev-list", "--all"])
    assert isinstance(commits_output, bytes)
    commits = [item for item in commits_output.decode("ascii").splitlines() if item]
    path_objects: set[tuple[str, str, str]] = set()
    for commit in commits:
        tree = _run_git(root, ["ls-tree", "-r", "-z", "--full-tree", commit])
        assert isinstance(tree, bytes)
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, path_bytes = record.split(b"\t", maxsplit=1)
            mode, object_type, object_id = metadata.split(b" ")
            if object_type != b"blob":
                continue
            path_objects.add(
                (
                    os.fsdecode(path_bytes).replace("\\", "/"),
                    object_id.decode("ascii"),
                    mode.decode("ascii"),
                )
            )
    blobs: list[tuple[str, str, str, int, bytes | None]] = []
    for path_text, object_id, mode in sorted(path_objects):
        size_output = _run_git(root, ["cat-file", "-s", object_id])
        assert isinstance(size_output, bytes)
        size = int(size_output.decode("ascii").strip())
        payload: bytes | None = None
        if size <= MAX_TEXT_BYTES:
            value = _run_git(root, ["cat-file", "blob", object_id])
            assert isinstance(value, bytes)
            payload = value
        blobs.append((path_text, object_id, mode, size, payload))
    return blobs


def _history_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path_text, object_id, mode, size, payload in history_blobs(root):
        origin = f"history:{object_id[:12]}"
        findings.extend(
            Finding(item.code, item.path, item.message, origin) for item in validate_path(path_text)
        )
        if mode == "120000":
            findings.append(
                Finding(
                    "symbolic-link",
                    path_text,
                    "historical symbolic links are not allowed",
                    origin,
                )
            )
        if size > MAX_TEXT_BYTES:
            findings.append(
                Finding(
                    "file-too-large",
                    path_text,
                    f"historical source exceeds {MAX_TEXT_BYTES} bytes",
                    origin,
                )
            )
        else:
            assert payload is not None
            findings.extend(validate_content(path_text, payload, origin=origin))
    return findings


def _history_completeness_findings(root: Path) -> list[Finding]:
    value = _run_git(root, ["rev-parse", "--is-shallow-repository"])
    assert isinstance(value, bytes)
    state = value.decode("ascii", errors="strict").strip()
    if state == "false":
        return []
    if state == "true":
        return [
            Finding(
                "history-incomplete",
                ".git",
                "history scanning requires a non-shallow repository",
                "history",
            )
        ]
    raise RuntimeError("git returned an invalid shallow-repository state")


def scan_repository(
    root: Path, *, strict_release: bool = False, include_history: bool = False
) -> list[Finding]:
    global _public_matrix_schema_module
    root = root.resolve()
    if not (root / ".git").exists():
        raise ValueError(f"not a Git repository root: {root}")
    entries = index_entries(root)
    previous_schema = _public_matrix_schema_module
    if strict_release:
        strict_schema, schema_findings = _strict_public_matrix_schema(root, entries)
        if schema_findings:
            return sorted(set(schema_findings))
        assert strict_schema is not None
        _public_matrix_schema_module = strict_schema
    try:
        paths = candidate_paths(root)
        findings = _candidate_findings(root, paths, strict=strict_release)
        findings.extend(_index_findings(root, entries))
        if include_history:
            completeness = _history_completeness_findings(root)
            findings.extend(completeness)
            if not completeness:
                findings.extend(_history_findings(root))
        return sorted(set(findings))
    finally:
        _public_matrix_schema_module = previous_schema


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to this tool's parent repository)",
    )
    parser.add_argument("--strict-release", action="store_true")
    parser.add_argument("--include-history", action="store_true")
    args = parser.parse_args(argv)
    try:
        findings = scan_repository(
            args.root,
            strict_release=args.strict_release,
            include_history=args.include_history,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"public release guard could not run: {exc}", file=sys.stderr)
        return 2
    if findings:
        print(f"public release guard FAILED with {len(findings)} finding(s):")
        for finding in findings:
            print(f"[{finding.code}] {finding.origin}:{finding.path}: {finding.message}")
        return 1
    print("public release guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
