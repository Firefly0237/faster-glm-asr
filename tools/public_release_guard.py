#!/usr/bin/env python3
"""Fail-closed safety checks for the public source release.

The default inventory is equivalent to the tracked files plus unignored
untracked files that could be added to Git.  ``--include-history`` also scans
every blob reachable from every local ref, catching material deleted from the
current tree.  Findings never echo matched content.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_TEXT_BYTES = 2 * 1024 * 1024

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


def _metadata_findings(root: Path, candidate_set: set[str], *, strict: bool) -> list[Finding]:
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

    license_path = root / "LICENSE"
    if license_path.is_file():
        license_text = license_path.read_text(encoding="utf-8", errors="replace")
        required_license_phrases = (
            "Apache License",
            "Version 2.0, January 2004",
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
            "END OF TERMS AND CONDITIONS",
        )
        if not all(phrase in license_text for phrase in required_license_phrases):
            findings.append(
                Finding("license-invalid", "LICENSE", "Apache-2.0 license text is incomplete")
            )
    notice_path = root / "NOTICE"
    if notice_path.is_file():
        notice_text = notice_path.read_text(encoding="utf-8", errors="replace")
        required_notice_phrases = (
            "edin-mls-26-spring",
            "CC0 1.0 Universal",
            "https://github.com/zai-org/GLM-ASR",
            "Hugging Face Transformers",
            "model artifacts are not bundled here",
        )
        if not all(phrase in notice_text for phrase in required_notice_phrases):
            findings.append(
                Finding("notice-invalid", "NOTICE", "required upstream boundaries are missing")
            )
    return findings


def _candidate_findings(root: Path, paths: Sequence[str], *, strict: bool) -> list[Finding]:
    findings: list[Finding] = []
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
        local_path = root.joinpath(*PurePosixPath(path_text).parts)
        if local_path.is_symlink():
            findings.append(Finding("symbolic-link", path_text, "symbolic links are not allowed"))
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
        findings.extend(validate_content(path_text, local_path.read_bytes()))
    findings.extend(_metadata_findings(root, set(paths), strict=strict))
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


def scan_repository(
    root: Path, *, strict_release: bool = False, include_history: bool = False
) -> list[Finding]:
    root = root.resolve()
    if not (root / ".git").exists():
        raise ValueError(f"not a Git repository root: {root}")
    paths = candidate_paths(root)
    findings = _candidate_findings(root, paths, strict=strict_release)
    if include_history:
        findings.extend(_history_findings(root))
    return sorted(set(findings))


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
