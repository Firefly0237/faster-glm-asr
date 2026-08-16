from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools" / "public_release_guard.py"
SPEC = importlib.util.spec_from_file_location("public_release_guard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
public_release_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = public_release_guard
SPEC.loader.exec_module(public_release_guard)


class PublicReleaseGuardTest(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )

    def _repository(self, root: Path, *, final_readme: bool = False) -> None:
        self._git(root, "init", "--quiet")
        (root / "LICENSE").write_text(
            "Apache License\n"
            "Version 2.0, January 2004\n"
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"
            "END OF TERMS AND CONDITIONS\n",
            encoding="utf-8",
        )
        (root / "NOTICE").write_text(
            "edin-mls-26-spring\n"
            "CC0 1.0 Universal\n"
            "https://github.com/zai-org/GLM-ASR\n"
            "Hugging Face Transformers\n"
            "model artifacts are not bundled here\n",
            encoding="utf-8",
        )
        readme = "README.md" if final_readme else "README.draft.md"
        (root / readme).write_text("# Fixture\n", encoding="utf-8")
        (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")

    def test_clean_candidate_and_final_metadata_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            (root / "src" / "package").mkdir(parents=True)
            (root / "src" / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            self.assertEqual(public_release_guard.scan_repository(root), [])
            findings = public_release_guard.scan_repository(root, strict_release=True)
            self.assertEqual(
                {item.code for item in findings}, {"draft-present", "missing-metadata"}
            )
            (root / "README.draft.md").rename(root / "README.md")
            self.assertEqual(public_release_guard.scan_repository(root, strict_release=True), [])

    def test_fail_closed_path_allowlist(self) -> None:
        excluded_area = "job" + "-packaging"
        cases = {
            f"docs/{excluded_area}/notes.md": "forbidden-path",
            "data/sample.json": "forbidden-top-level",
            "docs/sample.wav": "forbidden-extension",
            "misc/source.py": "top-level-not-allowlisted",
            "src/package/payload.exe": "text-type-not-allowlisted",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                codes = {item.code for item in public_release_guard.validate_path(path)}
                self.assertIn(expected, codes)
        self.assertEqual(public_release_guard.validate_path("src/faster_glm_asr/model.py"), [])
        self.assertEqual(
            public_release_guard.validate_path("src/faster_glm_asr/data/audio_manifest.py"),
            [],
        )
        self.assertEqual(public_release_guard.validate_path("tests/data/test_manifest.py"), [])
        self.assertEqual(public_release_guard.validate_path("MANIFEST.in"), [])

    def test_project_gitignore_keeps_package_data_in_release_inventory(self) -> None:
        project_root = SCRIPT.parents[1]
        candidates = set(public_release_guard.candidate_paths(project_root))
        self.assertIn(
            "src/faster_glm_asr/data/audio_manifest.py",
            candidates,
            "a root-local data ignore rule must not hide the importable data package",
        )
        top_level_findings = public_release_guard.validate_path("data/sample.json")
        self.assertTrue(any(item.code == "forbidden-top-level" for item in top_level_findings))

    def test_sensitive_content_is_rejected_without_value_in_finding(self) -> None:
        access_value = "AK" + "IA" + "A" * 16
        local_path = "/" + "ho" + "me/alice/project/file.txt"
        excluded_words = "".join(chr(value) for value in (0x6C42, 0x804C))
        excluded_english = "career" + "-preparation"
        for payload, rule in (
            (access_value, "aws-access-key"),
            (local_path, "linux-home-path"),
            (excluded_words, "career-material-zh-1"),
            (excluded_english, "career-material-en"),
        ):
            with self.subTest(rule=rule):
                findings = public_release_guard.validate_content(
                    "docs/note.md", payload.encode("utf-8")
                )
                self.assertTrue(any(rule in item.message for item in findings))
                self.assertTrue(all(payload not in item.message for item in findings))
        self.assertEqual(
            public_release_guard.validate_content("docs/example.md", b'api_key = "<redacted>"\n'),
            [],
        )

    def test_binary_oversize_and_lfs_pointer_are_rejected(self) -> None:
        self.assertEqual(
            public_release_guard.validate_content("docs/x.md", b"a\0b")[0].code,
            "binary-content",
        )
        oversized = b"a" * (public_release_guard.MAX_TEXT_BYTES + 1)
        self.assertEqual(
            public_release_guard.validate_content("docs/x.md", oversized)[0].code,
            "file-too-large",
        )
        marker = ("version https://git-" + "lfs.github.com/spec/v1\n").encode()
        self.assertEqual(
            public_release_guard.validate_content("docs/x.md", marker)[0].code,
            "git-lfs-pointer",
        )

    def test_utf8_csv_is_allowed_but_binary_csv_is_rejected(self) -> None:
        path = "configs/benchmark/inventory.example.csv"
        self.assertEqual(public_release_guard.validate_path(path), [])
        self.assertEqual(
            public_release_guard.validate_content(
                path, "sample_id,audio_path\n示例,inputs/example.wav\n".encode()
            ),
            [],
        )
        findings = public_release_guard.validate_content(path, b"sample_id\0audio_path\n")
        self.assertTrue(any(item.code == "binary-content" for item in findings))

    def test_history_scan_finds_content_deleted_from_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root, final_readme=True)
            self._git(root, "config", "user.name", "Release Guard Test")
            self._git(root, "config", "user.email", "guard@example.invalid")
            docs = root / "docs"
            docs.mkdir()
            leaked = "gh" + "p_" + "A" * 24
            note = docs / "note.md"
            note.write_text(leaked, encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "--quiet", "-m", "fixture with old blob")
            note.unlink()
            (docs / "safe.md").write_text("safe\n", encoding="utf-8")
            self._git(root, "add", "-A")
            self._git(root, "commit", "--quiet", "-m", "remove old blob")

            self.assertEqual(public_release_guard.scan_repository(root, strict_release=True), [])
            findings = public_release_guard.scan_repository(
                root, strict_release=True, include_history=True
            )
            historical = [
                item
                for item in findings
                if item.path == "docs/note.md" and item.origin.startswith("history:")
            ]
            self.assertTrue(historical)
            self.assertTrue(all(leaked not in item.message for item in historical))

    def test_cli_does_not_echo_matched_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repository(root)
            (root / "docs").mkdir()
            value = "hf" + "_" + "A" * 24
            (root / "docs" / "note.md").write_text(value, encoding="utf-8")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = public_release_guard.main(["--root", str(root)])
            self.assertEqual(status, 1)
            self.assertNotIn(value, captured.getvalue())
            self.assertIn("hugging-face-token", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
