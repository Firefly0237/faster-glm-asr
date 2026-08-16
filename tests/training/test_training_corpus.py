from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "tools" / "prepare_training_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_training_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_training_corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_training_corpus
SPEC.loader.exec_module(prepare_training_corpus)


DELIMITER = b"<|endoftext|>"


def _source(documents: list[bytes]) -> bytes:
    return DELIMITER.join(documents) + DELIMITER


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TrainingCorpusTest(unittest.TestCase):
    def _configuration(
        self,
        root: Path,
        payloads: dict[str, bytes],
        *,
        train_target: int,
        validation_target: int,
    ) -> Path:
        config = {
            "schema_version": "training-corpus-source-v0.1",
            "dataset": {
                "name": "fixture",
                "revision": "0" * 40,
            },
            "license": {
                "publisher_declared_identifier": "fixture-only",
                "legal_review": "not_legal_advice",
            },
            "source_files": [
                {
                    "role": role,
                    "local_filename": filename,
                    "url": f"https://fixture.invalid/{filename}",
                    "publisher_advertised_size_bytes": len(payloads[role]),
                    "publisher_advertised_sha256": _sha256(payloads[role]),
                    "local_verification_status": "pending_local_download",
                }
                for role, filename in (
                    ("train", "train.txt"),
                    ("validation", "validation.txt"),
                    ("license_evidence", "README.md"),
                )
            ],
            "document_format": {
                "encoding": "utf-8",
                "delimiter": DELIMITER.decode("ascii"),
                "max_document_bytes": 1024,
                "canonical_framing": "strip+LF+delimiter+LF",
            },
            "derivation": {
                "version": "complete-documents-byte-budget-v0.1",
                "max_padding_fraction": 0.05,
                "splits": [
                    {
                        "split": "train",
                        "source_role": "train",
                        "filename": "train.bin",
                        "target_bytes": train_target,
                    },
                    {
                        "split": "validation",
                        "source_role": "validation",
                        "filename": "validation.bin",
                        "target_bytes": validation_target,
                    },
                ],
            },
        }
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _payloads(self) -> dict[str, bytes]:
        return {
            "train": _source([bytes([value]) for value in range(ord("a"), ord("i"))]),
            # The first document duplicates train and must be excluded.  The
            # remaining six one-byte documents exactly fill the byte budget.
            "validation": _source([b"a"] + [bytes([value]) for value in range(ord("i"), ord("o"))]),
            "license_evidence": b"fixture license evidence\n",
        }

    def test_public_source_contract_pins_revision_hashes_license_and_budgets(self) -> None:
        raw = prepare_training_corpus._read_json(prepare_training_corpus.DEFAULT_CONFIG)
        sources, splits, _delimiter, _maximum, _padding = prepare_training_corpus._parse_config(raw)
        self.assertEqual(
            raw["dataset"]["revision"],
            "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        )
        self.assertEqual(raw["license"]["publisher_declared_identifier"], "CDLA-Sharing-1.0")
        source_by_role = {item.role: item for item in sources}
        self.assertEqual(
            source_by_role["train"].publisher_advertised_sha256,
            "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
        )
        self.assertEqual(
            source_by_role["validation"].publisher_advertised_sha256,
            "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
        )
        self.assertEqual(
            {item.split: item.target_bytes for item in splits},
            {"train": 64 * 1024 * 1024, "validation": 4 * 1024 * 1024},
        )
        revision = raw["dataset"]["revision"]
        self.assertTrue(all(revision in item.url for item in sources))

    def test_downloader_uses_the_public_project_user_agent(self) -> None:
        captured: list[object] = []

        def fake_open(request: object, *, timeout: int) -> io.BytesIO:
            captured.append(request)
            self.assertEqual(timeout, 60)
            return io.BytesIO(b"fixture")

        output = io.BytesIO()
        with mock.patch.object(
            prepare_training_corpus.urllib.request,
            "urlopen",
            side_effect=fake_open,
        ):
            prepare_training_corpus._download_url("https://fixture.invalid/source", output)
        self.assertEqual(output.getvalue(), b"fixture")
        self.assertEqual(
            captured[0].get_header("User-agent"),
            "faster-glm-asr-training-corpus/1.0",
        )

    def test_offline_fixture_build_is_exact_hash_locked_and_disjoint(self) -> None:
        payloads = self._payloads()
        framed_one_byte_document = 1 + 1 + len(DELIMITER) + 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            filenames = {
                "train": "train.txt",
                "validation": "validation.txt",
                "license_evidence": "README.md",
            }
            for role, payload in payloads.items():
                (source_dir / filenames[role]).write_bytes(payload)
            config = self._configuration(
                root,
                payloads,
                train_target=8 * framed_one_byte_document,
                validation_target=6 * framed_one_byte_document,
            )
            output = root / "derived-v1"

            manifest = prepare_training_corpus.prepare(
                config, source_dir, output, artifact_root=root
            )

            self.assertEqual(manifest["status"], "complete_local_verification")
            records = {item["split"]: item for item in manifest["outputs"]}
            self.assertEqual(records["train"]["selected_document_count"], 8)
            self.assertEqual(records["validation"]["selected_document_count"], 6)
            self.assertEqual(records["validation"]["duplicate_documents_skipped"], 1)
            self.assertEqual(records["train"]["padding_bytes"], 0)
            self.assertEqual(records["validation"]["padding_bytes"], 0)
            for record in records.values():
                path = output / record["filename"]
                self.assertEqual(path.stat().st_size, record["target_bytes"])
                self.assertEqual(_sha256(path.read_bytes()), record["sha256"])

            data_config = json.loads((output / "training-data.json").read_text())
            self.assertEqual(data_config["mode"], "byte_files")
            self.assertEqual(data_config["train_files"], [str((output / "train.bin").resolve())])
            self.assertEqual(
                data_config["validation_files"],
                [str((output / "validation.bin").resolve())],
            )
            expected_manifest_hash = _sha256((output / "corpus-manifest.json").read_bytes())
            checksum = (output / "corpus-manifest.sha256").read_text().split()[0]
            self.assertEqual(checksum, expected_manifest_hash)
            self.assertEqual(manifest["derivation"]["selected_document_sha256_overlap_count"], 0)

            with self.assertRaises(FileExistsError):
                prepare_training_corpus.prepare(config, source_dir, output, artifact_root=root)

    def test_missing_sources_stay_explicitly_pending_without_output(self) -> None:
        payloads = self._payloads()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._configuration(root, payloads, train_target=16, validation_target=16)
            output = root / "derived"
            with self.assertRaisesRegex(FileNotFoundError, "verification is pending"):
                prepare_training_corpus.prepare(
                    config, root / "sources", output, artifact_root=root
                )
            self.assertFalse(output.exists())

    def test_downloads_missing_sources_and_never_replaces_existing_source(self) -> None:
        payloads = self._payloads()
        framed_one_byte_document = 1 + 1 + len(DELIMITER) + 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "sources"
            source_dir.mkdir()
            train_path = source_dir / "train.txt"
            train_path.write_bytes(payloads["train"])
            original_inode = train_path.stat().st_ino
            config = self._configuration(
                root,
                payloads,
                train_target=8 * framed_one_byte_document,
                validation_target=6 * framed_one_byte_document,
            )
            by_name = {
                "validation.txt": payloads["validation"],
                "README.md": payloads["license_evidence"],
            }
            downloaded: list[str] = []

            def fake_download(url: str, output: object) -> None:
                name = Path(url).name
                downloaded.append(name)
                output.write(by_name[name])

            with mock.patch.object(
                prepare_training_corpus, "_download_url", side_effect=fake_download
            ):
                prepare_training_corpus.prepare(
                    config,
                    source_dir,
                    root / "derived",
                    download=True,
                    artifact_root=root,
                )
            self.assertEqual(sorted(downloaded), ["README.md", "validation.txt"])
            self.assertEqual(train_path.read_bytes(), payloads["train"])
            self.assertEqual(train_path.stat().st_ino, original_inode)
            self.assertFalse(list(source_dir.glob("*.part-*")))

    def test_rejects_corrupt_existing_source_and_writes_no_derived_artifact(self) -> None:
        payloads = self._payloads()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            (source_dir / "train.txt").write_bytes(b"corrupt")
            config = self._configuration(root, payloads, train_target=16, validation_target=16)
            output = root / "derived"
            with self.assertRaisesRegex(ValueError, "publisher-advertised"):
                prepare_training_corpus.prepare(
                    config,
                    source_dir,
                    output,
                    download=True,
                    artifact_root=root,
                )
            self.assertEqual((source_dir / "train.txt").read_bytes(), b"corrupt")
            self.assertFalse(output.exists())

    def test_rejects_path_traversal_and_non_https_source(self) -> None:
        payloads = self._payloads()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._configuration(root, payloads, train_target=16, validation_target=16)
            config = json.loads(config_path.read_text())
            config["source_files"][0]["local_filename"] = "../train.txt"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "simple filename"):
                prepare_training_corpus.prepare(
                    config_path,
                    root / "sources",
                    root / "derived",
                    artifact_root=root,
                )

            config = json.loads(
                self._configuration(
                    root, payloads, train_target=16, validation_target=16
                ).read_text()
            )
            config["source_files"][0]["url"] = "http://fixture.invalid/train.txt"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must use HTTPS"):
                prepare_training_corpus.prepare(
                    config_path,
                    root / "sources",
                    root / "derived",
                    artifact_root=root,
                )

    def test_public_defaults_stay_below_the_ignored_private_artifact_root(self) -> None:
        repository_root = SCRIPT.parents[1]
        artifact_root = prepare_training_corpus.DEFAULT_ARTIFACT_ROOT
        self.assertEqual(
            artifact_root,
            repository_root / "artifacts/private/training-data",
        )
        self.assertTrue(prepare_training_corpus.default_source_dir().is_relative_to(artifact_root))
        self.assertTrue(prepare_training_corpus.default_output_dir().is_relative_to(artifact_root))
        self.assertIn("/artifacts/", (repository_root / ".gitignore").read_text())

    def test_cli_uses_private_defaults_and_rejects_an_output_escape(self) -> None:
        payloads = self._payloads()
        framed_one_byte_document = 1 + 1 + len(DELIMITER) + 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts/private/training-data"
            config = self._configuration(
                root,
                payloads,
                train_target=8 * framed_one_byte_document,
                validation_target=6 * framed_one_byte_document,
            )
            with mock.patch.object(prepare_training_corpus, "DEFAULT_ARTIFACT_ROOT", artifact_root):
                source_dir = prepare_training_corpus.default_source_dir()
                source_dir.mkdir(parents=True)
                filenames = {
                    "train": "train.txt",
                    "validation": "validation.txt",
                    "license_evidence": "README.md",
                }
                for role, payload in payloads.items():
                    (source_dir / filenames[role]).write_bytes(payload)
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    status = prepare_training_corpus.main(["--config", str(config)])
                self.assertEqual(status, 0, stderr.getvalue())
                output_dir = prepare_training_corpus.default_output_dir()
                self.assertTrue((output_dir / "corpus-manifest.sha256").is_file())
                self.assertEqual(
                    json.loads(stdout.getvalue())["status"], "complete_local_verification"
                )

                escaped = root / "outside" / "derived"
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    status = prepare_training_corpus.main(
                        [
                            "--config",
                            str(config),
                            "--source-dir",
                            str(source_dir),
                            "--output-dir",
                            str(escaped),
                        ]
                    )
                self.assertEqual(status, 2)
                self.assertIn("private artifact root", stderr.getvalue())
                self.assertFalse(escaped.exists())


if __name__ == "__main__":
    unittest.main()
