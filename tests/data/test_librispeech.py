from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

from faster_glm_asr.data import librispeech as build_librispeech_subset


class LibriSpeechSubsetTests(unittest.TestCase):
    def test_public_source_lock_matches_builder_contract(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        source_lock = json.loads(
            (repository_root / "configs/data/librispeech-test-clean-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(source_lock["dataset"], "LibriSpeech")
        self.assertEqual(source_lock["subset"], "test-clean")
        self.assertEqual(source_lock["upstream_page"], build_librispeech_subset.UPSTREAM_URL)
        self.assertEqual(source_lock["license"], build_librispeech_subset.UPSTREAM_LICENSE)
        self.assertEqual(source_lock["archive_size_bytes"], 346_663_984)
        self.assertEqual(source_lock["archive_md5"], "32fa31d27d2e1cad72775fee3f4849a9")
        self.assertEqual(
            source_lock["selection"]["version"],
            build_librispeech_subset.SELECTION_VERSION,
        )
        self.assertEqual(
            source_lock["selection"]["seed"],
            build_librispeech_subset.DEFAULT_SEED,
        )

    def _write_wav(self, path: Path, duration_s: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = 16000 * duration_s
        silence = b"\x00\x00" * 16000
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            for _ in range(duration_s):
                stream.writeframes(silence)
        self.assertEqual(frames, duration_s * 16000)

    def _dataset(self, root: Path, durations: list[int]) -> Path:
        corpus = root / "LibriSpeech"
        test_clean = corpus / "test-clean"
        chapter = test_clean / "61" / "70968"
        corpus.mkdir(parents=True)
        (corpus / "LICENSE.TXT").write_text(
            "Synthetic test fixture; not the upstream license.\n", encoding="utf-8"
        )
        lines = []
        for index, duration_s in enumerate(durations):
            sample_id = f"61-70968-{index:04d}"
            self._write_wav(chapter / f"{sample_id}.wav", duration_s)
            lines.append(f"{sample_id} SYNTHETIC TRANSCRIPT NUMBER {index}\n")
        (chapter / "61-70968.trans.txt").write_text("".join(lines), encoding="utf-8")
        return test_clean

    def test_builds_balanced_deterministic_hash_locked_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_clean = self._dataset(root, [5, 6, 12, 13, 22, 23])
            license_file = test_clean.parent / "LICENSE.TXT"
            first = root / "subset-a"
            second = root / "subset-b"

            lock_a = build_librispeech_subset.build(
                test_clean,
                first,
                license_file=license_file,
                per_bucket=2,
                seed="fixed-seed",
            )
            lock_b = build_librispeech_subset.build(
                test_clean,
                second,
                license_file=license_file,
                per_bucket=2,
                seed="fixed-seed",
            )

            self.assertEqual(lock_a, lock_b)
            self.assertEqual(lock_a["selected_count"], 6)
            self.assertEqual(
                lock_a["candidate_counts"],
                {"short": 2, "medium": 2, "near-30s": 2},
            )
            records = [
                json.loads(line)
                for line in (first / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["duration_bucket"] for record in records],
                ["short", "short", "medium", "medium", "near-30s", "near-30s"],
            )
            self.assertTrue(all(record["shareable"] is False for record in records))
            self.assertTrue(
                all(record["manifest_parser_version"] == "asr-inventory-v0.2" for record in records)
            )
            manifest_hash = hashlib.sha256((first / "manifest.jsonl").read_bytes()).hexdigest()
            self.assertEqual(lock_a["manifest_sha256"], manifest_hash)
            stored_lock = json.loads((first / "selection-lock.json").read_text(encoding="utf-8"))
            self.assertEqual(stored_lock, lock_a)
            self.assertFalse(any(first.rglob("*.wav")))

    def test_rejects_existing_output_and_insufficient_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_clean = self._dataset(root, [5, 12, 22])
            license_file = test_clean.parent / "LICENSE.TXT"
            output = root / "subset"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                build_librispeech_subset.build(
                    test_clean,
                    output,
                    license_file=license_file,
                    per_bucket=1,
                    seed="fixed-seed",
                )
            output.rmdir()
            with self.assertRaisesRegex(ValueError, "requires 2"):
                build_librispeech_subset.build(
                    test_clean,
                    output,
                    license_file=license_file,
                    per_bucket=2,
                    seed="fixed-seed",
                )
            self.assertFalse(output.exists())

    def test_rejects_noncanonical_audio_and_malformed_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_clean = self._dataset(root, [5, 12, 22])
            wav_path = next(test_clean.rglob("*.wav"))
            with wave.open(str(wav_path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00\x00\x00" * 16000)
            with self.assertRaisesRegex(ValueError, "16 kHz mono"):
                build_librispeech_subset.discover(test_clean)

            transcript = next(test_clean.rglob("*.trans.txt"))
            transcript.write_text("NOT-A-LIBRISPEECH-ID text\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid utterance ID"):
                build_librispeech_subset.discover(test_clean)


if __name__ == "__main__":
    unittest.main()
