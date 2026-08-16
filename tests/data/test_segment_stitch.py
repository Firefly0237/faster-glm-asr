from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from faster_glm_asr.data import audio_manifest as build_audio_manifest
from faster_glm_asr.data import long_audio as build_long_audio_manifest
from faster_glm_asr.data import segment_stitch as stitch_asr_segments


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_map() -> dict[str, Any]:
    files = {
        "src/faster_glm_asr/benchmarking/runner.py": {
            "size_bytes": 123,
            "sha256": "c" * 64,
        }
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "faster-glm-asr-source-map-v1",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def _load_manifest(bundle: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (bundle / "segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SegmentStitchTest(unittest.TestCase):
    def _make_bundle(
        self, root: Path, *, with_reference: bool = True
    ) -> tuple[Path, list[dict[str, Any]]]:
        audio = root / "recording.wav"
        reference = root / "recording.vtt"
        inventory = root / "inventory.csv"
        source_manifest = root / "manifests" / "sources.jsonl"
        bundle = root / "bundle"

        pcm = np.arange(35_200, dtype=np.int32)
        pcm = ((pcm % 2048) - 1024).astype("<i2")
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(pcm.tobytes())
        if with_reference:
            reference.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:00.600\n"
                "alpha beta\n\n"
                "00:00:00.700 --> 00:00:01.500\n"
                "gamma delta\n\n"
                "00:00:01.600 --> 00:00:02.200\n"
                "epsilon zeta\n",
                encoding="utf-8",
                newline="\n",
            )
        with inventory.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(build_audio_manifest.KNOWN_COLUMNS))
            writer.writeheader()
            row = {
                "id": "recording-0001",
                "audio_path": audio.name,
                "language": "en",
                "collection": "synthetic-fixture",
                "speaker_count": "1",
                "split": "domain-test",
                "split_group_id": "session-0001",
                "shareable": "false",
                "notes": "synthetic test only",
            }
            if with_reference:
                row.update(
                    {
                        "reference_path": reference.name,
                        "authorization_id": "AUTH-SYNTHETIC",
                        "reference_quality": "human-corrected",
                    }
                )
            writer.writerow(row)

        self.assertEqual(build_audio_manifest.build(inventory, source_manifest), 0)
        self.assertEqual(
            build_long_audio_manifest.build(
                source_manifest,
                bundle,
                chunk_seconds="1",
                overlap_seconds="0.25",
            ),
            0,
        )
        return bundle, _load_manifest(bundle)

    def _make_result(
        self,
        bundle: Path,
        manifest: Sequence[dict[str, Any]],
        hypotheses: Sequence[str],
        output: Path,
    ) -> dict[str, Any]:
        self.assertEqual(len(manifest), len(hypotheses))
        samples = []
        excluded = {
            "sample_id",
            "audio_path",
            "audio_sha256",
            "language",
            "duration_s",
        }
        for index, (record, hypothesis) in enumerate(zip(manifest, hypotheses, strict=True)):
            token_ids = [index + 11]
            samples.append(
                {
                    "sample_id": record["sample_id"],
                    "audio_sha256": record["audio_sha256"],
                    "language": record["language"],
                    "duration_s": record["duration_s"],
                    "metadata": {
                        key: value for key, value in record.items() if key not in excluded
                    },
                    "hypothesis": hypothesis,
                    "hypothesis_sha256": hashlib.sha256(hypothesis.encode("utf-8")).hexdigest(),
                    "generated_tokens": len(token_ids),
                    "generated_token_ids": token_ids,
                    "generated_token_ids_sha256": hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "quality": None,
                }
            )
        result = {
            "schema_version": "0.3",
            "run": {
                "implementation": "custom_static_cache",
                "model_name": "synthetic/model",
                "model_revision": "a" * 40,
                "manifest_sha256": _sha256(bundle / "segments.jsonl"),
                "artifact_scope": "private",
                "measurement_profile": "canonical_latency",
                "metric_eligibility": {"canonical_latency": True},
                "phase_timing": False,
                "nvml_sample_ms": 0.0,
            },
            "git": {"commit": "b" * 40, "dirty": False},
            "sources": _source_map(),
            "environment": {"environment_lock_sha256": "e" * 64},
            "samples": samples,
        }
        _write_json(output, result)
        return result

    def test_stitches_longest_overlap_and_scores_at_source_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            output = root / "source-quality.json"
            self._make_result(
                bundle,
                manifest,
                [
                    "Alpha, beta gamma",
                    "BETA gamma delta epsilon",
                    "delta epsilon zeta!",
                ],
                benchmark_result,
            )

            self.assertEqual(stitch_asr_segments.build(benchmark_result, bundle, output), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "asr-source-stitch-v0.1")
            self.assertEqual(payload["artifact_scope"], "private")
            self.assertEqual(len(payload["source_groups"]), 1)
            source = payload["source_groups"][0]
            self.assertEqual(source["hypothesis"], "alpha beta gamma delta epsilon zeta")
            self.assertEqual(
                [decision["matched_overlap_words"] for decision in source["stitch_decisions"]],
                [0, 2, 2],
            )
            self.assertEqual(source["quality_status"], "evaluated")
            self.assertEqual(source["quality"]["wer"], 0.0)
            self.assertEqual(source["quality"]["cer"], 0.0)
            self.assertEqual(payload["aggregate_quality"]["sources_evaluated"], 1)
            self.assertEqual(
                payload["inputs"]["benchmark_result_sha256"],
                _sha256(benchmark_result),
            )
            self.assertEqual(
                payload["inputs"]["segment_manifest_sha256"],
                _sha256(bundle / "segments.jsonl"),
            )
            config_hash = hashlib.sha256(
                stitch_asr_segments._canonical_json_bytes(payload["stitch_configuration"])
            ).hexdigest()
            self.assertEqual(payload["stitch_configuration_sha256"], config_hash)
            self.assertEqual(
                payload["provenance"]["transform_sources"]["faster_glm_asr.data.segment_stitch"][
                    "sha256"
                ],
                _sha256(Path(stitch_asr_segments.__file__)),
            )

    def test_no_overlap_appends_every_normalized_word(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            output = root / "source-quality.json"
            self._make_result(
                bundle,
                manifest,
                ["one two", "three four", "five six"],
                benchmark_result,
            )

            stitch_asr_segments.build(benchmark_result, bundle, output)
            source = json.loads(output.read_text(encoding="utf-8"))["source_groups"][0]
            self.assertEqual(source["hypothesis"], "one two three four five six")
            self.assertEqual(
                [item["matched_overlap_words"] for item in source["stitch_decisions"]],
                [0, 0, 0],
            )

    def test_rejects_duplicate_missing_and_out_of_order_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            result = self._make_result(
                bundle,
                manifest,
                ["one", "two", "three"],
                benchmark_result,
            )

            duplicate = copy.deepcopy(result)
            duplicate["samples"][1] = copy.deepcopy(duplicate["samples"][0])
            duplicate_path = root / "duplicate.json"
            _write_json(duplicate_path, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate segment"):
                stitch_asr_segments.create_stitch_payload(duplicate_path, bundle)

            missing = copy.deepcopy(result)
            missing["samples"].pop()
            missing_path = root / "missing.json"
            _write_json(missing_path, missing)
            with self.assertRaisesRegex(ValueError, "completeness mismatch"):
                stitch_asr_segments.create_stitch_payload(missing_path, bundle)

            shuffled = copy.deepcopy(result)
            shuffled["samples"][0], shuffled["samples"][1] = (
                shuffled["samples"][1],
                shuffled["samples"][0],
            )
            shuffled_path = root / "shuffled.json"
            _write_json(shuffled_path, shuffled)
            with self.assertRaisesRegex(ValueError, "order differs"):
                stitch_asr_segments.create_stitch_payload(shuffled_path, bundle)

    def test_rejects_wrong_result_and_self_rehashed_sidecar_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            result = self._make_result(
                bundle,
                manifest,
                ["one", "two", "three"],
                benchmark_result,
            )
            result["run"]["manifest_sha256"] = "0" * 64
            _write_json(benchmark_result, result)
            with self.assertRaisesRegex(ValueError, "does not match the segment bundle"):
                stitch_asr_segments.create_stitch_payload(benchmark_result, bundle)

            result["run"]["manifest_sha256"] = _sha256(bundle / "segments.jsonl")
            _write_json(benchmark_result, result)
            sidecar_path = bundle / "source-references.private.jsonl"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["full_reference_text"] = "tampered reference"
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            lock_path = bundle / "bundle-lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["private_reference_sidecar_sha256"] = _sha256(sidecar_path)
            lock_path.write_bytes(build_long_audio_manifest._canonical_json_bytes(lock))
            with self.assertRaisesRegex(ValueError, "reference text SHA256 mismatch"):
                stitch_asr_segments.create_stitch_payload(benchmark_result, bundle)

    def test_empty_sidecar_and_empty_hypotheses_are_valid_without_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root, with_reference=False)
            benchmark_result = root / "private-benchmark.json"
            output = root / "source-quality.json"
            self._make_result(bundle, manifest, ["", "", ""], benchmark_result)
            self.assertEqual((bundle / "source-references.private.jsonl").read_bytes(), b"")

            stitch_asr_segments.build(benchmark_result, bundle, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            source = payload["source_groups"][0]
            self.assertEqual(source["hypothesis"], "")
            self.assertIsNone(source["quality"])
            self.assertEqual(source["quality_status"], "unavailable_no_reference")
            self.assertEqual(payload["aggregate_quality"]["sources_with_reference"], 0)
            self.assertEqual(payload["aggregate_quality"]["status"], "not_evaluated")

    def test_rejects_empty_reference_even_when_hashes_are_self_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            self._make_result(
                bundle,
                manifest,
                ["one", "two", "three"],
                benchmark_result,
            )
            sidecar_path = bundle / "source-references.private.jsonl"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["full_reference_text"] = ""
            sidecar["full_reference_text_sha256"] = hashlib.sha256(b"").hexdigest()
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            lock_path = bundle / "bundle-lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["private_reference_sidecar_sha256"] = _sha256(sidecar_path)
            lock_path.write_bytes(build_long_audio_manifest._canonical_json_bytes(lock))

            with self.assertRaisesRegex(ValueError, "must be non-empty text"):
                stitch_asr_segments.create_stitch_payload(benchmark_result, bundle)

    def test_identical_inputs_and_configuration_produce_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            first = root / "first.json"
            second = root / "second.json"
            self._make_result(
                bundle,
                manifest,
                ["alpha beta gamma", "beta gamma delta", "delta epsilon"],
                benchmark_result,
            )

            stitch_asr_segments.build(benchmark_result, bundle, first)
            stitch_asr_segments.build(benchmark_result, bundle, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_atomic_writer_refuses_overwrite_and_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, manifest = self._make_bundle(root)
            benchmark_result = root / "private-benchmark.json"
            output = root / "source-quality.json"
            self._make_result(
                bundle,
                manifest,
                ["alpha", "beta", "gamma"],
                benchmark_result,
            )
            output.write_text("sentinel\n", encoding="utf-8", newline="\n")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                stitch_asr_segments.build(benchmark_result, bundle, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
