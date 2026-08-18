from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.benchmarking.test_aggregation import MatrixFixture, _quality, _summary

from faster_glm_asr import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from faster_glm_asr.benchmarking import matrix_aggregator as aggregation
from faster_glm_asr.benchmarking import provenance, public_matrix
from faster_glm_asr.benchmarking import runner as benchmark_runner
from faster_glm_asr.data import librispeech

REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURE_DURATIONS = {
    "100-200-0001": 5.0,
    "101-201-0002": 15.0,
    "102-202-0003": 25.0,
}


def _fixture_audio_duration(path: Path) -> float:
    try:
        return FIXTURE_DURATIONS[path.stem]
    except KeyError as exc:
        raise AssertionError(f"unexpected fixture audio path: {path.name}") from exc


def _materialize_librispeech_manifest(root: Path, manifest: Path) -> None:
    shutil.copyfile(
        Path(benchmark_runner.__file__).resolve(),
        root / "src/faster_glm_asr/benchmarking/runner.py",
    )
    provenance_source = root / "src/faster_glm_asr/benchmarking/provenance.py"
    provenance_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(provenance.__file__).resolve(), provenance_source)
    test_clean_root = root / public_matrix.DATASET_EXTRACTED_ROOT / "test-clean"
    test_clean_root.mkdir(parents=True)
    license_path = test_clean_root.parent / "LICENSE.TXT"
    license_path.write_bytes(b"LibriSpeech license fixture\n")
    license_hash = hashlib.sha256(license_path.read_bytes()).hexdigest()
    selections = (
        ("short", 5.0, "100-200-0001"),
        ("medium", 15.0, "101-201-0002"),
        ("near-30s", 25.0, "102-202-0003"),
    )
    for _bucket, _duration, upstream_id in selections:
        speaker_id, chapter_id, _utterance_id = upstream_id.split("-", 2)
        chapter = test_clean_root / speaker_id / chapter_id
        chapter.mkdir(parents=True)
        (chapter / f"{speaker_id}-{chapter_id}.trans.txt").write_text(
            f"{upstream_id} PRIVATE REFERENCE {upstream_id}\n",
            encoding="utf-8",
        )
        audio = chapter / f"{upstream_id}.flac"
        audio.write_bytes(b"fLaC" + upstream_id.encode("ascii") + b"\x00fixture-audio")

    with mock.patch.object(librispeech, "_audio_duration", side_effect=_fixture_audio_duration):
        candidates = librispeech.discover(test_clean_root)
    selected = librispeech.select(
        candidates,
        per_bucket=1,
        seed=librispeech.DEFAULT_SEED,
    )
    rows = librispeech._manifest_records(selected, manifest.parent, license_hash)
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    manifest.write_text(manifest_text, encoding="utf-8", newline="\n")
    selection_lock = {
        "schema_version": librispeech.SCHEMA_VERSION,
        "selection_version": librispeech.SELECTION_VERSION,
        "seed": librispeech.DEFAULT_SEED,
        "per_bucket": 1,
        "duration_buckets": [
            {
                "name": name,
                "lower_s_inclusive": lower,
                "upper_s": upper,
                "upper_inclusive": inclusive,
            }
            for name, lower, upper, inclusive in librispeech.BUCKETS
        ],
        "candidate_counts": {
            name: sum(item.duration_bucket == name for item in candidates)
            for name, _lower, _upper, _inclusive in librispeech.BUCKETS
        },
        "candidate_inventory_sha256": librispeech._canonical_sha256(
            [
                {
                    "sample_id": item.sample_id,
                    "audio_sha256": item.audio_sha256,
                    "duration_s": round(item.duration_s, 9),
                    "duration_bucket": item.duration_bucket,
                    "transcript_file_sha256": item.transcript_file_sha256,
                }
                for item in sorted(candidates, key=lambda value: value.sample_id)
            ]
        ),
        "selected_upstream_ids": [row["upstream_utterance_id"] for row in rows],
        "selected_count": len(rows),
        "license_file_name": license_path.name,
        "license_file_sha256": license_hash,
        "upstream_url": librispeech.UPSTREAM_URL,
        "upstream_license_declared": librispeech.UPSTREAM_LICENSE,
        "manifest_file": "manifest.jsonl",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "builder_source_sha256": hashlib.sha256(
            (root / public_matrix.LIBRISPEECH_BUILDER_PATH).read_bytes()
        ).hexdigest(),
        "audio_copied": False,
        "publication_review_required": True,
    }
    (manifest.parent / "selection-lock.json").write_text(
        json.dumps(selection_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_config = root / public_matrix.CANONICAL_DATA_CONFIG_PATH
    data_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPOSITORY_ROOT / public_matrix.CANONICAL_DATA_CONFIG_PATH, data_config)

    archive_path = root / public_matrix.DATASET_ARCHIVE_PATH
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    def normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        return info

    dataset_root = test_clean_root.parent
    with (
        archive_path.open("wb") as raw_stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        archive.add(
            dataset_root,
            arcname="LibriSpeech",
            recursive=False,
            filter=normalize_tar_info,
        )
        for path in sorted(dataset_root.rglob("*")):
            archive.add(
                path,
                arcname=path.relative_to(dataset_root.parent).as_posix(),
                recursive=False,
                filter=normalize_tar_info,
            )
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive_member_count = len(archive.getmembers())
    archive_contract = {
        "size_bytes": archive_path.stat().st_size,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "member_count": archive_member_count,
    }
    configuration = json.loads(data_config.read_text(encoding="utf-8"))
    configuration.update(
        {
            "archive_size_bytes": archive_contract["size_bytes"],
            "archive_sha256": archive_contract["sha256"],
            "archive_member_count": archive_contract["member_count"],
        }
    )
    data_config.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_archive_member_and_rebind(
    root: Path,
    member: tarfile.TarInfo,
    payload: bytes = b"",
) -> None:
    archive_path = root / public_matrix.DATASET_ARCHIVE_PATH
    records: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(archive_path, mode="r:gz") as source:
        for existing in source.getmembers():
            stream = source.extractfile(existing) if existing.isfile() else None
            records.append((copy.copy(existing), stream.read() if stream is not None else None))
    temporary = archive_path.with_name("test-clean.rebuilt.tar.gz")
    with (
        temporary.open("wb") as raw_stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as rebuilt,
    ):
        for existing, data in records:
            rebuilt.addfile(existing, io.BytesIO(data) if data is not None else None)
        rebuilt.addfile(member, io.BytesIO(payload) if member.isfile() else None)
    temporary.replace(archive_path)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        member_count = len(archive.getmembers())
    contract = {
        "size_bytes": archive_path.stat().st_size,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "member_count": member_count,
    }
    data_config = root / public_matrix.CANONICAL_DATA_CONFIG_PATH
    configuration = json.loads(data_config.read_text(encoding="utf-8"))
    configuration.update(
        {
            "archive_size_bytes": contract["size_bytes"],
            "archive_sha256": contract["sha256"],
            "archive_member_count": contract["member_count"],
        }
    )
    data_config.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class PublicMatrixFixture(MatrixFixture):
    def __init__(self, *, mutator=None, provenance_mutator=None) -> None:
        def manifest_setup(root: Path, manifest: Path) -> None:
            _materialize_librispeech_manifest(root, manifest)
            if provenance_mutator is not None:
                provenance_mutator(root, manifest)

        super().__init__(
            mutator=mutator,
            dataset_label=public_matrix.EXPECTED_DATASET_LABEL,
            model_name=DEFAULT_MODEL_ID,
            model_revision=DEFAULT_MODEL_REVISION,
            manifest_relative=(
                "artifacts/private/datasets/librispeech-test-clean-3-v1/manifest.jsonl"
            ),
            manifest_setup=manifest_setup,
        )
        data_configuration = json.loads(
            (self.root / public_matrix.CANONICAL_DATA_CONFIG_PATH).read_text(encoding="utf-8")
        )
        self.archive_contract = {
            "size_bytes": data_configuration["archive_size_bytes"],
            "sha256": data_configuration["archive_sha256"],
            "member_count": data_configuration["archive_member_count"],
        }
        self.aggregate_directory = self.root / "artifacts/private/aggregated"
        aggregation.aggregate_matrix(
            self.plan_path,
            self.journal_path,
            repository_root=self.root,
            output_directory=self.aggregate_directory,
            repository_state=self.repository_state,
        )

    def _environment(self) -> dict:
        environment = super()._environment()
        environment.update(
            {
                "device_name": "NVIDIA GeForce RTX 3090",
                "device_uuid": "GPU-private-fixture-identifier",
                "cuda_visible_devices": "GPU-private-fixture-identifier",
                "compute_capability": [8, 6],
                "total_memory_bytes": 24 * 1024**3,
                "cudnn": 91002,
                "package_versions": {
                    "torch": "2.10.0+cu128",
                    "triton": "3.6.0",
                    "transformers": "5.14.0",
                    "accelerate": "1.12.0",
                    "huggingface-hub": "1.5.0",
                    "safetensors": "0.8.0",
                    "numpy": "2.3.5",
                    "scipy": "1.17.0",
                    "soundfile": "0.13.1",
                    "nvidia-ml-py": "13.610.43",
                },
            }
        )
        return environment

    def _artifact(self, task: dict) -> dict:
        payload = super()._artifact(task)
        original = payload["samples"][0]
        samples = []
        manifest_rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        excluded_metadata = {
            "sample_id",
            "audio_path",
            "audio_sha256",
            "reference_text",
            "language",
            "duration_s",
            "shareable",
        }
        for index, row in enumerate(manifest_rows):
            duration = row["duration_s"]
            sample = copy.deepcopy(original)
            sample["sample_id"] = row["sample_id"]
            sample["audio_sha256"] = row["audio_sha256"]
            sample["language"] = row["language"]
            sample["duration_s"] = duration
            sample["metadata"] = {
                **{key: value for key, value in row.items() if key not in excluded_metadata},
                "shareable": row["shareable"],
            }
            sample["hypothesis"] = f"private fixture transcript {index + 1}"
            hypothesis_hash = hashlib.sha256(sample["hypothesis"].encode("utf-8")).hexdigest()
            sample["hypothesis_sha256"] = hypothesis_hash
            token_ids = [7, 8 + index]
            token_hash = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            sample["generated_token_ids"] = token_ids
            sample["generated_tokens"] = len(token_ids)
            sample["generated_token_ids_sha256"] = token_hash
            sample["prepared_request"]["tensors"]["input_features_mask"] = {
                "shape": [1, 100],
                "dtype": (
                    "torch.float32"
                    if task["implementation"].startswith("custom_")
                    else "torch.int32"
                ),
                "device": "cuda:0",
                "content_sha256": "3" * 64,
            }
            sample["prepared_request"]["valid_feature_frames"] = 100
            for tensor in sample["prepared_request"]["tensors"].values():
                tensor["content_sha256"] = str(index + 1) * 64

            observation = sample["observations"][0]
            generation = float(observation["generation_ms"]) + index
            preprocess = float(observation["preprocess_ms"])
            decode = float(observation["decode_ms"])
            observation.update(
                {
                    "generation_ms": generation,
                    "processor_plus_generation_ms": preprocess + generation,
                    "request_no_file_io_ms": preprocess + generation + decode,
                    "generated_tokens": len(token_ids),
                    "hypothesis_sha256": hypothesis_hash,
                    "generated_token_ids_sha256": token_hash,
                    "rtf_generation": generation / 1000.0 / duration,
                    "rtf_request_no_file_io": (
                        (preprocess + generation + decode) / 1000.0 / duration
                    ),
                }
            )
            memory = observation["memory_bytes"]
            memory["after_generation"] = {
                "allocated": 30 + index,
                "reserved": 40 + index,
            }
            memory["generation_allocated_peak"] = 35 + index
            memory["generation_reserved_peak"] = 45 + index
            memory["generation_allocated_peak_delta"] = 15 + index
            nvml = observation["nvml_process_memory"]
            if nvml["enabled"]:
                nvml["samples"] = [
                    {"elapsed_ms": 0.0, "used_bytes": 100 + index},
                    {"elapsed_ms": 1.0, "used_bytes": 120 + index},
                ]
                nvml["first_used_bytes"] = 100 + index
                nvml["last_used_bytes"] = 120 + index
                nvml["observed_peak_used_bytes"] = 120 + index
            phases = observation["diagnostic_cuda_phases"]
            if phases is not None:
                phases["decode_step"] = {
                    "events_ms": [generation],
                    "summary_ms": _summary([generation]),
                    "total_ms": generation,
                }
            sample["summary"] = {
                metric: _summary([float(observation[metric])])
                for metric in aggregation.SUMMARY_METRICS
            }
            samples.append(sample)

        payload["samples"] = samples
        payload["aggregate"] = {
            metric: _summary(
                [
                    float(observation[metric])
                    for sample in samples
                    for observation in sample["observations"]
                ]
            )
            for metric in aggregation.AGGREGATE_METRICS
        }
        payload["aggregate"]["quality"] = _quality()
        return payload


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


class PublicMatrixSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        duration_patch = mock.patch.object(
            librispeech,
            "_audio_duration",
            side_effect=_fixture_audio_duration,
        )
        duration_patch.start()
        self.addCleanup(duration_patch.stop)
        self.fixture = PublicMatrixFixture()
        self.addCleanup(self.fixture.close)
        archive_contract_patch = mock.patch.object(
            public_matrix,
            "DATASET_ARCHIVE_CONTRACT",
            self.fixture.archive_contract,
        )
        archive_contract_patch.start()
        self.addCleanup(archive_contract_patch.stop)

    def _build(self) -> dict:
        with mock.patch.object(
            public_matrix,
            "DATASET_ARCHIVE_CONTRACT",
            self.fixture.archive_contract,
        ):
            return public_matrix.build_public_summary(
                self.fixture.plan_path,
                self.fixture.journal_path,
                self.fixture.aggregate_directory,
                evidence_root=self.fixture.root,
                repository_state=self.fixture.repository_state,
            )

    def test_rebuilds_minimal_allowlisted_summary_and_preserves_negative_speedup(self) -> None:
        with mock.patch.object(
            public_matrix.comparator,
            "compare",
            wraps=public_matrix.comparator.compare,
        ) as compare:
            summary = self._build()
        self.assertEqual(compare.call_count, len(public_matrix.COMPARISON_PAIRS))
        self.assertEqual(public_matrix.validate_public_summary(summary), [])
        self.assertEqual(summary["hardware"]["accelerator_model"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(summary["hardware"]["nominal_memory_gib"], 24)
        self.assertEqual(summary["software"]["packages"]["torch"], "2.10.0")
        self.assertEqual(
            set(summary["evidence"]),
            {
                "git_commit",
                "source_map_sha256",
                "plan_sha256",
                "journal_sha256",
                "aggregate_set_sha256",
                "dataset_archive_sha256",
                "exporter_source_sha256",
                "schema_source_sha256",
            },
        )
        self.assertEqual(summary["evidence"]["git_commit"], "d" * 40)
        self.assertEqual(
            summary["evidence"]["dataset_archive_sha256"],
            self.fixture.archive_contract["sha256"],
        )
        self.assertTrue(
            all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value)
                for value in summary["evidence"].values()
            )
        )
        self.assertEqual(len(summary["results"]), 6)
        self.assertTrue(all(len(row["duration_buckets"]) == 3 for row in summary["results"]))
        self.assertEqual(len(summary["comparisons"]), 5)
        self.assertTrue(all(row["exact_token_parity"] for row in summary["comparisons"]))
        self.assertTrue(
            any(
                bucket["generation_p95_speedup"] < bucket["generation_p50_speedup"]
                for comparison in summary["comparisons"]
                for bucket in comparison["duration_buckets"]
            )
        )
        final_pair = summary["comparisons"][-1]
        self.assertEqual(
            (final_pair["baseline"], final_pair["candidate"]),
            ("custom_full_prefix", "custom_static_cache"),
        )
        self.assertLess(final_pair["duration_buckets"][0]["generation_p50_speedup"], 1.0)
        self.assertTrue(
            summary["results"][0]["duration_buckets"][0]["request_memory"][
                "process_peak_observed_is_lower_bound"
            ]
        )

        keys = _all_keys(summary)
        self.assertTrue(keys.isdisjoint(public_matrix.FORBIDDEN_PUBLIC_KEYS))
        serialized = json.dumps(summary, sort_keys=True)
        for private_value in (
            "librispeech-100-200-0001",
            "PRIVATE REFERENCE 100-200-0001",
            "private fixture transcript",
            "GPU-private-fixture-identifier",
            "fixture-linux",
            "artifacts/private",
            "[7, 8]",
        ):
            self.assertNotIn(private_value, serialized)

    def test_prepared_requests_are_exact_only_within_runtime_families(self) -> None:
        self.assertEqual(
            public_matrix.PREPARED_REQUEST_FAMILIES,
            {
                "hf": ("hf_cached", "hf_no_cache"),
                "custom": (
                    "custom_full_prefix",
                    "custom_greedy_full_prefix",
                    "custom_tuple_cache",
                    "custom_static_cache",
                ),
            },
        )
        family_dtypes = {}
        for implementation in ("hf_cached", "custom_full_prefix"):
            filename = aggregation.FORMAL_OUTPUT_FILENAMES[(implementation, "canonical_latency")]
            payload = json.loads((self.fixture.aggregate_directory / filename).read_bytes())
            family_dtypes[implementation] = payload["samples"][0]["prepared_request"]["tensors"][
                "input_features_mask"
            ]["dtype"]
        self.assertEqual(family_dtypes["hf_cached"], "torch.int32")
        self.assertEqual(family_dtypes["custom_full_prefix"], "torch.float32")
        self.assertEqual(public_matrix.validate_public_summary(self._build()), [])

    def test_rejects_prepared_request_drift_within_each_runtime_family(self) -> None:
        self.fixture.close()
        for implementation, family, drift_dtype in (
            ("hf_no_cache", "hf", "torch.int64"),
            ("custom_tuple_cache", "custom", "torch.float64"),
        ):
            with self.subTest(implementation=implementation):

                def mutate_prepared_request(
                    task: dict,
                    payload: dict,
                    target: str = implementation,
                    replacement: str = drift_dtype,
                ) -> None:
                    if task["implementation"] != target:
                        return
                    for sample in payload["samples"]:
                        sample["prepared_request"]["tensors"]["input_features_mask"]["dtype"] = (
                            replacement
                        )

                fixture = PublicMatrixFixture(mutator=mutate_prepared_request)
                self.fixture = fixture
                try:
                    with self.assertRaisesRegex(
                        ValueError, f"prepared request differs within {family} family"
                    ):
                        self._build()
                finally:
                    fixture.close()

    def test_rejects_missing_extra_and_changed_aggregate_artifacts(self) -> None:
        filename = aggregation.FORMAL_OUTPUT_FILENAMES[("hf_cached", "canonical_latency")]
        target = self.fixture.aggregate_directory / filename
        original = target.read_bytes()
        target.unlink()
        with self.assertRaisesRegex(ValueError, "exactly the fixed 18"):
            self._build()

        target.write_bytes(original)
        extra = self.fixture.aggregate_directory / "unexpected.json"
        extra.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly the fixed 18"):
            self._build()
        extra.unlink()

        changed = json.loads(original)
        changed["hostname"] = "private-host"
        target.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not equal the validated rebuild"):
            self._build()

        changed = json.loads(original)
        changed["environment"]["device_index"] = False
        target.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not equal the validated rebuild"):
            self._build()

    def test_rejects_matrix_wide_token_or_bucket_drift(self) -> None:
        self.fixture.close()

        def mutate_tokens(task: dict, payload: dict) -> None:
            if task["implementation"] == "custom_static_cache":
                for sample in payload["samples"]:
                    sample["generated_token_ids"] = [7, 99]
                    digest = hashlib.sha256(b"[7,99]").hexdigest()
                    sample["generated_token_ids_sha256"] = digest
                    sample["observations"][0]["generated_token_ids_sha256"] = digest

        token_fixture = PublicMatrixFixture(mutator=mutate_tokens)
        self.fixture = token_fixture
        try:
            with self.assertRaisesRegex(ValueError, "token output differs"):
                self._build()
        finally:
            token_fixture.close()

        def mutate_buckets(_task: dict, payload: dict) -> None:
            payload["samples"][1]["metadata"]["duration_bucket"] = "short"

        bucket_fixture = PublicMatrixFixture(mutator=mutate_buckets)
        self.fixture = bucket_fixture
        try:
            with self.assertRaisesRegex(ValueError, "duration does not match"):
                self._build()
        finally:
            bucket_fixture.close()

        def mutate_metadata(task: dict, payload: dict) -> None:
            if task["implementation"] == "custom_static_cache":
                payload["samples"][0]["metadata"]["speaker_id"] = "999"

        metadata_fixture = PublicMatrixFixture(mutator=mutate_metadata)
        self.fixture = metadata_fixture
        try:
            with self.assertRaisesRegex(ValueError, "identity differs"):
                self._build()
        finally:
            metadata_fixture.close()

    def test_rejects_raw_memory_beyond_physical_and_recomputable_inconsistency(self) -> None:
        self.fixture.close()

        def beyond_physical(_task: dict, payload: dict) -> None:
            for sample in payload["samples"]:
                memory = sample["observations"][0]["memory_bytes"]
                memory["before_prepare"] = {
                    "allocated": 24 * 1024**3 + 1,
                    "reserved": 24 * 1024**3 + 1,
                }

        fixture = PublicMatrixFixture(mutator=beyond_physical)
        self.fixture = fixture
        try:
            with self.assertRaisesRegex(ValueError, "outside physical device memory"):
                self._build()
        finally:
            fixture.close()

        def allocated_above_reserved(_task: dict, payload: dict) -> None:
            for sample in payload["samples"]:
                sample["observations"][0]["memory_bytes"]["before_prepare"] = {
                    "allocated": 21,
                    "reserved": 20,
                }

        fixture = PublicMatrixFixture(mutator=allocated_above_reserved)
        self.fixture = fixture
        try:
            with self.assertRaisesRegex(ValueError, "allocated exceeds reserved"):
                self._build()
        finally:
            fixture.close()

        def allocator_peak_inconsistent(_task: dict, payload: dict) -> None:
            for sample in payload["samples"]:
                memory = sample["observations"][0]["memory_bytes"]
                memory["generation_allocated_peak"] = 50
                memory["generation_reserved_peak"] = 45
                memory["generation_allocated_peak_delta"] = 30

        fixture = PublicMatrixFixture(mutator=allocator_peak_inconsistent)
        self.fixture = fixture
        try:
            with self.assertRaisesRegex(ValueError, "allocated peak exceeds reserved peak"):
                self._build()
        finally:
            fixture.close()

        def nvml_summary_inconsistent(task: dict, payload: dict) -> None:
            if task["measurement_profile"] != "request_memory":
                return
            for sample in payload["samples"]:
                sample["observations"][0]["nvml_process_memory"]["first_used_bytes"] = 101

        fixture = PublicMatrixFixture(mutator=nvml_summary_inconsistent)
        self.fixture = fixture
        try:
            with self.assertRaisesRegex(ValueError, "NVML process memory summary is inconsistent"):
                self._build()
        finally:
            fixture.close()

    def test_rejects_evidence_source_map_and_git_state_drift(self) -> None:
        for label, _module, relative in public_matrix.CORE_SOURCE_MODULES:
            with self.subTest(core_source=label):
                source = self.fixture.root / relative
                original = source.read_bytes()
                source.write_bytes(original + b"# drift\n")
                with self.assertRaisesRegex(ValueError, f"executing {label} source differs"):
                    self._build()
                source.write_bytes(original)

        added_source = (
            self.fixture.root / "src/faster_glm_asr/benchmarking/exporter-added-after-evidence.py"
        )
        added_source.write_text("# evidence source drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "binding drift"):
            self._build()
        added_source.unlink()

        changed_state = {"commit": "e" * 40, "dirty": False}
        with self.assertRaisesRegex(ValueError, "Git commit/dirty state drift"):
            public_matrix.build_public_summary(
                self.fixture.plan_path,
                self.fixture.journal_path,
                self.fixture.aggregate_directory,
                evidence_root=self.fixture.root,
                repository_state=changed_state,
            )

    def test_rejects_manifest_selection_lock_config_and_anchor_drift(self) -> None:
        selection_path = self.fixture.manifest.parent / "selection-lock.json"
        selection_original = selection_path.read_bytes()
        mutations = {
            "manifest_sha256": lambda lock: lock.update(manifest_sha256="f" * 64),
            "candidate_counts": lambda lock: lock["candidate_counts"].update(short=123),
            "candidate_inventory_sha256": lambda lock: lock.update(
                candidate_inventory_sha256="a" * 64
            ),
            "license_file_name": lambda lock: lock.update(license_file_name="UNRELATED.TXT"),
        }
        # These post-plan mutations change reconstructed data semantics and
        # remain inadmissible even though no new plan is built for them.
        for label, mutate in mutations.items():
            with self.subTest(selection_lock=label):
                selection = json.loads(selection_original)
                mutate(selection)
                selection_path.write_text(json.dumps(selection), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"selection lock {label} binding"):
                    self._build()
                selection_path.write_bytes(selection_original)

        selection = json.loads(selection_original)
        selection["builder_source_sha256"] = "not-a-sha256"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "builder_source_sha256 is invalid"):
            self._build()
        selection_path.write_bytes(selection_original)

        data_path = self.fixture.root / public_matrix.CANONICAL_DATA_CONFIG_PATH
        data_original = data_path.read_bytes()
        data = json.loads(data_original)
        data["archive_member_count"] = True
        data_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not the admitted revision"):
            self._build()
        data_path.write_bytes(data_original)

        self.fixture.close()

        def mutate_anchor(_task: dict, payload: dict) -> None:
            payload["samples"][0]["audio_sha256"] = "f" * 64

        anchor_fixture = PublicMatrixFixture(mutator=mutate_anchor)
        self.fixture = anchor_fixture
        with self.assertRaisesRegex(ValueError, "anchor does not match manifest row"):
            self._build()

    def test_requires_fixed_complete_safe_archive_and_exact_extracted_tree(self) -> None:
        archive_path = self.fixture.root / public_matrix.DATASET_ARCHIVE_PATH
        archive_original = archive_path.read_bytes()
        archive_path.write_bytes(archive_original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "archive size/SHA-256 binding"):
            self._build()
        archive_path.write_bytes(archive_original)

        self.assertEqual(
            public_matrix.DATASET_EXTRACTED_ROOT.as_posix(),
            "artifacts/private/cache/datasets/librispeech/extracted-v1/LibriSpeech",
        )
        test_clean = self.fixture.root / public_matrix.DATASET_EXTRACTED_ROOT / "test-clean"
        extra = test_clean / "unexpected.private"
        extra.write_bytes(b"private")
        with self.assertRaisesRegex(ValueError, "extra or missing"):
            self._build()
        extra.unlink()

        transcript = test_clean / "100/200/100-200.trans.txt"
        transcript_original = transcript.read_bytes()
        transcript.unlink()
        with self.assertRaisesRegex(ValueError, "extra or missing"):
            self._build()
        transcript.write_bytes(transcript_original)

        self.fixture.close()
        cases = (
            ("absolute", "/absolute/private", tarfile.REGTYPE, "", b"x", "unsafe member"),
            ("traversal", "../escape", tarfile.REGTYPE, "", b"x", "unsafe member"),
            (
                "duplicate",
                "LibriSpeech/LICENSE.TXT",
                tarfile.REGTYPE,
                "",
                b"duplicate",
                "duplicate member",
            ),
            (
                "symbolic-link",
                "LibriSpeech/private-link",
                tarfile.SYMTYPE,
                "LICENSE.TXT",
                b"",
                "link or special",
            ),
            (
                "hard-link",
                "LibriSpeech/private-hardlink",
                tarfile.LNKTYPE,
                "LibriSpeech/LICENSE.TXT",
                b"",
                "link or special",
            ),
            (
                "device",
                "LibriSpeech/private-device",
                tarfile.CHRTYPE,
                "",
                b"",
                "link or special",
            ),
            (
                "wav",
                "LibriSpeech/test-clean/100/200/private.wav",
                tarfile.REGTYPE,
                "",
                b"RIFFprivate",
                "non-FLAC asset",
            ),
        )
        for label, name, member_type, linkname, payload, error in cases:
            with self.subTest(archive_member=label):

                def mutate_archive(
                    root: Path,
                    _manifest: Path,
                    *,
                    member_name: str = name,
                    type_code: bytes = member_type,
                    link_target: str = linkname,
                    data: bytes = payload,
                ) -> None:
                    member = tarfile.TarInfo(member_name)
                    member.type = type_code
                    member.linkname = link_target
                    member.size = len(data) if type_code == tarfile.REGTYPE else 0
                    _append_archive_member_and_rebind(root, member, data)

                fixture = PublicMatrixFixture(provenance_mutator=mutate_archive)
                self.fixture = fixture
                try:
                    with self.assertRaisesRegex(ValueError, error):
                        self._build()
                finally:
                    fixture.close()

    def test_accepts_historical_builder_audit_handle_and_rejects_manifest_forgery(self) -> None:
        self.fixture.close()

        def record_historical_builder_handle(_root: Path, manifest: Path) -> None:
            path = manifest.parent / "selection-lock.json"
            lock = json.loads(path.read_text(encoding="utf-8"))
            lock["builder_source_sha256"] = "f" * 64
            path.write_text(json.dumps(lock), encoding="utf-8")

        lock_fixture = PublicMatrixFixture(provenance_mutator=record_historical_builder_handle)
        self.fixture = lock_fixture
        summary = self._build()
        self.assertEqual(public_matrix.validate_public_summary(summary), [])
        lock_fixture.close()

        def forge_manifest_schema(_root: Path, manifest: Path) -> None:
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            rows[0]["unexpected_private_field"] = "forged"
            manifest.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            path = manifest.parent / "selection-lock.json"
            lock = json.loads(path.read_text(encoding="utf-8"))
            lock["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            path.write_text(json.dumps(lock), encoding="utf-8")

        manifest_fixture = PublicMatrixFixture(provenance_mutator=forge_manifest_schema)
        self.fixture = manifest_fixture
        with self.assertRaisesRegex(ValueError, "manifest row 1 schema"):
            self._build()

    def test_public_checker_rejects_unknown_private_fields_paths_and_nonfinite_values(self) -> None:
        summary = self._build()
        additions = {
            "sample_id": "sample-001",
            "transcript": "secret words",
            "generated_token_ids": [7, 8],
            "audio_sha256": "e" * 64,
            "manifest_sha256": "f" * 64,
            "prepared_request": {},
            "branch": "agent/private",
            "device_uuid": "GPU-private-fixture-identifier",
            "bdf": "0000:01:00.0",
            "cuda_visible_devices": "0",
            "hostname": "private-host",
            "command_argv": ["python", "private.py"],
            "logs": ["private.log"],
            "wer": 0.1,
        }
        for key, value in additions.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(summary)
                changed[key] = value
                errors = public_matrix.validate_public_summary(changed)
                self.assertTrue(errors)
                self.assertTrue(any("schema mismatch" in error for error in errors))

        absolute = copy.deepcopy(summary)
        absolute["software"]["python"] = "/" + "ho" + "me/private/venv/bin/python"
        errors = public_matrix.validate_public_summary(absolute)
        self.assertTrue(any("absolute path" in error for error in errors))

        nonfinite = copy.deepcopy(summary)
        nonfinite["results"][0]["duration_buckets"][0]["canonical"]["generation_p50_ms"] = math.nan
        self.assertTrue(public_matrix.validate_public_summary(nonfinite))

        duplicate = self.fixture.root / "artifacts/private/duplicate.json"
        duplicate.write_text('{"schema_version":"one","schema_version":"two"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            public_matrix._load_json_object(duplicate, "duplicate fixture")

    def test_checker_recomputes_all_speedups_and_enforces_ordered_float_metrics(self) -> None:
        summary = self._build()
        for comparison_index in range(len(public_matrix.COMPARISON_PAIRS)):
            for key in ("generation_p50_speedup", "generation_p95_speedup"):
                for delta in (-1e-13, 1e-13):
                    with self.subTest(comparison=comparison_index, key=key, delta=delta):
                        changed = copy.deepcopy(summary)
                        bucket = changed["comparisons"][comparison_index]["duration_buckets"][0]
                        bucket[key] += delta
                        errors = public_matrix.validate_public_summary(changed)
                        self.assertTrue(
                            any("does not match result rows" in error for error in errors)
                        )

        p95_cases = (
            ("canonical", "generation_p50_ms", "generation_p95_ms"),
            ("canonical", "rtf_p50", "rtf_p95"),
            (
                "request_memory",
                "allocator_peak_delta_p50_mib",
                "allocator_peak_delta_p95_mib",
            ),
            (
                "request_memory",
                "process_peak_observed_p50_mib",
                "process_peak_observed_p95_mib",
            ),
        )
        for section, p50_key, p95_key in p95_cases:
            with self.subTest(section=section, p95=p95_key):
                changed = copy.deepcopy(summary)
                metrics = changed["results"][0]["duration_buckets"][0][section]
                metrics[p95_key] = metrics[p50_key] / 2.0
                errors = public_matrix.validate_public_summary(changed)
                self.assertTrue(any("p95" in error and ">=" in error for error in errors))

        integer_float = copy.deepcopy(summary)
        integer_float["protocol"]["request_memory"]["poll_interval_ms"] = 10
        self.assertTrue(public_matrix.validate_public_summary(integer_float))

        unsafe_integer = copy.deepcopy(summary)
        canonical = unsafe_integer["results"][0]["duration_buckets"][0]["canonical"]
        canonical["generation_p50_ms"] = 2**53 + 1
        canonical["generation_p95_ms"] = 2**53
        self.assertTrue(public_matrix.validate_public_summary(unsafe_integer))

        enormous_integer = copy.deepcopy(summary)
        enormous_integer["results"][0]["duration_buckets"][0]["canonical"]["generation_p50_ms"] = (
            10**400
        )
        self.assertTrue(public_matrix.validate_public_summary(enormous_integer))

    def test_checker_requires_canonical_numeric_and_pep440_versions(self) -> None:
        summary = self._build()
        cases = (
            ("python", "3.11.9rc1"),
            ("cuda_runtime", "12.8+private"),
            ("nvidia_driver_version", "570-private"),
        )
        for key, version in cases:
            with self.subTest(key=key):
                changed = copy.deepcopy(summary)
                changed["software"][key] = version
                self.assertTrue(public_matrix.validate_public_summary(changed))
        for version in ("0!2.10.0", "1!2.10.0", "2.10.0+cu128", "02.10.0"):
            with self.subTest(package_version=version):
                changed = copy.deepcopy(summary)
                changed["software"]["packages"]["torch"] = version
                self.assertTrue(public_matrix.validate_public_summary(changed))

    def test_export_check_and_publish_are_atomic_and_no_overwrite(self) -> None:
        nested_publication_root = self.fixture.root / "publication-checkout"
        nested_publication_root.mkdir()
        with self.assertRaisesRegex(ValueError, "separate, non-overlapping"):
            public_matrix.export_summary(
                self.fixture.plan_path,
                self.fixture.journal_path,
                self.fixture.aggregate_directory,
                nested_publication_root / "artifacts/private/public-summary.json",
                evidence_root=self.fixture.root,
                publication_root=nested_publication_root,
                repository_state=self.fixture.repository_state,
            )

        publication_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(publication_temporary.cleanup)
        publication_root = Path(publication_temporary.name)
        draft_relative = Path("artifacts/private/public-summary.json")
        draft = publication_root / draft_relative
        result = public_matrix.export_summary(
            self.fixture.plan_path.relative_to(self.fixture.root),
            self.fixture.journal_path.relative_to(self.fixture.root),
            self.fixture.aggregate_directory.relative_to(self.fixture.root),
            draft_relative,
            evidence_root=self.fixture.root,
            publication_root=publication_root,
            repository_state=self.fixture.repository_state,
        )
        self.assertEqual(result, draft)
        before = draft.read_bytes()
        self.assertEqual(public_matrix.main(["check", str(draft)]), 0)
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            public_matrix.export_summary(
                self.fixture.plan_path,
                self.fixture.journal_path,
                self.fixture.aggregate_directory,
                draft,
                evidence_root=self.fixture.root,
                publication_root=publication_root,
                repository_state=self.fixture.repository_state,
            )
        self.assertEqual(draft.read_bytes(), before)

        published = publication_root / "benchmarks/results/rtx3090-matrix.json"
        public_matrix.export_summary(
            self.fixture.plan_path,
            self.fixture.journal_path,
            self.fixture.aggregate_directory,
            published,
            evidence_root=self.fixture.root,
            publication_root=publication_root,
            publish=True,
            repository_state=self.fixture.repository_state,
        )
        published_before = published.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            public_matrix.export_summary(
                self.fixture.plan_path,
                self.fixture.journal_path,
                self.fixture.aggregate_directory,
                published,
                evidence_root=self.fixture.root,
                publication_root=publication_root,
                publish=True,
                repository_state=self.fixture.repository_state,
            )
        self.assertEqual(published.read_bytes(), published_before)

        with self.assertRaisesRegex(ValueError, "direct children"):
            public_matrix._destination(
                Path("docs/result.json"),
                publication_root=publication_root,
                public=True,
            )


if __name__ == "__main__":
    unittest.main()
