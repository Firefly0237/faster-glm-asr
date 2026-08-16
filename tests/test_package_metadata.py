from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import faster_glm_asr

EXPECTED_CLI_SCRIPTS = {
    "faster-glm-asr-aggregate": ("faster_glm_asr.benchmarking.matrix_aggregator:main"),
    "faster-glm-asr-benchmark": "faster_glm_asr.benchmarking.runner:main",
    "faster-glm-asr-build-long-audio": "faster_glm_asr.data.long_audio:main",
    "faster-glm-asr-build-manifest": "faster_glm_asr.data.audio_manifest:main",
    "faster-glm-asr-check-public": ("faster_glm_asr.benchmarking.public_artifact:main"),
    "faster-glm-asr-compare": "faster_glm_asr.benchmarking.comparator:main",
    "faster-glm-asr-plan": "faster_glm_asr.benchmarking.matrix_planner:main",
    "faster-glm-asr-prepare-librispeech": "faster_glm_asr.data.librispeech:main",
    "faster-glm-asr-run-matrix": ("faster_glm_asr.benchmarking.matrix_executor:main"),
    "faster-glm-asr-stitch": "faster_glm_asr.data.segment_stitch:main",
    "faster-glm-asr-transcribe": "faster_glm_asr.transcribe:main",
}


def test_version_and_model_revision_are_pinned() -> None:
    assert faster_glm_asr.__version__ == "0.1.0.dev0"
    assert faster_glm_asr.DEFAULT_MODEL_ID == "zai-org/GLM-ASR-Nano-2512"
    assert len(faster_glm_asr.DEFAULT_MODEL_REVISION) == 40
    int(faster_glm_asr.DEFAULT_MODEL_REVISION, 16)


def test_pyproject_uses_src_layout_and_readme() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["name"] == "faster-glm-asr"
    assert payload["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert payload["project"]["readme"] == "README.md"
    assert payload["project"]["license"] == "Apache-2.0"
    assert not any(
        classifier.startswith("License ::") for classifier in payload["project"]["classifiers"]
    )
    assert payload["project"]["license-files"] == ["LICENSE", "NOTICE"]


def test_published_cli_set_is_exact_and_resolves_to_callable_mains() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]
    assert scripts == EXPECTED_CLI_SCRIPTS

    for entry_point in scripts.values():
        module_name, separator, attribute_name = entry_point.partition(":")
        assert separator == ":"
        assert attribute_name == "main"
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute_name))
