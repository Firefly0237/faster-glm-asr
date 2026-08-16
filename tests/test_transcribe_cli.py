from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from faster_glm_asr import DEFAULT_MODEL_REVISION
from faster_glm_asr import transcribe as transcribe_module
from faster_glm_asr.transcribe import TranscriptionResult, build_parser, transcribe_file


def _write_silent_wav(path: Path, *, seconds: float = 0.01) -> None:
    frames = max(1, round(16000 * seconds))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * frames)


def test_parser_exposes_four_community_facing_paths() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["sample.wav", "--model", "snapshot"])
    assert defaults.implementation == "custom-static-cache"
    for implementation in transcribe_module.IMPLEMENTATIONS:
        args = parser.parse_args(
            ["sample.wav", "--model", "snapshot", "--implementation", implementation]
        )
        assert args.implementation == implementation
        assert args.revision == DEFAULT_MODEL_REVISION
        assert args.max_new_tokens == 128


def test_runner_factory_maps_public_names_to_existing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from faster_glm_asr.benchmarking import runner as benchmark_runner

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[tuple[object, ...]] = []

    class FakeHuggingFaceRunner:
        def __init__(self, root: Path, use_cache: bool) -> None:
            calls.append(("hf", root, use_cache))

    class FakeCustomRunner:
        def __init__(self, implementation: str, root: Path, revision: str) -> None:
            calls.append(("custom", implementation, root, revision))

    monkeypatch.setattr(benchmark_runner, "HuggingFaceRunner", FakeHuggingFaceRunner)
    monkeypatch.setattr(benchmark_runner, "CustomRunner", FakeCustomRunner)

    transcribe_module._create_runner("hf-cached", snapshot, DEFAULT_MODEL_REVISION)
    for public_name in (
        "custom-greedy-full-prefix",
        "custom-tuple-cache",
        "custom-static-cache",
    ):
        transcribe_module._create_runner(public_name, snapshot, DEFAULT_MODEL_REVISION)

    assert calls == [
        ("hf", snapshot, True),
        ("custom", "custom_greedy_full_prefix", snapshot, DEFAULT_MODEL_REVISION),
        ("custom", "custom_tuple_cache", snapshot, DEFAULT_MODEL_REVISION),
        ("custom", "custom_static_cache", snapshot, DEFAULT_MODEL_REVISION),
    ]


def test_transcribe_dispatches_without_loading_real_model(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)
    calls: list[tuple[str, Path, str]] = []

    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeRunner:
        torch = SimpleNamespace(inference_mode=lambda: FakeInferenceMode())

        def prepare(self, audio: np.ndarray) -> dict[str, object]:
            assert audio.dtype == np.float32
            assert audio.flags.c_contiguous
            return {"input_ids": "prepared"}

        def generate(self, prepared: dict[str, object], max_new_tokens: int) -> str:
            assert prepared == {"input_ids": "prepared"}
            assert max_new_tokens == 7
            return "tokens"

        def decode(self, output: str, prepared: dict[str, object]) -> tuple[str, list[int]]:
            assert output == "tokens"
            assert prepared == {"input_ids": "prepared"}
            return "hello world", [10, 11]

    def fake_factory(implementation: str, root: Path, revision: str) -> FakeRunner:
        calls.append((implementation, root, revision))
        return FakeRunner()

    result = transcribe_file(
        snapshot,
        audio_path,
        implementation="custom-tuple-cache",
        max_new_tokens=7,
        _runner_factory=fake_factory,
    )

    assert result.text == "hello world"
    assert result.implementation == "custom-tuple-cache"
    assert result.audio_duration_s == pytest.approx(0.01)
    assert result.generated_tokens == 2
    assert result.token_ids == (10, 11)
    assert calls == [("custom-tuple-cache", snapshot.resolve(), DEFAULT_MODEL_REVISION)]


def test_main_prints_plain_text_and_forwards_cli_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)
    received: dict[str, object] = {}

    def fake_transcribe(model: Path, audio: Path, **kwargs: object) -> TranscriptionResult:
        received.update(model=model, audio=audio, **kwargs)
        return TranscriptionResult("hello", "hf-cached", 1.25, 1, (42,))

    monkeypatch.setattr(transcribe_module, "transcribe_file", fake_transcribe)

    assert (
        transcribe_module.main(
            [
                str(audio_path),
                "--model",
                str(snapshot),
                "--implementation",
                "hf-cached",
                "--max-new-tokens",
                "32",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "hello\n"
    assert received == {
        "model": snapshot,
        "audio": audio_path,
        "implementation": "hf-cached",
        "revision": DEFAULT_MODEL_REVISION,
        "max_new_tokens": 32,
    }


def test_main_json_output_is_machine_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)
    monkeypatch.setattr(
        transcribe_module,
        "transcribe_file",
        lambda *_args, **_kwargs: TranscriptionResult(
            "你好", "custom-static-cache", 0.5, 2, (7, 8)
        ),
    )

    assert transcribe_module.main([str(audio_path), "--model", str(snapshot), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "text": "你好",
        "implementation": "custom-static-cache",
        "audio_duration_s": 0.5,
        "generated_tokens": 2,
        "token_ids": [7, 8],
    }


@pytest.mark.parametrize(
    ("revision", "max_new_tokens", "message"),
    [
        ("main", 16, "revision must be"),
        (DEFAULT_MODEL_REVISION, 0, "max_new_tokens must be"),
    ],
)
def test_validation_fails_before_runner_creation(
    tmp_path: Path,
    revision: str,
    max_new_tokens: int,
    message: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)

    with pytest.raises(ValueError, match=message):
        transcribe_file(
            snapshot,
            audio_path,
            revision=revision,
            max_new_tokens=max_new_tokens,
            _runner_factory=lambda *_args: pytest.fail("runner must not be created"),
        )


@pytest.mark.parametrize("invalid", [True, 1.5, "8"])
def test_public_api_requires_integer_max_new_tokens(tmp_path: Path, invalid: Any) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)

    with pytest.raises(TypeError, match="max_new_tokens must be an integer"):
        transcribe_file(
            snapshot,
            audio_path,
            max_new_tokens=invalid,
            _runner_factory=lambda *_args: pytest.fail("runner must not be created"),
        )


def test_main_reports_user_facing_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_transcription(*_args: object, **_kwargs: object) -> TranscriptionResult:
        raise ValueError("snapshot is incomplete")

    monkeypatch.setattr(transcribe_module, "transcribe_file", fail_transcription)
    with pytest.raises(SystemExit) as raised:
        transcribe_module.main(["sample.wav", "--model", "snapshot"])

    assert raised.value.code == 2
    assert capsys.readouterr().err == "error: snapshot is incomplete\n"
