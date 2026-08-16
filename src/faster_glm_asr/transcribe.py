"""Transcribe one WAV file with a local GLM-ASR snapshot.

This module is intentionally a thin user-facing layer over the same inference
implementations used by the benchmark runner.  It performs no benchmarking or
artifact publication; it simply loads one local snapshot, runs deterministic
greedy generation, and returns the transcript.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from faster_glm_asr import DEFAULT_MODEL_REVISION

IMPLEMENTATIONS = (
    "hf-cached",
    "custom-greedy-full-prefix",
    "custom-tuple-cache",
    "custom-static-cache",
)
DEFAULT_IMPLEMENTATION = "custom-static-cache"
DEFAULT_MAX_NEW_TOKENS = 128
MAX_DIRECT_AUDIO_SECONDS = 600.0

_CUSTOM_IMPLEMENTATIONS = {
    "custom-greedy-full-prefix": "custom_greedy_full_prefix",
    "custom-tuple-cache": "custom_tuple_cache",
    "custom-static-cache": "custom_static_cache",
}
_IMMUTABLE_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


@dataclass(frozen=True)
class TranscriptionResult:
    """Result returned by :func:`transcribe_file`."""

    text: str
    implementation: str
    audio_duration_s: float
    generated_tokens: int
    token_ids: tuple[int, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _validate_inputs(
    model_snapshot: str | Path,
    audio_path: str | Path,
    implementation: str,
    revision: str,
    max_new_tokens: int,
) -> tuple[Path, Path]:
    snapshot = Path(model_snapshot).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()

    if not snapshot.is_dir():
        raise ValueError(f"model snapshot directory not found: {snapshot}")
    if not audio.is_file():
        raise ValueError(f"audio file not found: {audio}")
    if audio.suffix.lower() != ".wav":
        raise ValueError("audio input must be a WAV file")
    if implementation not in IMPLEMENTATIONS:
        choices = ", ".join(IMPLEMENTATIONS)
        raise ValueError(f"implementation must be one of: {choices}")
    if _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("revision must be a lowercase 40- or 64-character commit SHA")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    return snapshot, audio


def _create_runner(implementation: str, model_snapshot: Path, revision: str) -> Any:
    # Keep heavyweight model and benchmark-runner imports out of CLI help and
    # parser-only workflows.
    from faster_glm_asr.benchmarking.runner import CustomRunner, HuggingFaceRunner

    if implementation == "hf-cached":
        return HuggingFaceRunner(model_snapshot, use_cache=True)
    return CustomRunner(_CUSTOM_IMPLEMENTATIONS[implementation], model_snapshot, revision)


def transcribe_file(
    model_snapshot: str | Path,
    audio_path: str | Path,
    *,
    implementation: str = DEFAULT_IMPLEMENTATION,
    revision: str = DEFAULT_MODEL_REVISION,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    _runner_factory: Callable[[str, Path, str], Any] | None = None,
) -> TranscriptionResult:
    """Transcribe one WAV file using a local model snapshot.

    Loading is always offline: both the Hugging Face reference path and the
    custom paths pass ``local_files_only=True`` to their underlying loaders.
    Generation is deterministic greedy decoding on one CUDA device.

    Args:
        model_snapshot: Local directory containing the pinned Hugging Face
            snapshot.
        audio_path: WAV file to transcribe.  Audio is converted to mono and
            resampled to 16 kHz by the shared audio loader when needed.
        implementation: One of :data:`IMPLEMENTATIONS`.
        revision: Immutable upstream commit SHA associated with the snapshot;
            formal byte-level snapshot verification is handled by the benchmark
            workflow rather than this convenience API.
        max_new_tokens: Maximum number of transcript tokens to generate.

    Returns:
        The decoded text and generation metadata.
    """

    snapshot, audio_file = _validate_inputs(
        model_snapshot,
        audio_path,
        implementation,
        revision,
        max_new_tokens,
    )

    from faster_glm_asr.benchmarking.runner import _read_audio

    audio, duration_s = _read_audio(audio_file)
    if duration_s <= 0:
        raise ValueError("audio file contains no samples")
    if duration_s > MAX_DIRECT_AUDIO_SECONDS:
        raise ValueError(
            f"audio duration {duration_s:.1f}s exceeds the {MAX_DIRECT_AUDIO_SECONDS:g}s "
            "single-request limit; segment the recording before transcription"
        )

    factory = _runner_factory or _create_runner
    runner = factory(implementation, snapshot, revision)
    with runner.torch.inference_mode():
        prepared = runner.prepare(audio)
        output = runner.generate(prepared, max_new_tokens)
        text, token_ids = runner.decode(output, prepared)

    return TranscriptionResult(
        text=text,
        implementation=implementation,
        audio_duration_s=duration_s,
        generated_tokens=len(token_ids),
        token_ids=tuple(token_ids),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing Torch or Transformers."""

    parser = argparse.ArgumentParser(
        prog="faster-glm-asr-transcribe",
        description="Transcribe one WAV file with a local GLM-ASR snapshot.",
    )
    parser.add_argument("audio", type=Path, help="WAV file to transcribe")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        metavar="SNAPSHOT_DIR",
        help="local zai-org/GLM-ASR-Nano-2512 snapshot directory",
    )
    parser.add_argument(
        "--implementation",
        choices=IMPLEMENTATIONS,
        default=DEFAULT_IMPLEMENTATION,
        help=f"decode path (default: {DEFAULT_IMPLEMENTATION})",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help="immutable upstream commit SHA associated with the snapshot",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"maximum generated transcript tokens (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit structured JSON including generated token IDs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single-file transcription CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = transcribe_file(
            args.model,
            args.audio,
            implementation=args.implementation,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.json:
        print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
