# Faster GLM-ASR

[![CPU checks](https://github.com/Firefly0237/faster-glm-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/Firefly0237/faster-glm-asr/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A PyTorch/Triton inference path and reproducible benchmark suite for
[`zai-org/GLM-ASR-Nano-2512`](https://huggingface.co/zai-org/GLM-ASR-Nano-2512).

Faster GLM-ASR focuses on the autoregressive decode path: loading the pinned
official checkpoint into a custom model, making KV-cache behavior explicit, and
comparing alternative generation strategies under the same inputs and
measurement conditions. The goal is to study latency and GPU-memory efficiency
without changing the generated sequence.

The repository also includes long-audio preparation and evaluation utilities,
machine-readable benchmark artifacts, and a separate single-node distributed
training experiment for studying DDP systems behavior.

## Highlights

- A custom GLM-ASR model graph with strict loading for the pinned Transformers
  checkpoint and processor.
- Six named inference paths covering Hugging Face references, full-prefix
  decoding, dynamic tuple caches, and preallocated static caches.
- Torch reference implementations and Triton kernels for selected model
  operations.
- Paired correctness checks for prepared inputs, generated token IDs, and decoded
  hypotheses.
- Separate latency, memory, diagnostic-phase, and profiler runs to keep primary
  timing free from instrumentation overhead.
- Deterministic audio manifests, long-recording segmentation and stitching, and
  WER/CER evaluation.
- Reproducible environment, model, source, and input fingerprints in benchmark
  artifacts.
- Single-node DDP scaling, sampled byte-corpus validation, and checkpoint/replay
  diagnostics from four RTX 3090 GPUs.

## Quick start

The single-file CLI runs greedy transcription on one NVIDIA GPU. Start from a
Linux environment with Python 3.11 or 3.12 and a CUDA-compatible driver:

```bash
git clone https://github.com/Firefly0237/faster-glm-asr.git
cd faster-glm-asr
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=77" wheel
python -m pip install --no-build-isolation -e ".[gpu]"
```

Download the pinned model snapshot once. Inference uses this local directory
without a network fallback:

```bash
MODEL_DIR=artifacts/private/cache/model/glm-asr-nano-2512
hf download zai-org/GLM-ASR-Nano-2512 \
  --revision 61ba4e0b3309b6656edea3e93e419f7bd5c61957 \
  --local-dir "$MODEL_DIR"
```

Transcribe a WAV file with the static KV-cache path:

```bash
faster-glm-asr-transcribe path/to/sample.wav \
  --model "$MODEL_DIR" \
  --implementation custom-static-cache
```

The command prints the transcript. Add `--json` to include the implementation,
audio duration, generated-token count, and token IDs. The same interface also
offers `hf-cached`, `custom-greedy-full-prefix`, and `custom-tuple-cache` for
quick functional comparisons:

```bash
faster-glm-asr-transcribe path/to/sample.wav \
  --model "$MODEL_DIR" \
  --implementation hf-cached \
  --json
```

For controlled performance measurements, use the pinned environment and matrix
workflow in the [reproducibility guide](docs/reproducibility.md).

## Architecture

The runtime separates model semantics, generation policy, and measurement. This
keeps cache implementations independently testable and makes paired comparisons
easier to reproduce.

```mermaid
flowchart LR
    A[Audio + prompt] --> P[Pinned GLM-ASR processor]
    P --> H[Hugging Face references]
    P --> C[Custom Torch/Triton model]
    C --> F[Full-prefix decode]
    C --> T[Tuple KV cache]
    C --> S[Static KV cache]
    H & F & T & S --> G[Paired correctness checks]
    G --> M[Latency, memory, and quality artifacts]
```

The processor produces audio features, masks, and prompt tokens. The custom
runtime applies the audio encoder and projector, merges the projected audio with
text embeddings, and performs greedy autoregressive decoding through the text
decoder. See the [architecture guide](docs/architecture.md) for module boundaries
and the complete runtime data flow.

## Inference paths

| Implementation | Runtime | Prefix policy | KV storage |
|---|---|---|---|
| `hf_cached` | Transformers reference | Prefill once, then decode one token at a time | Framework cache |
| `hf_no_cache` | Transformers reference | Recompute the growing prefix | None |
| `custom_full_prefix` | Custom model | Recompute the growing prefix | None |
| `custom_greedy_full_prefix` | Custom model | Full-prefix decode with deterministic argmax | None |
| `custom_tuple_cache` | Custom model | Prefill once, then decode one token at a time | Per-layer tuple concatenation |
| `custom_static_cache` | Custom model | Prefill once, then decode one token at a time | Preallocated per-layer buffers |

The tuple and static paths share last-position logit projection and preallocated
token output. Their primary implementation difference is dynamic KV growth
versus in-place cache writes. `hf_no_cache` is a controlled ablation;
`hf_cached` is the standard cached reference. The custom runtime currently
targets single-request greedy decoding in FP32, with dense per-request storage
for the static cache.

## Benchmarking

For each admitted performance pair, the benchmark comparator requires identical
prepared-input contracts, generation settings, model bytes, and software
environments. Matrix-wide token parity also covers paths that are not admitted
as direct speedup pairs. The benchmark records:

- median and p95 request latency;
- real-time factor for speech workloads;
- framework allocator peaks and sampled process GPU memory;
- diagnostic CUDA-event phases and profiler metadata in separate runs;
- token and decoded-text parity between paired paths;
- WER/CER after source-level stitching for long audio.

### RTX 3090 inference results

Generation latency on the three-item LibriSpeech `test-clean` performance subset:

| Implementation | Short p50 (ms) | Medium p50 (ms) | Near-30s p50 (ms) |
|---|---:|---:|---:|
| `hf_cached` | 781.75 | 1,355.36 | 2,572.08 |
| `hf_no_cache` | 4,872.40 | 10,205.54 | 23,479.90 |
| `custom_full_prefix` | 1,154.78 | 3,123.37 | 8,204.01 |
| `custom_greedy_full_prefix` | 1,155.58 | 3,123.95 | 8,207.39 |
| `custom_tuple_cache` | 998.94 | 1,728.57 | 3,343.49 |
| `custom_static_cache` | 1,195.62 | 1,816.03 | 3,872.65 |

Method: one 24 GiB RTX 3090, FP32, batch size 1, concurrency 1, at most 128 new
tokens, and one LibriSpeech utterance in each duration bucket; three passes used
three warm-ups followed by ten measured requests per pass, giving 30 observations
per implementation and bucket, with canonical latency CUDA-synchronized around
generation and excluding preprocessing, output decoding, file I/O, and model
loading.

All 18 implementation/profile aggregates passed exact generated-token parity.
For the admitted within-family p50 comparisons, `hf_cached` produced
6.23×/7.53×/9.13× speedups over `hf_no_cache` across short/medium/near-30s;
`custom_tuple_cache` produced 1.16×/1.81×/2.45× over
`custom_greedy_full_prefix`. Static versus tuple cache measured
0.84×/0.95×/0.86×, so this static-cache prototype was slower in all three
buckets; relative to custom full-prefix decoding, it measured
0.97×/1.72×/2.12×. Hugging Face and custom requests use different prepared mask
dtypes, so no direct cross-family speedup is published.

See the [public result JSON](benchmarks/results/rtx3090-librispeech-test-clean-3-v1.json)
for the complete summary and the [benchmark contract](docs/benchmark-contract.md)
for comparison rules and metric boundaries. The
[reproducibility guide](docs/reproducibility.md) covers data preparation,
execution, aggregation, and artifact validation.

## Distributed-training experiment

The scaling and byte-corpus runs at revision `1d6e9e3` used one PHB-connected
node with four 24 GiB RTX 3090 GPUs, Python 3.11.8, PyTorch 2.10.0+cu128, CUDA
12.8, and NVIDIA driver 595.71.05. The 403,097,088-parameter BF16/FP32 workload
fixed 32,768 tokens per update; each world size contributed 90 post-warmup
observations across three interleaved trials.

| GPUs | Median tokens/s | p05–p95 tokens/s | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 14,212.555 | 14,156.341–14,358.586 | 1.0000× | 100.000% |
| 2 | 24,901.857 | 24,853.724–25,025.829 | 1.7521× | 87.605% |
| 4 | 30,708.305 | 30,434.381–31,002.553 | 2.16065× | 54.016% |

An independent 275,621,120-parameter decoder run on a TinyStories-derived byte
corpus reached sampled validation loss 1.1350639 at update 200 (byte-level
perplexity 3.11137); each evaluation sampled 65,536 next-byte targets from the
disjoint validation file. In the interruption experiment, the fixed-bucket
diagnostic matched all eight updates exactly, a result consistent with the DDP
reducer lifecycle affecting the resume boundary but not evidence of a unique
cause. The default and fixed-bucket replay diagnostics used independently
source-locked revisions `49864c9` and `71f6893`, respectively.

See [Distributed training](docs/distributed-training.md) for the experimental
contract, metric definitions, and reproduction workflow.

## Repository layout

```text
faster-glm-asr/
├── src/faster_glm_asr/
│   ├── modeling/          # Model graph, generation, and checkpoint loading
│   ├── kernels/           # Torch implementations and Triton kernels
│   ├── benchmarking/      # Runners, comparisons, matrices, and artifacts
│   └── data/              # Audio manifests, segmentation, and stitching
├── experiments/               # Standalone distributed-training experiments
├── configs/                   # Benchmark, data, environment, and training configs
├── docs/                      # Architecture and reproducibility guides
├── tests/                     # Unit, integration, and contract tests
└── tools/                     # Environment, data, and release utilities
```

## Documentation

| Guide | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Runtime data flow, package boundaries, and cache policies |
| [Benchmark contract](docs/benchmark-contract.md) | Comparison rules, measurement profiles, and metrics |
| [Reproducibility](docs/reproducibility.md) | End-to-end benchmark preparation and execution |
| [RTX 3090 environment](docs/rtx3090-environment.md) | Four-GPU environment setup and acceptance checks |
| [Distributed training](docs/distributed-training.md) | Single-node DDP scaling, replay, and profiling experiment |
| [Provenance](docs/provenance.md) | Upstream sources, licenses, and adaptation records |
| [Documentation index](docs/README.md) | Complete guide index |

## Development

The CPU test suite covers model and cache invariants, checkpoint mapping,
deterministic data transforms, benchmark schemas, and CLI contracts. From a
configured development environment, run:

```bash
python -m pytest
python -m ruff check .
python -m build --no-isolation
```

Python 3.11 and 3.12 are exercised in CI. GPU execution uses the pinned CUDA
environment described in [RTX 3090 environment](docs/rtx3090-environment.md).

## Contributing

Contributions to model compatibility, cache correctness, kernels, benchmarking,
and audio tooling are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow, validation requirements, and provenance guidelines. Bug
reports and feature proposals can be opened through
[GitHub Issues](https://github.com/Firefly0237/faster-glm-asr/issues). Please use
the process in [SECURITY.md](SECURITY.md) for security reports.

## Acknowledgments

Faster GLM-ASR builds on the official GLM-ASR model interfaces and Hugging Face
Transformers compatibility behavior. Adapted components, upstream revisions,
datasets, and their licenses are recorded in [NOTICE](NOTICE) and the
[provenance guide](docs/provenance.md).

## License

Project-authored source is licensed under the
[Apache License 2.0](LICENSE). Model weights, datasets, and third-party
components retain their respective licenses; see [NOTICE](NOTICE).
