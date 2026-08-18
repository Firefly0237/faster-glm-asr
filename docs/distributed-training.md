# Single-node decoder distributed-training experiment

This is a standalone decoder-only systems workload; it does not train GLM-ASR.
Synthetic byte streams provide controlled scaling and checkpoint/replay inputs,
while an independent TinyStories-derived byte corpus exercises the data,
training, checkpoint, and sampled-validation path.

## Results

The scaling, byte-corpus, and profiler measurements were collected at repository
revision `1d6e9e3` on one node with four 24 GiB RTX 3090 GPUs, Python 3.11.8,
PyTorch 2.10.0+cu128, CUDA 12.8, and NVIDIA driver 595.71.05. The GPUs shared a
PHB topology and exposed neither NVLink nor CUDA peer-to-peer access. The replay
diagnostics below used later source locks at revisions `49864c9` and `71f6893`.

### Strong scaling

All three world sizes used the 403,097,088-parameter configuration with a fixed
32,768 global tokens per optimizer update. Forward and backward used BF16
autocast, while parameters, gradients, AdamW state, and DDP reduction remained
FP32. For each world size, three interleaved trials ran 40 updates apiece; the
first 10 updates in every trial were warmup, yielding 90 eligible observations.

| World size | Median tokens/s | p05–p95 tokens/s | Speedup vs. 1 GPU | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 14,212.555 | 14,156.341–14,358.586 | 1.0000× | 100.000% |
| 2 | 24,901.857 | 24,853.724–25,025.829 | 1.7521× | 87.605% |
| 4 | 30,708.305 | 30,434.381–31,002.553 | 2.16065× | 54.016% |

The two- and four-GPU figures use the one-GPU median as their baseline. The
efficiency drop at four GPUs is consistent with communication occupying a larger
part of this fixed-workload step on the observed non-NVLink topology.

### Sampled byte-corpus validation

The separate TinyStories-derived byte-corpus run used four GPUs and the
275,621,120-parameter configuration for 200 optimizer updates. Sampled
validation loss was 2.64365 at update 20, 2.37070 at update 40, and 1.1350639 at
update 200; the last value corresponds to byte-level perplexity 3.11137. Each
evaluation consumes the next 128 deterministic pseudorandom windows, sampled
with replacement, representing 65,536 next-byte targets across four ranks from
a disjoint validation byte file pinned at upstream revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. These points therefore describe
sampled validation at each checkpoint, not a fixed exhaustive held-out-set
curve or a tokenizer-level language-model comparison.

### Checkpoint/replay diagnostic

The default DDP trajectory matched across two uninterrupted launches but showed
low-order gradient differences after checkpoint resume. The fixed-bucket
diagnostic, configured with `find_unused_parameters=True` and
`static_graph=False`, matched its uninterrupted control exactly across all
eight updates, including losses, gradient diagnostics, learning rates,
validation, final model and AdamW state, RNG state, and data cursors. The paired
result is consistent with the PyTorch reducer lifecycle affecting the resume
boundary; it does not isolate reducer bucket rebuilding as the unique cause.
The model has no unused parameters, and this diagnostic setting is excluded
from throughput measurements. The default diagnostic used revision `49864c9`;
the passing fixed-bucket diagnostic used revision `71f6893`. Its
`decoder-training-trajectory-comparison/1` artifact has SHA-256
`30a4fe0faf7630f3ffd6fdacd501f835be19c21dadcc7dfe8eb24b50ab6b203f`;
the corresponding source lock has SHA-256
`1e6d5d4f0066499db5873651c03c240cfa29a00bb4700eebdcd00d3b16b601f1`.

### Isolated profiler

The separate four-rank profile used the 275,621,120-parameter configuration.
Across three active steps, rank 0 recorded 93 NCCL FP32 ring LL kernels totaling
1.182699568 seconds of device duration; `ProfilerStep*` totaled 3.162648197
seconds of device duration, giving a 37.4% ratio. These are summed device-event
durations and may overlap with other device or host work, so the ratio is not a
wall-clock communication fraction and is not part of the throughput aggregate.
The denominator is the host-side `ProfilerStep*` aggregate row's
`device_time_total_us`. The NCCL numerator counts the 93 FP32
gradient-reduction kernels and excludes three uint32 finite-guard kernels.

## Setup

Each DDP rank holds a complete FP32 model, FP32 gradients, and two FP32 AdamW
moment buffers. The tensor-only capacity floor is about 6.45 GB for 403,097,088
parameters and 4.41 GB for 275,621,120 parameters, before activations, SDPA
workspaces, CUDA context, and allocator reservation. These are capacity
estimates; measured memory comes from the run artifacts.

The implementation is deliberately limited to single-node DDP/NCCL. It does not
implement or claim FSDP, tensor parallelism, pipeline parallelism, ZeRO, or
multi-node training.

### Model configurations

The primary model is exactly 403,097,088 unique trainable parameters:

| Component | Configuration |
|---|---|
| decoder | 16 layers, width 1536 |
| attention | 24 query heads, 8 KV heads, head width 64 |
| MLP | SwiGLU, hidden width 4096 |
| vocabulary | 256 byte values, tied output head |
| context | 512 tokens |

The fallback is exactly 275,621,120 parameters: 14 layers, width 1280, 20 query
heads, 4 KV heads, hidden width 4096, and 260 output logits. This systems run
uses only byte IDs 0–255; IDs 256–259 are reserved and unused. Both counts are
checked analytically by the release validator and at model construction time.

## Method

### Strong-scaling matrix

The canonical strong-scaling matrix fixes 32,768 tokens per optimizer update:

| World size | Microbatch/rank | Sequence | Accumulation | Global tokens/update |
|---:|---:|---:|---:|---:|
| 1 | 4 | 512 | 16 | 32,768 |
| 2 | 4 | 512 | 8 | 32,768 |
| 4 | 4 | 512 | 4 | 32,768 |

Each run has 40 updates. Updates 0–9 are timing warmup and updates 10–39 are
eligible, giving 30 observations. The runner predeclares three interleaved
passes—`1→2→4`, `4→2→1`, `2→4→1`—rather than running all trials for one world
size consecutively. Aggregation fails unless all nine runs end successfully and
each world size contributes exactly 90 eligible observations. Profiler output is
stored under a different subtree and can never enter this aggregate.

Canonical `elapsed_s` and `tokens_per_second` use a composed `core_update`
interval. The per-step rank barrier and CUDA peak-memory reset occur first;
timing then starts before `optimizer.zero_grad`. It includes CPU random window
sampling, host-to-device copies, BF16 forward/backward and DDP gradient
reduction, local loss-finite checks, learning-rate computation and assignment,
gradient clipping, the pre-guard CUDA synchronization, and the optimizer step
with its CUDA synchronization. The cross-rank finite-check all-reduce is timed
separately and excluded from `core_update`; mean-loss reduction, metrics I/O,
memory diagnostics, rank-diagnostic `all_gather_object`, evaluation,
checkpointing, and `profiler.step()` occur afterward. The aggregate uses the
slowest rank's `core_update` value for each eligible update.

## Reproduce

### Local validation

From the repository root:

```bash
python tools/run_training_matrix.py validate
python -m pytest tests/training/test_training_core.py \
  tests/training/test_training_release.py
python -m compileall -q experiments/distributed_training tools/run_training_matrix.py
python -m ruff check experiments/distributed_training \
  tools/run_training_matrix.py tests/training
```

The training tests cover RMSNorm, RoPE, GQA, causal masking, tied weights,
cross-entropy, the learning-rate schedule, byte-stream provenance, the direct
next-token label-shift oracle, batch cursor replay, checkpoint state, parameter
counts, matrix construction, source locking, warmup exclusion, private paths,
atomic no-overwrite behavior, and trajectory comparison. The GPU acceptance
test is skipped unless
`RUN_4X3090_ACCEPTANCE=1`; a CPU skip is not GPU evidence.

Generate and validate plans before GPU execution:

```bash
python tools/run_training_matrix.py source-lock
python tools/run_training_matrix.py matrix --dry-run
python tools/run_training_matrix.py trajectory --dry-run
python tools/run_training_matrix.py profile
python tools/run_training_matrix.py real-data
```

All training-runner outputs default below the ignored
`artifacts/private/training/` tree; readiness caches and derived data use
neighboring ignored `artifacts/private/` subtrees. Source locks must be
regenerated after any bound code, config, data manifest, environment candidate,
or readiness report changes.

### GPU environment acceptance

Run both the full readiness gate and the training-specific NCCL gate before a
formal training launch. The host phase is inventory-only and is not sufficient
training evidence.

1. Run the host inventory gate before installing or changing anything:

   ```bash
   python tools/preflight_rtx3090.py run \
     --phase host \
     --workdir . \
     --output artifacts/private/rtx3090-readiness/host.json
   ```

2. Select exactly one CUDA branch from the report. Keep the validated host
   NVIDIA driver unchanged inside the execution environment:

   ```bash
   python -m pip install -r requirements/rtx3090-cu128.in
   python -m pip install -r requirements/rtx3090-common.in
   python -m pip check
   ```

   Use `rtx3090-cu126.in` instead only when the driver gate selects that branch.

3. Populate the selected wheel cache and pinned model snapshot exactly as
   described in the [RTX 3090 environment runbook](rtx3090-environment.md),
   create their semantic manifests, and run the full readiness phase. The
   following uses the preferred `cu128` branch; use the selected `cu126` names
   consistently when the host report requires it:

   ```bash
   python tools/preflight_rtx3090.py manifest \
     --root artifacts/private/cache/wheels/cu128 \
     --kind wheel-cache \
     --cuda-branch cu128 \
     --output artifacts/private/cache/manifests/wheels-cu128.json

   python tools/preflight_rtx3090.py manifest \
     --root artifacts/private/cache/model/glm-asr-nano-2512 \
     --kind model-snapshot \
     --output artifacts/private/cache/manifests/model.json

   python tools/preflight_rtx3090.py run \
     --phase full \
     --offline \
     --workdir . \
     --stress-seconds 300 \
     --distributed-timeout 900 \
     --wheel-root artifacts/private/cache/wheels/cu128 \
     --wheel-manifest artifacts/private/cache/manifests/wheels-cu128.json \
     --model-root artifacts/private/cache/model/glm-asr-nano-2512 \
     --model-manifest artifacts/private/cache/manifests/model.json \
     --output artifacts/private/rtx3090-readiness/full.json
   ```

   Training remains blocked unless this report is `phase=full` with
   `status=pass`, `training_ready=true`, and every formal eligibility gate
   accepted by the validator. The current readiness schema retains the legacy
   compatibility field `day_rental_eligible=true`; this is a machine contract,
   not an execution recommendation. The runner also verifies the report tool
   hash, environment-contract hash,
   byte-identical pre/post `pip freeze --all`, package pins, both cache
   manifests, at least 300 seconds of stress, BF16/NCCL, thermal health, and Xid
   gates. A host-only, warning, incomplete, or hand-written substitute is
   rejected.

4. Run the additional training-specific four-rank NCCL/BF16 check:

   ```bash
   python -m torch.distributed.run --standalone --nproc_per_node=4 \
     -m experiments.distributed_training.preflight \
     --environment-candidate configs/environments/rtx3090-ddp-v1.json \
     --output artifacts/private/training/preflight-4x3090.json
   ```

This preflight requires four unique CUDA ranks, matching RTX 3090 names, at
least 23 GiB per device, BF16 support, a shared artifact filesystem, a correct
NCCL all-reduce, and an identical report hash visible from every rank. It also
records the P2P matrix, `nvidia-smi topo -m`, rank environments, and collective
timings. It supplements rather than replaces the full readiness report. A
warning, partial inventory, or short-only result is not a training gate pass.

GPU identity is fail-closed and does not depend on optional PCI attributes on
`torch.cuda.get_device_properties`. Rank 0 captures the physical
`nvidia-smi --query-gpu=index,uuid,pci.bus_id,...` inventory once and broadcasts
it to all ranks. Each rank enumerates the UUIDs of its CUDA-visible logical
devices, verifies that `LOCAL_RANK` selects the corresponding logical device,
and joins that UUID to the same `nvidia-smi` row to obtain the PCI BDF. All four
ranks must observe the same logical UUID order and form four unique UUID/BDF
pairs. Consequently, a reordered `CUDA_VISIBLE_DEVICES` list is supported, but
the physical `nvidia-smi` index is never assumed to equal `LOCAL_RANK`; missing,
ambiguous, or inconsistent UUID mappings abort before the NCCL timing probe.

### Canonical strong-scaling run

Freeze a fresh source lock after the checkout and environment candidate are
final:

```bash
python tools/run_training_matrix.py source-lock \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --preflight-report artifacts/private/training/preflight-4x3090.json \
  --output artifacts/private/training/source-lock.json

python tools/run_training_matrix.py matrix \
  --execute \
  --source-lock artifacts/private/training/source-lock.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --preflight-report artifacts/private/training/preflight-4x3090.json
```

Every update records global loss, gradient norm, synchronized critical-rank
latency, global tokens/s, measurement eligibility, and per-rank local latency,
loss, tokens/s, allocated/reserved memory, peak memory, and free/total device
memory. `run_start` records all rank environments, exact config/data/source
hashes, precision policy, world size, and parameter count. The optimizer is
single-tensor AdamW (`foreach=False`, `fused=False`); parameters, gradients,
AdamW state, and DDP buckets are checked as FP32 while forward/backward compute
uses BF16 autocast.

The canonical fields follow the composed `core_update` definition in
[Method](#method). The excluded cross-rank finite-check interval is reported as
`critical_rank_finite_guard_elapsed_s`, and the contiguous inclusive interval as
`critical_rank_guarded_update_elapsed_s`. Only `core_update` enters the scaling
aggregate.

Do not report scaling numbers until
`artifacts/private/training/strong-scaling/aggregate.json` exists and its nine
runs have passed. The aggregate embeds SHA-256 references for the frozen plan,
completed journal, source lock, both readiness reports, and every raw metrics
and final-checkpoint artifact.

### Four-rank checkpoint/replay run

The default trajectory workload is the short 275,621,120-parameter BF16
four-rank config, not the CPU smoke test:

```bash
python tools/run_training_matrix.py trajectory \
  --execute \
  --source-lock artifacts/private/training/source-lock.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --preflight-report artifacts/private/training/preflight-4x3090.json
```

The runner executes an uninterrupted eight-update control, a second run stopped
normally after the step-4 checkpoint, and a same-world-size resume in the same
output directory. `--stop-after-step` accepts only an interior checkpoint
boundary and never mutates config bytes. Comparison fails closed on missing or
duplicate steps, missing planned-stop/run-end events, any per-step global or
per-rank loss/gradient diagnostic difference, learning-rate or validation
difference, or any final model, AdamW, rank process-state, RNG, or
train/validation cursor digest difference.

Trajectory evidence uses the dedicated `trajectory-275m-w4.json` config. It
enables PyTorch deterministic algorithms while deliberately retaining TF32 and
automatic SDPA selection for the first attributable determinism check. Before
distributed initialization, each worker sets an absent
`CUBLAS_WORKSPACE_CONFIG` to `:4096:8` or rejects any conflicting value, and
records the effective value in `rank_environments`. The strong-scaling and
isolated-profiler configs remain unchanged. If this strict first-stage recipe
raises on an unsupported operation or still diverges, TF32 and SDPA backend
selection are separate follow-up experiments rather than bundled changes.

`trajectory-275m-w4-fixed-buckets.json` is a matched trajectory diagnostic with
the same model, data, precision, and update recipe. It sets DDP
`find_unused_parameters=True` with `static_graph=False`, changing the reducer
lifecycle observed around the resume boundary. Its exact match is consistent
with reducer lifecycle effects but does not establish a unique cause. The model
uses every parameter; this setting adds an autograd-graph traversal and is not
used in the throughput matrix. Select it explicitly with
`--config configs/training/trajectory-275m-w4-fixed-buckets.json`; its config
stem gives it a separate artifact directory from the default first-stage run.

The CPU smoke config exercises the same code in CI, but it is not evidence that
the four-rank path passed.

### Isolated profiler run

```bash
python tools/run_training_matrix.py profile \
  --execute \
  --source-lock artifacts/private/training/source-lock.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --preflight-report artifacts/private/training/preflight-4x3090.json
```

Each rank writes a Chrome trace and a key-averages JSON file below
`artifacts/private/training/profiler/`. The profiler schedule is separate from
the canonical matrix, and the profile journal explicitly marks it ineligible
for canonical aggregation.

### TinyStories-derived byte-corpus run

Prepare the corpus before GPU execution. The builder pins the upstream revision,
verifies source bytes, selects complete documents into disjoint
64 MiB/4 MiB train/validation files, and writes
`training-corpus-manifest-v0.1`. Audio, corpora, checkpoints, and reports remain
private and are never release files.

```bash
python tools/prepare_training_corpus.py --download

python tools/run_training_matrix.py source-lock \
  --data-manifest \
    artifacts/private/training-data/derived/tinystories-64mib-4mib-v1/corpus-manifest.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --preflight-report artifacts/private/training/preflight-4x3090.json \
  --output artifacts/private/training/real-data-source-lock.json

python tools/run_training_matrix.py real-data \
  --execute \
  --source-lock artifacts/private/training/real-data-source-lock.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --preflight-report artifacts/private/training/preflight-4x3090.json
```

The byte-corpus workload uses four ranks, 275,621,120 parameters, 200 updates,
and validation every 20 updates. At each evaluation, eight batches per rank,
four sequences per batch, and 512 next-byte targets per sequence produce 128
deterministic pseudorandom windows and 65,536 sampled targets across four ranks.
Window starts are sampled with replacement and may overlap; the validation
sampler advances its CPU RNG, so successive records use different positions
rather than one fixed exhaustive set. The runner also verifies the manifest
hashes and 64 MiB/4 MiB byte budgets before launch.

## Limitations and evidence boundaries

Safe claims require the corresponding private artifact:

- “DDP/NCCL ran on four RTX 3090s” requires both the pure-pass full readiness
  report and the passing training-specific NCCL report; every formal journal
  retains both report paths and SHA-256 hashes.
- Parameter counts and fixed tokens/update are code/config invariants.
- Throughput, scaling efficiency, and peak memory require the completed
  nine-run aggregate and raw JSONL.
- A checkpoint/replay match requires a passing trajectory comparison.
- Profiler findings require all four rank traces and key-averages files.
- Training/validation behavior requires the TinyStories-derived corpus
  manifest and the completed real-data journal.

The experiment is single-node DDP and does not measure FSDP, tensor, pipeline,
or multi-node parallelism. Dry-run plans, CPU smoke results, skipped GPU tests,
capacity estimates, synthetic losses, and incomplete journals do not
substantiate measured GPU results.
