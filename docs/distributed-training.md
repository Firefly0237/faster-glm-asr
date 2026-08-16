# Single-node decoder distributed-training experiment

This directory is an isolated systems experiment for a small decoder-only
Transformer. It is not GLM-ASR training, pretraining evidence for a foundation
model, or a claim about model quality. Synthetic streams are used only for
throughput controls and exact checkpoint replay. The separate TinyStories-derived
byte-file run is the only committed path that produces held-out validation loss
from non-synthetic data.

## Hardware decision

Use the four RTX 3090 instance, not four A4000s, for the frozen experiment. Each
3090 exposes 24 GiB, while an A4000 normally exposes 16 GiB. This code keeps a
complete FP32 model, FP32 gradients, and two FP32 AdamW moment buffers on every
DDP rank. The tensor-only floor is about 6.45 GB for 403,097,088 parameters and
4.41 GB for 275,621,120 parameters, before activations, SDPA workspaces, CUDA
context, and allocator reservation. Those are capacity estimates, not measured
results; only the rented-host JSONL records may be quoted as measurements.

The implementation is deliberately limited to single-node DDP/NCCL. It does not
implement or claim FSDP, tensor parallelism, pipeline parallelism, ZeRO, or
multi-node training.

## What is frozen before rental

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

## Local gates before starting paid time

From the repository root:

```bash
python tools/run_training_matrix.py validate
python -m pytest tests/training/test_training_core.py \
  tests/training/test_training_release.py
python -m compileall -q experiments/distributed_training tools/run_training_matrix.py
python -m ruff check experiments/distributed_training \
  tools/run_training_matrix.py tests/training
```

The 13 core tests cover RMSNorm, RoPE, GQA, causal masking, tied weights,
cross-entropy, the learning-rate schedule, byte-stream provenance, the direct
next-token label-shift oracle, batch cursor replay, and exact checkpoint state.
The release tests cover parameter counts, matrix construction, source locking,
warmup exclusion, private paths, atomic no-overwrite behavior, and trajectory
comparison. The GPU acceptance test is skipped unless
`RUN_4X3090_ACCEPTANCE=1`; a CPU skip is not GPU evidence.

Generate plans before rental:

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

## First hour on the rented host

Keep the instance hourly until the full readiness gate and the additional
training NCCL gate both pass. The host phase is diagnostic only: even a host
`pass` does not authorize training or a day rental.

1. Run the host inventory gate before installing or changing anything:

   ```bash
   python tools/preflight_rtx3090.py run \
     --phase host \
     --workdir . \
     --output artifacts/private/rtx3090-readiness/host.json
   ```

2. Select exactly one CUDA branch from the report. Do not replace the host
   NVIDIA driver inside the rented container:

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
   `status=pass`, `day_rental_eligible=true`, and `training_ready=true`. The
   runner also verifies the report tool hash, environment-contract hash,
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

## Canonical strong-scaling run

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

Canonical `elapsed_s` and `tokens_per_second` use the `core_update` boundary:
the synchronized forward/backward pass (including DDP gradient reduction),
gradient clipping, and optimizer step. The fail-fast cross-rank finite guard is
reported separately as `critical_rank_finite_guard_elapsed_s`; the inclusive
guarded loop is `critical_rank_guarded_update_elapsed_s`. Mean-loss reduction
runs after both timers. Only `core_update` enters the scaling aggregate, so the
extra diagnostic collective on multi-rank runs cannot masquerade as model
scaling time.

Do not report scaling numbers until
`artifacts/private/training/strong-scaling/aggregate.json` exists and its nine
runs have passed. The aggregate embeds SHA-256 references for the frozen plan,
completed journal, source lock, both readiness reports, and every raw metrics
and final-checkpoint artifact.

## Four-rank interruption and exact replay

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
duplicate steps, missing planned-stop/run-end events, any per-step loss
difference, or any final model, AdamW, rank process-state, RNG, or
train/validation cursor digest difference.

The CPU smoke config exercises the same code in CI, but it is not evidence that
the four-rank path passed.

## Isolated profiler run

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

## TinyStories-derived byte-file short run

Prepare data before or at the beginning of rental. The builder pins the upstream
revision, verifies source bytes, selects complete documents into disjoint
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

The committed real-data workload uses four ranks, 275,621,120 parameters, 200
updates, and validation every 20 updates. The runner verifies the manifest
hashes and exact 64 MiB/4 MiB byte budgets before launch, then refuses success
unless finite validation-loss records exist. These losses show only that the
pipeline trained and evaluated on the declared byte corpus; they are not a
quality comparison with a tokenizer-based language model.

## Evidence boundaries

Safe claims require the corresponding private artifact:

- “DDP/NCCL ran on four RTX 3090s” requires both the pure-pass full readiness
  report and the passing training-specific NCCL report; every formal journal
  retains both report paths and SHA-256 hashes.
- Parameter counts and fixed tokens/update are code/config invariants.
- Throughput, scaling efficiency, and peak memory require the completed
  nine-run aggregate and raw JSONL.
- Exact interruption/resume requires a passing trajectory comparison.
- Profiler findings require all four rank traces and key-averages files.
- Training/validation behavior requires the TinyStories-derived corpus
  manifest and the completed real-data journal.

Never convert a dry-run plan, CPU smoke result, skipped GPU test, theoretical
memory estimate, synthetic loss, or incomplete journal into a measured claim.
