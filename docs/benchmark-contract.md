# Benchmark contract

A benchmark is comparable or publishable only when another reviewer can identify
the exact bytes, machine decision, source state, command, and correctness outcome
that produced it. Validation is exact and fail-closed; a missing field is not an
unknown value that can be filled in later.

## Formal-run admission

A formal plan and every downstream execution step require all of the following:

- a real `faster-glm-asr` source checkout and an explicit `--repository-root`;
- a 40-hex Git commit with `dirty: false` at plan, execute, aggregate, and compare;
- the external model manifest schema `cache-file-manifest-v1`, kind
  `model-snapshot`, exactly ten safe ordinary files, and exact current size/SHA-256;
- semantic model repository and immutable revision equal to the run configuration;
- one shared local snapshot root for processor, HF model, and custom loader, with
  `local_files_only=True` and no network fallback;
- a readiness report with schema `rtx3090-readiness-v1`, phase `full`, status
  `pass`, `training_ready: true`, `day_rental_eligible: true`, no failed or warning
  gates, and checkout-matching contract and tool hashes;
- four-GPU identity plus topology, clock/power, and combined hardware evidence
  digests derived from the private readiness report;
- warm-up of at least one iteration. For custom implementations the first warm-up
  covers lazy host-to-device parameter materialization before measurement.

An ad-hoc or dirty run may be useful for diagnosis only. It must record
`formal_run: false` or a non-comparable reason, cannot set canonical metric
eligibility, cannot enter the strict comparator, and cannot become a public
artifact. Warm-up zero has the same boundary.

## Bound identity

Every accepted artifact binds:

- Git commit, branch in private evidence, and clean-tree decision;
- the sorted recursive source map for package `modeling`, `kernels`,
  `benchmarking`, and `data` Python files, including size and SHA-256;
- model/processor repository, immutable revision, external-manifest SHA-256,
  aggregate current model-byte SHA-256, file count, and total bytes;
- readiness report, contract, tool, GPU identity, topology, clock/power, and
  combined hardware evidence hashes;
- Python executable bytes, full environment-lock bytes, package versions, CUDA,
  driver, cuDNN, operating system, device model, memory, and math settings;
- input-manifest digest, each audio digest, normalized prepared tensor content,
  generation settings, implementation, cache policy, dtype, warm-up, repeat
  count, measurement profile, and synchronization boundary.

Raw UUIDs, private paths, transcripts, and report contents remain in the private
evidence root. Public output retains sanitized hardware identity and the hashes
needed to prove which admitted evidence was used.

## Paired-comparison gates

The six formal mechanisms are `hf_cached`, `hf_no_cache`,
`custom_full_prefix`, `custom_greedy_full_prefix`, `custom_tuple_cache`, and
`custom_static_cache`. In the current formal matrix all six use FP32 storage and
activations. A BF16-versus-FP32 pair is not accepted as cache evidence.

Before computing a ratio, the strict comparator requires exact equality of model
bytes, readiness binding, source map, Git commit, environment, manifest,
generation parameters, prepared input tensor shapes/dtypes/content hashes, token
IDs, and decoded hypotheses. Failed or divergent samples remain visible and
block comparison; they are never silently discarded.

Custom execution is recorded as `custom-torch-triton-hybrid`. The dispatch
contract is policy evidence, not a claim that every operation used Triton:

- supported norm, RoPE, embedding, and convolution components may dispatch to
  Triton automatically and otherwise fall back to Torch;
- unmasked decoder attention uses dense Triton only for supported padded inputs
  and head dimension at most 256, otherwise Torch dense attention; masked
  correctness fallbacks delegate to framework SDPA and are not called
  FlashAttention by this project;
- grouped-query KV heads are explicitly expanded;
- long 30-second inputs are expected to exercise the attention fallback often;
- no observed speed difference is attributed to a Triton kernel without a
  separate backend trace.

## Measurement profiles and metrics

Canonical latency, request-memory polling, and diagnostic CUDA phase timing are
separate runs. Instrumented memory or phase rows cannot supply canonical latency.
The balanced matrix interleaves implementations over three outer passes and the
aggregator recomputes statistics from raw observations.

- Latency includes count, median, p95, and the synchronization boundary.
- Real-time factor is wall-clock seconds divided by input audio seconds; its
  reciprocal must be labelled separately.
- GPU memory distinguishes framework allocated/reserved peaks from sampled
  device-process memory; polling interval and lower-bound semantics are recorded.
- Cold start and model loading require separate explicitly scoped experiments;
  runner construction is not silently relabelled as either metric.
- Throughput must identify its unit and concurrency. Scaling must state whether
  global or per-device work is fixed, exact world size, and backend.

## Publication boundary

Formal matrix outputs are private and no-overwrite. Aggregation requires all 54
successful tasks, a valid append-only journal, and unchanged output/log hashes;
it emits exactly 18 profile artifacts. A public benchmark artifact is produced by
the runner sanitizer during a fresh clean formal run and must pass the strict
public schema. Manual JSON field deletion is not a release mechanism.

Synthetic inputs validate control flow, not speech quality. The three-item
LibriSpeech subset is the performance matrix; the 24-item subset is one separate
quality evaluation, not a larger matrix. One RTX 3090 host does not establish
another GPU family or multi-node behavior. Missing measurements stay pending and
are never estimated from theoretical peak specifications.
