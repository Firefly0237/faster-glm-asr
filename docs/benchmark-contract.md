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

The public summary admits five ordered comparisons: `hf_no_cache` to
`hf_cached`; `custom_full_prefix` to `custom_greedy_full_prefix`;
`custom_greedy_full_prefix` to `custom_tuple_cache`; `custom_tuple_cache` to
`custom_static_cache`; and `custom_full_prefix` to `custom_static_cache`. It
does not publish a direct Hugging Face-to-custom speedup because those runtime
families do not produce byte-identical prepared-request tensor contracts. Exact
generated-token parity remains a matrix-wide correctness gate across all six
mechanisms; it does not establish prepared-input comparability for an otherwise
inadmissible latency ratio. The exporter therefore enforces sample identity and
token output globally, while enforcing prepared-request equality separately
inside the Hugging Face and custom families.

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

The matrix-level public summary has a stricter, sample-free publication path:

```bash
faster-glm-asr-public-matrix export \
  --evidence-root /worktrees/faster-glm-asr-evidence \
  --publication-root . \
  --plan artifacts/private/plans/librispeech-test-clean-3-v1.json \
  --journal artifacts/private/run-journals/librispeech-test-clean-3-v1/journal.json \
  --aggregate-dir artifacts/private/aggregated \
  --output artifacts/private/public-matrix-draft.json

faster-glm-asr-public-matrix check \
  artifacts/private/public-matrix-draft.json

faster-glm-asr-public-matrix publish \
  --evidence-root /worktrees/faster-glm-asr-evidence \
  --publication-root . \
  --plan artifacts/private/plans/librispeech-test-clean-3-v1.json \
  --journal artifacts/private/run-journals/librispeech-test-clean-3-v1/journal.json \
  --aggregate-dir artifacts/private/aggregated \
  --output benchmarks/results/rtx3090-librispeech-test-clean-3-v1.json
```

Both `export` and `publish` rebuild the summary from the private evidence and
refuse overwrite; `publish` additionally restricts the destination to a direct
child of `benchmarks/results/`. The exporter validates the exact 18-file mapping
against a fresh aggregate rebuild; revalidates the plan-bound three-row
LibriSpeech manifest, its selection lock, and the canonical data
configuration; binds every aggregate anchor row back to that manifest; checks
all generated-token sequences across all profiles; and runs the strict
canonical comparator for five fixed baseline-to-candidate pairs. It also checks
that the executing planner, executor, comparator, aggregator, runner,
provenance, and LibriSpeech-builder bytes equal their evidence-checkout
counterparts. Speedups are baseline latency divided by candidate latency, so a
value below one remains below one. The three-item performance summary
deliberately contains no WER or CER.

The selection lock's `builder_source_sha256` records the builder claimed by the
original subset-generation event. It is retained as a well-formed historical
audit handle, not treated as proof that unavailable historical source bytes
equal the builder used for the formal matrix. Data validation does not rely on
that claim: the executing LibriSpeech builder must byte-match the evidence
checkout and the plan's complete source map, then freshly reconstruct the
manifest rows, selected IDs, candidate counts and inventory, and license binding
from the pinned official archive and its exact extracted tree. Every other
selection-lock field is compared with those reconstructed, plan/config-bound
values. Without the historical source bytes or a separately trusted
attestation, substituting one syntactically valid builder audit handle for
another cannot be detected and does not change the reconstructed data semantics.

`export` and `publish` recompute the public evidence hashes from the admitted
plan, journal, canonical aggregate set, pinned dataset archive, source map, and
executing exporter/schema bytes. Those hashes are audit handles, not
self-authenticating proof. The sample-free `check` command and source-release
guard have no private evidence from which to reproduce that chain; they enforce
the exact public schema, metric consistency, and privacy boundary, but do not
prove that an otherwise valid JSON file originated from `publish`.

`--evidence-root` is the immutable evidence checkout, not necessarily the
checkout from which the installed CLI is launched. `--publication-root` is a
separate output checkout and is never used to resolve the plan, journal,
manifest, or aggregates. If the exporter itself was added after a matrix ran,
point `--evidence-root` at a separate clean worktree of the matrix's recorded
commit. Adding the exporter to that evidence worktree would change the source
map and must fail validation.

Synthetic inputs validate control flow, not speech quality. The three-item
LibriSpeech subset is the performance matrix; the 24-item subset is one separate
quality evaluation, not a larger matrix. One RTX 3090 host does not establish
another GPU family or multi-node behavior. Missing measurements stay pending and
are never estimated from theoretical peak specifications.
