# Reproducibility

This is the executable, fail-closed path from source data to a checked public
artifact. Run it from a clean `faster-glm-asr` source checkout on the accepted
Ubuntu 22.04 / Python 3.11 host. A wheel installation alone is insufficient for
a formal run because source and readiness-tool bytes are part of the evidence.

## Evidence states

| State | Meaning |
|---|---|
| `implemented` | code exists and applicable static tests pass |
| `validated-small` | a reduced model passed its stated oracle |
| `validated-checkpoint` | the pinned checkpoint passed input/token parity |
| `measured` | a frozen accepted host produced raw repeated observations |
| `publishable` | strict comparison and the public-artifact guard both pass |

CPU or synthetic results never establish target-GPU latency, memory, quality,
or scaling. Model, audio, report, and raw result bytes stay under the ignored
`artifacts/private/` tree.

## 1. Install and materialize immutable caches

Follow [`rtx3090-environment.md`](rtx3090-environment.md) to select `cu128` or
the documented `cu126` fallback. The commands below show the preferred branch
and the fixed cache layout:

```bash
python -m pip download \
  --dest artifacts/private/cache/wheels/cu128 \
  -r requirements/rtx3090-cu128.in
python -m pip download \
  --dest artifacts/private/cache/wheels/cu128 \
  -r requirements/rtx3090-common.in

hf download zai-org/GLM-ASR-Nano-2512 \
  --revision 61ba4e0b3309b6656edea3e93e419f7bd5c61957 \
  --local-dir artifacts/private/cache/model/glm-asr-nano-2512

python tools/preflight_rtx3090.py manifest \
  --kind wheel-cache --cuda-branch cu128 \
  --root artifacts/private/cache/wheels/cu128 \
  --output artifacts/private/cache/manifests/wheels-cu128.json
python tools/preflight_rtx3090.py manifest \
  --kind model-snapshot \
  --root artifacts/private/cache/model/glm-asr-nano-2512 \
  --output artifacts/private/cache/manifests/model.json

python tools/preflight_rtx3090.py verify-manifest \
  --cuda-branch cu128 \
  --root artifacts/private/cache/wheels/cu128 \
  --manifest artifacts/private/cache/manifests/wheels-cu128.json
python tools/preflight_rtx3090.py verify-manifest \
  --root artifacts/private/cache/model/glm-asr-nano-2512 \
  --manifest artifacts/private/cache/manifests/model.json
```

Success means both verify commands exit zero. The model manifest must describe
exactly ten ordinary files, bind repository identifier plus immutable revision,
and match every current byte. The benchmark never calls a Hub identifier: both
the Hugging Face and custom paths load this same local root with network fallback
disabled.

## 2. Build the public input manifests

The source archive contract is
[`configs/data/librispeech-test-clean-v1.json`](../configs/data/librispeech-test-clean-v1.json).
Download its fixed URL, then verify both the official MD5 and repository-pinned
SHA-256 before extraction:

```bash
mkdir -p artifacts/private/cache/datasets
curl --fail --location --output \
  artifacts/private/cache/datasets/test-clean.tar.gz \
  https://www.openslr.org/resources/12/test-clean.tar.gz
printf '%s  %s\n' \
  32fa31d27d2e1cad72775fee3f4849a9 \
  artifacts/private/cache/datasets/test-clean.tar.gz | md5sum --check -
printf '%s  %s\n' \
  39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23 \
  artifacts/private/cache/datasets/test-clean.tar.gz | sha256sum --check -
tar --extract --gzip \
  --file artifacts/private/cache/datasets/test-clean.tar.gz \
  --directory artifacts/private/cache/datasets

faster-glm-asr-prepare-librispeech \
  --test-clean-root artifacts/private/cache/datasets/LibriSpeech/test-clean \
  --license-file artifacts/private/cache/datasets/LibriSpeech/LICENSE.TXT \
  --per-bucket 1 \
  --output-dir artifacts/private/cache/manifests/librispeech-test-clean-3-v1
faster-glm-asr-prepare-librispeech \
  --test-clean-root artifacts/private/cache/datasets/LibriSpeech/test-clean \
  --license-file artifacts/private/cache/datasets/LibriSpeech/LICENSE.TXT \
  --per-bucket 8 \
  --output-dir artifacts/private/cache/manifests/librispeech-test-clean-24-v1

test "$(wc -l < artifacts/private/cache/manifests/librispeech-test-clean-3-v1/manifest.jsonl)" -eq 3
test "$(wc -l < artifacts/private/cache/manifests/librispeech-test-clean-24-v1/manifest.jsonl)" -eq 24
```

The three-row duration-stratified manifest is the only input to the 54-task
performance matrix. The 24-row manifest is reserved for one independent quality
run; do not multiply it across all matrix cells.

For authorized long recordings, first replace the example row with real local
paths and authorization metadata, then run the separate inventory-to-segment
pipeline:

```bash
cp configs/benchmark/inventory.example.csv artifacts/private/inventory.csv
# Edit artifacts/private/inventory.csv before continuing.
faster-glm-asr-build-manifest \
  --inventory artifacts/private/inventory.csv \
  --output artifacts/private/cache/manifests/authorized-sources.jsonl
faster-glm-asr-build-long-audio \
  --source-manifest artifacts/private/cache/manifests/authorized-sources.jsonl \
  --output-dir artifacts/private/cache/manifests/authorized-segments-v1
```

Success is an atomic `manifest.jsonl`, selection/bundle lock, and zero validation
errors. This private path does not replace the fixed public matrix manifest.

## 3. Bind a full machine-readiness decision

The host phase is diagnostic only. Run the five-minute full phase while still
on usage billing and give it an explicit stable output path:

```bash
python tools/preflight_rtx3090.py run \
  --phase full --offline \
  --wheel-root artifacts/private/cache/wheels/cu128 \
  --wheel-manifest artifacts/private/cache/manifests/wheels-cu128.json \
  --model-root artifacts/private/cache/model/glm-asr-nano-2512 \
  --model-manifest artifacts/private/cache/manifests/model.json \
  --stress-seconds 300 --distributed-timeout 900 \
  --output artifacts/private/rtx3090-readiness/full.json

mkdir -p artifacts/private/environment
python -m pip freeze --all > artifacts/private/environment/runtime.freeze.txt
jq --exit-status '
  .schema_version == "rtx3090-readiness-v1" and
  .phase == "full" and
  .overall.status == "pass" and
  .overall.training_ready == true and
  .overall.day_rental_eligible == true and
  (.overall.failed_gates | length) == 0 and
  (.overall.warning_gates | length) == 0
' artifacts/private/rtx3090-readiness/full.json
```

Only exit zero plus the `jq` predicate is accepted. `host`, warnings, missing
topology/clock/power evidence, failed cache checks, or a changed contract/tool
hash blocks planning. The formal artifact stores only sanitized hardware
identity and evidence hashes; the raw report remains private.

## 4. Plan and execute the 54-task matrix

Commit the intended source first. A formal planner, executor, aggregator, runner,
or comparator rejects a dirty checkout. Ignored private artifacts do not affect
this gate.

```bash
git status --porcelain
cp configs/benchmark/matrix.example.json \
  artifacts/private/matrix-librispeech-test-clean-3-v1.json

faster-glm-asr-plan \
  --repository-root . \
  --config artifacts/private/matrix-librispeech-test-clean-3-v1.json \
  --output artifacts/private/plans/librispeech-test-clean-3-v1.json

faster-glm-asr-run-matrix \
  --repository-root . \
  --plan artifacts/private/plans/librispeech-test-clean-3-v1.json
faster-glm-asr-run-matrix \
  --repository-root . \
  --plan artifacts/private/plans/librispeech-test-clean-3-v1.json \
  --execute
```

`git status --porcelain` must print nothing. Planning must report exactly 54
tasks: six implementations × three measurement profiles × three balanced outer
passes. The first executor command must return `status: validated`; the second
must return `status: complete`. Every formal task uses FP32 for both HF and
custom paths, warm-up 3, identical input bytes, and a clean source map.

After an interruption, inspect the private logs and resume the same immutable
plan and journal; never delete successful rows or generate a replacement plan:

```bash
faster-glm-asr-run-matrix \
  --repository-root . \
  --plan artifacts/private/plans/librispeech-test-clean-3-v1.json \
  --execute --resume
```

## 5. Aggregate and compare

```bash
faster-glm-asr-aggregate \
  --repository-root . \
  --plan artifacts/private/plans/librispeech-test-clean-3-v1.json \
  --journal artifacts/private/run-journals/librispeech-test-clean-3-v1/journal.json \
  --output-dir artifacts/private/aggregated

faster-glm-asr-compare \
  --baseline artifacts/private/aggregated/custom-full-prefix.json \
  --candidate artifacts/private/aggregated/custom-static-cache.json \
  --output artifacts/private/comparisons/full-prefix-vs-static-cache.json
```

Aggregation succeeds only after all 54 journaled tasks and their current output
and log hashes validate; it writes exactly 18 no-overwrite artifacts. Comparison
succeeds only when paired tokens, prepared tensors, FP32 dtype, model bytes,
readiness evidence, source map, environment, Git state, and generation settings
match exactly. A ratio is not publishable when any one of those gates fails.

## 6. Run the independent 24-item quality set

Use one implementation once; this is not a second performance matrix:

```bash
faster-glm-asr-benchmark \
  --repository-root . --formal --artifact-scope private \
  --implementation custom_static_cache \
  --manifest artifacts/private/cache/manifests/librispeech-test-clean-24-v1/manifest.jsonl \
  --model-snapshot-root artifacts/private/cache/model/glm-asr-nano-2512 \
  --model-snapshot-manifest artifacts/private/cache/manifests/model.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --environment-lock artifacts/private/environment/runtime.freeze.txt \
  --warmup 1 --repeats 1 --max-new-tokens 128 \
  --output artifacts/private/quality/librispeech-test-clean-24-v1.json
```

Success means all 24 items remain in the artifact and quality status is
`evaluated`. Do not quote its incidental single-repeat latency as matrix data.

## 7. Produce and check a public artifact

Public evidence is generated by the runner's sanitizer during a fresh formal
run, never by manually deleting fields from a private JSON:

```bash
faster-glm-asr-benchmark \
  --repository-root . --formal --artifact-scope public \
  --implementation custom_static_cache \
  --manifest artifacts/private/cache/manifests/librispeech-test-clean-3-v1/manifest.jsonl \
  --model-snapshot-root artifacts/private/cache/model/glm-asr-nano-2512 \
  --model-snapshot-manifest artifacts/private/cache/manifests/model.json \
  --readiness-report artifacts/private/rtx3090-readiness/full.json \
  --environment-lock artifacts/private/environment/runtime.freeze.txt \
  --warmup 3 --repeats 30 --max-new-tokens 128 \
  --output artifacts/private/public-candidates/custom-static-cache.json

faster-glm-asr-check-public \
  artifacts/private/public-candidates/custom-static-cache.json
```

The final command must exit zero and print `public artifact validation passed`.
It rejects raw text, private paths/IDs, dirty Git, ad-hoc runs, warm-up zero,
incomplete provenance, and schema drift.

Before publishing source, run:

```bash
python -m pytest
python -m ruff check .
python tools/public_release_guard.py --root . --strict-release --include-history
python -m build
```

No speedup, latency, memory, WER, or scaling value is a result until the exact
target command produces accepted raw evidence and the relevant gate above passes.
