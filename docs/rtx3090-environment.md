# Four-GPU RTX 3090 environment acceptance

This is a fail-closed acceptance procedure for one Linux host with exactly four
RTX 3090 devices. It is an execution contract, not evidence that a target host
has already passed. Driver state, package availability, topology, collectives,
thermals, and cache bytes remain pending until the private report from that host
has `status: pass` and `day_rental_eligible: true`.

## Fixed boundary

| Component | Accepted value |
|---|---|
| Operating system | Ubuntu 22.04 x86-64 |
| Python | 3.11.x |
| GPUs | exactly 4 × RTX 3090, at least 23,000 MiB each, compute capability 8.6 |
| Preferred CUDA branch | PyTorch 2.10.0 cu128, Linux driver at least 570.26.0 |
| Fallback CUDA branch | PyTorch 2.10.0 cu126, Linux driver at least 560.28.3 |
| Triton | 3.6.0 |
| Transformers | 5.14.0 |
| Tokenizers | 0.22.2 exactly |
| Hugging Face Hub | 1.5.0 exactly |
| Safetensors | 0.8.0 exactly |
| Editable-build bootstrap | Setuptools 80.10.2 and Wheel 0.46.3 exactly |
| Distributed runtime | 4 ranks, NCCL backend |
| Stress interval | at least 300 seconds |
| Thermal evidence | at least 240 one-second samples for each exact GPU UUID |

The Tokenizers, Hugging Face Hub, and Safetensors pins are explicit even though
their compatible ranges accept more than one release. They satisfy the
published core metadata of Transformers 5.14.0, Accelerate 1.12.0, and Datasets
4.5.0 and prevent a new resolver run from silently changing implementations.

Do not replace the host NVIDIA driver from a container. Select only the CUDA
wheel branch supported by the driver reported by the host phase.

The four device memories remain separate 24 GiB address spaces. An aggregate
capacity label does not make one larger device, so model and optimizer placement
must still fit the selected data-, tensor-, or sharding strategy.

## Prepare immutable caches

Prepare the wheel cache on Linux with the same Python minor version as the
target. Run the branch and common downloads separately so the PyTorch index does
not become the index for every dependency:

```bash
python -m pip download \
  --dest artifacts/private/cache/wheels/cu128 \
  -r requirements/rtx3090-cu128.in
python -m pip download \
  --dest artifacts/private/cache/wheels/cu128 \
  -r requirements/rtx3090-common.in
```

Use `cu126` in both paths and the branch input file only when the host phase
selects the fallback. Do not combine branches in one cache.

Materialize the pinned model into a self-contained directory:

```bash
hf download zai-org/GLM-ASR-Nano-2512 \
  --revision 61ba4e0b3309b6656edea3e93e419f7bd5c61957 \
  --local-dir artifacts/private/cache/model/glm-asr-nano-2512
```

Do not point the manifest builder at a standard Hugging Face snapshot directory
whose files link to `../../blobs`. The builder deliberately rejects every
symbolic link, including links that currently resolve inside a larger cache,
because a moved snapshot would no longer be self-contained. `--local-dir`
materializes ordinary files and is the supported layout.

Create manifests only after each cache is complete. Keep each manifest outside
the root it describes:

```bash
python tools/preflight_rtx3090.py manifest \
  --kind wheel-cache \
  --cuda-branch cu128 \
  --root artifacts/private/cache/wheels/cu128 \
  --output artifacts/private/cache/manifests/wheels-cu128.json

python tools/preflight_rtx3090.py manifest \
  --kind model-snapshot \
  --root artifacts/private/cache/model/glm-asr-nano-2512 \
  --output artifacts/private/cache/manifests/model.json
```

Each manifest records every relative path, byte count, and SHA-256. Wheel
metadata also binds the environment contract, CUDA branch, both requirements
inputs, and a deterministic dependency-closure summary. That summary records
the fixed marker environment, compatible wheel-tag policy, resolved
distributions, requested extras, active edge count, and an edge-set SHA-256.
Model metadata binds the contract, repository identifier, and immutable
revision. Added, missing, linked, resized, or changed files fail verification.
Manifest and report writes never overwrite existing evidence.

Before a wheel manifest can be created, every file must be a structurally valid
ZIP wheel with exactly one `.dist-info/METADATA`. The verifier uses a fixed
Ubuntu 22.04 x86_64 / CPython 3.11 marker environment, compatible tags, and
`Requires-Python`; it never consults the development host's platform tags.
Every exact top-level requirement must have one matching candidate, each
distribution must occur only once, and every active `Requires-Dist` edge must
resolve to that one compatible wheel at a satisfying version. Requested extras
propagate recursively (for example, `datasets` to `fsspec[http]` and then
Fsspec's HTTP dependencies).
Missing CUDA runtime wheels, a Windows/macOS tag, an absent transitive edge, or
two Fsspec/Setuptools versions is a hard error. Move any unselected duplicate to
the private quarantine only after recording the resolver result, then rebuild
the manifest. This static target-Linux closure is necessary but not sufficient:
the target Linux full phase must still complete its own network-disabled fresh
environment install.

## Hourly acceptance sequence

Start with usage-based billing. The host phase is a fast diagnostic only:

```bash
python tools/preflight_rtx3090.py run --phase host
```

It checks the exact device count and per-device memory, unique raw UUIDs, driver
branch, idle state, basic resources, endpoint reachability or both verified
caches, and recent NVIDIA Xid evidence. It also records current/maximum PCIe
generation and width, `lscpu` NUMA data, `nvidia-smi topo -m`, and the supported
`topo -p2p p/r/w/n` matrices. Drivers that do not expose a PCIe query field are
retried one field at a time and the remaining values are explicitly marked
unavailable. Missing topology evidence is at least a warning. PCIe generation,
lack of NVLink, or a particular NUMA layout is recorded for diagnosis but is not
by itself a hardware failure. A host-phase exit code of zero means only
`continue-hourly-to-full-validation`; it never authorizes a day rental.

### Xid time windows and provider evidence

The report records `preflight_started_at` before hardware inspection. The
pre-existing Xid window is exactly the 900 seconds ending at that instant. In a
full run, the new-event window begins at that same instant and ends at the
post-run capture. Both `journalctl` and the `dmesg` fallback receive explicit
`--since` and `--until` values; `dmesg` is never read as an unbounded history.
An event older than the lookback cannot become a permanent failure. An event in
the lookback or any new event during this preflight still fails its respective
gate.

An empty query window is a clean zero-event result only after a separate probe
proves that the underlying journal or kernel buffer is readable. Missing journal
storage, permission errors, an unreadable kernel buffer, or an empty Journal
probe that cannot establish readability remains unavailable. With no other
evidence this is a warning, never a clean result.

For a container where both local sources are unavailable, the CLI accepts two
separate provider evidence pairs. `before` covers the lookback; `after` covers
the preflight itself. The after paths may not exist at launch: a trusted
provider-side collector can publish them during the five-minute run. The tool
waits up to 30 seconds by default at each requested capture, bounded to 60
seconds by the CLI. This makes the fallback operational without allowing a
static file created before the run to certify a future interval.

Each JSON file has this exact schema; extra or missing fields fail validation:

```json
{
  "schema_version": "provider-gpu-health-evidence-v1",
  "provider": "example-provider",
  "instance_reference": "<provider-instance-reference>",
  "window_start": "2026-08-16T11:45:00+00:00",
  "window_end": "2026-08-16T12:05:30+00:00",
  "captured_at": "2026-08-16T12:05:31+00:00",
  "nvidia_xid_events": [
    {
      "timestamp": "2026-08-16T12:03:10+00:00",
      "code": 79,
      "summary": "example event description"
    }
  ]
}
```

Timestamps must be ordered, explicit UTC RFC3339 values. Every event must lie
inside the evidence window. The evidence window must cover the requested local
window, the report must be finalized near the requested end, and `captured_at`
must follow the evidence end within the contract's reporting delay. An event
outside the requested sub-window is retained only by the provider file; an
event inside it remains a hard failure.

For each JSON file, supply a second file containing exactly one lowercase
SHA-256 digest and an optional final newline. Obtain both through the trusted
provider channel. The detached digest detects byte substitution but is not a
cryptographic provider signature. Publish each JSON and checksum via temporary
names followed by atomic renames; a partial or mismatched pair remains
unavailable.

When local logs are unavailable, use one terminal or provider-side process to
publish the before pair immediately after launch and the after pair once its
window covers the completed workload. Point the preflight at the final paths:

```bash
python tools/preflight_rtx3090.py run \
  --phase full \
  --offline \
  --wheel-root artifacts/private/cache/wheels/cu128 \
  --wheel-manifest artifacts/private/cache/manifests/wheels-cu128.json \
  --model-root artifacts/private/cache/model/glm-asr-nano-2512 \
  --model-manifest artifacts/private/cache/manifests/model.json \
  --provider-health-evidence-before artifacts/private/health/xid-before.json \
  --provider-health-evidence-before-sha256 artifacts/private/health/xid-before.sha256 \
  --provider-health-evidence-after artifacts/private/health/xid-after.json \
  --provider-health-evidence-after-sha256 artifacts/private/health/xid-after.sha256 \
  --provider-health-evidence-wait-seconds 30 \
  --stress-seconds 300 \
  --distributed-timeout 900
```

The provider-specific export command is intentionally not invented here. If the
provider cannot supply timestamped Xid evidence through an authenticated
channel, leave the fallback absent and treat unavailable local logs as the
warning they are.

After the selected branch is known, create a clean Python 3.11 environment and
install the corresponding PyTorch input followed by the common input. When an
offline cache is used, keep index access disabled and install only from the
verified directory, for example:

```bash
python -m pip install --no-index \
  --find-links artifacts/private/cache/wheels/cu128 \
  torch==2.10.0
python -m pip install --no-index \
  --find-links artifacts/private/cache/wheels/cu128 \
  -r requirements/rtx3090-common.in
python -m pip check
```

Then run the complete acceptance while billing is still usage-based:

```bash
python tools/preflight_rtx3090.py run \
  --phase full \
  --offline \
  --wheel-root artifacts/private/cache/wheels/cu128 \
  --wheel-manifest artifacts/private/cache/manifests/wheels-cu128.json \
  --model-root artifacts/private/cache/model/glm-asr-nano-2512 \
  --model-manifest artifacts/private/cache/manifests/model.json \
  --stress-seconds 300 \
  --distributed-timeout 900
```

The full phase launches exactly four local ranks. Every rank must execute a BF16
matrix multiplication that passes an FP32 oracle. A 64 MiB all-reduce must be
correct, the complete directed peer-access matrix must be observed, and all four
raw UUIDs must remain present after stress with released CUDA allocations. A
complete matrix may contain `false` entries: they describe the topology and do
not create a warning. Failure to capture the whole matrix does create a warning
and therefore cannot yield the pure pass needed to switch billing mode. The tool
captures `pip freeze --all` and `pip check` both before and after; any byte drift
or dependency error fails the run.

Before GPU stress, the full phase also creates a new
`fresh-offline-venv` under the private run directory. With index access disabled,
it first installs `setuptools==80.10.2` and `wheel==0.46.3` from the verified
cache, then installs both fixed requirements files. The editable repository is
installed with `--no-build-isolation --no-deps`, so pip cannot create a hidden
network-backed build environment or resolve a second dependency graph. The gate
then runs `pip check`, imports the package, and hashes
`fresh-offline-freeze.txt`. Any missing bootstrap wheel, unresolved transitive
dependency, or build failure rejects the cache; the already-installed operator
environment cannot substitute for this proof.

The all-reduce fields use decimal gigabytes per second, named `*_gb_per_s`:

```text
algorithmic_gb_per_s = payload_bytes / elapsed_seconds / 1e9
bus_gb_per_s = algorithmic_gb_per_s * 2 * (world_size - 1) / world_size
```

The 1.0 GB/s minimum is only a functional sanity floor. It is not a predicted
topology limit, a scaling claim, or a substitute for reporting the measured
rank-level and aggregate observations from the accepted node.

## Decisions and evidence

| Exit | Status | Required action |
|---:|---|---|
| 0 | `pass` | For `host`, continue hourly. For `full`, day rental and training are eligible. |
| 3 | `pass-with-warnings` | Manual review; do not switch billing mode or start training. |
| 2 | `fail` | Reject or repair the instance; do not train. |

The default report path is a unique directory below
`artifacts/private/rtx3090-readiness/`, which Git ignores. Exact GPU UUIDs are
retained in that private report; the tool does not fabricate anonymous values.
Its console summary contains no UUID. Do not publish raw reports, freeze files,
model files, or caches.

A full pass requires all of these target-host observations:

- exact package pins and identical pre/post freezes;
- verified wheel and model cache contents and provenance;
- four visible CUDA devices with BF16 support;
- captured PCIe, CPU/NUMA, GPU matrix, and supported peer-capability topology;
- four-rank NCCL correctness and the 1.0 GB/s functional floor;
- a complete peer-access matrix; `false` links remain ordinary observations;
- at least five minutes of nonzero BF16 work and 240 thermal samples per UUID;
- temperatures at or below 85 °C, identical post-run UUIDs, released contexts,
  no Xid in the explicit pre-existing window, and no new Xid during preflight.

If kernel logs and strict provider evidence are unavailable, the Xid gate is a
warning and therefore cannot produce the pure full pass required for day-rental
eligibility. Do not relabel an empty, inaccessible, stale, future-certifying, or
hash-mismatched observation as a pass.

## Still pending before the first accepted run

The repository can validate schemas and failure behavior without CUDA. It
cannot establish the following until the rented node is available: its actual
driver, exact UUIDs, topology and peer matrix, NCCL rate, BF16 execution,
five-minute thermal trace, Xid history, post-run cleanup, installed freeze, and
the byte identity of the caches copied to that node. Any numeric performance or
scaling statement remains unsupported until those private artifacts exist and
the measurement workflow accepts them.
