# TinyStories byte-corpus preparation

The distributed decoder experiment uses a small, deterministic byte-level
corpus derived from TinyStories. Source and derived bytes are private run inputs;
this repository contains only the builder and its pinned source declaration.
No dataset file is bundled, and the corpus status remains pending until the
source files have been downloaded and verified locally.

This corpus exercises the training system. It is not an ASR fine-tuning dataset
and does not support a speech-quality claim.

## Pinned upstream inputs

The source declaration is
[`configs/training/tinystories-source.json`](../configs/training/tinystories-source.json).
It pins `roneneldan/TinyStories` at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

| Role | Upstream file | Advertised bytes | Advertised SHA-256 |
|---|---|---:|---|
| train | `TinyStories-train.txt` | 1,924,281,556 | `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f` |
| validation | `TinyStories-valid.txt` | 19,447,282 | `94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4` |
| license evidence | pinned dataset `README.md` | not advertised | verified and recorded after download |

The pinned dataset card declares `CDLA-Sharing-1.0`. The builder downloads that
card as license evidence and records its locally observed size and SHA-256 in
the completed manifest. The source declaration deliberately leaves a publisher
hash as `null` when the publisher has not supplied one; it never substitutes an
invented digest. Review the license terms before redistributing source bytes,
derived bytes, or model artifacts. This record is not legal advice.

## Private paths and commands

The default root is `artifacts/private/training-data/`, which Git ignores. The
CLI does not expose an artifact-root override, and rejects source or output
paths that escape that root.

To download the three pinned inputs, verify them, and build the fixed splits:

```bash
python tools/prepare_training_corpus.py --download
```

For an offline build, first place the three complete files under
`artifacts/private/training-data/sources/tinystories-f54c09fd/`, then omit the
download flag:

```bash
python tools/prepare_training_corpus.py
```

An absent source is reported as pending and produces no completed output. An
existing source is never replaced. A corrupt existing source fails before
derivation. The versioned output directory is also no-overwrite; choose a new
version only after intentionally changing the source or derivation contract.

## Deterministic derivation

The builder reads the multi-gigabyte train file incrementally and retains at
most one bounded document buffer plus selected document digests. It does not
load the full source into memory. Each non-empty UTF-8 document is stripped and
framed as:

```text
document + LF + <|endoftext|> + LF
```

Documents are selected in upstream order only when the complete framed document
fits the remaining byte budget. SHA-256 duplicates are skipped within a split
and across train/validation. The two splits also originate from distinct pinned
upstream files. Any small final gap is filled with LF bytes and its exact count
and fraction are recorded; a padding fraction above 0.5% fails closed.

The fixed outputs are:

| Split | Private path | Exact size |
|---|---|---:|
| train | `artifacts/private/training-data/derived/tinystories-64mib-4mib-v1/tinystories-train-64mib.bin` | 67,108,864 bytes |
| validation | `artifacts/private/training-data/derived/tinystories-64mib-4mib-v1/tinystories-validation-4mib.bin` | 4,194,304 bytes |

These are separate files, not slices of one shared byte stream.

## Evidence and loader contract

The output directory contains:

- `corpus-manifest.json`: `training-corpus-manifest-v0.1` source, license,
  derivation, per-split size/hash/document-count, padding, and isolation record;
- `corpus-manifest.sha256`: commit marker published after all other output files;
- `source-config.json`: exact snapshot of the public source declaration;
- `training-data.json`: the strict `byte_files` data object;
- the 64 MiB train and 4 MiB validation byte files.

`training-data.json` has exactly the keys accepted by
`experiments/distributed_training/train.py`:

```json
{
  "mode": "byte_files",
  "train_files": [
    "artifacts/private/training-data/derived/tinystories-64mib-4mib-v1/tinystories-train-64mib.bin"
  ],
  "validation_files": [
    "artifacts/private/training-data/derived/tinystories-64mib-4mib-v1/tinystories-validation-4mib.bin"
  ]
}
```

Run training commands from the repository root so these repository-relative
paths resolve consistently. The loader rejects duplicate or overlapping split
paths, hashes every byte file again, rejects identical train/validation file
content, and records the resolved files in the run's data fingerprint.

Do not add the generated manifest, data object, source files, derived files, or
their parent artifact directory to Git. Only the public source declaration,
builder, documentation, and offline fixture tests belong in a source release.
