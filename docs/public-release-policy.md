# Public release policy

`tools/public_release_guard.py` applies a fail-closed policy to the files that Git
would publish. The normal mode scans tracked and unignored untracked files. The
release mode additionally requires final repository metadata; history mode scans
every reachable Git blob so removing a file in the latest commit is insufficient.

## Allowlisted surface

Only standard package, test, configuration, benchmark-harness, example,
documentation, tool, and repository-automation directories are accepted. Root
files are limited to project metadata, build configuration, dependency locks, and
container/environment definitions. Unknown top-level paths fail review and must
be added intentionally in code and tests.

Allowed files are UTF-8 text from a conservative extension set and are bounded in
size. Symbolic links and binary payloads fail. Model files, audio, archives,
office documents, profiler captures, and raw generated results are not source
files. The sole generated-evidence exception is a checked public matrix summary
below `benchmarks/results/`. The release procedure creates that file with
`faster-glm-asr-public-matrix publish` from all 18 admitted private aggregates;
the guard can enforce its public shape and content policy, but cannot infer the
command that produced an otherwise valid JSON file.

The public matrix schema is an exact nested allowlist. It has no per-utterance
rows and rejects transcripts, generated-token values, audio or manifest
fingerprints, prepared-input fingerprints, absolute paths, Git branches, UUIDs,
PCI addresses, CUDA visibility settings, hostnames, commands, and logs. It also
omits WER/CER from the three-item performance subset. Only the hardware class,
software versions, protocol, duration-bucket latency/RTF, bounded request-memory
metrics, exact-token parity gate, and five fixed baseline-to-candidate comparisons
are admitted.

The five ratios stay within prepared-request-compatible runtime families: one
Hugging Face cache ablation and four custom-runtime cache/generation ablations.
No direct Hugging Face-to-custom speedup is published. Matrix-wide exact-token
parity still covers all six implementations as a correctness gate, but does not
assert that the Hugging Face and custom prepared-request tensor contracts are
identical.

For a strict release, the guard first requires the ordinary stage-zero schema
blob selected for commit to equal the non-reparse worktree schema byte for byte.
It then uses that verified current validator for every worktree, index, and
historical `benchmarks/results/*.json` blob; historical Python is never executed.
Invalid JSON, duplicate keys, non-finite numbers, validator import failure, and
schema-origin drift all fail closed. A published file is created atomically and
never overwrites an existing result.

The public evidence hashes are audit handles recomputed by `export` and
`publish`, not self-authenticating proof. `check` and the release guard validate
only the exact public structure, metric consistency, and privacy allowlist. In
the absence of the private evidence or a separately trusted attestation, those
hash strings do not prove that a file originated from `publish`.

## Content checks

The guard rejects common credential formats, private-key blocks, authorization
headers, local absolute paths, embedded Git credential URLs, Git LFS pointers,
and terminology associated with excluded personal or non-public project material.
It also checks that LICENSE and NOTICE preserve the expected license and upstream
boundaries.

Pattern scanning reduces accidental disclosure; it cannot prove that prose,
images, or obfuscated values are safe. A human reviewer must still inspect the
complete release diff and third-party provenance.

## Commands

During development:

```bash
python tools/public_release_guard.py --root .
```

For a release candidate after the final `README.md` exists and the draft is gone:

```bash
python tools/public_release_guard.py \
  --root . \
  --strict-release \
  --include-history
```

The command exits non-zero and lists every finding. There is no warning-only mode
and no path-level suppression flag; policy changes require a reviewed code change
with a regression test.
