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
office documents, profiler captures, and generated results are not source files.

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
