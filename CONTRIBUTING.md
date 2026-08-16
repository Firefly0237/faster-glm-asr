# Contributing

Thank you for improving Faster GLM-ASR. Keep each change reviewable, reproducible,
and explicit about its provenance.

## Development workflow

1. Create a focused branch from the current default branch.
2. Use Python 3.11 or 3.12 in an isolated environment.
3. Install the package and development tools with `python -m pip install -e ".[dev]"`.
4. Add or update tests for every behavior change.
5. Run `python -m pytest` and the public-release guard before opening a pull request.
6. Explain correctness checks, hardware assumptions, and any untested boundary in
   the pull request description.

Run the release guard from the repository root:

```bash
python tools/public_release_guard.py --root .
```

Maintainers run the stricter history-aware form for a release candidate:

```bash
python tools/public_release_guard.py --root . --strict-release --include-history
```

## Correctness and benchmark changes

- Compare optimized paths against a pinned reference implementation across
  representative shapes, dtypes, boundary lengths, and cache states.
- Treat warm-up, synchronization, sample count, generation settings, hardware,
  software versions, and input manifest as part of the benchmark definition.
- Commit schemas and small synthetic fixtures, not generated performance output.
- Never report a performance, memory, or quality number without the raw artifact
  and environment metadata required to reproduce it.
- Mark design-only, fixture-only, and unverified hardware paths clearly.

## Provenance and data safety

- Record the source URL, immutable revision, component license, and modifications
  for adapted code.
- Preserve applicable source headers and NOTICE text.
- Do not submit model weights, private audio, datasets, access credentials,
  profiler captures, machine-specific paths, or generated checkpoints.
- Use synthetic fixtures for tests. Public datasets must be downloaded separately
  under their own terms and must not be silently vendored.

## Licensing contributions

Unless you state otherwise in writing, a contribution intentionally submitted for
inclusion is provided under the Apache License 2.0, as described in section 5 of
the project LICENSE. Only submit work you have the right to contribute.
