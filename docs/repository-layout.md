# Repository layout

The repository separates importable product code, experiment definitions,
generated evidence, and local-only material. Only the first two categories and
small synthetic tests belong in source control.

| Path | Owner and contract | Published content |
|---|---|---|
| `src/faster_glm_asr/` | Importable package | Python/Triton source and small text metadata |
| `configs/` | Experiment definitions | Schemas and reviewed JSON/JSONL/YAML configurations |
| `experiments/` | Isolated systems experiments | Source/config-driven experiments, never model-feature claims |
| `requirements/` | Environment inputs | Reviewed top-level dependency pins; generated freezes remain private |
| `tests/` | Automated verification | Source tests and tiny synthetic text fixtures |
| `tools/` | Repository operations | Validators and release-safety tooling |
| `docs/` | Technical documentation | Architecture, protocols, provenance, and limitations |
| `benchmarks/results/` | Reviewed public summaries | Only allowlisted matrix-summary JSON produced by `faster-glm-asr-public-matrix publish` |
| `.github/` | Repository automation | Least-privilege CI and contribution templates |

The following artifact classes are deliberately outside version control:

- downloaded model files and checkpoints;
- source or derived audio and third-party datasets;
- profiler captures and generated benchmark results;
- host-specific environments, caches, logs, and access material;
- unpublished institutional material and unrelated private planning notes.

An experiment may write local artifacts below an ignored directory. A public
matrix summary is rebuilt from a complete plan, append-only journal, and the
exact 18 private aggregates; it is not made by deleting fields manually. The
public schema contains only duration-bucket metrics and fixed protocol metadata,
and `publish` atomically creates a new file without an overwrite option. The
evidence checkout supplied by `--evidence-root` may be separate from the output
checkout supplied by `--publication-root`; private inputs are resolved only
against the former, while publication is restricted to a direct child of the
latter's `benchmarks/results/` directory.
Generated numbers do not become source truth merely because they are copied
into Markdown.

The package must not depend on files outside this repository. Any compatibility
reference to upstream code is identified by URL and revision in
[provenance.md](provenance.md).
