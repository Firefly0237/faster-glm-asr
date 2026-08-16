# Provenance and third-party boundaries

This file records the source classes used to design Faster GLM-ASR. It is not a
substitute for per-file headers when code is copied or modified.

| Source | Fixed reference used for review | Declared license | Repository use |
|---|---|---|---|
| Edinburgh Machine Learning Systems 2026 | `525c3c4c3c584ab4c9e0ec7ff8d8ee202933d374` at `https://github.com/ed-aisys/edin-mls-26-spring` | CC0 1.0 Universal | Starting point for the model topology, weight mapping, full-prefix path, and selected kernels |
| Z.ai GLM-ASR source | `a324aa59a30952c54e8097c85efea592b7fafd81` | Apache License 2.0 | Architecture and public inference reference |
| GLM-ASR-Nano-2512 artifacts | `61ba4e0b3309b6656edea3e93e419f7bd5c61957` | MIT, as declared by the pinned model card | Runtime dependency downloaded by the user; never bundled |
| Hugging Face Transformers | `v5.14.0` | Apache License 2.0 | Native GLM-ASR compatibility reference |
| TinyStories dataset | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | CDLA-Sharing-1.0, as declared by the pinned dataset card | Optional private runtime input for the isolated decoder experiment; never bundled |
| LibriSpeech `test-clean` | OpenSLR resource 12, archive MD5 `32fa31d27d2e1cad72775fee3f4849a9` | CC BY 4.0 | Optional public-source ASR quality input; audio is downloaded separately and never bundled |

## Adaptation record requirements

For every copied or materially adapted source file, a change should record:

1. upstream repository and immutable revision;
2. upstream file path and license;
3. whether the local file is copied, adapted, or independently implemented from
   documented behavior;
4. a concise description of local modifications;
5. the correctness test that compares it with the selected reference.

Do not remove an upstream copyright, patent, trademark, attribution, or NOTICE
statement that still pertains to the distributed component. CC0 reference
material should still be identified for technical provenance even when
attribution is not a license condition.

## Model and data separation

The source-code license does not relicense model artifacts or datasets. A model
loader must expose the selected model identifier and revision, but the repository
must not vendor those files. Dataset and audio preparation is similarly external:
only schemas and synthetic test fixtures may be committed.

The pinned model-card license declaration is available at
[`zai-org/GLM-ASR-Nano-2512@61ba4e0`](https://huggingface.co/zai-org/GLM-ASR-Nano-2512/blob/61ba4e0b3309b6656edea3e93e419f7bd5c61957/README.md).
The TinyStories source declaration and redistribution caution are recorded in
[`configs/training/tinystories-source.json`](../configs/training/tinystories-source.json).
The LibriSpeech archive and deterministic subset policy are recorded in
[`configs/data/librispeech-test-clean-v1.json`](../configs/data/librispeech-test-clean-v1.json).

Project names are used to explain compatibility and origin. They do not imply
endorsement, partnership, or ownership of upstream trademarks.

The pinned Edinburgh reference history identifies Yangshen Deng and Yeqi Huang
(Chivier Humber) as upstream contributors to the relevant files. The repository
does not attribute that reference implementation to the maintainer of Faster
GLM-ASR. Ownership of later local cache and compatibility changes must be
confirmed before a release may describe them as an individual's work.
