# Reproducible requirements inputs

The `.in` files are exact top-level candidates, not resolved transitive locks.
The complete wheel cache manifest and two identical `pip freeze --all` captures
form the acceptance evidence for a specific rented host.

`rtx3090-common.in` explicitly pins Python packages shared by both CUDA branches,
including `tokenizers==0.22.2`, `huggingface-hub==1.5.0`,
`safetensors==0.8.0`, `setuptools==80.10.2`, and `wheel==0.46.3`. Published
metadata for Transformers 5.14.0 requires Hub
`>=1.5.0,<2.0`, Tokenizers `>=0.22.0,<=0.23.0`, and Safetensors `>=0.8.0`;
Accelerate 1.12.0 and Datasets 4.5.0 accept the selected Hub release. Exact
candidates remove resolver ambiguity.

A cache is rejected if an exact top-level wheel is missing, any file has a
non-Linux/non-universal tag, or one distribution has multiple candidate
versions. Even a clean byte manifest remains candidate evidence until the full
gate creates a fresh Python 3.11 environment on the target Linux host, installs
exact Setuptools and Wheel first, installs both inputs, and installs this
repository with `--no-index --no-build-isolation --no-deps`. It must then pass
`pip check` and import, and hash the resulting freeze.

The preflight selects exactly one PyTorch branch from the host driver:

```bash
# Preferred branch when the cu128 driver gate passes.
python -m pip install -r requirements/rtx3090-cu128.in

# Or the sanctioned fallback, never both.
python -m pip install -r requirements/rtx3090-cu126.in

python -m pip install -r requirements/rtx3090-common.in
python -m pip check
```

`ci.in` is the small CPU-only dependency set used by repository CI; it is not a
GPU runtime lock and must not be substituted for either RTX 3090 branch.

Do not replace the host NVIDIA driver from inside a rented container. Run the
host-only gate first under hourly billing, select the branch reported by that
artifact, install from a prepared cache or approved network source, and then run
the full four-rank acceptance gate.
