# Security policy

## Supported versions

This project is pre-release. Security fixes are applied to the current default
branch; older commits and development snapshots are not supported release lines.

## Reporting a vulnerability

Please use the repository's **Private vulnerability reporting** form under the
GitHub Security tab. Do not open a public issue for a suspected vulnerability,
credential exposure, private-data leak, unsafe model-loading path, or dependency
compromise.

Include the affected revision, platform, minimal reproduction, impact, and any
suggested mitigation. Remove model weights, audio, access tokens, signed URLs,
hostnames, and personal data from the report unless they are strictly necessary;
when they are necessary, describe them first and agree on a secure transfer
method with the maintainers.

Maintainers will acknowledge a report when available, assess scope, and coordinate
a fix and disclosure. This document intentionally makes no fixed response-time or
remediation-time promise.

## Scope and trust boundaries

Security reports may cover this repository's Python/Triton code, artifact parsing,
model-loading controls, benchmark input validation, and dependency configuration.
Model behavior, upstream model files, dataset licensing, and vulnerabilities in
third-party services should also be reported to the relevant upstream maintainer.

Never load untrusted pickle-based checkpoints. Prefer immutable revisions and
safe tensor formats, verify downloaded artifacts, and review any upstream custom
code before enabling it.
