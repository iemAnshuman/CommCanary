# Release runbook

This runbook prepares and verifies a release; it does not authorize creating a
tag, publishing a GitHub release, or uploading to PyPI. Those are explicit
maintainer actions after review.

## 1. Freeze release identity

Choose one version and make these agree:

- `[project].version` in `pyproject.toml`;
- the `CHANGELOG.md` release heading and date;
- Git tag `v<version>`; and
- GitHub release name `v<version>` (an empty release name is also accepted by
  the workflow).

Review the format support matrix, public API/CLI changes, security fixes,
migrations, platform support, and experiment claims. The release commit must be
clean; do not build from an unreviewed dirty checkout.

During development keep the heading `## <version> - Unreleased`. Immediately
before the release-candidate gate, replace `Unreleased` with the intended ISO
release date. Release mode rejects a missing, duplicate, malformed, or still
unreleased version heading.

## 2. Install the constrained release toolchain

```bash
python3 -m venv .release-venv
source .release-venv/bin/activate
python -m pip install --disable-pip-version-check ".[dev]"
```

The declared ranges make upgrades deliberate but are not a hash-locked release
environment. Record `python -m pip freeze` with release evidence until a fully
hashed toolchain lock is adopted.

## 3. Run the exact-artifact gate

Start with empty output paths:

```bash
python -m tools.verify --release \
  --expected-version 0.3.0 \
  --artifact-dir dist \
  --metadata-dir release-metadata
```

The command runs the canonical source gate, builds under a fixed
`SOURCE_DATE_EPOCH`, normalizes and compares two builds byte-for-byte, inspects
wheel/sdist contents and metadata, installs the exact wheel outside the checkout
with source-path overrides removed, runs installed-package/CLI/type tests, and
only then writes:

- the tested wheel and sdist under `dist/`;
- `release-metadata/SHA256SUMS`;
- the canonical archive/member inventory; and
- an SPDX 2.3 JSON SBOM.

Metadata generation rehashes the same in-memory artifact set and rejects drift
from the post-install-test hashes. Output directories are collision-safe and
must be absent or empty.

Review archive contents, checksums, SBOM, optional dependency metadata, README
rendering, and the installed `commcanary --version` output. Preserve the gate
log, Python/toolchain freeze, commit, and generated metadata as release evidence.
The wheel contains only the typed runtime package and packaged schemas. The
sdist additionally contains the reviewed docs, examples, top-level schemas,
benchmarks, experiment sources, tests, and verification tools referenced by the
project. The historical paper draft is deliberately not a release artifact
until a new complete campaign regenerates its tables and claims. Private
context, local results, caches, and generated metadata are rejected by the
staging and member-policy checks.

## 4. Create and publish deliberately

After review, create the signed/annotated tag and GitHub release through the
normal maintainer process. Publishing the GitHub release triggers
`.github/workflows/publish.yml` at that exact tag.

The workflow deliberately separates authority:

1. the `build` job has read-only repository permission, checks identity, runs
   the release gate, and uploads immutable `release-distributions` and
   `release-metadata` workflow artifacts;
2. the `publish` job receives only the exact distribution artifact, runs in the
   protected `pypi` environment, and alone receives `id-token: write`; and
3. the pinned PyPI publisher uses trusted publishing and uploads signed digital
   attestations for each distribution.

Do not rebuild in the publish job, manually substitute a file in `dist/`, use a
long-lived PyPI token, or bypass the protected environment.

## 5. Post-publish verification

In a new environment with no checkout path overrides:

```bash
python3 -m venv .post-release-venv
source .post-release-venv/bin/activate
python -m pip install --disable-pip-version-check "commcanary==0.3.0"
commcanary --version
commcanary --help
```

Compare downloaded distribution hashes with `SHA256SUMS` and PyPI's published
hashes/attestations. Run a tiny documented compile/replay/compare pipeline, then
confirm the Pages demo and links. If any identity, hash, install, help/version,
or smoke check fails, stop promotion and publish an incident/yank decision; do
not overwrite an existing release artifact.

## Rollback and yanking

PyPI files and Git tags are immutable evidence. Never replace an uploaded file
under the same version. For a serious defect, document it, yank the affected
version when appropriate, fix forward with a new version, and preserve the old
release metadata for auditability.
