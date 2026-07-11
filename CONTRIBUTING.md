# Contributing to CommCanary

CommCanary artifacts make integrity and regression claims, so changes are
reviewed as contract changes first and code changes second. Keep patches small,
preserve independent verification paths, and add an executable boundary or
mutation test for every changed claim.

## Development setup

CommCanary supports CPython 3.9 through 3.13. Create an isolated environment and
install the constrained development extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the same fast gate used by CI while iterating:

```bash
python -m tools.verify --fast
```

Before requesting review, run the full artifact gate:

```bash
python -m tools.verify
```

The full gate builds a wheel and sdist, inspects their contents and metadata,
installs the exact wheel outside the checkout with `PYTHONPATH` removed, and
runs the installed-package tests and command-line smoke checks. Release changes
must also pass `python -m tools.verify --release` under a fixed source epoch.

Do not rely on `PYTHONPATH=src`; it can hide packaging failures. Do not commit
caches, generated benchmark data, experiment raw results, build directories, or
editable-install metadata.

## Change discipline

- Fix or characterize behavior before moving its implementation.
- Keep the producer and the independent verifier separate. A shared codec is
  appropriate; a shared calculation of the value being attested is not.
- Treat changes to canonical JSON, hashes, schemas, defaults, reason codes,
  exit codes, and public imports as compatibility changes.
- Never raise a resource ceiling implicitly to accept a fixture. Pass an
  explicit `ResourceLimits` policy and test the exact boundary.
- Public functions consume caller-owned input as read-only and return detached
  JSON snapshots.
- Every persisted artifact or experiment record needs a stable schema ID,
  collision-safe write policy, and tamper/staleness test.
- Keep physical-cluster instructions and site credentials out of normal tests.
  Local experiment tests must not invoke SLURM or contact a cluster.

Use literal golden fixtures where exact bytes or hashes are the contract. Use
parameterized/property-style tests for malformed-field families. A test should
assert a stable status/reason code when one exists rather than parse prose.

## Pull requests

Describe:

1. the user-visible or wire-contract change;
2. the independent evidence that detects a broken implementation;
3. compatibility, privacy, resource, and performance consequences;
4. the exact local verification commands run; and
5. any gate that could not be run and why.

Generated output belongs in a reviewable commit only when it is intentionally
part of the contract. Do not mix unrelated formatting and semantic changes.

## Reporting security problems

Do not open a public issue for a suspected vulnerability, unsafe parser case,
path escape, denial-of-service vector, or disclosure of sensitive trace
metadata. Follow [SECURITY.md](SECURITY.md).

