# Security policy

## Supported versions

Security fixes are provided for the latest released minor version and the
current default branch. Older artifact formats may remain readable for
compatibility, but an old package release is not thereby security-supported.

## Report a vulnerability privately

Use GitHub's private vulnerability-reporting flow for this repository:

<https://github.com/iemAnshuman/commcanary/security/advisories/new>

If that flow is unavailable, contact the maintainer through the repository
owner's GitHub profile and request a private channel. Do not include exploit
details, real workload traces, cluster identifiers, credentials, hostnames, or
other sensitive metadata in a public issue.

Please include, when safe:

- affected version/commit and platform;
- the smallest synthetic reproducer;
- expected and observed behavior;
- whether the issue involves path containment, unbounded work, integrity status,
  aliasing, command execution, or metadata disclosure; and
- any known mitigations.

You should receive an acknowledgement within seven days. Disclosure timing and
credit will be coordinated after the issue is reproduced and a mitigation is
available. Please do not probe infrastructure or data you do not own or have
permission to test.

## Security model

CommCanary treats input files, compressed expansion counts, capture metadata,
and experiment records as untrusted. The shared resource policy, path
containment, duplicate-key rejection, checked arithmetic, immutable experiment
records, and independent verification paths are defense-in-depth boundaries.
They do not make the Python process a sandbox.

SHA-256 commitments are tamper evidence relative to trusted source material;
they are not signatures and do not authenticate a producer. See
[`docs/integrity.md`](docs/integrity.md). Deployments handling hostile input
should also use OS-level time, memory, filesystem, and process isolation with a
stricter [`ResourceLimits`](docs/resource-limits.md) policy.

The capture command executes an explicitly supplied workload command. Treat it
with the same authority as running that command directly. CommCanary does not
download or execute cluster jobs as part of its normal library or test suite.

