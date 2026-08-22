# CommCanary documentation

Use this index to find the document that matches the work at hand.

## Understand the product boundary

- [Product status](product-status.md): Current readiness, qualified scope, and the public-claims boundary.
- [Architecture](architecture.md): Package responsibilities and dependency direction.
- [Platform support](platform-support.md): Supported Python versions, operating systems, and excluded environments.

## Run the tool

- [Command-line contract](cli.md): Commands, output channels, exit codes, and machine-readable output.
- [Python API stability](api.md): Stable imports and the documented API tiers.
- [Local scale benchmarks](benchmarks.md): Deterministic fixture generation and benchmark execution.
- [Operator quick start](operator-quickstart.md): Local orientation and the boundary before physical execution.
- [Artifact evaluation](artifact-evaluation.md): Repository verification and the Rostam evidence procedure.
- [Release runbook](release.md): Release preparation, package checks, and maintainer-only publication steps.

## Understand the contracts

- [Resource limits](resource-limits.md): Shared limits applied to untrusted input and expansion work.
- [Physical decision canaries](physical-canary.md): Bundle states, inputs, and decision-preservation requirements.
- [Wire-format compatibility](formats/compatibility.md): Format identifiers, schemas, producers, consumers, and migration policy.
- [Wire-contract decisions](formats/decisions.md): Normative decisions for loading, validation, canonical data, and compatibility.
- [Characterized Python imports and types](formats/public-api.md): Executable examples and the characterized public import surface.
- [Architecture decision records](adr/README.md): Accepted design decisions and their tradeoffs.

## Understand integrity and privacy

- [Integrity and claim dimensions](integrity.md): Assurance states and the properties each verification path demonstrates.
- [Metadata privacy and redaction](privacy.md): Disclosure risks, redaction boundaries, and review guidance.
