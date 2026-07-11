# Architecture decision records

Architecture decision records capture durable engineering choices that are
easy to lose when code moves. They describe the accepted decision and its
tradeoffs; executable contracts remain in the test suite and machine-readable
schemas.

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-assurance-terminology.md) | Accepted | Use a cumulative assurance ladder and do not call hashes authenticity |
| [0002](0002-shared-resource-policy.md) | Accepted | Pass one immutable resource policy through all untrusted-input work |
| [0003](0003-hash-taxonomy.md) | Accepted | Name digest aliases honestly and version every projection change |
| [0004](0004-api-stability.md) | Accepted | Keep a small stable API and explicit adapter and experimental tiers |
| [0005](0005-dependency-policy.md) | Accepted | Keep runtime dependency-free and pin development and CI dependencies |
| [0006](0006-experiment-boundary.md) | Accepted | Separate immutable planned inputs from append-only observed execution evidence |
| [0007](0007-platform-support.md) | Accepted | Claim only the platforms exercised by the supported verification matrix |
| [0008](0008-paper-publication-boundary.md) | Accepted | Keep the historical draft outside product distributions until validated evidence regenerates it |
