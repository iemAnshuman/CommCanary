# ADR 0003: Hash taxonomy and evolution

- Status: Accepted
- Date: 2026-07-11

## Context

CommCanary artifacts contain several SHA-256 fields. Some protect different
projections, while two historical pairs are aliases: source-normalized and
source-trace commitments currently cover the same bytes, and scheduler and
execution commitments currently cover the same execution projection. Distinct
names can otherwise suggest independent guarantees that do not exist.

Changing a projection in place would make a format identifier describe two
different protocols and would invalidate cross-version golden vectors.

## Decision

Every digest is identified by its exact projection and canonicalization
version. Documentation calls an equal-value pair an alias, not an independent
attestation. The 0.3 reader and writer preserve the existing fields for v2
compatibility; a future intentional format revision may remove an alias or give
it a genuinely different, newly named projection.

Projection or canonicalization changes require a new version identifier,
literal cross-language vectors, a compatibility-table update, and an explicit
migration decision. Volatile build or run provenance belongs in a manifest or
an excluded provenance field, not in a semantic digest merely to make the hash
look more comprehensive.

Embedded hashes provide tamper evidence relative to trusted content or a
trusted digest. They do not authenticate a producer. Source-assisted and
model-recomputed verifiers remain separate from producer calculations when
their value comes from independent disagreement.

## Consequences

- Equal historical aliases remain readable without overstating their meaning.
- A semantic hash never changes silently under an existing format ID.
- Golden vectors are protocol fixtures and are not regenerated to accommodate
  an accidental implementation change.
- Signatures, package attestations, and source correspondence remain separate
  assurance mechanisms.

## Alternatives considered

- Giving aliases different values without a format revision was rejected as a
  wire-contract break.
- Removing them immediately was rejected because v2 compatibility is promised.
- Treating every SHA-256 field as an independent proof was rejected because the
  protected projections, not the labels, determine the guarantee.
