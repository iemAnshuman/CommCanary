# ADR 0009: License under Apache 2.0 and reserve the project name

Status: Accepted (2026-07-26)

## Context

CommCanary was published under the MIT license. The author, who holds sole
copyright, wanted the license to reserve rights: recognition for authorship and
the ability to derive income from the work.

A license cannot supply either of those directly. Copyright covers this
expression of the design, not the ideas behind it, and those ideas are
deliberately published in the accompanying paper. Anyone willing to spend the
effort can reimplement the tool regardless of the license chosen here. Only a
patent could prevent that, and no patent is sought.

Copyleft licenses were considered. AGPL-3.0 would force anyone distributing a
derivative to publish their source, and it would preserve a dual-licensing
option. It would also suppress adoption: many organizations refuse strong
copyleft dependencies by policy, and it would foreclose contributing this work
upstream to Apache-2.0 ecosystems such as MLCommons and Chakra. Source-available
licenses such as PolyForm Noncommercial reserve commercial use outright, at the
cost of no longer being open source, which weakens the artifact's standing for
publication and artifact evaluation.

Recognition follows citation and adoption; adoption is what restrictive terms
reduce.

## Decision

License the project under Apache License 2.0, and reserve the marks separately.

Apache 2.0 keeps the project open source and upstreamable while adding what MIT
lacks: an express patent grant with retaliation, and a mandatory attribution
requirement. Redistributors must preserve the copyright notice and the `NOTICE`
file, so the author's name travels with every derivative as a license condition
rather than as a courtesy.

Section 6 of the license grants no trademark rights. The `NOTICE` file states
this explicitly: derivative works may say they are based on CommCanary, but may
not adopt the name to brand or promote themselves. The name, not the code, is
the durable identifier of authorship.

The author retains copyright and may license future versions under different
terms, including a commercial or copyleft license, if a concrete opportunity
arises. Versions already released stay under the license they carried.

## Consequences

- `LICENSE` holds the verbatim Apache 2.0 text; `NOTICE` carries the copyright
  line and the trademark reservation, and ships in the wheel and sdist.
- Package metadata, the SBOM license identifier, and the release-metadata
  default declare `Apache-2.0`; the release gate asserts the declared value.
- Contributions are licensed under Apache 2.0 by section 5 unless a contributor
  states otherwise, so no separate contributor agreement is required for the
  current single-author project.
- Relicensing again is cheap only while authorship stays sole. Accepting outside
  contributions without a contributor licensing arrangement forecloses it.
- MIT grants already made for previously published versions are irrevocable and
  are not affected by this decision.
