# Future plan: A-Square Systems + CommCanary

> [!IMPORTANT]
> This is a forward-looking business and product plan, not a record of verified
> CommCanary evidence. `HANDOFF.md` remains authoritative for the live repository
> and Rostam campaign checkpoint. Before executing cluster work, reconcile every
> study step, hash, runner, and scheduler action against that handoff.

## 1. Business structure

Do not create two companies or two unrelated products.

```text
A-Square Systems
│
├── Distributed Inference Performance Diagnostic
│     Immediate revenue and customer discovery
│
├── Optimization Sprint
│     Implement measurable performance improvements
│
├── Private performance-delivery toolkit
│     Make the service repeatable and profitable
│
└── CommCanary
      Flagship R&D project
      Internal service advantage
      Potential recurring software product
```

The commercial mission is:

> Make expensive distributed inference faster, cheaper and safer to change.

CommCanary's product hypothesis is:

> Given a private distributed-AI workload and regression policy, produce the
> shortest physical test that safely predicts whether a stack change should ship,
> with enough evidence to audit the decision.

Chakra is the execution-trace carrier. CommCanary owns policy-conditioned
minimization, asymmetric regression safety, and the evidence bundle.

## 2. Codebase diagnosis

CommCanary's technical implementation is ahead of its commercial validation. The
repository already has:

- Chakra ET ingestion and graph validation.
- Dependency-closed physical-region selection.
- Counterexample-guided candidate search.
- Training and holdout separation.
- vLLM and SGLang runner contracts.
- Content-addressed artifacts and append-only experiment evidence.
- JSON, HTML, JUnit, and SARIF gate outputs.
- Signed private-exchange bundles.
- Integrity, privacy, packaging, release, and supply-chain controls.

The existing exact-work experiment is encouraging: 26 of 28 configuration pairs
agreed, Kendall tau-b was 0.857, and the point-estimate criteria passed. The final
result was correctly declared **inconclusive** because confidence intervals
crossed decision boundaries and one configuration was unstable. That experiment
replayed the complete source-derived program, so it proves neither physical
reduction nor cost savings.

The current constraints are:

1. Stable physical decision evidence.
2. A workflow that an independent operator can use.
3. A customer who values the outcome.

Freeze broad feature development. The next unit of progress is evidence or
revenue, not another subsystem.

### Technical readiness checkpoint — August 12, 2026

The local implementation boundary for the future plan is complete:

- `docs/product-status.md` defines the evidence and claims boundary.
- `claims/public-claims.yaml` is schema-validated and mirrored in the package.
- `docs/operator-quickstart.md` covers blocked orientation, runner and study
  handoff, terminal statuses, verification, recovery, and friction reporting.
- The README now presents the narrow product surface; the longer exact-replay
  and qualification material lives in `RESEARCH_HISTORY.md`.
- Instrumented capture can carry the source-bound reduction and GEMM recipe
  needed to emit a physical Chakra projection.
- The local `capture -> build` orientation produces a verified
  `blocked_missing_physical_oracle_corpus` bundle instead of inventing evidence.
- The canonical fast and reproducible verification gates pass.

This does **not** make CommCanary a validated product. The remaining technical
work is physical: build the digest-pinned runner, freeze and execute the vLLM
study, record the held-out outcome, and run the independent-operator trial.

## 3. Scope freeze

The first physical study supports exactly:

```text
single-node
four NVIDIA GPUs
tensor-parallel dense-decoder inference
vLLM first, SGLang second
GEMM + all-reduce dependency structure
```

Four-to-eight NVIDIA GPUs is the intended near-term commercial scope. Do not
claim that range until the four-GPU study passes and a separately frozen
eight-GPU study or customer workload supplies evidence.

The codebase does not yet qualify expert-parallel all-to-all, multi-node
communication, CUDA graphs, multi-stream execution, KV-cache pressure, or network
congestion. Do not advertise those capabilities.

Defer:

- Multi-node support.
- AMD or TPU support.
- Training workloads.
- Expert parallelism.
- Generic profiler functionality.
- A hosted dashboard.
- Kubernetes control planes.
- Automated NCCL tuning.
- Another benchmark runner.
- Another startup product.
- General "GPU observability."

The existing `capture -> build -> gate` product surface is sufficient. It already
exists and verifies generated bundles immediately.

## 4. Artifacts to maintain

Maintain five artifacts.

### Public artifacts

#### CommCanary

The technical product and research proof.

#### Evidence-led case studies

Each case study includes:

- The exact workload.
- Hardware and software environment.
- Frozen policy.
- Full application ground truth.
- Canary result.
- Runtime and GPU-cost reduction.
- False negatives and false positives.
- Configuration agreement.
- Raw or appropriately redacted evidence.
- Limitations.
- Reproduction instructions.

#### Company website

The commercial explanation, service offer, application form, and case studies.

### Private artifacts

#### `asquare-performance-kit`

A private service-delivery repository:

```text
asquare-performance-kit/
├── intake/
├── environment-collector/
├── benchmark-plans/
├── experiment-ledger/
├── profiler-adapters/
├── bottleneck-checklists/
├── report-generator/
├── roi-calculator/
├── redaction/
└── engagement-templates/
```

It should automate:

```text
customer intake
      ↓
environment snapshot
      ↓
reproducible baseline
      ↓
profiling
      ↓
experiment matrix
      ↓
before/after validation
      ↓
ROI estimate
      ↓
technical + executive report
```

This repository remains private. It is a delivery tool, not a separate product.

#### `social-narrator`

The private `NARRATIVE.md` ingestion, X drafting, verification, and scheduling
system.

## 5. Execution timeline

### August 13-16, 2026: freeze and present

#### Product

Create `docs/product-status.md` containing:

- Current validated capabilities.
- Current unvalidated hypotheses.
- Allowed public claims.
- Forbidden claims.
- Supported domain.
- Evidence required for each future claim.

Create a machine-readable claim registry:

```yaml
# claims/public-claims.yaml

product_status: research_alpha

allowed:
  - capture_build_gate_exists
  - chakra_dependency_closed_selection_exists
  - exact_work_point_estimates_published
  - private_exchange_is_implemented

qualified_only_after_evidence:
  - predicts_stack_change_decisions
  - reduces_physical_runtime_by_10x
  - safe_for_ci_gating
  - preserves_held_out_regressions

forbidden:
  - production_validated
  - privacy_safe
  - works_across_arbitrary_models
  - multi_node_qualified
```

The X narrator, website generator, and README should consume this file so that
marketing language cannot drift beyond the evidence.

#### Website

Put the company website live without waiting for CommCanary qualification. Keep
the existing GitHub Pages demo as a technical demo. It demonstrates the older
synthetic compile/replay/compare path and says that reduced physical decision
fidelity remains unproven; it is not the commercial homepage.

#### Social system

Create the narrative format and run the X generator in dry-run mode.

#### Sales

Build the first 30-50-account prospect list and request five warm introductions.

### August 17-31, 2026: execute the vLLM study

This is the most important technical work. The predeclared study has:

- One baseline.
- Seven training perturbations.
- Three untouched holdout perturbations.
- Eight independent application allocations across at least two days.
- Real vLLM application measurements.
- Active physical-candidate execution.
- Exact replay, random, stratified, ddmin, and communication-only baselines.
- Selection frozen before holdout evidence becomes available.

Execute it in this order:

1. Build and verify the digest-pinned vLLM runner.
2. Review the workload and projection.
3. Freeze the physical-canary policy.
4. Freeze the training and holdout perturbations.
5. Freeze the study directory.
6. Verify every input and runner artifact.
7. Run the application baseline allocations.
8. Run active candidate synthesis on training perturbations.
9. Freeze the selected candidate.
10. Open the holdout callback.
11. Execute held-out application and candidate measurements.
12. Generate the result, evidence bundle, and audit bundle.
13. Reproduce the analysis from immutable bytes.
14. Do not change thresholds after seeing results.

The cluster workflow for runner construction, study freezing, submission,
resumption, and final evidence generation already exists. Follow the live hashes,
attempt inventory, and operator instructions in `HANDOFF.md`.

#### Required decision gates

Do not weaken these gates because a near-pass looks commercially useful:

- At least 10x physical runtime reduction.
- Zero false negatives for severe held-out regressions.
- No more than 10% false positives.
- At least 90% pair-decision agreement.
- Stability across independent allocations and days.
- A second engine or materially different workload.
- An independent operator completing the workflow.
- A digest-pinned runner.

#### Distribution during the study

Post only what the evidence supports:

- Experimental setup.
- Failures.
- Methodological decisions.
- Surprising observations.
- Why holdout evidence is unavailable during selection.
- Why a promising point estimate can still be inconclusive.

Send 15 researched prospect messages per week. Social activity does not replace
outbound.

### September 1-15, 2026: branch on evidence

#### Outcome A: all core gates pass

1. Publish the complete vLLM case study.
2. Conduct the independent-operator trial.
3. Freeze and run the SGLang replication.
4. Open three private design-partner slots.
5. Run the release gate.
6. Publish CommCanary 0.3.0 as a clearly labelled research alpha.
7. Add one physical-product demo to the website.

#### Outcome B: the result is inconclusive

1. Publish the inconclusive result with its limitations.
2. Determine whether instability came from the application, runner, hardware
   allocation, or policy.
3. Design a new predeclared study, not a favourable retry.
4. Keep selling performance diagnostics.
5. Keep CommCanary internal during customer work.

#### Outcome C: the reduced canary fails safety or 10x reduction

1. Do not sell it as a fast performance gate.
2. Reframe it as a verifiable exact workload capsule or research tool.
3. Test whether simple stratified sampling performs equally well.
4. Use service engagements to find a stronger product wedge.

If the narrow physical study misses its gates, describe the product as an exact
workload capsule, not a fast regression canary. A negative result invalidates one
product hypothesis, not A-Square Systems.

### September 16-October 31, 2026: secure customer one

#### Founding offer

> **7-Day Distributed Inference Performance Diagnostic**

Price the first two engagements at **$4,000 each**.

Deliver:

- Reproducible production-shaped benchmark.
- Hardware, software, and environment manifest.
- Throughput, TTFT, TPOT, and tail-latency baseline.
- GPU, runtime, and communication bottleneck analysis.
- Ranked experiment plan.
- Implementation of one or two low-risk fixes.
- Before/after measurements.
- Technical report.
- One-page executive report.
- A regression test or CommCanary artifact only where technically valid.

Terms:

- 100% payment before kickoff.
- Credit the entire diagnostic fee toward an optimization sprint started within
  30 days.
- No guaranteed improvement percentage.
- No 24/7 support.
- The customer supplies representative hardware and a reproducible workload.

After two strong engagements, increase the diagnostic price to
**$7,500-$12,500**.

#### Optimization sprint

- Two to four weeks.
- Initially $15,000-$30,000.
- Fixed scope.
- 50% at signing, 25% after baseline acceptance, and 25% at delivery.

#### First-customer targets

Prioritize teams with:

- Self-hosted open-weight inference.
- vLLM or SGLang.
- Approximately 4-64 NVIDIA GPUs.
- A small infrastructure team.
- A recent migration, performance regression, or cost problem.
- A technical buyer who can approve a $4,000 experiment without enterprise
  procurement.

Avoid API-only startups and teams with no reproducible workload.

#### Weekly acquisition cadence

Every week:

- Research 20 companies.
- Send 15 personalized emails or X DMs.
- Request two warm introductions.
- Hold three customer-research or sales conversations.
- Publish three to five technical X posts.
- Make ten useful technical replies manually.
- Prepare at least one proposal when the pipeline supports it.

Cold-message template:

> I saw that you're using `<runtime/configuration>` and currently
> `<hiring/migrating/scaling>`.
>
> I work on distributed runtime and collective-communication performance,
> including HPX collectives through GSoC. I'm opening two fixed-scope diagnostics
> for multi-GPU inference teams: establish a reproducible baseline, identify the
> expensive bottlenecks, implement the highest-confidence fixes, and leave a
> regression test.
>
> Is `<specific performance problem>` currently important, or is another part of
> the stack more painful?

Attach one relevant experiment, not a pitch deck.

### November 2026-January 2027: turn delivery into a system

Targets:

- Three to five paying customers.
- Two publishable case studies.
- At least one referral.
- One recurring performance contract.
- A record of which parts of CommCanary customers repeatedly use.
- A private delivery toolkit that reduces engagement effort.

Do not productize CommCanary because it is technically impressive. Productization
requires:

1. Three customers with the same recurring expensive workflow.
2. Two willing to pay recurring software fees.
3. At least 10x test-cost or runtime reduction.
4. Held-out regression safety.
5. Installation in less than one working day.
6. Customer data remaining inside their infrastructure.
7. An independent operator succeeding from documentation.
8. No simple script or existing tool solving the complete workflow.

### February-August 2027: productized service or software

#### If CommCanary proves valuable

Sell a private design partnership:

> **CommCanary Design Partnership: $15,000-$25,000 for six months**

Include:

- Self-hosted installation.
- One supported runtime.
- Three generated and validated canaries.
- Monthly refresh.
- Stack-change qualification.
- Direct engineering support.
- No exclusivity.

#### If another recurring problem is stronger

Productize that problem instead. Candidate problems include upgrade
certification, reproducible inference-performance environments, configuration
selection, communication regression diagnosis, and topology qualification.
CommCanary can remain an internal measurement engine, open-source artifact, or
research project.

## 6. Required repository updates

### Simplify the README

The README currently explains the research history, exact capsule, old simulator
flow, physical path, qualification path, and product path together. Its opening
should contain only:

1. The problem CommCanary aims to solve.
2. Current status.
3. Supported domain.
4. The three-command `capture -> build -> gate` workflow.
5. One result/status table.
6. Links to research history and detailed contracts.

Move older exact-replay and qualification material into clearly labelled research
documentation. Preserve it, but do not require every new reader to work through it
first.

### Add a claim-status table

| Claim | Status |
| --- | --- |
| Chakra physical graph ingestion | Implemented and tested |
| Dependency-closed candidate construction | Implemented and tested |
| Training/holdout isolation | Implemented and tested |
| Exact-work configuration fidelity | Promising but inconclusive |
| Reduced physical runtime | Unmeasured until study |
| Held-out regression safety | Unproven |
| Production CI gate | Not validated |
| Multi-node support | Unsupported |

Preserve the README's current evidence boundary: 0.3.0 is unreleased, no
application corpus or held-out result is checked in, and CommCanary is not yet a
validated performance gate.

### Add an operator quick start

Create `docs/operator-quickstart.md`. It should let an independent engineer:

1. Install a pinned artifact.
2. Run a local blocked or synthetic example.
3. Understand why it is blocked.
4. Build a runner.
5. Freeze a study.
6. Execute or hand off the physical steps.
7. Verify the result.
8. Interpret every terminal status.
9. Recover from partial runs.
10. Report friction.

The operator trial is a release gate, not only a documentation test.

### Delay PyPI publication

CommCanary is not yet on PyPI, and the final release-identity gate has not run.
Publish 0.3.0 only after:

- Product-status documentation is complete.
- The vLLM study has a recorded outcome.
- An independent operator completes the workflow.
- `python -m tools.verify --release` passes.
- The website and README make no stronger claim than the evidence.

The release can remain a research alpha after a negative study result.

## 7. X narrative system

### Principle

> The AI may select, compress, and phrase what happened. It may never invent what
> happened, how the author felt, or what the evidence means.

This system distributes recorded work without interrupting the work. `GUIDE.md`
remains the factuality and writing layer. It forbids fabricated facts,
first-person experience, fake typos, staged slang, and engineered sentence
variance. It also requires social posts to follow platform norms without
manufactured hooks, engagement prompts, or hashtag blocks.

### Workspace structure

```text
~/work/
├── .agent/
│   └── skills/
│       └── x-narrator/
│           └── SKILL.md
│
├── social/
│   ├── VOICE.md
│   ├── NEVER.md
│   ├── CLAIMS.yaml
│   ├── projects.yaml
│   ├── queue.sqlite
│   ├── post-log.jsonl
│   ├── scheduler.yaml
│   └── audit/
│
├── CommCanary/
│   └── NARRATIVE.md
│
├── hpx-gpu-aware/
│   └── NARRATIVE.md
│
└── other-project/
    └── NARRATIVE.md
```

Each `NARRATIVE.md` should be gitignored by default. Public entries may be
exported later through an explicit decision.

### Narrative entry format

```md
## cc-2026-08-12-001: exact-work result stayed inconclusive

timestamp: 2026-08-12T21:30:00+05:30
visibility: public
claim_status: measured
arc: physical-decision-fidelity
share_after: 2026-08-13
expires_after: null

What happened:
Exact-work replay agreed on 26/28 configuration pairs.
The point-estimate gates passed, but the policy result was inconclusive.

Why:
One configuration exceeded the stability limit and bootstrap intervals
crossed decision boundaries.

My reaction:
Annoying, but the refusal is correct.

Evidence:
- experiments/rostam/results/...
- commit: <sha>
- policy: <sha>

Uncertainty:
This was complete-program replay. It says nothing yet about physical
runtime reduction.

Next:
Run the reduced physical-canary study against actual application decisions.
```

Required fields:

- Stable entry ID.
- Timestamp.
- Visibility.
- Claim status.
- Narrative arc.
- What happened.
- Evidence.
- Uncertainty.
- Next action.

`My reaction` must be written by the author or omitted. The agent cannot infer
frustration, excitement, embarrassment, or confidence from a failed command.

Corrections are append-only:

```text
correction_of: cc-2026-08-12-001
```

Never silently rewrite a historical entry after it has generated a post.

### Automatic narrative capture

The coding agent may append objective events after:

- A benchmark completes.
- A test fails.
- A bug is located.
- A PR is opened or merged.
- An experiment changes direction.
- A hypothesis is rejected.
- A result crosses a predeclared gate.
- A reviewer changes the author's understanding.

It may automatically capture:

- Commands.
- Exit status.
- Diff summary.
- Metrics.
- Artifact paths.
- Commit SHA.
- Test names.
- Error messages.

It may not automatically capture:

- Emotions.
- Motives.
- Social opinions.
- Claims about another person.
- Explanations unsupported by evidence.

For subjective context, use a small command:

```text
/narrate "I expected the reducer to preserve more than one event. this is funny and slightly worrying."
```

### Narrator pipeline

```text
NARRATIVE.md entries
        ↓
visibility + embargo filter
        ↓
interestingness selection
        ↓
narrative-arc grouping
        ↓
candidate draft
        ↓
claim verification
        ↓
privacy and secret scan
        ↓
voice transformation
        ↓
duplicate/staleness check
        ↓
approval classification
        ↓
stochastic scheduler
        ↓
X API
        ↓
immutable post log
```

#### Interestingness signals

Select entries containing at least one of:

- A surprising discrepancy.
- A failed assumption.
- A concrete metric.
- A difficult bug.
- A reversal.
- A decision with meaningful trade-offs.
- A result that changes the next action.
- A technically useful question.
- A small moment of elegance or absurdity.

Routine commits should usually produce nothing.

#### Narrative arcs

Do not turn every event into a post. Hold connected entries until they form:

```text
problem
  ↓
first theory
  ↓
failed attempt
  ↓
counterexample
  ↓
revised explanation
  ↓
measured result
```

One arc may produce a small observation while unresolved, a measured result, and
a longer retrospective after resolution.

### Approval classes

#### Class A: may auto-publish after calibration

- Facts already public on GitHub.
- Previously published benchmark results.
- Non-sensitive engineering failures.
- Open technical questions.
- Short observations with no commercial or interpersonal claim.
- Follow-ups whose source entry and evidence are public.

#### Class B: requires batch approval

- New benchmark numbers.
- A claim involving another project, company, or person.
- Product-launch announcements.
- Pricing or commercial calls to action.
- Criticism of tools or maintainers.
- Security-adjacent observations.
- Anything derived from embargoed research.

#### Class C: permanently blocked

- Customer data.
- Private-repository contents.
- Unpublished paper results under embargo.
- Security vulnerabilities.
- Credentials, paths, hostnames, or cluster identities.
- Claims rejected by `CLAIMS.yaml`.
- Invented first-person reactions.
- Unverified explanations.
- Automated replies or DMs.

For the first **30 approved posts**, run Class A in review-only mode. Approve them
in one weekly batch; the scheduler handles publication. Enable Class A
autopublishing only after there have been no factual, privacy, or tone failures.

### Scheduling

Target an average of four to six original posts per week. Do not fill a quota with
weak material.

```text
maximum posts per day:          2
minimum gap:                    6 hours
zero-post days:                 common
same-project gap:               18 hours
same-claim repetition:          30 days
normal freshness window:        6-72 hours
retrospective arc maximum age:  30 days
```

Publishing windows in IST:

- 13:00-16:00.
- 20:30-00:30.

Choose a random eligible window and jitter the time. Treat these as operating
windows, not claims of optimal engagement.

Pause the queue when:

- A major result has just been published.
- An error needs correction.
- A sensitive event is under review.
- Two scheduled posts make the same point.
- There is nothing worth saying.

Use X's official API instead of browser scripting. Avoid duplicate or spam-like
automated posts and do not automate likes. Replies, follows, likes, and DMs remain
manual. Review the current [X automation rules][x-automation] and [authenticity
policy][x-authenticity] before enabling publication.

### Transparency

Use the author's real account. A pinned note may say:

> I keep an append-only log while I build. Some build notes here are scheduled
> from it. The work, numbers, and opinions are mine; replies are always me.

### Voice system

Do not copy a contemporary novelist sentence by sentence. Use an original
archetype:

> Austen-era restraint, modern terminal brain, warmer than Darcy.

The voice is calm and self-possessed. It admits mistakes without performing
vulnerability or asking for approval. It uses exact numbers, concrete mechanisms,
technical curiosity, dry understatement, and occasional humor that arises from
the event.

Example tone:

> 26/28 configuration pairs agreed. every point estimate passed.
>
> the verdict is still "inconclusive" because one configuration refused to stay
> stable across runs.
>
> annoying. correct, but annoying.

Or:

> a performance canary that replays the entire workload is wonderfully faithful.
>
> it is also, in a small but relevant sense, the workload.

Or:

> spent the week removing ways the tool could lie more elegantly.
>
> not glamorous. probably the useful part.

These examples remain within the repository's evidence boundary.

#### `VOICE.md`

```md
calm
technically exact
dry
mildly mischievous
restrained
occasionally sincere

Write as though telling one intelligent friend what happened.

Prefer:
- concrete numbers
- mechanisms
- understatement
- uncertainty where real
- short observations
- occasional longer stories
- natural lowercase
- ordinary words
- one unexpected phrase when earned

Never manufacture:
- typos
- profanity
- slang
- personal emotions
- drama
- contrarianism
- flirtation
- confidence
```

#### `NEVER.md`

```md
"I'm excited to announce"
"here are 5 lessons"
"game changer"
"let that sink in"
"the future of..."
"X isn't Y. It's Z."
"what nobody tells you"
"this changes everything"
emoji bullets
hashtag blocks
fake quotes
fake conversations
fake embarrassment
startup grind posts
wealth goals
audience-growth commentary
engagement questions
daily-building streaks
```

`GUIDE.md` remains the factuality and anti-generic-writing layer. It requires
concrete anchors, plain words, and support for numerical claims.

### X content distribution

Use these approximate proportions:

- 40-50%: what happened while building.
- 20-30%: measured technical conclusions.
- 10-15%: unresolved questions.
- 10%: elegant code, papers, or systems ideas.
- 5-10%: CommCanary or service calls to action.

Use one commercial call to action for roughly every 8-12 useful posts.

```text
interesting X observation
        ↓
GitHub evidence or case study
        ↓
reader sees that the work is real
        ↓
profile links to A-Square Systems
        ↓
performance-diagnostic application
```

Track:

- Replies from relevant systems engineers.
- Profile visits from target companies.
- GitHub traffic.
- Diagnostic applications.
- Discovery conversations.
- Proposals and revenue.

Do not optimize for total impressions or follower count.

## 8. Reddit plan

Reddit remains manual and pseudonymous, but never deceptive. A pseudonym is fine;
when discussing CommCanary, say "I built this" or "I'm the author." Do not pretend
to have discovered it independently.

Use the narrative system to draft posts, then review and submit each one manually
because community rules and culture differ.

Post only for a substantial result:

- `r/LocalLLaMA`: end-user and inference implications.
- `r/mlops`: deployment and regression-testing implications.
- Relevant systems or HPC communities: methodology and physical evidence.
- Runtime-specific communities where permitted.

Good title:

> I tested whether an eight-minute workload could preserve vLLM configuration
> decisions: results and failure cases

Bad title:

> I built an AI platform that revolutionizes GPU optimization

Put the full result in the Reddit post. Link GitHub or the case study at the end.
Do not automate comments, cross-post identical text, or use Reddit for cold
selling. One strong Reddit post per major experiment is enough.

## 9. GitHub plan

GitHub is the evidence layer. Every substantial public X or Reddit claim should
map to at least one of:

- A commit.
- An issue.
- A benchmark manifest.
- A generated report.
- A case study.
- An immutable experiment archive.
- A release.

Pin:

1. CommCanary.
2. The strongest distributed-systems contribution or case-study repository.
3. The company or evidence repository, if useful.

Use GitHub Discussions for technical design-partner questions only after explicit
approval for the specific discussion. Do not enter an upstream issue and
immediately pitch the service. Reproduce the problem, contribute useful technical
evidence, and keep commercial conversations separate.

## 10. Website plan

### Publish date

Publish by **August 16, 2026**, without waiting for a qualified CommCanary result.
The immediate offer is the performance service.

### Site structure

```text
/
├── performance-diagnostic
├── commcanary
├── evidence
├── notes
├── about
├── apply
├── privacy
└── terms
```

### Homepage

Headline:

> Make distributed inference faster, cheaper and safer to change.

Subheading:

> I run fixed-scope performance diagnostics for teams serving open models on
> NVIDIA multi-GPU systems: establish a reproducible baseline, isolate runtime
> and communication bottlenecks, implement the highest-confidence fixes, and
> leave a regression test.

Primary call to action:

> Apply for a 7-day diagnostic

Secondary call to action:

> Read the CommCanary research

Use "I," not "we," until there is a team.

### CommCanary page

Separate:

- Product target.
- Current implementation.
- Existing promising evidence.
- Unproven claims.
- Supported domain.
- Design-partner application.

Call the project a "research alpha," not a "production-ready GPU regression
platform."

### Evidence page

Before customer one, publish:

- The exact-work decision-fidelity study.
- The vLLM reduced-canary study when complete.
- Public performance experiments.
- Relevant HPX and collective work.

After customer one, publish only with written permission. Use exact baselines and
methods, a verified customer quotation, and a clear distinction between estimated
and measured savings.

### Application form

Ask for:

- Company.
- Role.
- Serving runtime.
- Model family.
- GPU type and count.
- Single-node or multi-node deployment.
- Approximate monthly GPU-spend range.
- Current throughput or latency problem.
- Planned stack change.
- Ability to run inside the customer's environment.
- Desired start date.

Qualify applicants before providing a calendar link.

## 11. Licensing and IP

The repository is Apache-2.0, with the CommCanary name and marks separately
reserved. Already released versions retain their original licenses. Accepting
outside contributions without an appropriate contributor agreement can
complicate future relicensing. The `NOTICE` file reserves the CommCanary name
against misleading derivative branding.

### Recommended split

Keep these public under Apache-2.0:

- Artifact schemas.
- Chakra ingestion.
- Capture adapters.
- Core CLI.
- Bundle verification.
- Basic synthesis already published.
- Public examples.
- Research baselines.
- Reproduction tooling.

Keep these private or commercial:

- Customer benchmark orchestration.
- Workload-specific adapters.
- Fleet and history management.
- Drift monitoring.
- Enterprise CI integration.
- Policy management.
- Report automation.
- Managed design-partner deployment.
- Future differentiated optimization modules developed separately.
- Customer-support tooling.

Do not attempt to close already published code retroactively.

Before accepting substantial outside contributions, either commit to keeping that
component Apache-2.0 or use a lawyer-prepared contributor license agreement that
preserves the required commercial rights. A Developer Certificate of Origin does
not by itself grant broad relicensing rights.

### Customer contracts

Contracts should distinguish:

- **Background IP:** CommCanary, reusable tools, and the performance methodology
  remain the supplier's.
- **Customer data:** customer code, models, traces, and configurations remain the
  customer's.
- **Customer-specific patches:** ownership or license is explicit.
- **General improvements:** reusable, non-confidential improvements remain the
  supplier's.
- **Upstream contributions:** contributions use the upstream project's terms and
  require customer permission.

Do not sign a blanket work-for-hire clause that transfers CommCanary or the
performance toolkit. Use an Indian lawyer and chartered accountant before the
first substantial foreign contract or invoice.

## 12. Security and confidentiality

Communication traces are not automatically anonymous. Message sizes, operation
order, topology, timing, and provenance can reveal deployment details.

Operating policy:

- Run inside the customer's VPC or cluster.
- Do not copy raw prompts or weights.
- Collect only required measurements.
- Redact before hashing or compiling.
- Use separate customer credentials and workspaces.
- Delete temporary customer artifacts after the contractual retention period.
- Never commit customer artifacts to a personal GitHub account.
- Never send customer code, traces, or architecture to a consumer AI service
  without written approval.

Redact at the source boundary. Public artifacts should omit hostnames, account
names, paths, job IDs, customer identifiers, and unnecessary environment metadata.

### Social-narrator security

The narrator must:

- Read only allowlisted local repositories.
- Default every narrative entry to `visibility: private`.
- Refuse customer directories.
- Honor embargo dates.
- Scan secrets, hostnames, absolute paths, emails, and identifiers.
- Compare every proposed claim against `CLAIMS.yaml`.
- Keep X API tokens outside the repository.
- Have a global `AUTOPUBLISH=false` kill switch.
- Preserve an immutable log of source entry IDs, generated text, approval status,
  and resulting post ID.

The Python process is not a sandbox. `capture` executes the supplied workload
command with the same authority as running it directly.

## 13. AI subscription and tooling

The planned starting point is ChatGPT Pro at $100, with an upgrade to $200 only
when limits repeatedly interrupt customer or CommCanary work. Do not pay for both
ChatGPT and Claude Max initially.

As checked on August 12, 2026, the [official OpenAI plan
documentation][openai-pro] says that both Pro tiers have the same core
capabilities. The $100 tier has 5x Plus usage, while the $200 tier has 20x Plus
usage. This comparison is time-sensitive; recheck it before purchasing or
upgrading.

Use:

- ChatGPT and Codex for repository implementation, research synthesis, and
  business operations.
- Small API spending for the narrative pipeline.
- A second model through an API only for difficult reviews.
- Local deterministic checks for factuality, privacy, and evidence binding.

No AI model is the sole authority for publishing benchmark claims.

Consider the $200 plan when at least one condition holds:

- Usage limits are reached twice a week.
- Two customer engagements run in parallel.
- Long Codex sessions are blocked by allowance.
- The additional usage saves more than $100 per month.

## 14. Weekly operating system

Until the first customer:

| Work | Share |
| --- | ---: |
| CommCanary physical evidence | 45% |
| Customer discovery and outbound | 25% |
| Public case studies and X/Reddit | 15% |
| Service-delivery toolkit | 10% |
| Administration | 5% |

A representative week:

- **Monday:** pipeline, outreach, and study status.
- **Tuesday:** physical study or benchmark work.
- **Wednesday:** physical study or implementation.
- **Thursday:** prospect research, conversations, and proposals.
- **Friday:** analysis, reports, and case-study writing.
- **Saturday:** technical depth, upstream contribution, or independent review.
- **Sunday:** scorecard, narrative approval batch, and recovery.

Every week must produce at least one evidence increment, one revenue action, and
one public proof increment. Coding alone cannot count for all three.

## 15. Scorecard

### Product

- Physical runtime reduction.
- GPU-second reduction.
- Severe held-out false negatives.
- False-positive rate.
- Pair-decision agreement.
- Stability across allocations and days.
- Setup time.
- Independent-operator completion.
- Comparison against simple sampling.

### Sales

- Qualified accounts researched.
- Personalized messages sent.
- Positive replies.
- Discovery conversations.
- Proposals.
- Paid diagnostics.
- Average contract value.
- Referrals.
- Revenue.
- Cash collected.

### Distribution

- Posts grounded in evidence.
- Relevant engineer replies.
- Relevant profile visits.
- GitHub repository visits.
- Case-study reads.
- Inbound diagnostic applications.
- Calls attributable to X, Reddit, or GitHub.

Do not measure writing quality with AI-detector scores. `GUIDE.md` explains why
optimizing detector proxies damages the writing.

## 16. Kill and reframe rules

### CommCanary

Reframe the product when:

- It misses a severe held-out regression.
- It cannot achieve meaningful runtime reduction.
- Stratified sampling performs equally well.
- Decisions are unstable across allocations.
- An independent operator cannot use it without the author.
- Customer workloads fall outside the supported domain.
- Three service customers do not care about repeated regression qualification.

### Offer

Change positioning when:

- 100 well-targeted contacts produce fewer than five serious conversations.
- Conversations occur but nobody values the problem enough to pay $4,000.
- Proposals repeatedly fail for the same reason.
- Customers consistently buy another outcome.

Do not respond to weak demand by adding product features.

### Social system

Disable autopublishing when:

- One factual claim lacks evidence.
- One private identifier leaks.
- The voice repeats recognizable templates.
- Posts become more frequent than the underlying work.
- Automated posts create conversations that repeatedly go unanswered.
- The system starts optimizing engagement instead of preserving the narrative.

## 17. Financial targets

These are targets, not forecasts.

### First 90 days

- One or two $4,000 diagnostics.
- One $15,000-$25,000 optimization sprint.
- One strong public case study.
- One referral.

### First six months

- Three to five paying companies.
- $40,000-$100,000 in cumulative revenue.
- One recurring customer.
- Clear evidence for or against CommCanary productization.

### First year

- **Base success:** $60,000-$100,000 in revenue.
- **Strong year:** $100,000-$160,000 in revenue.
- **Stretch:** $160,000-$220,000 in revenue, making approximately $100,000 in
  operating profit possible with controlled GPU, contractor, and legal costs.

The route to $1 million is:

```text
high-margin service revenue
        ↓
repeated customer workflow
        ↓
recurring software
        ↓
valuable ownership
```

## 18. Exact next actions

### August 13

- Freeze the supported domain.
- Add `docs/product-status.md`.
- Add `claims/public-claims.yaml`.
- Stop unrelated feature work.

### August 14

- Put `asquare.site` live.
- Publish the diagnostic offer.
- Add the qualification form.
- Set up domain email.

### August 15

- Add `NARRATIVE.md` to each active project.
- Create `VOICE.md`, `NEVER.md`, and the narrator skill.
- Backfill objective CommCanary milestones from Git history and experiment records.
- Leave historical emotional fields blank.

### August 16

- Generate the first 20 candidate X posts.
- Reject weak or repetitive drafts.
- Approve an initial queue.
- Publish the transparent pinned note.
- Begin manual technical replies.

### August 17-20

- Build and verify the vLLM runner.
- Finalize the policy and perturbation split.
- Freeze the physical study.
- Verify every input.

### August 21 onward

- Execute the product study.
- Publish only evidence-safe development notes.
- Research 30-50 target companies.
- Send the first 15 personalized messages.
- Request five warm introductions.
- Hold the first three customer conversations.

For the next month, the only outcomes that matter are:

> A held-out CommCanary result, a public case study grounded in verifiable
> evidence, and the first paid performance diagnostic.

[openai-pro]: https://help.openai.com/en/articles/9793128-what-is-chatgpt-pro
[x-authenticity]: https://help.x.com/en/rules-and-policies/authenticity
[x-automation]: https://help.x.com/es/rules-and-policies/x-automation
