# Active plan: remaining work

Last updated: 2026-08-12

This file is the execution ledger for the work that remains. `future_plan.md`
contains the wider A-Square Systems strategy. `HANDOFF.md` remains the source
for the live repository and Rostam checkpoint, and `docs/product-status.md`
defines the public evidence boundary.

CommCanary is locally ready to enter its first physical product study. It is
still an unreleased research alpha. Physical runtime reduction, held-out
regression safety, independent operation, and production CI use remain
unproven.

## Completion target

The current cycle ends when all three outcomes exist:

1. A frozen, reproducible vLLM study with a recorded held-out outcome.
2. A public case study that reports the result and its limits from immutable
   evidence.
3. The first paid A-Square Systems performance diagnostic, or documented sales
   evidence strong enough to revise the offer.

The vLLM result may pass, fail, or remain inconclusive. Completion requires a
valid predeclared result, not a favourable result.

## Current checkpoint

| Area | State | Remaining boundary |
|---|---|---|
| Local product implementation | Complete for study entry | Preserve it in a reviewed source commit. |
| Product status and claims | Complete locally | Connect the website and narrator in their own repositories. |
| Operator quick start | Written and locally exercised | An independent operator must complete it. |
| Local verification | Green | Re-run after the source freeze and at every release boundary. |
| Product runner | Unbuilt | Build and verify the digest-pinned vLLM OCI/SIF runner on Rostam. |
| Physical study | Unfrozen and unexecuted | Freeze inputs, run training and holdout, and publish the outcome. |
| SGLang replication | Gated | Start only after the vLLM study passes its core gates. |
| 0.3.0 release | Deferred | Requires a recorded study outcome, operator trial, release gate, and explicit publication decision. |
| Website, narrator, and delivery kit | External state unverified | Inspect their actual repositories before marking any task complete. |
| Customer evidence | Not recorded here | Launch the diagnostic offer and run outbound in parallel. |

## Roles and authority

### Technical lead: Codex

- Own scope, task decomposition, contract interpretation, patch review, and
  evidence validation.
- Reject changes that weaken fail-closed behavior or convert fixtures into
  product evidence.
- Run the relevant focused tests and canonical verification gates.
- Keep public claims aligned with `claims/public-claims.yaml`.
- Never infer permission to commit, push, publish, file issues, or mutate
  Rostam.

### Implementation agents: OpenCode DeepSeek

- Implement bounded local tasks in isolated copies when explicitly assigned.
- Return a patch, focused tests, and an evidence report for technical-lead
  review.
- Do not decide study policy, interpret a physical result, edit immutable
  evidence, publish claims, or mutate the cluster.
- Work in parallel only where file ownership and acceptance tests are separate.

### User/operator

- Approve commits, Rostam access, scheduler mutation, release, and publication
  as separate actions.
- Open the user-managed SSH ControlMaster when cluster access is authorized.
- Select the independent operator and approve customer, pricing, legal, and
  public communication decisions.

## Approval gates

No later gate implies an earlier or broader permission.

- [ ] **A1: source-freeze approval.** Approve the exact proposed commit scope.
- [ ] **A2: read-only Rostam approval.** Permit inspection of the checkout,
  queue, fleet, existing attempts, storage, and build prerequisites.
- [ ] **A3: runner-build approval.** Permit creation of new runner artifacts on
  Rostam. This does not permit study submission.
- [ ] **A4: study-freeze approval.** Approve the policy, perturbations, runner,
  source inputs, manifest, and submission plan.
- [ ] **A5: scheduler-mutation approval.** Permit the reviewed CommCanary jobs
  on `cuda-A100`. Cancellation still requires approval of literal job IDs.
- [ ] **A6: holdout-open approval.** Confirm that selection is immutable before
  the callback can expose holdout evidence.
- [ ] **A7: publication approval.** Approve the exact case-study and evidence
  scope. A Git push is a publication action.
- [ ] **A8: release approval.** Approve the exact 0.3.0 tag and distribution
  artifacts after the release gate passes.

## Critical path: CommCanary physical evidence

### Phase 0: freeze a reviewable source state

Target: August 12–13.

- [ ] Inventory every modified and untracked path. Classify it as intended
  product work, intended documentation/evidence, private operational state, or
  unrelated user work.
- [ ] Produce a proposed commit manifest. Do not stage all dirty files by
  default.
- [ ] Review schema mirrors, package inventory, README links, claims, and the
  product-study scripts against the same source state.
- [ ] Run:

  ```console
  python -m pytest -q tests/experiments/rostam
  python -m ruff check experiments/rostam tests/experiments/rostam
  git diff --check
  python -m tools.verify --fast
  python -m tools.verify --reproducible
  ```

- [ ] After A1, create a local commit or a small reviewable commit series. Do
  not push it.
- [ ] Record the resulting Git commit and tested artifact digests in the live
  handoff without presenting uncommitted bytes as frozen inputs.

Exit criteria:

- The intended study implementation is represented by Git.
- The tree used to build the runner is clean.
- Fast and reproducible verification pass from that exact commit.
- No unrelated user change was staged, reverted, or deleted.

### Phase 1: perform a read-only Rostam preflight

Target: immediately after A2 and source freeze.

- [ ] Confirm the SSH ControlMaster is user-managed and available. Never copy
  credentials into commands, files, output, or conversation.
- [ ] Read the current Rostam Git HEAD, tracked cleanliness, branch divergence,
  and prior CommCanary attempt inventory.
- [ ] Inspect `cuda-A100`, the current queue, and node health. Treat other
  users, unrelated partitions, and Jenkins allocations as read-only.
- [ ] Probe `nvidia-smi` inside an allocation before attributing a missing-GPU
  failure to CommCanary.
- [ ] Verify available disk, Apptainer, OCI build tooling, Python, driver, CUDA,
  and expected architecture.
- [ ] Reconcile every live finding with `HANDOFF.md`. Update assumptions before
  any mutation.

Exit criteria:

- The exact source transfer or guarded cherry-pick path is known.
- Runner prerequisites and current fleet health are recorded.
- No repository, queue, or result state was changed.

### Phase 2: build and verify the vLLM runner

Target: August 13–17, subject to cluster availability.

- [ ] Build a wheel from the frozen source and record its SHA-256.
- [ ] Review the digest-pinned base image, vLLM dependency lock, Python and CUDA
  compatibility, runner protocol, Linux/architecture identity, and dummy-model
  configuration.
- [ ] After A3, build the OCI archive and Apptainer SIF into a new output
  directory.
- [ ] Verify manifest and config descriptors, content-addressed blobs, declared
  sizes, SHA-256 values, Linux/architecture identity, SIF bytes, installed
  versions, and the runtime probe.
- [ ] Preserve the complete descriptor, logs, archive, SIF, version inventory,
  build inputs, and wheel together.
- [ ] Do not reuse an existing output unless independent verification proves it
  byte-identical.

Exit criteria:

- A digest-pinned runner descriptor verifies from exact bytes.
- The SIF can import and invoke the intended vLLM application driver.
- The study policy can bind the exact runner OCI digest and world size four.

### Phase 3: review and freeze the vLLM study

Target: August 17–20.

- [ ] Review a complete four-rank Chakra ET and matching source projection.
- [ ] Freeze one baseline, seven training perturbations, and three untouched
  holdout perturbations.
- [ ] Freeze the four-GPU policy before measurements. It must include:

  - at least 10× physical runtime reduction;
  - zero severe held-out false negatives;
  - no more than 10% false positives;
  - at least 90% pair-decision agreement;
  - stability across independent allocations and at least two days;
  - the runtime, privacy, runner, protocol, and supported-domain bindings.

- [ ] Freeze exact replay, random, stratified, ddmin, and
  communication-only baselines for comparison.
- [ ] Freeze eight independent application allocations across at least two
  days. Do not treat repetitions inside one allocation as independent
  allocations.
- [ ] Verify every source, policy, model configuration, perturbation catalog,
  wheel, runner artifact, execution script, and site-contract hash.
- [ ] Generate the canonical submission plan and review cell ownership, output
  paths, resource limits, and retry semantics.
- [ ] After A4, freeze the study directory. Any bound-byte change requires a
  replacement study, not an edit to the frozen study.

Exit criteria:

- Study verification passes with no missing or mutable input.
- Holdout bytes are inaccessible to mutable selection code.
- The reviewed plan names one owner for every expected cell.

### Phase 4: execute baseline and training evidence

Target: August 21–31, subject to queue time.

- [ ] After A5, submit one small execution canary for the new runner and
  workload shape.
- [ ] Verify its scheduler record, node, partition, GPU visibility, runner
  identity, attempt ownership, output hashes, and application measurement.
- [ ] Submit the remaining work in chunks of 16–24 cells.
- [ ] Record every retry under a new attempt ID. Preserve failed, cancelled,
  and successful attempts.
- [ ] Run the application baseline allocations and training perturbations.
- [ ] Run active candidate synthesis using training evidence only.
- [ ] Compare the selected candidate against exact replay and the frozen simple
  baselines.
- [ ] Inspect queue state without changing unrelated jobs. Use literal job IDs
  with `sacct` after completion.

Exit criteria:

- Every expected baseline and training cell has at least one terminal attempt.
- The selected attempt inventory verifies.
- Candidate selection can be reproduced without holdout evidence.

### Phase 5: freeze selection and execute holdout

Target: immediately after complete training evidence.

- [ ] Persist exactly one selected candidate and its content identity.
- [ ] Prove that the selection and training ledger are immutable.
- [ ] Review the selection record and training completeness verdict.
- [ ] After A6, open the holdout callback.
- [ ] Execute held-out application and candidate measurements without changing
  the policy, thresholds, perturbations, runner, or selected candidate.
- [ ] Select exactly one terminal attempt for every expected holdout cell.
- [ ] Persist a fail-closed completeness verdict with zero unresolved issues.

Exit criteria:

- Holdout access occurred after selection freeze.
- Every expected cell has one verified selected terminal attempt.
- No threshold or study input changed after observing results.

### Phase 6: generate and reproduce the result

Target: September 1–5.

- [ ] Generate the physical gate result, product bundle, full audit bundle, raw
  archive descriptor, and publication files from selected evidence.
- [ ] Recompute all corpus rows, decisions, error rates, agreement, runtime
  reduction, stability, and privacy checks from immutable bytes.
- [ ] Regenerate the analysis into a new temporary destination and require a
  byte-for-byte match.
- [ ] Verify every historical artifact still verifies after the new result is
  added.
- [ ] Record one exact outcome: qualified, failed held-out validation, failed
  privacy policy, or one of the defined blocked states.
- [ ] Update `docs/product-status.md` and `claims/public-claims.yaml` only with
  claims supported by that outcome.

Exit criteria:

- The result and publication regenerate deterministically.
- Raw archives bind exact manifest, selection, and completeness hashes.
- The public wording matches the machine-readable claim category.

### Phase 7: publish the vLLM case study

Target: September 1–15, after A7.

- [ ] Write the exact workload, hardware/software manifest, policy, full
  application ground truth, canary result, runtime and GPU-second comparison,
  false-negative and false-positive counts, pair agreement, stability,
  limitations, and reproduction instructions.
- [ ] Link every quantitative claim to a generated report, manifest, archive,
  or commit.
- [ ] Redact source identifiers before hashing or publication. Do not describe
  redaction as anonymity or a privacy guarantee.
- [ ] Publish a negative or inconclusive result without a favourable retry.
- [ ] Update the website and social queue through the claims registry.

Exit criteria:

- A reader can reproduce the analysis from the published bytes.
- The case study distinguishes measured savings from estimates.
- No customer, cluster, hostname, path, account, or secret crosses the public
  boundary.

## Evidence-dependent branches

### If all core vLLM gates pass

- [ ] Run the independent-operator trial.
- [ ] Freeze and execute the SGLang replication or a materially different
  workload.
- [ ] Open at most three private design-partner slots.
- [ ] Add one physical-product demo to the website.
- [ ] Consider the 0.3.0 research-alpha release after the release gate.

### If the result is inconclusive

- [ ] Publish the inconclusive result and its exact uncertainty or stability
  failure.
- [ ] Attribute instability only where retained evidence supports the cause.
- [ ] Design a new predeclared study. Do not rerun selectively or pool attempts
  after inspecting the answer.
- [ ] Keep CommCanary internal during performance diagnostics.

### If safety or 10× reduction fails

- [ ] Do not sell CommCanary as a fast regression gate.
- [ ] Compare against stratified sampling and the other frozen simple
  baselines.
- [ ] Reframe the public artifact as an exact workload capsule or research
  tool where the evidence supports that description.
- [ ] Use customer work to test a different recurring product problem.

## Independent operation and release

### Independent-operator trial

- [ ] Select an engineer who did not prepare the source or study.
- [ ] Give the operator a pinned artifact and `docs/operator-quickstart.md`
  without private verbal steps.
- [ ] Require the operator to install, run the blocked orientation, interpret
  the status, verify the runner and study, recover a partial run, verify the
  final bundle, and report friction.
- [ ] Treat undisclosed assistance or an unrecoverable workflow as a failed
  release gate.
- [ ] Fix documentation or implementation problems, then repeat with a fresh
  operator where the change affects independence.

### 0.3.0 research-alpha release

- [ ] Record the vLLM study outcome, including a negative or inconclusive one.
- [ ] Complete the independent-operator trial.
- [ ] Reconcile README, website, narrator, and claims-registry wording.
- [ ] Use a clean, reviewed source state and run:

  ```console
  python -m tools.verify --release
  ```

- [ ] Inspect the exact wheel, sdist, checksums, inventory, and SBOM.
- [ ] After A8, tag and publish exactly those artifacts. Do not rebuild after
  approval.

Release does not establish product qualification. A negative study may still
ship as a clearly labelled research alpha.

## Parallel commercial lane

Technical qualification and customer discovery run in parallel. CommCanary
does not need to pass before A-Square Systems sells a fixed-scope performance
diagnostic.

### Company website and offer

Target: August 16 or earlier.

- [ ] Inspect the actual website repository and deployment state before making
  changes or marking tasks complete.
- [ ] Publish the A-Square Systems homepage, seven-day diagnostic page,
  CommCanary research-alpha page, evidence page, application form, privacy
  page, and terms page.
- [ ] Use first-person singular until a team exists.
- [ ] Put the diagnostic application ahead of a generic calendar link.
- [ ] Make the website consume or validate against
  `claims/public-claims.yaml`.
- [ ] Configure domain email.

### Customer discovery and sales

- [ ] Build a 30–50-account list of teams using self-hosted vLLM or SGLang on
  roughly 4–64 NVIDIA GPUs.
- [ ] Request five warm introductions.
- [ ] Each week, research 20 companies, send 15 personalized messages, request
  two warm introductions, and hold three conversations.
- [ ] Attach one relevant experiment instead of a pitch deck.
- [ ] Offer the first two seven-day diagnostics at $4,000, paid before kickoff,
  under the terms in `future_plan.md`.
- [ ] Track qualified accounts, replies, conversations, proposals, paid
  diagnostics, cash collected, and repeated customer problems.
- [ ] Revisit positioning after 100 well-targeted contacts or repeated proposal
  failures. Do not answer weak demand with more CommCanary features.

### Evidence-led distribution

- [ ] Inspect the external narrator repository before marking its state.
- [ ] Keep Class A in review-only mode for the first 30 approved posts.
- [ ] Connect drafts to the claims registry and immutable evidence paths.
- [ ] Keep replies, DMs, follows, and likes manual.
- [ ] Publish one substantial manual Reddit post per major experiment, with
  authorship disclosed when discussing CommCanary.
- [ ] Track relevant engineer replies, repository visits, case-study reads,
  applications, conversations, proposals, and revenue. Do not optimize the
  system for impressions alone.

### Private performance-delivery kit

- [ ] Inspect or create the private `asquare-performance-kit` repository.
- [ ] Implement only the workflow needed for the first diagnostic: intake,
  environment snapshot, reproducible baseline, experiment ledger, redaction,
  before/after validation, ROI estimate, and technical/executive reports.
- [ ] Keep customer adapters, histories, credentials, and reports private.
- [ ] Use the first two engagements to decide what to automate next.
- [ ] Do not market this repository as another product.

### Legal and customer-data boundary

- [ ] Before the first substantial foreign contract or invoice, obtain Indian
  legal and accounting review.
- [ ] Separate background IP, customer data, customer-specific patches,
  reusable improvements, and upstream contributions in contracts.
- [ ] Avoid blanket work-for-hire language that transfers CommCanary or the
  private delivery kit.
- [ ] Run inside the customer environment, collect only required measurements,
  and apply the contractual retention and deletion period.
- [ ] Never send customer code, traces, models, prompts, or architecture to a
  consumer AI service without written approval.

## Weekly operating rule

Until the first customer, allocate approximately:

| Work | Share |
|---|---:|
| CommCanary physical evidence | 45% |
| Customer discovery and outbound | 25% |
| Public evidence and distribution | 15% |
| Private delivery kit | 10% |
| Administration | 5% |

Every week must produce one evidence increment, one revenue action, and one
public proof increment. A code change counts only for the category it actually
advances.

## Immediate next actions

1. Prepare the exact source-freeze manifest from the current dirty worktree.
2. Obtain A1 and create the reviewed local commit or commit series.
3. Re-run fast and reproducible verification from the committed state.
4. Obtain A2 and perform the read-only Rostam preflight.
5. Review the preflight evidence before requesting A3.

Do not start another subsystem while these five actions remain open.
