# Volume VIII — Recommendation Engine

Status: Foundation aligned through Phase 4D Step 3
Version: 1.2.0
Last revised: 2026-07-19

## 1. Purpose

Define the research-recommendation subsystem: deterministic identity, candidate
generation, persistence, duplicate handling, scoring/ranking, explicit human
review, immutable advisory evidence, the explicitly separated manual conversion
chain, read-only campaign planning/review, and explicit non-executing campaign
approval.

## 2. Scope

### 2.1 In scope

Recommendation policy and identities, pure candidates, scans, source evidence,
duplicates, immutable scores/components, deterministic ranks/explanations,
terminal review transitions, immutable review events, and deterministic
evaluation classification/history, ranking snapshots and comparisons, plus the
pure manually invoked proposed-experiment specification contract.
The explicit Phase 4C manual conversion chain and Phase 4D campaign planning,
review, and explicit non-executing approval are also in scope.

### 2.2 Out of scope

Automatic proposal creation, campaign execution, profitability claims,
recommendation-driven scheduler polling, worker execution, and automatic
approval/rejection/expiration.

### 2.3 Current implementation status

Phase 4A Steps 1–5, Phase 4B Steps 1–2, Phase 4C Steps 1–6, and Phase 4D Steps 1–3
implement the in-scope capabilities.
Phase 4B Step 1 classifies current persisted provenance and reuses the Step 4
score formula unchanged. Step 2 ranks only persisted Step 1 results, stores
exact snapshot membership, and compares compatible persisted components.
Phase 4C provides the explicit proposal, review, paused conversion, activation,
and workflow-observation chain. Phase 4D Step 1 reads one explicit immutable
ranking snapshot and Phase 4C workflow evidence to produce a deterministic,
bounded campaign plan without persisting or executing it. Step 2 validates and
reviews that plan with deterministic duplicates and coverage. Step 3 records
one explicit immutable operator approval or rejection for an exactly
reconstructed review without executing it. Detailed contracts
remain in the Phase 4 documents referenced in §12.

## 3. Responsibilities

### 3.1 Owned responsibilities

Own deterministic proposal identity/generation, authoritative persistence and
duplicate provenance, advisory score history, explicit review disposition, and
their presentation.

### 3.2 Dependencies

Consumes completed experiment/final-analysis evidence from Volumes VI/VII.
Only the explicit Phase 4C operator commands cross into experiment lifecycle;
ranking, campaign planning, campaign review, and campaign approval have no
downstream execution dependency.

### 3.3 Prohibited responsibilities

Advisory evaluation, ranking, campaign planning, campaign review, and campaign
approval MUST NOT create experiments, queue work, mutate experiment or
scheduler state, infer reviewer identity, use
score thresholds for authorization, or claim expected profitability. Phase 4C
may create and activate exactly one experiment only through its separately
invoked, audited manual commands.

## 4. Architecture

### 4.1 Components

- Pure identity/policy and candidate-generation domain components.
- Pure manual conversion eligibility, source-consistency, and proposed-
  specification domain component.
- Repository-owned PostgreSQL mapping, scans, duplicates, scores, and reviews.
- Append-only conversion-proposal persistence with exact canonical retries.
- Services for explicit generation, scoring, evaluation, ranking/comparison,
  inspection, and review use cases.
- CLI parsing/validation/presentation outside scheduler polling.

### 4.2 Control flow

Completed evidence → deterministic source selection → pure candidates →
collision-aware persistence → optional explicit score/evaluation/ranking. The
Phase 4C proposal → review → paused execution → activation chain consists of
separately invoked and audited manual actions. Phase 4D reads ranking and Phase
4C history to plan and review; campaign approval records only an explicit human
decision and adds no implicit arrow to mutation or execution.

The governed campaign path is:

```text
recommendation ranking
-> read-only campaign plan
-> read-only campaign review
-> explicit persisted campaign approval
-> future campaign execution
-> existing Phase 4C per-recommendation manual workflow
```

The first four stages are implemented through Phase 4D Step 3.

### 4.3 Ownership boundaries

Canonical text decides identity; repository transactions decide persistence;
pure scoring, planning, and campaign-review policies decide advisory output; pure review rules
decide legal transitions; and the operator separately supplies review,
conversion, and activation actions. Only Phase 4C Steps 4–5 may create or
activate the one provenance-linked experiment. Campaign planning, review, and
approval never do.

## 5. Data model

### 5.1 Authoritative entities

Recommendation policy, semantic/invocation configuration, recommendation scan,
recommendation, score run/result/component, evaluation run/result/component,
ranking snapshot/member, and review event.
Conversion proposal, review decision, execution, activation, and campaign
approval records are separate immutable audit entities; only their explicitly linked experiment is
an experiment-lifecycle entity. Campaign plans and reviews are reconstructed,
not persisted; Step 3 persists only their exact approved/rejected provenance.

### 5.2 Provenance and versions

Canonical text is authoritative; tagged hashes accelerate lookup. Persisted
records retain source experiment/analysis/model, scan, policy, identity,
structural mutation, source rank, score policy/components, and review snapshot.

### 5.3 Invariants and legacy data

Scan-associated rows require positive source rank and cannot detach from their
originating scan. Proposed/approved identities remain active; rejected/expired
matches remain historical blockers. Legacy NULL/NULL scan/rank rows are not
backfilled and are excluded from operations requiring provenance.

Review transitions are only proposed → approved, rejected, or expired. Review
events are append-only to the runtime role. Approval is advisory and leaves
`approved_experiment_id` null under Step 5.

## 6. Transactions

### 6.1 Read paths

Evidence, list, detail, score history, evaluation history, ranking input, and
review history use typed read-only repository paths. Ranking does not reopen
source experiment metrics.

### 6.2 Write paths

Candidate persistence uses short collision-aware transactions. Score result and
ordered components persist atomically. Evaluation results and their unchanged
Step 4 components persist atomically without locking experiment evidence.
Review locks one recommendation and updates status plus one immutable event in
the same transaction. Ranking persists its complete member set atomically and
then completes a ranking-owned snapshot after an exact count check.
Conversion-proposal insertion locks only its accelerator-hash bucket, compares
canonical text exactly under the bytewise ``C`` collation, and inserts one
complete row in one transaction. The collision ordinal is storage metadata;
canonical text alone remains authoritative.

### 6.3 Failure semantics

Invalid commands create no scan/event. Candidate failures are counted per scan.
Score retries compare complete immutable results. Evaluation retains completed
per-recommendation results if a later item fails, marks the run failed, and
forbids adding new results to that terminal run; exact existing results remain
verifiable. Review event failure rolls back status; terminal retries create no
event. A ranking-member failure leaves no partial membership.

## 7. Concurrency

### 7.1 Conflict domain

Generation conflicts on semantic configuration plus policy; scoring retries on
score run plus recommendation; evaluation conflicts on canonical evaluation-run
identity and evaluation run plus recommendation; ranking conflicts on canonical
snapshot identity and snapshot plus evaluation result; review conflicts on one
recommendation row.

### 7.2 Locking and serialization

Generation uses a transaction advisory key plus canonical rechecks and active
uniqueness. Scoring uses uniqueness and complete retry comparison. Evaluation
uses a short shared lock on its own run row while inserting an atomic result;
it does not lock experiment evidence. Review uses `SELECT ... FOR UPDATE` and
one-event uniqueness. Ranking serializes identical member verification with a
short `FOR UPDATE` lock on its ranking-owned snapshot. Unrelated snapshots and
unrelated work remain concurrent.

### 7.3 Winner, loser, and retry outcomes

Concurrent identical generation returns one created/one existing result.
Equivalent score retry reuses only an exact persisted result. Concurrent review
produces one terminal winner and one `recommendation_review_status_conflict`;
exactly one event exists. Concurrent identical ranking requests converge on one
snapshot with an exactly verified member set.
Concurrent identical conversion proposals converge on one exact canonical row;
same-hash different canonical rows remain distinct.

## 8. CLI

### 8.1 Commands and validation

Generation, scoring, evaluation, ranking/comparison, their inspection commands,
review actions, and review inspection are explicit families. Evaluation,
ranking, and review mutation
are mutually exclusive with generation, scoring, scheduler, continuation, and
experiment-lifecycle commands.

### 8.2 Machine output

`EXPERIMENT_RECOMMENDATION_*` records use stable fields, percent-escaped text,
explicit `NULL`, deterministic ordering, and separate conflict/failure events.

### 8.3 Human output

Human score/evaluation/review summaries are concise, control-byte safe, and
state that the result is advisory and created/queued no experiment. Ranking
separates advisory-ready, blocked, and non-actionable presentation.

## 9. Testing

### 9.1 Pure tests

Cover canonical identity/collisions, eligibility, candidates, scoring formulas,
ranking, evaluation classification/identity, review parsing/reasons, and all
transition outcomes.

### 9.2 Persistence and migration tests

Cover scans, duplicates, source rank/scan provenance, score ownership and
immutability, review constraints, runtime privileges, SQLSTATEs, and migration
repeatability. Evaluation tests additionally cover canonical idempotency,
append-only result/component privileges, nullable blocked scores, and evidence
foreign keys. Ranking tests cover exact membership, bucket/rank uniqueness,
narrow privileges, scope/captured-membership trigger enforcement, bounded
evidence loading, atomic member persistence, and Step 1 immutability.
Conversion-proposal tests cover complete invocation round trips, exact retries,
hash collisions, concurrent insertion, restrictive privileges, and unchanged
experiment fixtures.

### 9.3 Concurrency and integration tests

Cover same-candidate generation, score retries, evaluation run/result retries
and partial failures, review action races, unrelated reviews, rollback, exact
cleanup, and before/after experiment/identity/score digests.

### 9.4 Regression boundaries

Steps 1–5 remain individually stable. Changes MUST NOT alter training,
inference, continuation, experiment lifecycle, or scheduler behavior.

## 10. Operational safety

### 10.1 Runtime isolation

No recommendation command enters scheduler polling. Integration uses disposable
rows and exact cleanup without disturbing genuine experiments.

### 10.2 Permissions and destructive operations

Runtime access to review history and evaluation results/components is
SELECT/INSERT only; evaluation-run updates are limited to lifecycle columns.
Ranking members are append-only and ranking-snapshot updates are limited to
lifecycle/count columns. Owner/test connections perform exact fixture cleanup.
Conversion proposals grant runtime ``SELECT``/``INSERT`` access. Manual review
decisions grant runtime ``SELECT`` plus column-limited ``INSERT`` for decision
payload only; generated decision IDs and timestamps are not caller-writable.
Their source-provenance foreign keys are restrictive. Review history is
append-only, and the greatest generated decision ID yields the current
pending/approved/rejected administrative disposition. This sequence-ID order is
authoritative even when concurrent transactions commit in another order.
Conversion activations likewise grant only ``SELECT`` and column-limited
``INSERT`` plus sequence usage. Activation evidence is append-only; runtime
``UPDATE``, ``DELETE``, and ``TRUNCATE`` are denied.
Other immutable histories follow Volume I §7.3.

### 10.3 Observability and recovery

Scans/runs expose completion and counters; scores/events retain immutable
provenance; explicit conflicts distinguish safe rejection from system failure.

## 11. Future extensions

### 11.1 Approved extension points

Versioned policies, pure candidate/scoring components, typed repository/service
APIs, and separate inspection commands.

### 11.2 Deferred capabilities

Experiment queueing/execution from conversion, multi-source attribution,
automatic research campaigns, profitability evidence, and scheduler-managed
recommendation work. Phase 4C Step 1 provides the pure proposed-
specification contract, Step 2 provides durable proposal audit evidence, and
Step 3 provides explicit append-only manual proposal review. Step 4 permits one
explicit approved proposal to create one paused experiment. Step 5 permits a
separate explicit operator action to make that exact experiment pending under
the existing scheduler lifecycle.

### 11.3 Required decisions

ADR-0005 defines explicit conversion authorization, implementation identity,
idempotent experiment creation, transactions, and audit history. Approval alone
does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
remain deferred under ADR-0004.

Phase 4C Step 2 adds the append-only
``experiment_recommendation_conversion_proposal`` record and a narrow
repository. Exact canonical text remains authoritative; same-hash distinct
canonical identities receive separate collision ordinals, while concurrent
identical inserts converge on one proposal. Optional ranking provenance is
advisory, excluded from identity, and cannot authorize conversion. Runtime
access is limited to ``SELECT``/``INSERT``; no scheduler path reads these rows.

Phase 4C Step 3 adds explicit ``approve``/``reject`` decisions against an exact
proposal primary key. A caller-visible request token makes CLI retries
idempotent per proposal. New request tokens permit auditable reversals; the
greatest decision ID defines the current disposition. Approval remains
administrative evidence only and never creates or queues an experiment.

Phase 4C Step 4 adds one explicit manual conversion transaction. It takes a
proposal-specific transaction advisory lock, revalidates the immutable
proposal, requires the latest serialized decision to be ``approve``, creates
one ``paused`` experiment, and records immutable provenance linking the
proposal, approving decision, and experiment. Unique proposal and experiment
references make retries and concurrent requests converge. The scheduler does
not poll conversion rows and a converted experiment is not queued or started.

Phase 4C Step 5 adds one explicit activation transaction. It revalidates the
Step 4 execution and exact created invocation, requires the experiment to
remain in the pristine ``paused/train`` state, records immutable activation
evidence, and changes only that experiment to ``pending/train``. The audit and
lifecycle update are atomic. Exact retries return the same activation without
reapplying the transition. The scheduler is unchanged and never reads the
activation table; no worker is started by activation.

Phase 4C Step 6 adds a read-only workflow aggregate over proposal, greatest-ID
review disposition, exact execution authorization, activation evidence, and
current experiment lifecycle. Stable states and deterministic diagnostics make
healthy and inconsistent chains inspectable without adding schema, privileges,
workflow mutation, scheduler polling, or worker behavior.

Phase 4D Step 1 adds a pure, versioned campaign policy and a read-only aggregate
over one explicit completed ranking snapshot, persisted recommendation source
metrics, and the existing Phase 4C workflow derivation. Ranking global ordinal,
recommendation ID, ranking member ID, then canonical evidence define the full
presentation/selection order. Every considered
candidate is retained with one decision and stable reasons; plan identity binds
the full policy, scope, snapshot, evidence, workflow provenance, decisions, and
reasons while excluding display time. There is no campaign table or scheduler
consumer. Because no authoritative profitability evidence exists, a requested
profitability threshold yields ``profitability_metric_unavailable`` rather than
an inferred proxy.

Phase 4D Step 2 adds a pure review contract over the Step 1 plan and reuses the
same read-only loader and planner. It verifies plan identity, policy/scope,
counts, result shape, and deterministic ordering; preserves selected and
excluded rows with their ordered reasons; and derives exact canonical duplicate
groups plus deterministic family, symbol, and horizon coverage. Review identity
binds these results while excluding display time. Step 2 adds no migration,
privilege, persisted campaign, workflow mutation, scheduler consumer, or worker
behavior.

Phase 4D Step 3 adds one immutable approval/rejection audit row for one exact
Step 2 review. One PostgreSQL transaction loads the explicit ranking snapshot
and Phase 4C workflow evidence, reconstructs and validates the plan and review,
requires an operator-supplied expected review hash, and inserts the decision.
Canonical review text is authoritative; identical retries converge, changed
decision/reviewer/reason payload conflicts, and a zero-selection campaign may
only be rejected. Runtime privileges are append-only and column-limited. The
approval row never creates or modifies an experiment and is not consumed by
the scheduler.

## 12. References

- [Volume I §§5–10](Volume_I_Foundation.md)
- [ADR-0003](adr/ADR-0003-advisory-recommendation-evaluation.md)
- [ADR-0005](adr/ADR-0005-manual-recommendation-conversion.md)
- [Recommendation foundation](../Phase4AExperimentRecommendationFoundation.rst)
- [Recommendation persistence](../Phase4AExperimentRecommendationPersistence.rst)
- [Recommendation scoring](../Phase4AExperimentRecommendationScoring.rst)
- [Recommendation review](../Phase4AExperimentRecommendationReview.rst)
- [Recommendation evaluation](../Phase4BExperimentRecommendationEvaluation.rst)
- [Recommendation ranking](../Phase4BExperimentRecommendationRanking.rst)
- [Manual conversion contract](../Phase4CExperimentRecommendationConversion.rst)
- [Manual conversion proposal persistence](../Phase4CExperimentRecommendationConversionPersistence.rst)
- [Manual conversion proposal review](../Phase4CExperimentRecommendationConversionProposalReview.rst)
- [Manual conversion execution](../Phase4CExperimentRecommendationConversionExecution.rst)
- [Manual conversion activation](../Phase4CExperimentRecommendationConversionActivation.rst)
- [Manual conversion workflow observability](../Phase4CExperimentRecommendationConversionWorkflow.rst)
- [Read-only campaign planning](../Phase4DExperimentRecommendationCampaignPlanning.rst)
- [Read-only campaign review](../Phase4DExperimentRecommendationCampaignReview.rst)
- [Explicit campaign approval](../Phase4DExperimentRecommendationCampaignApproval.rst)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the Step 1–5-aligned recommendation architecture outline. | ADR-0003 |
| 0.2.0 | 2026-07-15 | Recorded Phase 4B Step 1 deterministic advisory evaluation evidence without changing execution ownership. | ADR-0003, ADR-0004 |
| 0.3.0 | 2026-07-15 | Recorded Phase 4B Step 2 immutable advisory ranking snapshots and policy-aware comparison. | ADR-0001, ADR-0003, ADR-0004 |
| 0.4.0 | 2026-07-18 | Recorded the pure Phase 4C Step 1 manually authorized proposed-experiment contract; durable conversion and execution remain deferred. | ADR-0003, ADR-0004 |
| 0.5.0 | 2026-07-18 | Recorded immutable Phase 4C Step 2 conversion-proposal persistence; experiment creation and execution remain deferred. | ADR-0001, ADR-0003, ADR-0004 |
| 0.6.0 | 2026-07-18 | Recorded append-only, idempotent Phase 4C Step 3 manual proposal review; approval remains non-executing. | ADR-0001, ADR-0003, ADR-0004 |
| 0.7.0 | 2026-07-19 | Recorded explicit, idempotent Phase 4C Step 4 conversion to one paused experiment. | ADR-0001, ADR-0004, ADR-0005 |
| 0.8.0 | 2026-07-19 | Recorded explicit, atomic Phase 4C Step 5 activation of a converted experiment into the existing pending lifecycle. | ADR-0001, ADR-0004, ADR-0005 |
| 0.9.0 | 2026-07-19 | Recorded read-only Phase 4C Step 6 end-to-end workflow state and integrity observability. | ADR-0001, ADR-0004, ADR-0005 |
| 1.0.0 | 2026-07-19 | Recorded deterministic read-only Phase 4D Step 1 campaign planning from explicit persisted ranking and workflow evidence. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
| 1.1.0 | 2026-07-19 | Recorded deterministic read-only Phase 4D Step 2 campaign review, duplicate findings, and coverage. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
| 1.2.0 | 2026-07-19 | Recorded explicit immutable Phase 4D Step 3 campaign approval/rejection for one exact reconstructed review without execution. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
