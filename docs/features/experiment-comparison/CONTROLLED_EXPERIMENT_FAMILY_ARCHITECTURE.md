# Declared controlled experiment family architecture

## Decision

Add a first-class **declared controlled experiment family** lifecycle for a
scientifically new, outcome-blind experiment matrix.  It is deliberately not a
mode of `ExperimentReplicationMaterialization`.

Replication starts with an existing experiment and derives a fresh experiment
from its persisted configuration.  A declared family starts with an explicit,
complete scientific specification and creates its first experiment rows only
after that specification, its deterministic expansion, and an approval have
been frozen.  The two mechanisms may share comparators, scheduler provenance,
and canonicalization utilities; neither should call or emulate the other's
materializer.

This document is a design only.  Names below are proposed names, not current
schema objects or commands.

## Repository evidence and architectural boundary

The authoritative schema is `Database/LSTM_schema.sql`.  The relevant current
contracts are:

- `experiment` contains most queue-time configuration, lifecycle state,
  pause/resume fields, `fresh_initialization_seed`, calendar ID/hash, model
  input identity, training-objective canonical/hash, checkpoint policy, and
  continuation policy.  Migration 095 makes the fresh seed part of the
  current experiment uniqueness identity.
- `experiment_scheduler_worker_attempt` (migration 096) records the selected
  layout, width, role, source commit, executable SHA-256, runtime identity,
  and canonical manifest path.  `model` and `inference_eval_result` bind their
  producing attempt.  This is actual execution provenance, not a future
  execution requirement.
- `experiment_analysis_result`, `inference_eval_result`, and
  `inference_profitability_observation` are output/evidence tables.  They are
  not inputs to a new declared configuration.
- `ExperimentReplicationMaterialization` opens one SERIALIZABLE transaction,
  takes `LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE`, reloads source
  evidence, checks equivalence, then inserts a complete paused wave.  Its
  repository copies configured columns from the source experiment and changes
  only the seed.  That is correct for replication and cannot author a new
  scientific configuration.
- The scheduler currently loads `experiment` rows with `status='pending'` and
  claims them by changing them to `running`.  Paused rows are not candidates.
  Existing pause/resume and priority behavior remains the operational source
  of truth; a new family gate must be an additional admission condition, not a
  competing scheduler state machine.
- Continuation scans completed rows where `continuation_policy_enabled=true`.
  Therefore a mere convention not to enable it is insufficient for a
  no-continuation controlled study.
- Recommendation campaign tables persist a ranking-derived selection of
  existing Phase 4C conversion proposals.  Their approval/materialization
  evidence is a useful pattern, but their source recommendation/ranking
  semantics cannot represent a new scientific matrix.

The source-clone Fibonacci materialization report already established that the
frozen layout-9/103 source pair does not exist and that layout-7/8 candidates
are scientifically incompatible.  This design accepts that conclusion; it
does not revisit or work around it.

## Vocabulary and lifecycle

Use **declared controlled experiment family** (short: *family*) for the
logical study and **family version** for one immutable scientific plan.  The
word *declared* distinguishes it from source-derived replication.

```
draft input -> frozen family version -> review evidence -> approval
     -> atomic materialization -> paused members -> execution authorization
```

Draft input is an in-memory/file request, not a mutable durable scientific
row.  `prepare` validates it and inserts one complete immutable family version
and its expansion in a single transaction.  Consequently `frozen` is an
inherent property of a stored version, rather than a state in which editable
columns might accidentally remain mutable.

Review is an append-only evidence record and a read-only rendering, not a
mutable state.  Approval is a separate append-only decision bound to the
exact version/specification/plan/review canonical identities.  Materialization
and execution authorization are also append-only records.  A status view may
derive `frozen`, `reviewed`, `approved`, `materialized`, and
`execution_authorized`; no later event edits prior evidence.

Changing science creates a new version, linked by `supersedes_family_version_id`.
It cannot update, unapprove, or reuse an approval for an earlier version.
An approved/materialized/executed version remains historical forever;
operational pausing or authorization revocation never changes its approval.

## Three deliberately separate domains

### Scientific configuration

This is hashed, reviewed, immutable, and is identical across a matched pair
except for explicitly declared arm dimensions.  It includes the full
construction contract:

- canonical symbol, horizon, paired fresh seed, target epoch budget, fresh
  versus resume initialization, train/inference half-open UTC ranges, and
  threshold;
- model-input semantic layout and width; feature warmup scope; Donchian mode
  and lookback; canonical feature-ablation mask; and a versioned
  target-generation/label/normalization contract;
- architecture and train contract: window, batch, hidden size, layer count,
  base learning rate, core/head-weight/head-bias multipliers, optimizer,
  class weights, clipping, accumulation/precision, and checkpoint interval;
- full training-objective canonical text/hash/version, auxiliary loss and all
  nullable regression/robust/clipping/normalization fields;
- checkpoint-inference and checkpoint-policy contract (explicitly disabled is
  a value, not an omitted default); continuation policy contract; and final
  evaluation scope/range;
- immutable calendar, model-input semantic-layout, target-generation, feature
  family/mask, and objective resource identities; and
- required execution-resource identities described below.

Some of these values are currently persisted only in produced
`train_config_meta`, hard-coded worker behavior, or scheduler command
construction.  A new family must not infer them from whichever binary happens
to run.  The implementation must extend the typed configuration passed to the
worker and/or persist an authoritative declared training contract before
launch.  A family request fails closed if a required value cannot be expressed
and enforced.  `train_config_meta` remains valuable completed-run evidence,
but is not an adequate pre-execution specification.

### Operational scheduling configuration

This is not a treatment effect but must be explicit for reproducibility and
audit: initial `scheduler_priority`, requested capacity class, initial
paused-at-birth behavior, authorization scope, and operator/reason for a
release.  It is stored in the version/materialization/authorization evidence,
not silently taken from current CLI defaults.  Priority is not included in the
scientific pair-equality projection unless a future protocol explicitly says
it affects the scientific question; its exact value is still frozen as an
operational contract.

`status`, `phase`, `resume_requested`, worker PID, leases, timestamps, and
current capacity are live operational state and are never part of scientific
identity.

### Actual execution provenance

The selected `experiment_scheduler_worker_attempt`, its model-producing TRAIN
attempt, and final-INFER producing attempt are facts after execution.  They
must never be copied into the proposed scientific configuration.  The family
instead declares what identity is required; scheduler admission proves the
selected artifact equals it, and analysis proves actual recorded provenance
equals it.

## Family expansion and pair invariants

A family version defines ordered dimensions and arms, then persists the full
expansion before approval:

```
family version
  +- arm definitions (one or more)
  +- cells: (symbol, horizon, fresh seed)
       +- one member per arm
```

`controlled_experiment_family_cell` is the matched-pair unit even when a
future study has more than two arms.  It has a canonical cell key and a
one-based ordinal.  `controlled_experiment_family_member` is the unique
cell/arm member and has its own canonical planned-experiment identity.

For a two-arm study, the pair comparison declares a direction and control arm.
For more arms, each arm names a comparison baseline and comparison direction;
the expansion remains a set of cells, not an arbitrary experiment DSL.

The version stores a closed allow-list of treatment dimensions per arm, for
example `{feature_ablation_mask}`.  Preparation constructs a normalized
scientific field map for every member, compares it to its cell baseline, and
rejects any difference absent from that allow-list.  The same verification is
repeated from persisted rows at approval, materialization, and audit.  Thus a
seed, dates, objective, width, calendar, worker requirement, or hidden
training parameter cannot become an undeclared treatment difference.

## Canonical identities

Canonical text, bytewise `C` collation, and exact canonical equality are
authoritative, following the repository's campaign conventions.  A
`fnv1a64:<16 lowercase hex>` hash is an index/lock accelerator, never the
sole proof of identity; each hash-keyed uniqueness rule includes a collision
ordinal and exact canonical comparison.  Retain a rendered JSON snapshot only
as a convenience/audit artifact, never as the sole source enforced by the
materializer or scheduler.

Versioned, length-framed canonical encoding is required.  It includes contract
versions and named fields in fixed bytewise field order.  Lists use an explicit
order-policy marker:

- symbols are normalized lowercase and stored/rendered in declared lexical
  order; duplicate symbols reject rather than collapse;
- horizons and seeds retain the declared order and reject duplicates;
- arms use a stable ordinal/key; cells sort `(symbol, horizon, seed)`;
- masks are parsed by `FeatureAblationMask`, then stored as its canonical
  Tensor-column order text and its identity hash.  Empty mask is a distinct
  empty value, never NULL;
- all timestamps are UTC RFC3339 with a fixed microsecond representation and
  half-open boundary notation; date-only inputs are converted before hashing;
- integer fields are base-10 without leading signs/zeros except zero; finite
  scientific decimal values use a specified exact decimal representation.
  Reject NaN, infinity, locale formatting, and implicit binary-float
  formatting;
- enums are closed lowercase ASCII tokens.  NULL, false, zero, and empty
  string use distinct length-framed forms;
- referenced resources include both stable logical identifiers where useful
  and immutable content/semantic hashes: calendar ID+hash, objective
  canonical+hash, layout identity/version+width, feature-mask hash, and each
  required worker's role/capabilities/source/SHA/runtime/manifest identity.

The specification identity includes all scientific fields, normalized arm and
cell sets, declared-difference policy, and resource identities.  The plan
identity additionally includes exact member ordinals and each expanded member
identity/count.  Operator names, reasons, generated primary keys, timestamps,
and live scheduler state are excluded from those two identities but are bound
in approval/materialization/authorization identities.

## Proposed additive schema

The following is concrete PostgreSQL design, not a migration.  All referenced
history uses `ON DELETE RESTRICT`; runtime privileges allow column-limited
`INSERT` but no application `UPDATE`/`DELETE` on immutable evidence.

### `controlled_experiment_family`

Stable logical study registry: `controlled_experiment_family_id bigint PK`,
`family_key text COLLATE "C" UNIQUE`, `created_at`, `created_by`, and optional
`retired_at/retired_reason` for naming administration only.  It has no
scientific mutable fields.  `family_key` is a human locator, not identity.

### `controlled_experiment_family_version`

One immutable frozen specification/plan:

```sql
controlled_experiment_family_version_id bigint primary key,
controlled_experiment_family_id bigint not null references controlled_experiment_family,
version_ordinal integer not null,
supersedes_family_version_id bigint null references controlled_experiment_family_version,
specification_contract_version integer not null,
planning_contract_version integer not null,
specification_identity_canonical text collate "C" not null,
specification_identity_hash text not null,
specification_hash_collision_ordinal integer not null default 0,
plan_identity_canonical text collate "C" not null,
plan_identity_hash text not null,
plan_hash_collision_ordinal integer not null default 0,
expected_cell_count integer not null,
expected_member_count integer not null,
expected_arm_count integer not null,
scientific_configuration columns ... not null,
initial_scheduler_priority text not null,
continuation_mode text not null,
created_at timestamptz not null default clock_timestamp(),
created_by text not null,
creation_reason text not null,
rendered_specification_snapshot jsonb not null
```

The normalized scientific columns include every shared field listed in
"Scientific configuration" (ranges, threshold, epochs, fresh-start shape,
input layout/width, warmup/Donchian, training/target/optimizer contracts,
objective fields, checkpoint contract, calendar ID+hash, final-evidence
contract, and continuation contract).  Use `numeric` or canonical text for
new exact decimals instead of PostgreSQL `double precision` where an exact
hashing contract is needed; the materializer converts to the existing
experiment representation only after validation.  `continuation_mode` is
`prohibited` or `declared_policy`; the latter requires a complete frozen
policy canonical/hash.

Important constraints: unique `(family_id, version_ordinal)`; unique exact
specification and plan identities with collision ordinal; nonempty canonical
text/hash format; positive counts; `expected_member_count = expected_cell_count
* expected_arm_count`; required finalized calendar identity; and a fresh-start
check (`resume_model_id IS NULL`, `resume_expand_input_width=false`) when
declared.  A trigger rejects UPDATE/DELETE after insert and validates the
snapshot is a faithful rendering of normalized fields.

### Arms, cells, members, and execution requirements

`controlled_experiment_family_arm` has `family_version_id`, `arm_ordinal`,
stable `arm_key`, display role, optional `baseline_arm_key`, comparison
direction, `declared_difference_canonical/hash`, canonical ablation mask/hash,
and immutable per-arm override fields.  Unique `(version, arm_ordinal)` and
`(version, arm_key)`.  The declared-difference map is closed and checked
against the normalized baseline; for common two-arm studies it is one row for
control and one for treatment.

`controlled_experiment_family_cell` has `family_version_id`, `cell_ordinal`,
`symbol`, `prediction_horizon`, `fresh_initialization_seed`, canonical/hash.
Unique `(version, symbol, horizon, seed)` and `(version, cell_ordinal)`.

`controlled_experiment_family_member` has `family_version_id`, `cell_id`,
`arm_id`, `member_ordinal`, a complete normalized
`planned_experiment_identity_canonical/hash`, collision ordinal, and nullable
`experiment_id`.  Unique `(cell_id, arm_id)`, `(version, member_ordinal)`,
and `(version, planned identity hash, collision ordinal)`.  It stores the
resolved configuration projection needed for materialization or foreign keys
to the corresponding normalized version/arm/cell values; it does not rely on
JSON or mutable defaults.  Its triggers prove membership belongs to the same
version, counts/ordinals are complete at deferred transaction end, and all
non-declared pair fields agree.

`controlled_experiment_family_execution_requirement` is one immutable row per
`(family_version_id, phase)` (initially `train` and `infer`): semantic layout,
input width, semantic worker role, sorted required capabilities canonical/hash,
source commit, executable SHA-256, runtime identity, canonical manifest path,
and manifest SHA-256 when available.  It makes a required worker an explicit
resource binding rather than a later registry preference.  A future
phase-specific requirement is possible without changing member identity
rules.

### Review, approval, materialization, and authorization

`controlled_experiment_family_review_event` is append-only review evidence:
version FK, contract version, reconstructed specification/plan canonical/hash,
review result (`reviewed` or `rejected`), reviewer, reason, rendered review
hash, and timestamp.  It may record multiple reviews, but approval chooses one
specific valid reviewed event.

`controlled_experiment_family_approval` is append-only and has version/review
FKs; copied exact specification, plan, and review canonical/hash fields;
decision (`approved`/`rejected`); approver/reason; approval identity/hash and
collision ordinal; timestamp.  A composite FK/trigger requires every copied
identity to equal the referenced immutable version/review.  One terminal
approval per exact review canonical identity; identical retry returns it,
changed payload conflicts.  Only `approved` may be materialized.

`controlled_experiment_family_materialization` is one immutable manifest per
version: approval FK, copied approval/version/plan identities, materializer
and reason, materialization identity/hash, expected counts, and timestamp.
Unique `family_version_id`; no reused members are permitted in v1.  Its child
`controlled_experiment_family_materialization_member` maps every member to
exactly one newly inserted `experiment_id`, with member ordinal and identity
copies.  Unique member and unique experiment mappings.  A deferred completeness
trigger requires precisely all ordinals before commit.

`controlled_experiment_family_execution_authorization` is an append-only
operational release decision: version/materialization FKs; exact materialized
member-set identity/hash; scope (`family`, `cell_set`, or `member_set`);
authorizer/reason; grant identity/hash; optional later superseding revocation
event; timestamp.  Its child table lists every authorized member explicitly,
including a whole-family grant, so a changing query cannot alter scope.  A
derived active-grant view resolves supersession.  It is not an approval row and
cannot edit approval history.

Finally, add nullable `experiment.controlled_experiment_family_member_id`
with a unique restrictive FK to `controlled_experiment_family_member`.
Historical rows remain NULL.  A trigger permits setting it only during the
matching materialization transaction, proves all configured experiment columns
equal the member's frozen projection, forces initial
`status='paused', phase='train', resume_requested=false`, and disallows a
second-family attachment.  This direct link makes experiment-to-protocol audit
cheap and prevents dual incompatible memberships.

## Approval and materialization protocol

### Prepare/review/approve

`prepare` parses a typed specification file/arguments, resolves and validates
immutable resource IDs/hashes, canonicalizes masks, expands all cells/members,
and performs no experiment write.  It either writes a complete immutable
version plus child rows in one transaction or writes nothing.  A review command
reloads only those rows and renders: every cell/arm/member, total counts,
dates, mask, calendar, objective, train/infer requirements, continuation mode,
and all undeclared-difference checks.

Approval reloads the version and reconstructs the exact review.  The caller
supplies the expected plan/review hash.  The service compares canonical text as
well as hash, inserts an approval bound to that version only, and makes no
experiment or scheduler change.  Thus approval of A cannot materialize B.

### One all-family materialization transaction

For the Fibonacci-size matrix, **96 members/48 cells in one PostgreSQL
transaction is the required boundary**.  It is small; partial scientific
matrices are unacceptable.  Use this sequence:

1. Begin SERIALIZABLE; acquire a transaction-scoped advisory lock on the
   exact version canonical identity, then the current materializer's
   `SHARE ROW EXCLUSIVE` lock on `experiment`, then lock version/approval rows
   in one documented order.
2. Reload and validate the approved immutable version, review/approval binding,
   child completeness, every pair invariant, immutable resource finalization,
   and exact worker requirements.  Preflight both arms/phases through the
   registry and require equality to the frozen requirements; do not select a
   newer preferred worker.
3. Under the experiment-table lock, check every planned member's exact
   materializable identity for occupancy.  Existing exact controlled binding,
   unbound matching experiment, ambiguous legacy evidence, or any partial
   family is a conflict, never a fuzzy reuse.
4. Insert the immutable materialization manifest, all 96 experiment rows with
   explicitly supplied values and paused/train/no-resume lifecycle, all member
   bindings, and all 96 manifest children.  No source experiment is read or
   cloned.  Each experiment receives fresh administrative `duplicate_nonce`
   only as needed by the legacy unique index; it is excluded from scientific
   identity.
5. Deferred triggers verify 96 members, 48 cells, one row per arm/cell, exact
   mappings, and paused-at-birth.  Commit once; publish success output only
   after commit.

The experiment table lock follows the existing, proven writer-serialization
boundary and blocks scheduler/continuation/conversion experiment writers while
the reload/check/insert boundary is active.  Standardize lock order to avoid
deadlock.  SERIALIZABLE serialization failure retries re-run from step 1;
unique bindings/manifest identity prevent duplicates.  A 96-row transaction is
far below a problematic PostgreSQL transaction size.  There is no valid
per-cell commit mode in v1.

### Idempotency and uncertain commit

| Situation | Required result |
| --- | --- |
| Never materialized, no collision | Insert complete manifest and all members atomically. |
| Identical complete manifest | Validate approval/version/member IDs and return `existing_identical`; no rows change. |
| Any partial mapping/manifest | `materialization_integrity_failure`; do not repair by inserting a remainder. Investigate/restore before a separately authorized remedy. |
| Any conflicting experiment or unprovable legacy equivalent | Abort the whole family before first insert with exact IDs/reasons. |
| Transaction/insertion failure | Roll back all inserts; retry begins with authoritative reload. |
| Client disconnect or `pqxx::in_doubt_error` | Return `outcome_unknown`; the recovery command performs only an immutable-manifest audit. It reports complete exact replay, absent, or integrity failure; it never guesses or inserts. |

The current broad `experiment` unique index is necessary but not sufficient as
the family idempotency key.  The version-to-manifest unique key, member-to-
experiment unique keys, exact canonical identities, and table lock make the
new contract deterministic.

## Execution authorization and scheduler integration

Materialization does not authorize execution.  The materializer always creates
paused members.  `authorize` records an immutable grant, explicitly lists the
member set, and in the same transaction applies the existing release behavior
only to those rows: `paused -> pending`, `resume_requested=true`, a new closed
`scheduler_resume_origin='controlled_family_authorization'`, and the frozen
priority.  It neither starts a worker nor bypasses capacity admission.

The scheduler's pending-load and atomic claim queries must additionally require
an active authorization grant for any non-NULL controlled-family member.  The
same predicate is required in every requeue/recovery/resume path.  Existing
pause/resume state remains the live source of operational truth; the family
grant is an immutable eligibility gate.  Individual resume, campaign group
resume, orphan recovery, direct transition helpers, and administrative CLI
must refuse a controlled member without that active grant.  A grant revocation
is an append-only event followed by existing pause controls; it is not an
"unapproval" and cannot erase execution history.

Authorization may target the whole family, an explicit cell set, or explicit
members.  A multi-arm cell selection must normally contain all its arms; a
partial-cell release requires an explicit protocol capability and is visibly
reported as non-paired operational execution.  Fibonacci v1 should authorize
only the whole 96-member family (or complete cells) to avoid accidental
asymmetric execution.

## Continuation contract

For `continuation_mode='prohibited'`, materialization writes all existing
continuation columns in their disabled/NULL/not-requested shape.  More
importantly, add family-aware enforcement:

- continuation candidate scans exclude controlled members with prohibited
  mode;
- continuation decision/child creation rejects such a source member;
- triggers reject changing a prohibited member to
  `continuation_policy_enabled=true`, inheritance, queued-child, or
  continuation lineage; and
- scheduler restart, orphan recovery, and ordinary resume retain the family
  membership and therefore apply the same rule.

For a future continuation-enabled family, freeze its complete symmetric
continuation policy canonical/hash in the version and require its child policy
and authorization policy to be declared before materialization.  It must not
fall back to the current per-experiment outcome-driven continuation logic.

## Required versus actual worker identity

The version's execution requirement is authoritative before launch.  At each
TRAIN/INFER reservation, the scheduler selects from the registry as usual,
then requires exact equality of role, layout, width, required capability set,
source commit, executable SHA-256, runtime identity, and manifest identity to
the requirement.  Missing, ambiguous, unavailable, or different selection is
an admission failure, not an opportunity to select a current default.

After execution, analysis joins each member through model and final inference
producer attempts and compares the attempt fields to the required fields.
The pair comparator also requires the actual control/treatment identities to
match each other where the protocol requires common execution.  Missing or
mismatched actual provenance makes the pair scientifically unavailable;
metrics remain raw evidence but cannot enter strict family analysis.

This cleanly separates configured requirements from facts.  The requirement
does not claim execution occurred; a worker attempt does not retroactively
change approved science.

## Relationship to replication and campaigns

Keep `ExperimentReplicationMaterialization` unchanged for TG4 and every
source-derived replication.  The shared long-term pieces should be small:
canonical identity framing, exact scientific configuration projection,
worker-provenance comparator, paused-at-birth inserter primitives, and the
experiment table serialization convention.  A future replication may create a
declared family version whose specification explicitly cites source evidence,
but that is a new feature; existing replication must not be rerouted.

Do not repurpose recommendation-campaign tables.  Their materialization
selects ranking members and creates/reuses Phase 4C **conversion proposals**
from existing recommendations; campaign approval is not scientific protocol
approval, and campaign execution/activation has different operational
semantics.  The useful common pattern is append-only canonical review,
approval, and manifest evidence.  If a generic immutable-decision helper is
ever extracted, it may be shared below both domain tables, but no FK should
pretend a declared family was ranking-derived.

## Operator CLI/API shape

Suggested commands, matching existing long-option style:

```text
LSTM_Release --prepare-controlled-experiment-family=SPEC_FILE
LSTM_Release --show-controlled-experiment-family-version=ID
LSTM_Release --review-controlled-experiment-family=ID
LSTM_Release --approve-controlled-experiment-family=ID \
  --expected-controlled-plan-hash=fnv1a64:... \
  --controlled-reviewer=... --controlled-review-reason=...
LSTM_Release --materialize-controlled-experiment-family=ID \
  --expected-controlled-approval-hash=fnv1a64:... \
  --controlled-materialized-by=... --controlled-materialization-reason=...
LSTM_Release --audit-controlled-experiment-family=ID
LSTM_Release --authorize-controlled-experiment-family=ID \
  --controlled-authorization-scope=family \
  --controlled-authorizer=... --controlled-authorization-reason=... --yes
```

Read-only commands render every arm/cell/member and the exact frozen fields.
Mutating commands require expected hashes and explicit identity/reason; only
execution authorization requires `--yes`.  Materialization output identifies
one transaction, exact counts/IDs, `queued=false`, and `started=false`.
Authorization output says rows were released to ordinary pending admission,
not launched.  A recovery/audit command is read-only and reports manifest
completeness, grants, member state, actual/required provenance, continuation
mode, and pair invariants.

## Audit queries/reports

Provide repository-backed reports for: experiment -> member -> cell/arm ->
family/version; frozen version/specification/plan and approval/reviewer/time;
materialization manifest and atomic member mapping; active/past execution
grants; required worker resources versus TRAIN/final-INFER attempt resources;
continuation mode/attempts; pair invariant differences; and experiment
configuration drift.  The family audit must render `unavailable` rather than
silently accepting missing output provenance.  Views may accelerate this, but
the immutable normalized tables are the authority.

## Migration and compatibility

Use additive migrations after the current 097 sequence: new family tables,
append-only enforcement triggers, nullable unique `experiment` FK, scheduler
authorization predicate, and continuation guards.  Build indexes concurrently
where PostgreSQL/migration tooling permits; keep short `ALTER TABLE` lock
windows and deploy schema before the binary that relies on it.  Backfill is
not required.  Historical experiments retain NULL family membership and all
current source-clone replication and campaign behavior remains unchanged.

## Test and qualification matrix

Implementation must include disposable-DB repository tests plus scheduler
service tests for:

- canonical byte identity (order, masks, decimal/timestamp/null forms),
  collision handling, and immutable version/supersession;
- deterministic expansion, exact seed pairing, multi-arm cell shape, declared
  treatment differences, and rejection of every undeclared difference;
- review/approval hash binding, mutation-after-approval rejection, replay and
  conflicting approval behavior;
- one-family atomic materialization, rollback, duplicate conflict, partial
  manifest rejection, concurrent convergence, SERIALIZABLE retry, and
  unknown-commit audit recovery;
- 96-member Fibonacci fixture: 6 x 2 x 4 x 2, 48 complete cells, paused rows,
  zero source requirement;
- scheduler cannot see/claim paused rows, cannot see a controlled pending row
  without active authorization, and only claims after explicit authorization;
- operator resume, requeue, orphan recovery, campaign pause/resume, and
  preemption all preserve the authorization gate;
- no-continuation service and database enforcement across restart/scan/resume;
- exact worker admission, unavailable/mismatched worker rejection, actual vs
  required TRAIN/INFER provenance, and strict pair unavailability on mismatch;
- calendar and objective ID/hash binding, finalized-resource validation, and
  member experiment projection equality;
- existing TG4 replication, replication materialization/concurrency,
  scheduler, continuation, semantic-worker registry, and recommendation
  campaign regression suites.

No production family is a qualification fixture.  The Fibonacci matrix uses a
new disposable database only after implementation is separately authorized.

## Adversarial review and closures

| Attack | Closure |
| --- | --- |
| Mutable defaults leak into a family | Full typed contract is frozen; materializer accepts no default-derived scientific field. |
| Approved science is edited or approval replayed to another version | Immutable version children; approval copies and verifies exact version/plan/review canonical identities. |
| Partial materialization or duplicate after retry/disconnect | One manifest/version, complete deferred member checks, member/experiment uniqueness, audit-only uncertain-commit recovery. |
| Scheduler claims too early or operator resume bypasses | Paused birth plus authorization predicate in pending-load, claim, requeue, and resume paths. |
| Continuation creates descendants | Family-level prohibited-mode checks in scans, child creation, and triggers. |
| Worker/calendar/objective selection drifts | Requirement/resource ID+content identity is frozen; scheduler and post-run provenance compare exact values. |
| Seeds or masks diverge | Cell seed and canonical arm mask are persisted; pair-difference validation repeats at each boundary. |
| Actual worker mismatch still yields paired result | Strict comparator makes the pair scientifically unavailable. |
| Old replication invokes new path | Separate commands/services; shared utilities only. |
| Scheduler/materializer deadlock or SERIALIZABLE duplicate | Documented advisory/table/row lock order and retry with unique manifests/bindings. |
| One experiment belongs to two families or history is erased | Unique nullable experiment member FK and restrictive append-only FKs; no unapproval/delete path. |

## Recommended implementation phases

1. **Family domain schema and repository model.**  Add additive tables,
   canonicalizer, immutable triggers, typed version/arm/cell/member model, and
   disposable DB tests.  Likely new `Sources/ControlledExperimentFamily*`
   modules plus a migration.  Qualification: identity, immutability, expansion.
2. **Prepare/review/approval CLI.**  Add deterministic specification parsing,
   resource inspection, rendering, review/approval repositories/services, and
   read-only CLI.  Qualification: exact replay, stale hash, pair invariants.
3. **Atomic materializer.**  Add declared-row inserter, exact occupancy
   projection, lock/retry/recovery behavior, manifest/binding enforcement, and
   96-member disposable fixture.  Qualification: rollback, concurrency,
   uncertain commit, paused birth.  No production materialization.
4. **Scheduler execution gate and required worker admission.**  Update
   `SchedulerCore` pending/claim/recovery paths and worker selection/reservation
   plumbing; add explicit authorization CLI.  Qualification: no bypass and
   exact requirements/attempt provenance.
5. **Continuation enforcement and audit/comparison.**  Add family-aware
   continuation guards, actual-vs-required audit, strict family availability,
   documentation, and broad regression.  Qualification: restart/orphan and
   TG4/campaign compatibility.
6. **Separately authorized Fibonacci disposable qualification, then an
   operator review.**  Only after every prior phase is accepted should a
   disposable 96-member qualification prove the frozen protocol can be
   represented.  Production materialization remains a separate authorization.

## Fibonacci paper validation

The frozen protocol `causal-fibonacci-layout9-controlled-lstm-v1` fits one
declared family version exactly:

| Frozen item | Proposed binding |
| --- | --- |
| 6 lexical symbols x H4/H6 x seeds `43,47,53,59` x two arms | 48 cells and 96 members; each cell has control and treatment with the same seed. |
| Complete control | Arm `fibonacci_complete`; canonical empty mask; launch has no `--ablate-features`. |
| Treatment | Arm `fibonacci_family_ablated`; only declared difference is the canonical mask below; launch has exactly that argument. |
| Layout/width | Version `model_input_semantic_layout_version=9`, `model_input_width=103` for both arms. |
| Dates/epochs/fresh start | Version train `[2010-01-01T00:00:00Z,2022-01-01T00:00:00Z)`, infer `[2022-01-01T00:00:00Z,2025-01-01T00:00:00Z)`, target 20, no resume. |
| Calendar | Version resource `economic_calendar_snapshot_id=1`, `fnv1a64:67610f94f5c8e7cc`. |
| TRAIN requirement | Requirement row: role `train`, layout/width 9/103, source `e964fa9e335e9ae63918187a7ffee7aa77f32b4b`, SHA-256 `f56342895009e19cc69751259590cd945d64e5bfb5ffcd330f6aefc8d3fd26e9`, runtime `6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e`, capabilities `train,train_feature_ablation_v1`. |
| No continuation | Version `continuation_mode=prohibited`, disabled persisted policy and service/trigger guards. |

The exact canonical 23-column treatment mask (identity
`fnv1a64:a3f595680caadb2e`) is:

```text
fib_recent_price_scale_valid,fib_up_recent_union_count_log,fib_up_recent_h1_count_log,fib_up_recent_h2_count_log,fib_up_recent_h1_h2_both_count_log,fib_up_recent_h1_youngest_age_20,fib_up_recent_h2_youngest_age_20,fib_up_recent_median_1272_signed_atr,fib_up_recent_median_1618_signed_atr,fib_up_recent_median_pullback_0382_signed_atr,fib_up_recent_median_pullback_0500_signed_atr,fib_up_recent_median_pullback_0618_signed_atr,fib_down_recent_union_count_log,fib_down_recent_h1_count_log,fib_down_recent_h2_count_log,fib_down_recent_h1_h2_both_count_log,fib_down_recent_h1_youngest_age_20,fib_down_recent_h2_youngest_age_20,fib_down_recent_median_1272_signed_atr,fib_down_recent_median_1618_signed_atr,fib_down_recent_median_pullback_0382_signed_atr,fib_down_recent_median_pullback_0500_signed_atr,fib_down_recent_median_pullback_0618_signed_atr
```

The deterministic counts are exactly `6 * 2 * 4 * 2 = 96` members and
`6 * 2 * 4 = 48` matched cells.  Materialization would create every member
paused; no member is runnable until an explicit grant, both arms must reserve
the identical required TRAIN identity, and continuation remains prohibited.
This demonstration creates no rows and does not alter the frozen protocol.

## Final recommendation and open implementation decisions

Adopt the declared-family architecture as a new, additive domain.  Do not
extend source-clone replication or campaign materialization to fill this gap.
Before Phase 1 implementation, resolve only these implementation-level
questions: whether the existing worker command can carry every frozen training
contract field or needs a versioned contract argument; whether manifest SHA is
available for every supported role; and the exact database role/trigger
privilege boundary for immutable rows.  None requires a scientific decision or
changes the frozen Fibonacci protocol.
