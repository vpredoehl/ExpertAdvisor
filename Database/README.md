# LSTM Database Migrations

Run migrations whenever a branch adds or changes LSTM database schema objects.

```bash
./migrate_lstm_db.sh
```

Defaults:

- `LSTM_DB_HOST=127.0.0.1`
- `LSTM_DB_NAME=LSTM`
- `LSTM_DB_ADMIN_USER=$USER`, or `vjp` if `$USER` is unset

Override example:

```bash
LSTM_DB_HOST=127.0.0.1 LSTM_DB_NAME=LSTM LSTM_DB_ADMIN_USER=vjp ./migrate_lstm_db.sh
```

The migration user must have enough PostgreSQL privileges to create tables,
create indexes, and grant privileges to the runtime user `pqxx`. Campaign
Operations migration `045` additionally requires authority to create or manage
its NOLOGIN owner and capability roles and to transfer object ownership to its
owner role.

The LSTM runtime user `pqxx` should not need schema-creation privileges after
migrations are applied. It only needs DML privileges on runtime tables such as
`inference_eval_result`, `experiment`, and `experiment_analysis_result`.

Applied migrations are tracked in `schema_migrations`:

- `version text primary key`
- `filename text not null`
- `checksum text not null`
- `applied_at timestamptz not null default now()`

Experiment scheduling tables are created by:

- `005_experiment_scheduler.sql`: `experiment`
- `006_experiment_analysis.sql`: `experiment_analysis_result`
- `046_global_experiment_control.sql`: database-authoritative global desired
  execution state, administrative request/outcome audit, cancellation targets,
  and managed worker PID/process-group/executable/process-start identity
- `050_experiment_current_operation_canonicalization.sql`: reconciles legacy
  operation/control labels and enforces the sole persisted
  `current_operation` values `train`, `infer`, and `analyze`

Recommendation conversion and campaign-approval history is created by:

- `036_experiment_recommendation_conversion_proposal.sql`:
  `experiment_recommendation_conversion_proposal`
- `037_experiment_recommendation_conversion_review.sql`:
  `experiment_recommendation_conversion_review_decision`
- `038_experiment_recommendation_conversion_execution.sql`:
  `experiment_recommendation_conversion_execution`
- `039_experiment_recommendation_conversion_activation.sql`:
  `experiment_recommendation_conversion_activation`
- `040_experiment_recommendation_campaign_approval.sql`:
  `experiment_recommendation_campaign_approval`
- `041_experiment_recommendation_campaign_materialization.sql`:
  `experiment_recommendation_campaign_materialization` and
  `experiment_recommendation_campaign_materialization_member`, with
  invoker-rights provenance and deferred completeness enforcement
- `042_experiment_recommendation_campaign_follow_up_proposal.sql`:
  immutable exact Phase 6A proposal manifests and ordered members for Phase 6B
  read-only operator preview
- `043_experiment_recommendation_campaign_follow_up_proposal_review.sql`:
  one immutable approved/rejected Phase 6C administrative review event per
  exact persisted Phase 6B proposal, without action authority
- `044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql`:
  one immutable Phase 6D governance ratification per exact eligible approved
  Phase 6C review, with mandatory reviewer/ratifier separation and no Phase 6E
  or operational authority
- `045_campaign_operations_foundation.sql`:
  immutable Campaign Operations V1 campaign, optional exact Phase 6D
  provenance, serialized authorization evidence, same-transaction audit
  references, and disabled-by-default capability roles; no operational workflow
- `047_campaign_operations_budget_request_acceptance.sql`:
  append-only materialized-member budget ledger, guarded held reservations,
  durable ready requests, acquisition events, accounting/status views,
  same-transaction audit completeness, and separate disabled-by-default budget
  administrator and request acceptor roles; no dispatch or lifecycle authority

These append-only tables record manually prepared proposals and their explicit
operator review decisions. An approval is administrative evidence for possible
later conversion; it is not an experiment queue, does not create an experiment,
and is not read by the scheduler. The greatest review-decision ID for a proposal
defines its current review disposition. That generated sequence-ID order, not
transaction commit time or the descriptive timestamp, is authoritative.
An explicit Step 4 conversion records the exact approving decision and creates
one paused experiment. It does not queue, start, or schedule that experiment.
An explicit Step 5 activation records a separate immutable audit event and
changes only that existing experiment from `paused/train` to `pending/train`.
It creates no experiment, starts no worker, and adds no scheduler dependency on
Phase 4C tables.
Phase 4C Step 6 adds no schema object or privilege. Its read-only workflow view
joins these existing audit records with current experiment lifecycle state and
reports deterministic integrity diagnostics.
Phase 4D Steps 1 and 2 add no schema object or privilege. Their
read-only campaign planner and review consume one explicit completed ranking
snapshot plus existing
recommendation and Phase 4C workflow evidence in a PostgreSQL read transaction.
They do not persist a plan or advance a sequence.
Phase 4D Step 3 reconstructs that exact plan and review in the same transaction
that inserts one immutable operator approval or rejection. Runtime ``pqxx``
has ``SELECT``, column-limited payload ``INSERT``, and sequence ``USAGE`` only;
it cannot provide generated IDs/timestamps or update, delete, or truncate
approval history. Canonical review text is authoritative, identical retries
return the existing row, and changed payload for the same review conflicts.
Campaign approval does not create or modify an experiment and does not execute
the campaign.
Phase 4D Step 4 atomically reconstructs one approved campaign and creates or
reuses only its exact ordered Phase 4C conversion-proposal set. One immutable
manifest and ordered member links preserve approval, ranking, review, and
proposal provenance. Runtime access is append-only and column-limited.
First-time materialization reconstructs current authoritative evidence; exact
retry validates and returns the immutable manifest because its Phase 4C
proposals intentionally change later planning evidence.
Materialization creates no conversion review, execution, activation, or
experiment and does not involve the scheduler or workers.
Phase 4D Step 5 adds no schema object or privilege. Its read-only campaign
handoff projection treats the Step 4 manifest/member rows as authoritative
membership, selects the greatest Phase 4C review-decision ID for current
disposition, and validates linked proposal, execution, activation, and
experiment evidence in one read transaction. It advances no sequence and
uses `REPEATABLE READ` for one consistent snapshot; it never repairs or
progresses a workflow.
Phase 4D Step 6 also adds no schema object or privilege. One explicit operator
command validates the exact immutable Step 4 member set and atomically appends
one ordinary Phase 4C review row per member. A deterministic request ID binds
the materialization, decision, operator, reason, and operation version for exact
retry recognition. The greatest Phase 4C review-decision ID remains authoritative;
no campaign-level status authority, execution, activation, experiment, scheduler,
or worker behavior is added.
Phase 5 Step 1 adds no schema object or privilege. Its explicit confirmed
campaign command validates one exact Phase 4D materialization and uses the
existing Phase 4C execution transaction primitive to create all corresponding
``paused/train`` experiments and immutable conversion-execution rows atomically.
Exact all-member retries insert nothing; partial prior execution conflicts. It
does not activate or queue experiments, modify scheduler state, or start workers.
Phase 5 Step 2 likewise adds no schema object or privilege. It validates the
same immutable membership and atomically reuses the existing Phase 4C
activation insert plus ``paused/train`` to ``pending/train`` experiment update.
Exact all-member retries insert nothing; mixed prior activation conflicts. It
does not start the scheduler, launch workers, or perform a follow-up command.
Phase 5 Step 3 adds no schema object or privilege. Its explicit confirmed
launch command composes the existing transaction-bound Phase 4C execution and
activation authorities in one outer transaction for the exact immutable
materialization. It either creates all required paused experiments/executions
and all activations, reuses all executions before activation, or reports an
exact already-satisfied result; partial prior execution or activation fails
closed. The resulting ordinary experiments are ``pending/train`` but the
command neither polls nor signals the scheduler and launches no worker.
Phase 5 Step 4 also adds no schema object or privilege. Its standalone status
command uses one repeatable-read, read-only transaction to validate immutable
materialization membership and project bounded exact Phase 4C, experiment,
model, final-inference, and final-analysis evidence. It takes no advisory or row
lock, advances no sequence, writes no cached campaign state, and neither polls
nor controls the scheduler or workers.
Phase 6B adds append-only persistence for the exact immutable Phase 6A
follow-up proposal. Canonical text remains authoritative, exact retries are
idempotent, hash collisions remain distinct, and deferred completeness plus
upstream-provenance triggers protect the ordered manifest. Runtime access is
limited to `SELECT`, column-scoped `INSERT`, and sequence `USAGE`. The
read-only preview path adds no approval, activation, execution, experiment,
queue, scheduler, worker, or follow-up authorization behavior.
Phase 6C adds one append-only administrative review-event table with a
restrictive foreign key and exact version/canonical/hash binding to a persisted
Phase 6B proposal. One proposal has at most one immutable approved or rejected
decision; exact replay is idempotent and any changed decision, reviewer, reason,
or identity conflicts. Runtime access remains `SELECT`, column-scoped `INSERT`,
and sequence `USAGE`; generated IDs/timestamps and update/delete/truncate are
denied. Approval means only administrative approval for possible consideration
by a later explicitly authorized phase. It does not activate, execute,
authorize follow-up, queue, schedule, signal/start the scheduler, launch a
worker, create/modify an experiment, or declare campaign success.
Phase 6D adds one separate append-only governance-ratification event table.
It asks whether a governance actor ratifies the exact merits-approved proposal
for entry into the next separately controlled phase. Only an exact persisted
Phase 6C `approved` review is eligible, the fixed role is
`follow_up_governance_ratifier`, and the ratifier must differ from the Phase 6C
reviewer. Restrictive review/proposal foreign keys, checks, and an
invoker-rights provenance trigger verify the complete copied
version/canonical/hash chain and separation of duties. Exact replay is
idempotent; any changed ratifier, basis, role, or identity conflicts.
Runtime access is limited to `SELECT`, column-scoped `INSERT`, and sequence
`USAGE`, with explicit PUBLIC and trigger-function revocation. Phase 6D
ratification remains non-operational evidence: it grants no Phase 6E
capability and does not authorize follow-up or execution, activate, queue,
schedule, signal/start the scheduler, launch a worker, create/modify an
experiment or model, or declare campaign success.

Campaign Operations Phase 1 adds only its foundational domain persistence.
One immutable operational campaign binds one exact Phase 4D materialization;
row existence directly derives `awaiting_operational_authorization`. Optional
Phase 6D evidence is provenance/prerequisite only. Authorization history is an
append-only, fork-resistant chain whose persisted kinds are exactly `granted`,
`revoked`, and `expiry_observed`; supersession is represented by one successor
`granted` row. The migration creates separate NOLOGIN capability roles but does
not grant them to `pqxx`, so no runtime workflow is enabled. It creates no
budget, reservation, request, dispatch, cancellation, completion, lifecycle,
scheduler, worker, UI, or CLI behavior.

Campaign Operations Phase 2 adds the first bounded operational authority.
Budget grants, amendments, revocations, and explicit supersession form one
append-only ledger per operational campaign. A request acceptance transaction
locks authorization, budget, and campaign in the accepted order, validates the
current operational grant and active budget, reserves exactly the immutable
materialization member count, and atomically inserts one ``held`` reservation,
one ``ready`` request, its acquisition event, and audit evidence. The
authorization trigger takes the same authorization-domain lock as the
repository, including for direct capability-role inserts. The logical
operation key makes identical retries return the existing request and makes any
changed actor, reason, authorization, expiry, or payload conflict. Deferred
constraints prevent a budget entry, reservation, or request from committing
without its required audit/acquisition evidence. Request evidence is bound
exactly to the accepting authorization's action, scope, prerequisite policy,
and optional governance provenance. Cause-specific audit constraints bind
actor, reason, capability, versions, and causal IDs to the authoritative
mutation, and PostgreSQL rejects a non-null reservation expiry that is not
later than ``transaction_timestamp()``. Request status reports evidence
consistent only after matching the exact request, reservation, authorization,
prerequisite/provenance, budget, acquisition, and audit relationships.

Migration 047 creates separate NOLOGIN
``campaign_operations_budget_administrator`` and
``campaign_operations_request_acceptor`` capabilities and grants neither to
``pqxx``. Capability roles receive only the reads, column-scoped inserts,
sequence usage, and narrow campaign-lock function needed by their workflows.
They receive no update, delete, truncate, dispatch, scheduler, worker, or
experiment-lifecycle privilege. Assigning either capability to a deployment
principal is a separate reviewed administrator action.

The scheduler and analyzer expect these migrations to be applied before running
`--schedule-experiments`, `--enqueue-experiment`, or leaderboard commands.
Migration 046 is additionally required before starting the scheduler or using
``--pause-all-experiments``, ``--resume-all-experiments``, or
``--cancel-all-experiments``. Migration 050 must be applied before relying on
the database-enforced canonical scheduler operation contract. Apply migrations
with ``migrate_lstm_db.sh`` so reconciliation, trigger/constraint installation,
and migration bookkeeping commit atomically. Migration 050 takes a brief
exclusive lock on ``experiment``; deploy the corrected executable first and
apply it in a monitored maintenance window after validating unsupported rows
and taking a backup. Its compatibility trigger remains only until every
pre-050 scheduler, worker, and administrative executable has exited; remove it
later with a reviewed migration while retaining the constraint. See
``docs/GlobalExperimentControls.rst`` for locking, process validation,
checkpoint cancellation, inference, restart, dry-run, and audit semantics.

## Database Backups

Use the LSTM executable to create a PostgreSQL custom-format snapshot before
important migrations or research milestones:

```bash
./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release --backup-database
```

Backups are written under `Database/backups/` as `.dump` files by default and
include both schema and data. A JSON manifest is written beside each dump with
the code commit, schema version, and table counts when available.

To overwrite a stable snapshot path instead of creating a timestamped dump on
each run:

```bash
./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  --backup-database \
  --backup-output=Database/backups/LSTM_latest.dump
```

Backups are data and runtime state snapshots. Migrations remain the
source-controlled schema history, not data backups.

Run backups before `./migrate_lstm_db.sh` when you need a rollback point. Dump
files are ignored by default to avoid accidental large commits. The stable
`Database/backups/LSTM_latest.dump` path is explicitly allowed by `.gitignore`
for deliberate milestone snapshots.
