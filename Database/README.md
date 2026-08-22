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
- `073_inference_profitability_observation.sql`: immutable, explicitly scoped
  final/checkpoint terminal-horizon directional log-return observations. It
  performs no historical backfill and activates no recommendation, campaign,
  continuation, or checkpoint policy. See
  `docs/InferenceProfitabilityPersistence.rst`.
- `046_global_experiment_control.sql`: database-authoritative global desired
  execution state, administrative request/outcome audit, cancellation targets,
  and managed worker PID/process-group/executable/process-start identity
- `050_experiment_current_operation_canonicalization.sql`: reconciles legacy
  operation/control labels and enforces the sole persisted
  `current_operation` values `train`, `infer`, and `analyze`
- `051_scheduler_ownership_and_worker_attempts.sql`: immutable scheduler
  invocations, fenced singleton ownership, durable global worker
  attempts/capacity, exact lifecycle attempt links, and conservative
  `legacy_unverified` reservations for existing active rows
- `055_campaign_operations_production_admission_foundation.sql`: additive H1
  production enablement/admission evidence, Attempt V2 shape and reconstruction,
  exact fixed-transition replay, sealed owner-safe transaction context, full
  acquisition lock order, exact fail-closed sealed-role/object/ACL preflight,
  deterministic read-only deployment audit, deferred consistency, inert NOLOGIN
  capabilities, and read-only readiness/status; use
  `Scripts/CampaignOperationsH1DeploymentAudit.sh` at every documented upgrade,
  restore, and pre-enablement stage. Before catalog comparison the audit
  validates the versioned, checksummed object, explicit ACL, default ACL, and
  column ACL manifests in `manifests/`. Actual catalog discovery is independent
  from the expectation rows, preserves exact NULL-versus-explicit ACL origin,
  and performs two-way explicit/default ACL set differences. Logical restore
  workflows use `Scripts/CampaignOperationsH1RestoreAclOrigin.sh` to restore the
  frozen explicit ACL origin that `pg_dump` otherwise normalizes. It grants no LOGIN membership and implements no
  production dispatch, enable/disable service, Manager, scheduler, lifecycle,
  or worker behavior
- `058_campaign_operations_h3_manager_run_once.sql`: additive H3 complete
  Manager source-canonical evidence keyed to immutable Attempt V2 rows, with
  a deferred COMMIT-time reverse guarantee so every reserved `mgr-v1:` Attempt
  V2 has exactly one matching source row; it also snapshots the finite,
  immutable set of complete pre-058 H2 caller-keyed `mgr-v1:` Attempt V2
  operations for exact H2 replay/recovery only.  No post-058 caller-keyed
  acquisition may use that namespace, and the compatibility rows never create
  or backfill Manager source evidence; no durable batch identity, LOGIN grant,
  scheduler capability, or worker control

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
- `062_campaign_operations_pre_phase_h_view_access.sql`: restores the explicit
  `SELECT` required by the NOLOGIN owner of the Phase 2 status views after H1
  seals the operational-request table; it creates no LOGIN or capability grant
- `063_campaign_operations_h1_owner_read_acl_reconciliation.sql`: restores the
  four explicit H1-relation `SELECT` tuples required by the existing
  `campaign_operations_owner` SECURITY DEFINER read paths. H1 ownership,
  ordinary-role isolation, and all mutation ACLs remain unchanged.
- `064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql`: restores
  only the direct campaign-lock EXECUTE edge required by the Phase A-G budget
  administrator and request acceptor repository paths. The H1 boundary
  authority remains the helper owner; PUBLIC, `pqxx`, dispatcher, Phase-5,
  and all other sealed lock-helper grants remain prohibited. Its ACL is checked
  by the read-only compatibility overlay in
  `manifests/064_campaign_operations_pre_phase_h_acl_manifest.sql`. Migration
  064 does not rewrite or invoke the frozen migration-055 H1 deployment audit;
  the overlay is the independent post-H1 acceptance contract.

Campaign Operations deployment also requires an explicit, separately reviewed
pre-Phase-H LOGIN selected with `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER`.
For the admission path, grant the existing
`campaign_operations_pre_phase_h_login` exactly these three NOLOGIN
capabilities:

- `campaign_operations_campaign_creator`
- `campaign_operations_budget_administrator`
- `campaign_operations_request_acceptor`

Do not grant that LOGIN any Phase-H production capability. Admission is the
only path that creates an operational campaign from an existing recommendation
campaign materialization; it is an explicit mutation and does not create
budget, request, dispatch, experiment, or scheduler state.

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
``pqxx``. Migration 064 reconciles the H1 deployment drift by restoring only
the narrow campaign-lock function needed by their workflows. They receive no
update, delete, truncate, dispatch, scheduler, worker, or
experiment-lifecycle privilege. Assigning either capability to a deployment
principal is a separate reviewed administrator action.

Campaign Operations Phase 3 migration 048 adds durable lease acquisition,
immutable dispatch attempt/outcome evidence, complete ordered bindings,
permanent V1 control ownership, and atomic held-to-committed/bound Phase 5
handoff. Its dispatcher and transaction-bound Phase 5 roles are separate
NOLOGIN capabilities. Production dispatch remains constrained false.

Migrations 049 through 052 remain the authoritative global-control and
scheduler-hardening history: global selective resume, canonical
``current_operation`` values, scheduler ownership/worker attempts, and
generation-52 protocol/exact-attempt hardening. They are prerequisites for the
integrated Phase F migration and are neither renumbered nor absorbed into
Campaign Operations authority.

Campaign Operations Phase 4 migration 053 implements architectural Phase F.
It adds append-only pause/resume controls, cancellation intent, lifecycle
cancellation evidence, cancellation settlement, reconciliation observations,
resolutions and cursors, and cause-specific control audit. Each observation is
linked to one durable cursor identity; cursor and exact membership commit
atomically under a deferred database completeness constraint, and replay loads
by identity rather than run-key/request ranges. PostgreSQL admits only one
cancellation owner for each request (or campaign-only target), formats
reconciliation timestamps as fixed UTC microseconds, and creates resolutions
only through capability-specific functions that validate and store typed
causal settlement or dispatch-outcome references.
Pause gates request
acceptance and every Phase 3 selection/acquisition/handoff check without
signaling a scheduler or worker. Unbound cancellation atomically releases held
units and cancels the request; bound cancellation delegates through a narrow
lifecycle capability and never releases committed units. Expired dispatch
leases return to ``ready`` only after PostgreSQL time and exact absence of
binding, downstream execution, and current-attempt outcome are revalidated.
Reconciliation observation and recovery transactions use bounded
whole-transaction retries; an uncertain commit is resolved by exact durable
cursor or resolution lookup before any retry.
All new capability roles are NOLOGIN and are not granted to ``pqxx``.
If an earlier staged form of migration 053 already contains observations
without durable cursor membership, the corrected migration aborts atomically
instead of guessing membership from the superseded run-key/request-range
scheme. No production Phase 4 contract was released with that representation;
preserve such staged evidence for review rather than rewriting it.
Campaign Operations Phase 5 migration 054 implements architectural Phase G.
It adds one unique immutable completion event per campaign, a matching
same-transaction audit reference, exhaustive fail-closed evidence/blocker and
classification functions, completion gates, and a rebuildable read-only
status view. Completion binds exact budget, reservation, request, binding,
lifecycle, cancellation, and reconciliation evidence without changing any
owning row. Update/delete triggers protect completion history and there is no
force, reopen, supersede, override, or physical archival path.

Apply 054 only after 053. It is replay-idempotent and additive; it does not
rewrite existing Campaign Operations, recommendation, lifecycle, scheduler,
worker-attempt, budget, cancellation, or reconciliation history. The new
``campaign_operations_completion_writer`` role is NOLOGIN and is not granted
to ``pqxx``. It has column-scoped insert rights and the narrow reads/lock
functions required by complete-if-settled, but no lifecycle, scheduler,
worker, cancellation-settlement, reconciliation-resolution, update, or delete
authority. The function owner receives read access only to
``experiment_id``, ``status``, ``phase``, and ``updated_at`` for lifecycle
evidence. Assigning the completion capability to a production service
principal remains a separate reviewed administrator action. Architectural
Phase H production enablement is not part of migration 054.

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

Migration 051 is the durable lease/attempt foundation and must precede 052.
It does not by itself authorize corrected scheduler startup: a pre-051 binary
cannot honor the lease. Migration 051 does not assert ambiguous legacy rows
are live; it conservatively reserves their capacity until exact reconciliation.

Migration 052 is the technical scheduler-protocol cutover and exact-attempt
hardening migration. It is additive to 051 and must not be replaced by editing
an already applied 051 contract. It adds protocol generation/cutover state,
durable `checkpoint_analyze` attempts, exact administrative outcome attempt
identity, active-attempt shape triggers, and supporting indexes/constraints.
The migration is transactional and replay-idempotent.

After applying 052, corrected scheduler startup is rejected without mutation
until this explicit command succeeds:

```bash
./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  --complete-scheduler-protocol-cutover --yes
```

The command performs process inspection and records positive evidence that no
old or corrected scheduler dispatch process is alive. Inspection failure,
partial/failed cutover, stale metadata without positive process absence, or a
live scheduler fails closed. Replaying an already completed cutover is
idempotent. Do not use `UPDATE` to manufacture a completed cutover.

Deployment order is: install the generation-52 binary at a new path; stop and
verify old scheduler dispatch processes without stopping validated workers;
take a backup; apply 051 then 052; run the cutover command; start exactly one
generation-52 scheduler; inspect ownership, capacity, analyze attempts, and
unresolved legacy no-PID status. Do not roll back only the executable after
cutover. Legacy no-PID attempts remain capacity-consuming through the bounded
grace period and are released only by exact, audited reconciliation after
cutover evidence proves old dispatch authority absent.

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
