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
create indexes, and grant privileges to the runtime user `pqxx`.

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

The scheduler and analyzer expect these migrations to be applied before running
`--schedule-experiments`, `--enqueue-experiment`, or leaderboard commands.

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
