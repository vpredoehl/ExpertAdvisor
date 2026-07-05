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
