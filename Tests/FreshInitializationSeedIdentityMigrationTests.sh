#!/usr/bin/env bash
set -euo pipefail

# This is deliberately a predecessor-schema fixture: the corrected index is
# created only by the real 095 migration, never reproduced in test SQL.
root="$(cd "$(dirname "$0")/.." && pwd)"; work="$(mktemp -d "${TMPDIR:-/tmp}/ea_fresh_seed_identity.XXXXXX")"
pgdata="$work/pgdata"; socket="$work/socket"; db="ea_fresh_seed_identity_${$}"; port="$((57000 + ($$ % 1000)))"; postgres=/opt/homebrew/opt/postgresql@17/bin; started=false; created=false
trap 's=$?; if $created; then "$postgres/dropdb" -h "$socket" -p "$port" -U pqxx "$db" >/dev/null 2>&1 || true; fi; if $started; then "$postgres/pg_ctl" -D "$pgdata" -m fast -w stop >/dev/null 2>&1 || true; fi; rm -rf "$work"; exit $s' EXIT
mkdir -p "$socket"; "$postgres/initdb" -D "$pgdata" --username=pqxx --auth=trust --no-locale >/dev/null; "$postgres/pg_ctl" -D "$pgdata" -o "-F -k $socket -p $port -h ''" -w start >/dev/null; started=true; "$postgres/createdb" -h "$socket" -p "$port" -U pqxx "$db"; created=true
psql=("$postgres/psql" -X -v ON_ERROR_STOP=1 -q -h "$socket" -p "$port" -U pqxx -d "$db")
"${psql[@]}" <<'SQL'
CREATE TABLE experiment (
 experiment_id bigserial PRIMARY KEY, symbol text NOT NULL, prediction_horizon integer NOT NULL,
 c_next_threshold double precision NOT NULL, core_lr_mult double precision, head_lr_mult double precision,
 target_epochs integer NOT NULL, checkpoint_interval integer NOT NULL, train_start date NOT NULL, train_end date NOT NULL,
 infer_start timestamptz, infer_end timestamptz, resume_model_id bigint, donchian20_mode text NOT NULL,
 donchian_lookback integer NOT NULL, feature_warmup_scope text NOT NULL, feature_ablation_mask text NOT NULL,
 resume_expand_input_width boolean NOT NULL DEFAULT false, training_objective_hash text NOT NULL DEFAULT '',
 model_input_width integer, model_input_semantic_layout_version integer,
 economic_calendar_snapshot_id bigint, economic_calendar_snapshot_hash text, duplicate_nonce integer NOT NULL,
 status text NOT NULL
);
SQL
"${psql[@]}" -f "$root/Database/migrations/094_fresh_initialization_seed.sql"
"${psql[@]}" -f "$root/Database/migrations/095_fresh_initialization_seed_identity.sql"
sql(){ "${psql[@]}" -c "$1"; }
insert(){ sql "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,target_epochs,checkpoint_interval,train_start,train_end,donchian20_mode,donchian_lookback,feature_warmup_scope,feature_ablation_mask,training_objective_hash,duplicate_nonce,status,fresh_initialization_seed) VALUES ('fresh-seed',4,.0008,1,1,'2025-01-01','2025-01-02','enabled',20,'legacy_cold_boundary','','objective',99,'pending',$1);"; }
insert 41; insert 42
if sql "$(printf "%s" "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,target_epochs,checkpoint_interval,train_start,train_end,donchian20_mode,donchian_lookback,feature_warmup_scope,feature_ablation_mask,training_objective_hash,duplicate_nonce,status,fresh_initialization_seed) VALUES ('fresh-seed',4,.0008,1,1,'2025-01-01','2025-01-02','enabled',20,'legacy_cold_boundary','','objective',99,'pending',41);")" >/dev/null 2>&1; then
  echo 'identical fresh seed unexpectedly bypassed unique identity' >&2; exit 1
fi
[[ "$("${psql[@]}" -At -c "SELECT count(*) FROM experiment WHERE symbol='fresh-seed'")" == 2 ]]
echo FRESH_INITIALIZATION_SEED_IDENTITY_MIGRATION_095_DISPOSABLE_TESTS=passed
