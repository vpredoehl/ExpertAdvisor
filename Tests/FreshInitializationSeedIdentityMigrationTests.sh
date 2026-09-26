#!/usr/bin/env bash
set -euo pipefail

# This is deliberately a predecessor-schema fixture: identity indexes are
# created only by the real migrations, never reproduced in test SQL.
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
insert_identity(){
  sql "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,target_epochs,checkpoint_interval,train_start,train_end,resume_model_id,donchian20_mode,donchian_lookback,feature_warmup_scope,feature_ablation_mask,training_objective_hash,model_input_width,model_input_semantic_layout_version,economic_calendar_snapshot_id,economic_calendar_snapshot_hash,duplicate_nonce,status,resume_expand_input_width,fresh_initialization_seed) VALUES ('identity',4,.0008,1,1,'2025-01-01','2025-01-02',$1,'enabled',20,'legacy_cold_boundary','$6','$7',$4,$5,$8,'snapshot-$8',$3,'$9',${10},$2);"
}
reject_identity(){
  if insert_identity "$@" >/dev/null 2>&1; then
    echo "identity unexpectedly bypassed unique index: $*" >&2
    exit 1
  fi
}

# Migration 095 preserves distinct fresh initializations.
insert_identity NULL 41 99 80 8 '' objective-A 1 pending false
insert_identity NULL 42 99 80 8 '' objective-A 1 pending false
if insert_identity NULL 41 99 80 8 '' objective-A 1 pending false >/dev/null 2>&1; then
  echo 'identical fresh seed unexpectedly bypassed unique identity' >&2; exit 1
fi

"${psql[@]}" -f "$root/Database/migrations/097_resume_seed_conditional_identity.sql"

# Fresh seed remains scientific identity after 097; resume seed does not.
insert_identity NULL 43 99 80 8 '' objective-A 1 pending false
insert_identity 1001 42 99 80 8 '' objective-A 1 pending false
reject_identity 1001 43 99 80 8 '' objective-A 1 pending false

# Every other index dimension remains deliberate identity, including the
# explicit duplicate nonce escape hatch and the cancelled-row exclusion.
insert_identity 1002 43 99 80 8 '' objective-A 1 pending false
insert_identity 1001 43 99 80 8 '' objective-A 1 pending true
insert_identity 1001 43 99 81 8 '' objective-A 1 pending false
insert_identity 1001 43 99 80 9 '' objective-A 1 pending false
insert_identity 1001 43 99 80 8 mask-a objective-A 1 pending false
insert_identity 1001 43 99 80 8 '' objective-B 1 pending false
insert_identity 1001 43 99 80 8 '' objective-A 2 pending false
insert_identity 1001 43 100 80 8 '' objective-A 1 pending false
insert_identity 1001 43 99 80 8 '' objective-A 1 cancelled false

# A collision in historical resumed compatibility values must reject the
# migration atomically; neither row nor the prior 095 index may be lost.
sql 'DROP TABLE experiment;'
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
insert_identity 1001 42 99 80 8 '' objective-A 1 pending false
insert_identity 1001 43 99 80 8 '' objective-A 1 pending false
set +e
"${psql[@]}" -f "$root/Database/migrations/097_resume_seed_conditional_identity.sql" >/dev/null 2>&1
collision_status=$?
set -e
test "$collision_status" -ne 0
[[ "$("${psql[@]}" -At -c "SELECT count(*) FROM experiment")" == 2 ]]
[[ "$("${psql[@]}" -At -c "SELECT pg_get_indexdef('experiment_unique_identity_uidx'::regclass)")" == *"fresh_initialization_seed"* ]]
echo FRESH_INITIALIZATION_SEED_CONDITIONAL_IDENTITY_MIGRATION_097_DISPOSABLE_TESTS=passed
