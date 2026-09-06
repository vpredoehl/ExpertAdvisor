#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/economic_calendar_snapshot"
BIN="$BUILD_DIR/EconomicCalendarSnapshotTests"
SCHEDULER_BINARY="${1:-}"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-vjp}"
SOURCE_DB="${LSTM_DB_NAME:-LSTM}"
BASE_DB="ea_phase9_snapshot_base_${$}"
FULL_DB="ea_phase9_snapshot_full_${$}"
FORWARD_DB="ea_phase9_snapshot_forward_${$}"
REVERSE_DB="ea_phase9_snapshot_reverse_${$}"
TEMP_DIR="$(mktemp -d /tmp/ea-phase9-snapshot.XXXXXX)"

for database in "$BASE_DB" "$FULL_DB" "$FORWARD_DB" "$REVERSE_DB"; do
    case "$database" in
        ea_phase9_snapshot_*_[0-9]*) ;;
        *) exit 90 ;;
    esac
done

admin_psql() {
    psql -X -q -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$DB_ADMIN_USER" --dbname="$1" "${@:2}"
}

runtime_psql() {
    psql -X -q -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$DB_USER" --dbname="$1" "${@:2}"
}

drop_test_database() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        "$1" >/dev/null 2>&1 || true
}

cleanup() {
    drop_test_database "$REVERSE_DB"
    drop_test_database "$FORWARD_DB"
    drop_test_database "$FULL_DB"
    drop_test_database "$BASE_DB"
    rm -r -- "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a PQXX_CFLAGS <<< "$(pkg-config --cflags libpqxx)"
    read -r -a PQXX_LIBS <<< "$(pkg-config --libs libpqxx)"
else
    PQXX_PREFIX="$(brew --prefix libpqxx)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"
    PQXX_CFLAGS=("-I${PQXX_PREFIX}/include" "-I${LIBPQ_PREFIX}/include")
    PQXX_LIBS=("-L${PQXX_PREFIX}/lib" "-L${LIBPQ_PREFIX}/lib" -lpqxx -lpq)
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Sources/EconomicEventRepository.cpp" \
    "$ROOT/Tests/EconomicCalendarSnapshotTests.cpp" \
    "${PQXX_LIBS[@]}" -o "$BIN"

for database in "$BASE_DB" "$FULL_DB" "$FORWARD_DB" "$REVERSE_DB"; do
    if admin_psql postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$database'" | grep -q 1; then
        printf 'refusing to reuse existing database: %s\n' "$database" >&2
        exit 1
    fi
done

createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" --owner="$DB_USER" \
    --template=template0 "$BASE_DB"

# Production is inspected only through a read-only schema dump. The clone is
# advanced from the deployed schema-090 shape through 091 and then 092.
PGOPTIONS='-c default_transaction_read_only=on' \
    pg_dump -s --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        --dbname="$SOURCE_DB" |
    admin_psql "$BASE_DB" >"$TEMP_DIR/schema-restore.log"

admin_psql "$BASE_DB" \
    -f "$ROOT/Database/migrations/091_weekly_claims_historical_consensus.sql" \
    >"$TEMP_DIR/migration-091.log"

# Historical rows are deliberately created before 092. They must survive and
# remain NULL-bound after the migration.
runtime_psql "$BASE_DB" <<'SQL'
INSERT INTO experiment(
    symbol,prediction_horizon,c_next_threshold,target_epochs,
    checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce,
    donchian20_mode,feature_warmup_scope,donchian_lookback,
    feature_ablation_mask,model_input_width,
    model_input_semantic_layout_version)
VALUES('phase9legacy',4,0.0008,1,0,'2020-01-01','2020-01-02',
       'completed','done',9009001,'enabled','full_history_warmup',20,'',77,6);
INSERT INTO model(name,experiment_id)
SELECT 'phase9-legacy-model',experiment_id
FROM experiment WHERE symbol='phase9legacy';
SQL

legacy_before="$(runtime_psql "$BASE_DB" -tAc "SELECT e.experiment_id||'|'||m.model_id||'|'||e.symbol||'|'||e.status||'|'||e.phase FROM experiment e JOIN model m ON m.experiment_id=e.experiment_id WHERE e.symbol='phase9legacy'")"

admin_psql "$BASE_DB" \
    -f "$ROOT/Database/migrations/092_economic_calendar_snapshot.sql" \
    >"$TEMP_DIR/migration-092.log"

test "$(runtime_psql "$BASE_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot")" = 0
test "$(runtime_psql "$BASE_DB" -tAc "SELECT count(*) FROM experiment WHERE symbol='phase9legacy' AND economic_calendar_snapshot_id IS NULL AND economic_calendar_snapshot_hash IS NULL")" = 1
test "$(runtime_psql "$BASE_DB" -tAc "SELECT count(*) FROM model WHERE name='phase9-legacy-model' AND economic_calendar_snapshot_id IS NULL AND economic_calendar_snapshot_hash IS NULL")" = 1
legacy_after="$(runtime_psql "$BASE_DB" -tAc "SELECT e.experiment_id||'|'||m.model_id||'|'||e.symbol||'|'||e.status||'|'||e.phase FROM experiment e JOIN model m ON m.experiment_id=e.experiment_id WHERE e.symbol='phase9legacy'")"
test "$legacy_before" = "$legacy_after"

# Direct replay is the idempotence convention used by neighboring migrations.
admin_psql "$BASE_DB" \
    -f "$ROOT/Database/migrations/092_economic_calendar_snapshot.sql" \
    >"$TEMP_DIR/migration-092-replay.log"
test "$(runtime_psql "$BASE_DB" -tAc "SELECT count(*) FROM pg_trigger WHERE tgname='calendar_snapshot_content_immutable' AND NOT tgisinternal")" = 4

for database in "$FULL_DB" "$FORWARD_DB" "$REVERSE_DB"; do
    createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" --owner="$DB_USER" \
        --template="$BASE_DB" "$database"
done

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_ADMIN_USER="$DB_ADMIN_USER" \
    LSTM_TEST_DB_NAME="$FULL_DB" "$BIN"

forward_output="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_ADMIN_USER="$DB_ADMIN_USER" \
    LSTM_TEST_DB_NAME="$FORWARD_DB" "$BIN" --hash-order forward)"
reverse_output="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_ADMIN_USER="$DB_ADMIN_USER" \
    LSTM_TEST_DB_NAME="$REVERSE_DB" "$BIN" --hash-order reverse)"
forward_hash="${forward_output#HASH=}"
reverse_hash="${reverse_output#HASH=}"
test "$forward_hash" = "$reverse_hash"
case "$forward_hash" in fnv1a64:[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) exit 2 ;; esac

if [[ -n "$SCHEDULER_BINARY" ]]; then
    [[ -x "$SCHEDULER_BINARY" ]]
    cli_env=(
        LSTM_DB_HOST="$DB_HOST"
        LSTM_DB_USER="$DB_USER"
        LSTM_DB_NAME="$FORWARD_DB"
    )
    env "${cli_env[@]}" "$SCHEDULER_BINARY" \
        --create-economic-calendar-snapshot --dry-run \
        >"$TEMP_DIR/cli-dry-run.out"
    grep -q '^ECONOMIC_CALENDAR_SNAPSHOT_DRY_RUN,snapshot_id=DRY_RUN,' \
        "$TEMP_DIR/cli-dry-run.out"
    grep -Eq 'content_hash=fnv1a64:[0-9a-f]{16}' \
        "$TEMP_DIR/cli-dry-run.out"
    grep -q 'canonical_event_count=4' "$TEMP_DIR/cli-dry-run.out"
    grep -q 'selected_consensus_count=1' "$TEMP_DIR/cli-dry-run.out"
    grep -q 'release_actual_count=1' "$TEMP_DIR/cli-dry-run.out"
    grep -q 'proven_first_release_actual_count=1' \
        "$TEMP_DIR/cli-dry-run.out"
    grep -q 'provenance_unavailable_count=3' "$TEMP_DIR/cli-dry-run.out"
    grep -q 'source_family_counts=' "$TEMP_DIR/cli-dry-run.out"
    grep -q 'reused=0,dry_run=1' "$TEMP_DIR/cli-dry-run.out"
    test "$(runtime_psql "$FORWARD_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot")" = 0

    env "${cli_env[@]}" "$SCHEDULER_BINARY" \
        --create-economic-calendar-snapshot \
        >"$TEMP_DIR/cli-create.out"
    grep -q '^ECONOMIC_CALENDAR_SNAPSHOT_FINALIZED,snapshot_id=[0-9]' \
        "$TEMP_DIR/cli-create.out"
    grep -q 'reused=0,dry_run=0' "$TEMP_DIR/cli-create.out"
    env "${cli_env[@]}" "$SCHEDULER_BINARY" \
        --create-economic-calendar-snapshot \
        >"$TEMP_DIR/cli-reuse.out"
    grep -q 'reused=1,dry_run=0' "$TEMP_DIR/cli-reuse.out"
    test "$(runtime_psql "$FORWARD_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot")" = 1

    queue_env=(
        LSTM_DB_HOST="$DB_HOST"
        LSTM_DB_USER="$DB_USER"
        LSTM_DB_NAME="$REVERSE_DB"
    )
    env "${queue_env[@]}" "$SCHEDULER_BINARY" \
        --queue-experiment --symbol=eurusdrmp --prediction-horizon=4 \
        --target-epochs=1 >"$TEMP_DIR/queue-fresh.out"
    grep -q '^QUEUE_EXPERIMENT_CREATED' "$TEMP_DIR/queue-fresh.out"
    test "$(runtime_psql "$REVERSE_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot WHERE snapshot_state='finalized'")" = 1
    test "$(runtime_psql "$REVERSE_DB" -tAc "SELECT count(*) FROM experiment WHERE symbol='eurusdrmp' AND economic_calendar_snapshot_id IS NOT NULL AND economic_calendar_snapshot_hash IS NOT NULL")" = 1
    set +e
    env "${queue_env[@]}" "$SCHEDULER_BINARY" \
        --queue-experiment --symbol=eurusdrmp --prediction-horizon=4 \
        --target-epochs=1 >"$TEMP_DIR/queue-repeat.out" 2>&1
    queue_repeat_status=$?
    set -e
    test "$queue_repeat_status" = 3
    grep -q '^QUEUE_ALREADY_EXISTS' "$TEMP_DIR/queue-repeat.out"
    test "$(runtime_psql "$REVERSE_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot")" = 1
    test "$(runtime_psql "$REVERSE_DB" -tAc "SELECT count(*) FROM experiment WHERE symbol='eurusdrmp'")" = 1
    printf 'SNAPSHOT_CLI_AND_FRESH_QUEUE=PASS\n'
fi

expect_sql_failure() {
    local label="$1"
    local statement="$2"
    if runtime_psql "$FULL_DB" -c "$statement" \
        >"$TEMP_DIR/${label}.out" 2>&1; then
        printf 'expected SQL failure succeeded: %s\n' "$label" >&2
        exit 3
    fi
}

snapshot_one="$(runtime_psql "$FULL_DB" -tAc "SELECT min(economic_calendar_snapshot_id) FROM economic_calendar_snapshot WHERE snapshot_state='finalized'")"
snapshot_two="$(runtime_psql "$FULL_DB" -tAc "SELECT max(economic_calendar_snapshot_id) FROM economic_calendar_snapshot WHERE snapshot_state='finalized'")"
hash_one="$(runtime_psql "$FULL_DB" -tAc "SELECT content_hash FROM economic_calendar_snapshot WHERE economic_calendar_snapshot_id=$snapshot_one")"
hash_two="$(runtime_psql "$FULL_DB" -tAc "SELECT content_hash FROM economic_calendar_snapshot WHERE economic_calendar_snapshot_id=$snapshot_two")"

expect_sql_failure incomplete_experiment_binding \
    "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,target_epochs,checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce,donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,model_input_width,model_input_semantic_layout_version,economic_calendar_snapshot_hash) VALUES('phase9incomplete',4,0.0008,1,0,'2024-01-01','2024-02-01','pending','train',0,'enabled','full_history_warmup',20,'',77,6,'$hash_one')"
expect_sql_failure malformed_experiment_binding \
    "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,target_epochs,checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce,donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,model_input_width,model_input_semantic_layout_version,economic_calendar_snapshot_id,economic_calendar_snapshot_hash) VALUES('phase9malformed',4,0.0008,1,0,'2024-01-01','2024-02-01','pending','train',0,'enabled','full_history_warmup',20,'',77,6,$snapshot_one,'fnv1a64:0000000000000000')"
expect_sql_failure active_legacy_binding \
    "UPDATE experiment SET economic_calendar_snapshot_id=$snapshot_one,economic_calendar_snapshot_hash='$hash_one' WHERE symbol='phase9legacy'"

runtime_psql "$FULL_DB" <<SQL
INSERT INTO experiment(
    symbol,prediction_horizon,c_next_threshold,target_epochs,
    checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce,
    donchian20_mode,feature_warmup_scope,donchian_lookback,
    feature_ablation_mask,model_input_width,
    model_input_semantic_layout_version,economic_calendar_snapshot_id,
    economic_calendar_snapshot_hash)
VALUES
('phase9dedup',4,0.0008,1,0,'2024-01-01','2024-02-01','pending','train',
 9009,'enabled','full_history_warmup',20,'',77,6,$snapshot_one,'$hash_one'),
('phase9dedup',4,0.0008,1,0,'2024-01-01','2024-02-01','pending','train',
 9009,'enabled','full_history_warmup',20,'',77,6,$snapshot_two,'$hash_two');
INSERT INTO model(name,experiment_id)
SELECT 'phase9-inherited-model',experiment_id FROM experiment
WHERE symbol='phase9dedup' AND economic_calendar_snapshot_id=$snapshot_one;
SQL

test "$(runtime_psql "$FULL_DB" -tAc "SELECT count(*) FROM experiment WHERE symbol='phase9dedup'")" = 2
test "$(runtime_psql "$FULL_DB" -tAc "SELECT count(*) FROM model m JOIN experiment e USING(experiment_id) WHERE m.name='phase9-inherited-model' AND m.economic_calendar_snapshot_id=e.economic_calendar_snapshot_id AND m.economic_calendar_snapshot_hash=e.economic_calendar_snapshot_hash")" = 1

expect_sql_failure model_experiment_disagreement \
    "INSERT INTO model(name,experiment_id,economic_calendar_snapshot_id,economic_calendar_snapshot_hash) SELECT 'phase9-conflicting-model',experiment_id,$snapshot_two,'$hash_two' FROM experiment WHERE symbol='phase9dedup' AND economic_calendar_snapshot_id=$snapshot_one"

creating_id="$(runtime_psql "$FULL_DB" -tAc "INSERT INTO economic_calendar_snapshot(hash_contract_version,content_hash,created_by) VALUES(1,'fnv1a64:1111111111111111','phase9-test') RETURNING economic_calendar_snapshot_id")"
expect_sql_failure unfinalized_experiment_binding \
    "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,target_epochs,checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce,donchian20_mode,feature_warmup_scope,donchian_lookback,feature_ablation_mask,model_input_width,model_input_semantic_layout_version,economic_calendar_snapshot_id,economic_calendar_snapshot_hash) VALUES('phase9creating',4,0.0008,1,0,'2024-01-01','2024-02-01','pending','train',0,'enabled','full_history_warmup',20,'',77,6,$creating_id,'fnv1a64:1111111111111111')"

# Replay remains non-mutating after finalized materialization and bindings exist.
finalized_before="$(runtime_psql "$FULL_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot WHERE snapshot_state='finalized'")"
admin_psql "$FULL_DB" \
    -f "$ROOT/Database/migrations/092_economic_calendar_snapshot.sql" \
    >"$TEMP_DIR/migration-092-materialized-replay.log"
test "$(runtime_psql "$FULL_DB" -tAc "SELECT count(*) FROM economic_calendar_snapshot WHERE snapshot_state='finalized'")" = "$finalized_before"
test "$(runtime_psql "$FULL_DB" -tAc "SELECT count(*) FROM experiment WHERE symbol='phase9bound' AND economic_calendar_snapshot_id=$snapshot_one AND economic_calendar_snapshot_hash='$hash_one'")" = 1

printf 'MIGRATION_092_COMPATIBILITY=PASS\n'
printf 'SNAPSHOT_HASH_ORDER_INDEPENDENCE=PASS,hash=%s\n' "$forward_hash"
printf 'SNAPSHOT_BINDING_AND_IMMUTABILITY=PASS\n'
printf 'ECONOMIC_CALENDAR_SNAPSHOT_TESTS=PASS\n'
