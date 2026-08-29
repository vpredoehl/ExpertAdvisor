#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
maintenance_db="${LSTM_DB_NAME:-LSTM}"
db_name="lstm_campaign_profitability_phase12_${$}_$(date +%s)"

cleanup() {
    dropdb -h "${db_host}" -U "${admin_user}" --if-exists "${db_name}" >/dev/null
}
trap cleanup EXIT

createdb -h "${db_host}" -U "${admin_user}" \
    --maintenance-db="${maintenance_db}" "${db_name}"

psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" <<'SQL'
CREATE TABLE experiment(
 experiment_id bigint PRIMARY KEY,symbol text NOT NULL,
 prediction_horizon integer NOT NULL,c_next_threshold double precision NOT NULL,
 infer_end timestamptz NOT NULL,last_model_id bigint);
CREATE TABLE model(
 model_id bigint PRIMARY KEY,experiment_id bigint REFERENCES experiment,
 parent_model_id bigint);
ALTER TABLE experiment ADD CONSTRAINT experiment_last_model_fk
 FOREIGN KEY(last_model_id) REFERENCES model(model_id) DEFERRABLE INITIALLY DEFERRED;
CREATE TABLE experiment_recommendation_evaluation_run(
 recommendation_evaluation_run_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_ranking_snapshot(
 recommendation_ranking_snapshot_id bigint PRIMARY KEY,status text NOT NULL,
 evaluation_run_filter bigint REFERENCES experiment_recommendation_evaluation_run);
CREATE TABLE experiment_recommendation_ranking_member(
 recommendation_ranking_member_id bigint PRIMARY KEY,
 recommendation_ranking_snapshot_id bigint REFERENCES
  experiment_recommendation_ranking_snapshot,
 source_experiment_id bigint REFERENCES experiment,
 source_model_id bigint REFERENCES model);
BEGIN;
INSERT INTO experiment VALUES(1,'audcadrmp',4,0.0008,'2026-01-01',NULL);
INSERT INTO model VALUES(10,1,NULL),(11,1,NULL);
UPDATE experiment SET last_model_id=10 WHERE experiment_id=1;
INSERT INTO experiment_recommendation_evaluation_run VALUES(6);
INSERT INTO experiment_recommendation_ranking_snapshot VALUES(5,'completed',6);
INSERT INTO experiment_recommendation_ranking_member VALUES(1,5,1,10);
COMMIT;
SQL

psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -f \
    "${repo_root}/Database/migrations/085_campaign_profitability_prospective_outcome.sql"

insert_sql="INSERT INTO campaign_profitability_prospective_outcome_result(
 validation_cohort_identity_hash,ranking_snapshot_id,source_evaluation_run_id,
 source_experiment_id,source_model_id,symbol,prediction_horizon,threshold_logret,
 window_size,label_rule_id,target_type,input_width,outcome_start,outcome_end,
 job_identity_hash,feature_semantic_hash,model_lineage_hash,
 model_artifact_content_hash,metric_definition_canonical,
 metric_definition_hash,source_content_hash,prediction_count,actionable_count,
 winning_actionable_count,losing_actionable_count,
 gross_positive_terminal_horizon_log_return_sum,
 gross_negative_terminal_horizon_log_return_sum,
 aggregate_terminal_horizon_log_return_sum,
 average_terminal_horizon_log_return_per_actionable_prediction,
 inference_accuracy,outcome_identity_canonical,outcome_identity_hash)
 VALUES('fnv1a64:fe7aee4a1aed8a5e',5,6,1,10,'audcadrmp',4,0.0008,
 64,1,2,53,'2026-08-31','2026-09-30','fnv1a64:1111111111111111',
 'fnv1a64:2222222222222222','fnv1a64:3333333333333333',
 'fnv1a64:4444444444444444','metric','fnv1a64:5555555555555555',
 'fnv1a64:6666666666666666',10,4,2,2,0.02,-0.01,0.01,0.0025,0.75,
 'outcome-one','fnv1a64:7777777777777777')"

psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -c "${insert_sql};"
psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -c "${insert_sql} ON CONFLICT (outcome_identity_canonical) DO NOTHING;"

test "$(psql -X -Atq -h "${db_host}" -U "${admin_user}" -d "${db_name}" \
    -c 'SELECT count(*) FROM campaign_profitability_prospective_outcome_result;')" = "1"

if psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -c \
    "UPDATE campaign_profitability_prospective_outcome_result SET inference_accuracy=0.8;" \
    >/dev/null 2>&1; then
    echo "expected immutable update rejection" >&2
    exit 1
fi
if psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -c \
    "DELETE FROM campaign_profitability_prospective_outcome_result;" \
    >/dev/null 2>&1; then
    echo "expected immutable delete rejection" >&2
    exit 1
fi

collision_sql="${insert_sql/outcome-one/outcome-two}"
collision_sql="${collision_sql/fnv1a64:7777777777777777/fnv1a64:8888888888888888}"
if psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -c "${collision_sql};" >/dev/null 2>&1; then
    echo "expected deterministic job collision rejection" >&2
    exit 1
fi

nonfinal_sql="${insert_sql/,5,6,1,10,/,5,6,1,11,}"
nonfinal_sql="${nonfinal_sql/fnv1a64:1111111111111111/fnv1a64:9999999999999999}"
nonfinal_sql="${nonfinal_sql/outcome-one/outcome-nonfinal}"
nonfinal_sql="${nonfinal_sql/fnv1a64:7777777777777777/fnv1a64:aaaaaaaaaaaaaaaa}"
if psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" -c "${nonfinal_sql};" >/dev/null 2>&1; then
    echo "expected non-final source-model rejection" >&2
    exit 1
fi

test "$(psql -X -Atq -h "${db_host}" -U "${admin_user}" -d "${db_name}" \
    -c 'SELECT last_model_id FROM experiment WHERE experiment_id=1;')" = "10"
test "$(psql -X -Atq -h "${db_host}" -U "${admin_user}" -d "${db_name}" \
    -c 'SELECT count(*) FROM model;')" = "2"

echo "CAMPAIGN_PROFITABILITY_PHASE12_MIGRATION_TESTS_PASS"
