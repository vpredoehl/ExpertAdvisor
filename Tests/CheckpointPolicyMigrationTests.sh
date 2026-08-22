#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
schema="checkpoint_policy_identity_test_${$}"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
db_name="${LSTM_DB_NAME:-LSTM}"
db_admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
psql_admin=(psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${db_admin_user}" -d "${db_name}")

cleanup() {
    "${psql_admin[@]}" -c "DROP SCHEMA IF EXISTS \"${schema}\" CASCADE;" >/dev/null
}
trap cleanup EXIT

"${psql_admin[@]}" <<SQL
CREATE SCHEMA "${schema}";
SET search_path TO "${schema}", public;

CREATE TABLE model (model_id bigint PRIMARY KEY);
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    checkpoint_policy_enabled boolean NOT NULL DEFAULT false,
    checkpoint_policy_last_decision text,
    checkpoint_policy_last_decision_at timestamptz,
    checkpoint_policy_last_checkpoint_eval_id bigint,
    checkpoint_policy_last_reason text,
    stop_after_checkpoint_epoch integer,
    active_scheduler_worker_attempt_id bigint
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigint PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_epoch integer NOT NULL,
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id)
);
CREATE TABLE experiment_analysis_result (
    analysis_id bigint PRIMARY KEY
);
CREATE TABLE inference_eval_result (
    id bigint PRIMARY KEY
);
CREATE TABLE experiment_scheduler_worker_attempt (
    worker_attempt_id bigint PRIMARY KEY
);
CREATE TABLE experiment_checkpoint_decision (
    checkpoint_decision_id bigserial PRIMARY KEY,
    checkpoint_eval_id bigint NOT NULL REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_epoch integer NOT NULL,
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id),
    decision text NOT NULL,
    reason text NOT NULL,
    leader_score double precision,
    infer_accuracy double precision,
    rank_value integer,
    rank_scope text,
    requested_stop_epoch integer,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX experiment_checkpoint_decision_eval_uidx
    ON experiment_checkpoint_decision(checkpoint_eval_id);

INSERT INTO model VALUES (10), (11);
INSERT INTO experiment(experiment_id) VALUES (1);
INSERT INTO experiment_checkpoint_eval VALUES (100, 1, 20, 10);
INSERT INTO experiment_analysis_result VALUES (200), (201);
INSERT INTO inference_eval_result VALUES (300), (301);
INSERT INTO experiment_scheduler_worker_attempt VALUES (400);
INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,decision,reason)
VALUES (100,1,20,10,'continue','legacy preserved');

\i '${repo_root}/Database/migrations/075_checkpoint_policy_decision_identity.sql'
\i '${repo_root}/Database/migrations/075_checkpoint_policy_decision_identity.sql'
\i '${repo_root}/Tests/CheckpointPolicyMigrationTests.sql'
SQL

echo "CheckpointPolicyMigrationTests passed"
