#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/Build/inference_profitability_repository_tests"
binary="${build_dir}/InferenceProfitabilityRepositoryTests"
schema="inference_profitability_test_${$}"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
db_port="${LSTM_DB_PORT:-5432}"
db_name="${LSTM_DB_NAME:-LSTM}"
db_admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
psql_admin=(psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -p "${db_port}" -U "${db_admin_user}" -d "${db_name}")

cleanup() {
    "${psql_admin[@]}" -c "DROP SCHEMA IF EXISTS \"${schema}\" CASCADE;" >/dev/null
}
trap cleanup EXIT

"${psql_admin[@]}" <<SQL
CREATE SCHEMA "${schema}";
GRANT USAGE ON SCHEMA "${schema}" TO pqxx;
SET search_path TO "${schema}", public;
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    last_model_id bigint,
    recommendation_score_guard double precision NOT NULL,
    campaign_guard text NOT NULL,
    continuation_policy_guard text NOT NULL,
    checkpoint_policy_guard text NOT NULL
);
CREATE TABLE model (
    model_id bigint PRIMARY KEY,
    experiment_id bigint REFERENCES experiment(experiment_id)
);
CREATE TABLE matrix (
    model_id bigint NOT NULL REFERENCES model(model_id),
    param_name text NOT NULL,
    row_idx integer NOT NULL,
    col_idx integer NOT NULL,
    value double precision NOT NULL
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigint PRIMARY KEY,
    parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id)
);
CREATE TABLE inference_eval_result (
    id bigint PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES model(model_id),
    status text NOT NULL,
    inference_scope text NOT NULL,
    checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    parent_experiment_id bigint REFERENCES experiment(experiment_id),
    symbol text NOT NULL,
    prediction_horizon bigint NOT NULL,
    threshold_logret double precision NOT NULL,
    window_size bigint NOT NULL,
    label_rule_id integer NOT NULL,
    target_type integer NOT NULL,
    from_date text NOT NULL,
    to_date text NOT NULL,
    completed_epochs bigint,
    accept_model boolean,
    completed_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT ON experiment, model, experiment_checkpoint_eval,
    inference_eval_result, matrix TO pqxx;
\i '${repo_root}/Database/migrations/073_inference_profitability_observation.sql'
SQL

mkdir -p "${build_dir}"
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx)"
    libpq_prefix="$(brew --prefix libpq)"
    pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib" -lpqxx -lpq)
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${repo_root}/Tests/InferenceProfitabilityRepositoryTests.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

LSTM_DB_HOST="${db_host}" \
LSTM_DB_PORT="${db_port}" \
LSTM_DB_USER="${LSTM_DB_USER:-pqxx}" \
LSTM_DB_NAME="${db_name}" \
LSTM_PROFITABILITY_TEST_SCHEMA="${schema}" \
"${binary}"
