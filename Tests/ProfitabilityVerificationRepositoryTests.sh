#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
test_user="${LSTM_DB_USER:-pqxx}"
maintenance_db="${LSTM_DB_NAME:-LSTM}"
db_name="lstm_profitability_phase8_${$}_$(date +%s)"
build_dir="${repo_root}/DerivedData/Development/ProfitabilityPhase8/RepositoryTests"
binary="${build_dir}/ProfitabilityVerificationRepositoryTests"

echo "PROFITABILITY_PHASE8_DISPOSABLE_DB,name=${db_name}"
cleanup() {
    dropdb -h "${db_host}" -U "${admin_user}" --if-exists "${db_name}"
    echo "PROFITABILITY_PHASE8_DISPOSABLE_DB_DROPPED,name=${db_name},confirmed=true"
}
trap cleanup EXIT

createdb -h "${db_host}" -U "${admin_user}" \
    --maintenance-db="${maintenance_db}" "${db_name}"

psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" <<SQL
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    last_model_id bigint
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
    accept_model boolean
);
CREATE TABLE experiment_analysis_result (
    experiment_analysis_result_id bigint PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    model_id bigint NOT NULL REFERENCES model(model_id)
);
GRANT SELECT, INSERT ON experiment, model, matrix,
    experiment_checkpoint_eval, inference_eval_result,
    experiment_analysis_result TO "${test_user}";
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

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${repo_root}/Sources/ProfitabilityVerification.cpp" \
    "${repo_root}/Sources/ProfitabilityVerificationRepository.cpp" \
    "${repo_root}/Tests/ProfitabilityVerificationRepositoryTests.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

LSTM_DB_HOST="${db_host}" LSTM_DB_ADMIN_USER="${admin_user}" \
LSTM_DB_NAME="${db_name}" "${binary}" setup

PGOPTIONS='-c default_transaction_read_only=on' \
LSTM_DB_HOST="${db_host}" LSTM_DB_USER="${test_user}" \
LSTM_DB_NAME="${db_name}" "${binary}" verify
