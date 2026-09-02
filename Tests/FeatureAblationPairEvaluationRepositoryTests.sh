#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="${EA_CONSENSUS_EFFICACY_TEST_ROOT:-${repo_root}/DerivedData/Development/ConsensusEfficacyAnalysis/Tests}"
build_dir="$(mktemp -d "${test_root}/RepositoryBuild.XXXXXX")"
# PostgreSQL passes its -k value through a server option string; using a short
# no-space runtime path avoids ambiguity from the repository volume name.
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_consensus_efficacy.XXXXXX")"
cluster_dir="${runtime_dir}/pgdata"
socket_dir="${runtime_dir}/socket"
binary="${build_dir}/FeatureAblationPairEvaluationRepositoryTests"
database_name="ea_consensus_efficacy_${$}_${RANDOM}"
database_created=false
database_dropped=false
server_started=false
postgres_bin="/opt/homebrew/opt/postgresql@17/bin"
port="$((55000 + ($$ % 1000)))"

verify_database_name() {
    printf 'DISPOSABLE_DATABASE_NAME=%s\n' "${database_name}"
    if [[ ! "${database_name}" =~ ^ea_consensus_efficacy_[A-Za-z0-9_]+$ ]]; then
        printf 'Rejected unsafe disposable database name: %s\n' \
            "${database_name}" >&2
        exit 1
    fi
}

cleanup() {
    local status=$?
    if [[ "${database_created}" == true ]]; then
        verify_database_name
        if "${postgres_bin}/dropdb" -h "${socket_dir}" -p "${port}" \
            -U pqxx "${database_name}"; then
            database_dropped=true
            database_created=false
        fi
    fi
    if [[ "${server_started}" == true ]]; then
        "${postgres_bin}/pg_ctl" -D "${cluster_dir}" -m fast -w stop \
            >/dev/null || true
        server_started=false
    fi
    printf 'DISPOSABLE_DATABASE_DROPPED=%s\n' "${database_dropped}"
    rm -rf "${runtime_dir}" "${build_dir}"
    exit "${status}"
}
trap cleanup EXIT

mkdir -p "${socket_dir}"
verify_database_name
"${postgres_bin}/initdb" -D "${cluster_dir}" --username=pqxx \
    --auth=trust --no-locale >/dev/null
"${postgres_bin}/pg_ctl" -D "${cluster_dir}" \
    -o "-F -k ${socket_dir} -p ${port} -h ''" -w start >/dev/null
server_started=true
"${postgres_bin}/createdb" -h "${socket_dir}" -p "${port}" -U pqxx \
    "${database_name}"
database_created=true

"${postgres_bin}/psql" -X -v ON_ERROR_STOP=1 -q \
    -h "${socket_dir}" -p "${port}" -U pqxx -d "${database_name}" <<SQL
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    core_lr_mult double precision,
    head_lr_mult double precision,
    target_epochs integer NOT NULL,
    checkpoint_interval integer NOT NULL,
    train_start timestamptz NOT NULL,
    train_end timestamptz NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    status text NOT NULL,
    phase text NOT NULL,
    last_model_id bigint,
    resume_model_id bigint,
    stopped_at_checkpoint_epoch integer,
    stopped_at_checkpoint_model_id bigint,
    duplicate_nonce bigint NOT NULL,
    donchian20_mode text NOT NULL,
    donchian_lookback integer NOT NULL,
    feature_warmup_scope text NOT NULL,
    feature_ablation_mask text NOT NULL,
    resume_expand_input_width boolean NOT NULL DEFAULT false,
    checkpoint_infer_enabled boolean NOT NULL DEFAULT false,
    checkpoint_infer_min_epoch integer,
    checkpoint_infer_interval integer,
    checkpoint_policy_enabled boolean NOT NULL DEFAULT false,
    checkpoint_policy_min_leader_score double precision,
    checkpoint_policy_min_infer_accuracy double precision,
    checkpoint_policy_top_n integer,
    checkpoint_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
    checkpoint_policy_stop_mode text NOT NULL DEFAULT 'next_checkpoint',
    checkpoint_policy_grace_evals integer NOT NULL DEFAULT 1,
    checkpoint_policy_revision bigint NOT NULL DEFAULT 1,
    checkpoint_policy_hash text,
    git_commit text,
    git_branch text,
    git_dirty boolean,
    build_config text,
    compiler_version text,
    schema_version text,
    scheduler_version text,
    binary_name text
);

CREATE TABLE model (
    model_id bigint PRIMARY KEY,
    experiment_id bigint REFERENCES experiment(experiment_id),
    parent_model_id bigint REFERENCES model(model_id)
);

CREATE TABLE matrix (
    model_id bigint NOT NULL REFERENCES model(model_id),
    param_name text NOT NULL,
    row_idx integer NOT NULL,
    col_idx integer NOT NULL,
    n_rows integer NOT NULL,
    n_cols integer NOT NULL,
    value double precision NOT NULL,
    PRIMARY KEY (model_id,param_name,row_idx,col_idx)
);

CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigint PRIMARY KEY,
    parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id),
    checkpoint_epoch integer
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
    accuracy double precision,
    accept_model boolean,
    pred_down double precision,
    pred_neutral double precision,
    pred_up double precision,
    completed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE experiment_analysis_result (
    analysis_id bigint PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    model_id bigint REFERENCES model(model_id),
    analysis_scope text NOT NULL,
    checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    parent_experiment_id bigint REFERENCES experiment(experiment_id),
    analysis_status text NOT NULL,
    infer_accuracy double precision,
    accept_accuracy double precision,
    accept_rate double precision,
    leader_score double precision
);

\i '${repo_root}/Database/migrations/073_inference_profitability_observation.sql'
\i '${repo_root}/Database/migrations/079_training_objective_provenance.sql'
SQL

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
    "${pqxx_cflags[@]}" -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/FeatureAblationPairEvaluationRepositoryTests.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluation.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluationRepository.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluationService.cpp" \
    "${repo_root}/Sources/FeatureAblationReplicationEvaluation.cpp" \
    "${repo_root}/Sources/FeatureAblationReplicationEvaluationService.cpp" \
    "${repo_root}/Sources/PairedTrainingObjectiveEvaluation.cpp" \
    "${repo_root}/Sources/PairedTrainingObjectiveEvaluationRepository.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

EA_CONSENSUS_EFFICACY_DB_HOST="${socket_dir}" \
EA_CONSENSUS_EFFICACY_DB_PORT="${port}" \
EA_CONSENSUS_EFFICACY_DB_USER=pqxx \
EA_CONSENSUS_EFFICACY_DB_NAME="${database_name}" \
    "${binary}"

verify_database_name
"${postgres_bin}/dropdb" -h "${socket_dir}" -p "${port}" -U pqxx \
    "${database_name}"
database_created=false
database_dropped=true
printf 'DISPOSABLE_DATABASE_USED=%s\n' "${database_name}"
printf 'DISPOSABLE_DATABASE_DROPPED=true\n'
