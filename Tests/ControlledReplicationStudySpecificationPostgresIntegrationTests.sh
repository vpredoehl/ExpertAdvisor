#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
postgres_bin="/opt/homebrew/opt/postgresql@17/bin"
runtime_dir="$(mktemp -d "/tmp/ea_controlled_study_pg.XXXXXX")"
cluster_dir="${runtime_dir}/pgdata"
socket_dir="${runtime_dir}/socket"
build_dir="${repo_root}/DerivedData/Development/ControlledReplicationStudy/PostgresTests"
database_name="ea_controlled_study_${$}_${RANDOM}"
port="$((59000 + (RANDOM % 700)))"
server_started=false
database_created=false

cleanup() {
    local status=$?
    if [[ "${database_created}" == true ]]; then
        "${postgres_bin}/dropdb" -h "${socket_dir}" -p "${port}" -U pqxx \
            "${database_name}" >/dev/null 2>&1 || true
    fi
    if [[ "${server_started}" == true ]]; then
        "${postgres_bin}/pg_ctl" -D "${cluster_dir}" -m immediate -w stop \
            >/dev/null 2>&1 || true
    fi
    rm -rf "${runtime_dir}"
    exit "${status}"
}
trap cleanup EXIT

for command_name in initdb pg_ctl createdb dropdb psql; do
    test -x "${postgres_bin}/${command_name}"
done

mkdir -p "${socket_dir}" "${build_dir}"
"${postgres_bin}/initdb" -D "${cluster_dir}" --username=pqxx \
    --auth=trust --no-locale >/dev/null
"${postgres_bin}/pg_ctl" -D "${cluster_dir}" \
    -l "${runtime_dir}/postgres.log" \
    -o "-F -k ${socket_dir} -p ${port} -h ''" -w start >/dev/null
server_started=true
"${postgres_bin}/createdb" -h "${socket_dir}" -p "${port}" -U pqxx \
    "${database_name}"
database_created=true

"${postgres_bin}/psql" -X -v ON_ERROR_STOP=1 -q \
    -h "${socket_dir}" -p "${port}" -U pqxx -d "${database_name}" \
    -v repo_root="${repo_root}" <<'SQL'
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY, symbol text NOT NULL,
    prediction_horizon integer NOT NULL, c_next_threshold double precision NOT NULL,
    core_lr_mult double precision, head_lr_mult double precision,
    target_epochs integer NOT NULL, checkpoint_interval integer NOT NULL,
    train_start timestamptz NOT NULL, train_end timestamptz NOT NULL,
    infer_start timestamptz, infer_end timestamptz, resume_model_id bigint,
    duplicate_nonce bigint NOT NULL DEFAULT 0, status text NOT NULL,
    phase text NOT NULL, donchian20_mode text NOT NULL,
    donchian_lookback integer NOT NULL, feature_warmup_scope text NOT NULL,
    feature_ablation_mask text NOT NULL, fresh_initialization_seed bigint NOT NULL DEFAULT 42,
    resume_expand_input_width boolean NOT NULL DEFAULT false,
    git_commit text, git_branch text, git_dirty boolean, build_config text,
    compiler_version text, schema_version text, scheduler_version text,
    binary_name text, last_model_id bigint,
    training_objective_id text NOT NULL DEFAULT 'legacy_first_hit_weighted_ce_v1',
    training_objective_version integer NOT NULL DEFAULT 1,
    loss_definition_version integer NOT NULL DEFAULT 1,
    training_objective_canonical text NOT NULL DEFAULT 'training_objective_configuration_v1;',
    training_objective_hash text NOT NULL DEFAULT 'fnv1a64:0000000000000000',
    auxiliary_loss_mode text NOT NULL DEFAULT 'disabled',
    auxiliary_loss_coefficient double precision NOT NULL DEFAULT 0,
    regression_target_definition text, regression_normalization_identity text,
    robust_loss_definition text, robust_loss_delta double precision,
    target_clipping_definition text NOT NULL DEFAULT 'none',
    objective_normalization_identity text NOT NULL DEFAULT 'fixture_objective',
    model_input_width integer, model_input_semantic_layout_version integer,
    economic_calendar_snapshot_id bigint, economic_calendar_snapshot_hash text,
    checkpoint_infer_enabled boolean NOT NULL DEFAULT false,
    checkpoint_infer_min_epoch integer, checkpoint_infer_interval integer,
    checkpoint_policy_enabled boolean NOT NULL DEFAULT false,
    checkpoint_policy_min_leader_score double precision,
    checkpoint_policy_min_infer_accuracy double precision,
    checkpoint_policy_top_n integer, checkpoint_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
    checkpoint_policy_stop_mode text NOT NULL DEFAULT 'next_checkpoint',
    checkpoint_policy_grace_evals integer NOT NULL DEFAULT 1,
    checkpoint_policy_revision bigint NOT NULL DEFAULT 1, checkpoint_policy_hash text,
    continuation_policy_enabled boolean NOT NULL DEFAULT false,
    continuation_policy_target_epochs integer, continuation_policy_min_evals integer NOT NULL DEFAULT 2,
    continuation_policy_patience integer NOT NULL DEFAULT 2,
    continuation_policy_min_leader_score double precision,
    continuation_policy_min_infer_accuracy double precision,
    continuation_policy_min_profit_actionable_count bigint,
    continuation_policy_min_profit_aggregate_log_return_sum double precision,
    continuation_policy_min_profit_average_log_return double precision,
    continuation_policy_min_improvement double precision,
    continuation_policy_max_degradation double precision, continuation_policy_top_n integer,
    continuation_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
    continuation_policy_trend_mode text NOT NULL DEFAULT 'none',
    continuation_policy_source_mode text NOT NULL DEFAULT 'best_checkpoint',
    continuation_policy_include_excluded boolean NOT NULL DEFAULT false,
    continuation_candidate_excluded boolean NOT NULL DEFAULT false,
    continuation_policy_inherit_to_child boolean NOT NULL DEFAULT false,
    continuation_policy_progression_mode text, continuation_policy_target_sequence integer[],
    continuation_policy_target_increment integer, continuation_policy_max_target_epochs integer,
    continuation_policy_inherited boolean NOT NULL DEFAULT false,
    continuation_policy_inherited_from_experiment_id bigint,
    continuation_policy_inherited_from_revision bigint,
    continuation_policy_inherited_from_hash text,
    continuation_policy_inheritance_status text NOT NULL DEFAULT 'not_requested',
    continuation_policy_revision bigint NOT NULL DEFAULT 1,
    scheduler_priority text NOT NULL DEFAULT 'normal', worker_pid integer,
    stopped_at_checkpoint_epoch integer, stopped_at_checkpoint_model_id bigint,
    stop_after_checkpoint_epoch integer, current_epoch integer,
    started_at timestamptz, completed_at timestamptz,
    scheduler_version_unused text
);
CREATE TABLE model (
    model_id bigint PRIMARY KEY, experiment_id bigint REFERENCES experiment(experiment_id),
    parent_model_id bigint REFERENCES model(model_id), comment text,
    producer_worker_attempt_id bigint
);
CREATE TABLE experiment_scheduler_worker_attempt (
    worker_attempt_id bigint PRIMARY KEY, experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    worker_kind text NOT NULL, lifecycle_phase text NOT NULL, capacity_class text NOT NULL,
    lifecycle_state text NOT NULL, reserved_at timestamptz NOT NULL, completed_at timestamptz,
    exit_code integer, reconciliation_result text, checkpoint_eval_id bigint,
    semantic_layout_version integer, model_input_width integer, semantic_worker_role text,
    source_commit text, executable_sha256 text, runtime_identity text,
    canonical_manifest_path text, canonical_executable_path text
);
CREATE TABLE matrix (
    model_id bigint NOT NULL REFERENCES model(model_id), param_name text NOT NULL,
    row_idx integer NOT NULL, col_idx integer NOT NULL, n_rows integer NOT NULL,
    n_cols integer NOT NULL, value double precision NOT NULL,
    PRIMARY KEY(model_id,param_name,row_idx,col_idx)
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigint PRIMARY KEY, parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id), checkpoint_epoch integer
);
CREATE TABLE inference_eval_result (
    id bigint PRIMARY KEY, model_id bigint NOT NULL REFERENCES model(model_id),
    status text NOT NULL, inference_scope text NOT NULL, checkpoint_eval_id bigint,
    parent_experiment_id bigint, symbol text NOT NULL, prediction_horizon bigint NOT NULL,
    threshold_logret double precision NOT NULL, window_size bigint NOT NULL,
    label_rule_id integer NOT NULL, target_type integer NOT NULL, from_date text NOT NULL,
    to_date text NOT NULL, completed_epochs bigint, accuracy double precision,
    accept_model boolean, pred_down double precision, pred_neutral double precision,
    pred_up double precision, producer_worker_attempt_id bigint
);
CREATE TABLE experiment_analysis_result (
    analysis_id bigint PRIMARY KEY, experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    model_id bigint, analysis_scope text NOT NULL, checkpoint_eval_id bigint,
    parent_experiment_id bigint, analysis_status text NOT NULL,
    infer_accuracy double precision, accept_accuracy double precision,
    accept_rate double precision, leader_score double precision,
    pred_down_count bigint, pred_neutral_count bigint, pred_up_count bigint,
    accept_count bigint
);

SQL

for migration in \
    073_inference_profitability_observation.sql \
    079_training_objective_provenance.sql \
    096_scientific_execution_provenance.sql; do
    "${postgres_bin}/psql" -X -v ON_ERROR_STOP=1 -q \
        -h "${socket_dir}" -p "${port}" -U pqxx -d "${database_name}" \
        -f "${repo_root}/Database/migrations/${migration}"
done

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    -I/opt/homebrew/opt/libpqxx@7.10.1/include \
    -I/opt/homebrew/opt/libpq/include \
    "${repo_root}/Tests/ControlledReplicationStudySpecificationPostgresIntegrationTests.cpp" \
    "${repo_root}/Sources/ControlledReplicationStudySpecification.cpp" \
    "${repo_root}/Sources/ControlledReplicationStudySpecificationService.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparison.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparisonService.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluation.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluationRepository.cpp" \
    "${repo_root}/Sources/PairedTrainingObjectiveEvaluation.cpp" \
    "${repo_root}/Sources/PairedTrainingObjectiveEvaluationRepository.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    -L/opt/homebrew/opt/libpqxx@7.10.1/lib -L/opt/homebrew/opt/libpq/lib \
    -lpqxx -lpq -pthread -o "${build_dir}/ControlledReplicationStudySpecificationPostgresIntegrationTests"

EA_STUDY_TEST_DB_HOST="${socket_dir}" \
EA_STUDY_TEST_DB_PORT="${port}" \
EA_STUDY_TEST_DB_NAME="${database_name}" \
    "${build_dir}/ControlledReplicationStudySpecificationPostgresIntegrationTests"

printf '%s\n' 'Controlled replication study PostgreSQL integration tests passed'
