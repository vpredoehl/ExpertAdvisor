#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
test_user="${LSTM_DB_USER:-pqxx}"
maintenance_db="${LSTM_DB_NAME:-LSTM}"
db_name="lstm_campaign_profitability_phase8_${$}_$(date +%s)"
build_dir="${repo_root}/DerivedData/Development/ProfitabilityPhase8/CampaignReadinessTests"
binary="${build_dir}/CampaignProfitabilityReadinessRepositoryTests"

echo "CAMPAIGN_PROFITABILITY_PHASE8_DISPOSABLE_DB,name=${db_name}"
cleanup() {
    dropdb -h "${db_host}" -U "${admin_user}" --if-exists "${db_name}"
    echo "CAMPAIGN_PROFITABILITY_PHASE8_DISPOSABLE_DB_DROPPED,name=${db_name},confirmed=true"
}
trap cleanup EXIT

createdb -h "${db_host}" -U "${admin_user}" \
    --maintenance-db="${maintenance_db}" "${db_name}"

psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${admin_user}" \
    -d "${db_name}" <<SQL
CREATE SEQUENCE campaign_read_sentinel;
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    last_model_id bigint,
    target_epochs integer,
    status text NOT NULL,
    phase text NOT NULL,
    worker_pid integer,
    current_operation text,
    current_epoch integer,
    updated_at timestamptz NOT NULL
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
    completed_at timestamptz NOT NULL DEFAULT '2026-08-01 00:00:00+00'
);
CREATE TABLE experiment_recommendation_evaluation_run(
    recommendation_evaluation_run_id bigint PRIMARY KEY,
    status text NOT NULL,
    evaluation_run_identity_canonical text NOT NULL,
    evaluation_run_identity_hash text NOT NULL,
    recommendations_evaluated integer NOT NULL,
    recommendations_eligible integer NOT NULL,
    recommendations_blocked integer NOT NULL,
    evaluation_errors integer NOT NULL
);
CREATE TABLE experiment_recommendation_ranking_snapshot(
    recommendation_ranking_snapshot_id bigint PRIMARY KEY,
    status text NOT NULL,
    ranking_snapshot_identity_canonical text NOT NULL,
    ranking_snapshot_identity_hash text NOT NULL,
    ranking_policy_canonical text NOT NULL,
    ranking_policy_hash text NOT NULL,
    ranking_version integer NOT NULL,
    scope_type text NOT NULL DEFAULT 'evaluation_run',
    scope_canonical text NOT NULL DEFAULT 'legacy_scope',
    scope_hash text NOT NULL DEFAULT 'legacy_scope_hash',
    evaluation_run_filter bigint,
    requested_limit integer NOT NULL DEFAULT 1000,
    source_membership_canonical text NOT NULL DEFAULT
        'experiment_recommendation_ranking_membership_v1;count=0',
    source_membership_hash text NOT NULL DEFAULT 'legacy_membership_hash',
    ranking_snapshot_identity_version integer NOT NULL DEFAULT 1,
    population_semantic_state text NOT NULL,
    scoring_semantic_canonical text,
    scoring_semantic_hash text,
    scoring_semantic_version integer,
    evaluation_semantic_canonical text,
    evaluation_semantic_hash text,
    evaluation_semantic_version integer,
    distinct_scoring_semantic_count integer NOT NULL,
    distinct_evaluation_semantic_count integer NOT NULL,
    homogeneity_validation_result text NOT NULL,
    member_count integer NOT NULL,
    advisory_ready_count integer NOT NULL DEFAULT 0,
    blocked_count integer NOT NULL DEFAULT 0,
    non_actionable_count integer NOT NULL DEFAULT 0,
    completed_at timestamptz NOT NULL DEFAULT '2026-09-01 00:00:00+00'
);
CREATE TABLE experiment_recommendation_ranking_member(
    recommendation_ranking_member_id bigint PRIMARY KEY,
    recommendation_ranking_snapshot_id bigint NOT NULL,
    recommendation_evaluation_result_id bigint,
    recommendation_id bigint NOT NULL,
    source_experiment_id bigint NOT NULL,
    source_model_id bigint,
    global_ordinal integer NOT NULL,
    bucket text NOT NULL,
    final_score double precision,
    symbol text NOT NULL,
    horizon integer NOT NULL,
    family text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT '2026-08-01 00:00:00+00'
);
CREATE TABLE experiment_recommendation(
    recommendation_id bigint PRIMARY KEY,
    source_experiment_id bigint NOT NULL,
    source_model_id bigint,
    source_symbol text NOT NULL,
    source_prediction_horizon integer NOT NULL,
    changed_parameter text NOT NULL,
    source_leader_score double precision NOT NULL,
    source_infer_accuracy double precision NOT NULL,
    source_predicted_neutral_proportion double precision,
    semantic_configuration_canonical text NOT NULL,
    semantic_hash text NOT NULL,
    invocation_configuration_canonical text NOT NULL,
    invocation_hash text NOT NULL,
    final_profitability_provenance_version integer,
    source_final_inference_eval_result_id bigint,
    source_final_profitability_observation_id bigint,
    source_final_profitability_unavailable_reason text,
    source_final_profitability_inference_scope text,
    source_final_profitability_inference_start text,
    source_final_profitability_inference_end text,
    source_final_profitability_actionable_count bigint,
    source_final_profitability_aggregate_return double precision,
    source_final_profitability_average_return double precision,
    source_final_profitability_metric_definition_hash text,
    source_final_profitability_source_content_hash text,
    source_final_profitability_observation_identity_hash text,
    created_at timestamptz NOT NULL DEFAULT '2026-08-01 00:00:00+00'
);
CREATE TABLE experiment_recommendation_evaluation_result(
    recommendation_evaluation_result_id bigint PRIMARY KEY,
    recommendation_evaluation_run_id bigint NOT NULL,
    recommendation_id bigint NOT NULL,
    evaluation_identity_canonical text NOT NULL,
    evaluation_identity_hash text NOT NULL,
    source_experiment_id bigint NOT NULL,
    source_model_id bigint,
    final_score double precision,
    eligibility text NOT NULL,
    disposition text NOT NULL,
    result_status text NOT NULL,
    final_profitability_provenance_version integer,
    source_final_inference_eval_result_id bigint,
    source_final_profitability_observation_id bigint,
    source_final_profitability_unavailable_reason text,
    source_final_profitability_inference_scope text,
    source_final_profitability_inference_start text,
    source_final_profitability_inference_end text,
    source_final_profitability_actionable_count bigint,
    source_final_profitability_aggregate_return double precision,
    source_final_profitability_average_return double precision,
    source_final_profitability_metric_definition_hash text,
    source_final_profitability_source_content_hash text,
    source_final_profitability_observation_identity_hash text,
    created_at timestamptz NOT NULL DEFAULT '2026-08-01 00:00:00+00'
);
CREATE TABLE experiment_recommendation_conversion_proposal(
    recommendation_conversion_proposal_id bigint PRIMARY KEY,
    recommendation_id bigint NOT NULL,
    source_experiment_id bigint NOT NULL,
    conversion_contract_version integer NOT NULL,
    conversion_identity_canonical text NOT NULL,
    conversion_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL
);
CREATE TABLE experiment_recommendation_conversion_review_decision(
    recommendation_conversion_review_decision_id bigint PRIMARY KEY,
    recommendation_conversion_proposal_id bigint NOT NULL,
    decision text NOT NULL,
    decided_at timestamptz NOT NULL
);
CREATE TABLE experiment_recommendation_conversion_execution(
    recommendation_conversion_execution_id bigint PRIMARY KEY,
    recommendation_conversion_proposal_id bigint NOT NULL,
    recommendation_conversion_review_decision_id bigint NOT NULL,
    experiment_id bigint NOT NULL,
    execution_contract_version integer NOT NULL,
    authorization_decision text NOT NULL,
    execution_identity_canonical text NOT NULL,
    execution_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL
);
CREATE TABLE experiment_recommendation_conversion_activation(
    recommendation_conversion_activation_id bigint PRIMARY KEY,
    recommendation_conversion_execution_id bigint NOT NULL,
    recommendation_conversion_proposal_id bigint NOT NULL,
    recommendation_conversion_review_decision_id bigint NOT NULL,
    experiment_id bigint NOT NULL,
    activation_contract_version integer NOT NULL,
    previous_status text NOT NULL,
    previous_phase text NOT NULL,
    resulting_status text NOT NULL,
    resulting_phase text NOT NULL,
    activation_identity_canonical text NOT NULL,
    activation_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL
);
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO "${test_user}";
\i '${repo_root}/Database/migrations/073_inference_profitability_observation.sql'
GRANT SELECT ON campaign_read_sentinel TO "${test_user}";
SQL

mkdir -p "${build_dir}"
pqxx_prefix="/opt/homebrew/Cellar/libpqxx@7.10.1/7.10.1"
libpq_prefix="/opt/homebrew/opt/libpq"
pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
pqxx_libs=(
    "-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib"
    "-Wl,-rpath,${pqxx_prefix}/lib" "-Wl,-rpath,${libpq_prefix}/lib"
    -lpqxx -lpq
)

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationScoring.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationEvaluation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationRanking.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationCampaignPlanning.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationCampaignPlanningRepository.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationConversionWorkflow.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${repo_root}/Sources/ProfitabilityDistribution.cpp" \
    "${repo_root}/Sources/ProfitabilityCalibration.cpp" \
    "${repo_root}/Sources/ProfitabilityVerification.cpp" \
    "${repo_root}/Sources/ProfitabilityVerificationRepository.cpp" \
    "${repo_root}/Sources/ProfitabilityVerificationService.cpp" \
    "${repo_root}/Tests/CampaignProfitabilityReadinessRepositoryTests.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

otool -L "${binary}" | grep -E 'pqxx|libpq'

LSTM_DB_HOST="${db_host}" LSTM_DB_ADMIN_USER="${admin_user}" \
LSTM_DB_NAME="${db_name}" "${binary}" setup

PGOPTIONS='-c default_transaction_read_only=on' \
LSTM_DB_HOST="${db_host}" LSTM_DB_USER="${test_user}" \
LSTM_DB_NAME="${db_name}" "${binary}" verify
