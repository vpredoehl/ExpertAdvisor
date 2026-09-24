#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
postgres_bin="/opt/homebrew/opt/postgresql@17/bin"
runtime_dir="$(mktemp -d "/tmp/ea_replication_repo.XXXXXX")"
cluster_dir="${runtime_dir}/pgdata"
socket_dir="${runtime_dir}/socket"
build_dir="${repo_root}/DerivedData/Development/ExperimentReplicationMaterialization/RepositoryTests"
database_name="ea_replication_materializer_repo_${$}_${RANDOM}"
port="$((58000 + (RANDOM % 1000)))"
server_started=false

cleanup() {
    local status=$?
    if [[ "${server_started}" == true ]]; then
        "${postgres_bin}/pg_ctl" -D "${cluster_dir}" -m immediate -w stop \
            >/dev/null || true
    fi
    rm -rf "${runtime_dir}"
    exit "${status}"
}
trap cleanup EXIT

mkdir -p "${socket_dir}" "${build_dir}"
"${postgres_bin}/initdb" -D "${cluster_dir}" --username=pqxx \
    --auth=trust --no-locale >/dev/null
"${postgres_bin}/pg_ctl" -D "${cluster_dir}" \
    -l "${runtime_dir}/postgres.log" \
    -o "-F -k ${socket_dir} -p ${port} -h ''" -w start >/dev/null
server_started=true
"${postgres_bin}/createdb" -h "${socket_dir}" -p "${port}" -U pqxx \
    "${database_name}"

"${postgres_bin}/psql" -X -v ON_ERROR_STOP=1 -q \
    -h "${socket_dir}" -p "${port}" -U pqxx -d "${database_name}" <<'SQL'
CREATE TABLE experiment (
 experiment_id bigserial PRIMARY KEY, symbol text NOT NULL,
 prediction_horizon int NOT NULL, c_next_threshold double precision NOT NULL,
 core_lr_mult double precision, head_lr_mult double precision,
 target_epochs int NOT NULL, checkpoint_interval int NOT NULL,
 train_start timestamptz NOT NULL, train_end timestamptz NOT NULL,
 infer_start timestamptz, infer_end timestamptz, resume_model_id bigint,
 duplicate_nonce bigint NOT NULL DEFAULT 0, status text NOT NULL DEFAULT 'pending',
 phase text NOT NULL DEFAULT 'train', invocation_mode text,
 donchian20_mode text NOT NULL DEFAULT 'enabled',
 feature_warmup_scope text NOT NULL DEFAULT 'full_history_warmup',
 donchian_lookback int NOT NULL DEFAULT 20,
 feature_ablation_mask text NOT NULL DEFAULT '',
 fresh_initialization_seed bigint NOT NULL DEFAULT 42,
 resume_expand_input_width boolean NOT NULL DEFAULT false,
 training_objective_canonical text NOT NULL, training_objective_hash text NOT NULL,
 training_objective_id text NOT NULL, training_objective_version int NOT NULL,
 loss_definition_version int NOT NULL, auxiliary_loss_mode text NOT NULL,
 auxiliary_loss_coefficient double precision NOT NULL,
 regression_target_definition text, regression_normalization_identity text,
 robust_loss_definition text, robust_loss_delta double precision,
 target_clipping_definition text NOT NULL, objective_normalization_identity text NOT NULL,
 model_input_width int, model_input_semantic_layout_version int,
 economic_calendar_snapshot_id bigint, economic_calendar_snapshot_hash text,
 opportunistic_checkpoint_infer boolean NOT NULL DEFAULT false,
 checkpoint_infer_enabled boolean NOT NULL DEFAULT false,
 checkpoint_infer_min_epoch int, checkpoint_infer_interval int,
 checkpoint_policy_enabled boolean NOT NULL DEFAULT false,
 checkpoint_policy_min_leader_score double precision,
 checkpoint_policy_min_infer_accuracy double precision, checkpoint_policy_top_n int,
 checkpoint_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
 checkpoint_policy_stop_mode text NOT NULL DEFAULT 'next_checkpoint',
 checkpoint_policy_grace_evals int NOT NULL DEFAULT 1,
 checkpoint_policy_revision bigint NOT NULL DEFAULT 1, checkpoint_policy_hash text,
 continuation_policy_enabled boolean NOT NULL DEFAULT false,
 continuation_policy_target_epochs int, continuation_policy_min_evals int NOT NULL DEFAULT 2,
 continuation_policy_patience int NOT NULL DEFAULT 2,
 continuation_policy_min_leader_score double precision,
 continuation_policy_min_infer_accuracy double precision,
 continuation_policy_min_profit_actionable_count bigint,
 continuation_policy_min_profit_aggregate_log_return_sum double precision,
 continuation_policy_min_profit_average_log_return double precision,
 continuation_policy_min_improvement double precision,
 continuation_policy_max_degradation double precision, continuation_policy_top_n int,
 continuation_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
 continuation_policy_trend_mode text NOT NULL DEFAULT 'none',
 continuation_policy_source_mode text NOT NULL DEFAULT 'best_checkpoint',
 continuation_policy_include_excluded boolean NOT NULL DEFAULT false,
 continuation_candidate_excluded boolean NOT NULL DEFAULT false,
 continuation_policy_inherit_to_child boolean NOT NULL DEFAULT false,
 continuation_policy_progression_mode text, continuation_policy_target_sequence int[],
 continuation_policy_target_increment int, continuation_policy_max_target_epochs int,
 continuation_policy_inherited boolean NOT NULL DEFAULT false,
 continuation_policy_inherited_from_experiment_id bigint,
 continuation_policy_inherited_from_revision bigint,
 continuation_policy_inherited_from_hash text,
 continuation_policy_inheritance_status text NOT NULL DEFAULT 'not_requested',
 continuation_policy_revision bigint NOT NULL DEFAULT 1,
 scheduler_priority text NOT NULL DEFAULT 'normal', last_model_id bigint,
 stopped_at_checkpoint_epoch integer, stopped_at_checkpoint_model_id bigint,
 stop_after_checkpoint_epoch integer,
 current_epoch int, worker_pid int, started_at timestamptz, completed_at timestamptz,
 parent_experiment_id bigint, continuation_source_experiment_id bigint,
 continuation_source_model_id bigint, continuation_source_epoch integer,
 continuation_decision_id bigint, continuation_generation integer NOT NULL DEFAULT 1,
 train_log_path text, infer_log_path text, analysis_log_path text,
 exit_code integer, error_message text, current_operation text,
 worker_started_at timestamptz, active_scheduler_worker_attempt_id bigint,
 operator_forced_final_inference_rerun_requested boolean NOT NULL DEFAULT false,
 resume_requested boolean NOT NULL DEFAULT false,
 scheduler_resume_origin text NOT NULL DEFAULT 'none',
 git_commit text, git_branch text, git_dirty boolean, build_config text,
 compiler_version text, schema_version text, scheduler_version text,
 binary_name text,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO experiment (
 experiment_id,symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,
 target_epochs,checkpoint_interval,train_start,train_end,infer_start,infer_end,
 status,phase,donchian20_mode,feature_warmup_scope,donchian_lookback,
 feature_ablation_mask,fresh_initialization_seed,training_objective_canonical,
 training_objective_hash,training_objective_id,training_objective_version,
 loss_definition_version,auxiliary_loss_mode,auxiliary_loss_coefficient,
 target_clipping_definition,objective_normalization_identity,model_input_width,
 model_input_semantic_layout_version,checkpoint_policy_revision)
VALUES (10,'fixture',4,0.0008,120,25,80,20,'2010-01-01','2025-01-01',
 '2025-01-01','2026-01-01','completed','done','enabled','full_history_warmup',20,
 'mask_a',43,'objective','fnv1a64:0000000000000001','legacy',1,1,'disabled',0,
 'none','fixture_v1',80,8,7),
(11,'fixture',4,0.0008,120,25,80,20,'2010-01-01','2025-01-01',
 '2025-01-01','2026-01-01','completed','done','enabled','full_history_warmup',20,
 'mask_b',43,'objective','fnv1a64:0000000000000001','legacy',1,1,'disabled',0,
 'none','fixture_v1',80,8,7);
UPDATE experiment SET
 last_model_id=9000+experiment_id,current_epoch=80,worker_pid=4321,
 started_at='2025-01-01',completed_at='2025-02-01',
 parent_experiment_id=7000,continuation_source_experiment_id=7000,
 scheduler_priority='high',train_log_path='/tmp/train',infer_log_path='/tmp/infer',
 analysis_log_path='/tmp/analysis',exit_code=0,error_message='source result',
 current_operation='analyze',worker_started_at='2025-01-01',
 active_scheduler_worker_attempt_id=8000,
 operator_forced_final_inference_rerun_requested=true,resume_requested=true,
 scheduler_resume_origin='preemption',stop_after_checkpoint_epoch=80,
 stopped_at_checkpoint_epoch=80,stopped_at_checkpoint_model_id=9000+experiment_id,
 continuation_source_model_id=9000+experiment_id,continuation_source_epoch=80,
 continuation_decision_id=6000,continuation_generation=3,
 git_commit='source_commit',git_branch='source_branch',git_dirty=true,
 build_config='Release',compiler_version='source_compiler',
 schema_version='source_schema',scheduler_version='source_scheduler',
 binary_name='LSTM_Release';
CREATE UNIQUE INDEX experiment_unique_identity_uidx ON experiment (
 symbol,prediction_horizon,c_next_threshold,
 COALESCE(core_lr_mult,'-infinity'::double precision),
 COALESCE(head_lr_mult,'-infinity'::double precision),target_epochs,
 checkpoint_interval,train_start,train_end,
 COALESCE(infer_start,'-infinity'::timestamptz),
 COALESCE(infer_end,'-infinity'::timestamptz),
 COALESCE(resume_model_id,-1),donchian20_mode,donchian_lookback,
 feature_warmup_scope,feature_ablation_mask,resume_expand_input_width,
 duplicate_nonce) WHERE status <> 'cancelled';
SELECT setval(pg_get_serial_sequence('experiment','experiment_id'),11,true);

CREATE TABLE model (
 model_id bigint PRIMARY KEY, experiment_id bigint REFERENCES experiment(experiment_id),
 parent_model_id bigint REFERENCES model(model_id),
 producer_worker_attempt_id bigint, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE experiment_scheduler_worker_attempt (
 worker_attempt_id bigint PRIMARY KEY,
 experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
 worker_kind text NOT NULL, lifecycle_phase text NOT NULL,
 capacity_class text NOT NULL, lifecycle_state text NOT NULL,
 reserved_at timestamptz NOT NULL, completed_at timestamptz,
 exit_code integer, reconciliation_result text,
 semantic_layout_version integer, model_input_width integer,
 semantic_worker_role text, source_commit text, executable_sha256 text,
 runtime_identity text, canonical_manifest_path text,
 canonical_executable_path text
);
CREATE TABLE matrix (
 model_id bigint NOT NULL REFERENCES model(model_id), param_name text NOT NULL,
 row_idx integer NOT NULL, col_idx integer NOT NULL,
 n_rows integer NOT NULL, n_cols integer NOT NULL, value double precision NOT NULL,
 PRIMARY KEY(model_id,param_name,row_idx,col_idx)
);
CREATE TABLE experiment_checkpoint_eval (
 checkpoint_eval_id bigint PRIMARY KEY,
 parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
 checkpoint_model_id bigint NOT NULL REFERENCES model(model_id),
 checkpoint_epoch integer
);
CREATE TABLE inference_eval_result (
 id bigint PRIMARY KEY, model_id bigint NOT NULL REFERENCES model(model_id),
 status text NOT NULL, inference_scope text NOT NULL,
 checkpoint_eval_id bigint, parent_experiment_id bigint,
 symbol text NOT NULL, prediction_horizon bigint NOT NULL,
 threshold_logret double precision NOT NULL, window_size bigint NOT NULL,
 label_rule_id integer NOT NULL, target_type integer NOT NULL,
 from_date text NOT NULL, to_date text NOT NULL, completed_epochs bigint,
 accuracy double precision, accept_model boolean, pred_down double precision,
 pred_neutral double precision, pred_up double precision,
 producer_worker_attempt_id bigint
);
CREATE TABLE experiment_analysis_result (
 analysis_id bigint PRIMARY KEY, experiment_id bigint NOT NULL,
 model_id bigint, analysis_scope text NOT NULL, checkpoint_eval_id bigint,
 parent_experiment_id bigint, analysis_status text NOT NULL,
 infer_accuracy double precision, accept_accuracy double precision,
 accept_rate double precision, leader_score double precision,
 pred_down_count bigint, pred_neutral_count bigint, pred_up_count bigint,
 accept_count bigint
);
SQL

"${postgres_bin}/psql" -X -v ON_ERROR_STOP=1 -q \
    -h "${socket_dir}" -p "${port}" -U pqxx -d postgres \
    -c "ALTER DATABASE ${database_name} SET lock_timeout='5s';" \
    -c "ALTER DATABASE ${database_name} SET statement_timeout='20s';"

binary="${build_dir}/ExperimentReplicationMaterializationRepositoryTests"
"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    -I/opt/homebrew/opt/libpqxx@7.10.1/include \
    -I/opt/homebrew/opt/libpq/include \
    "${repo_root}/Tests/ExperimentReplicationMaterializationRepositoryTests.cpp" \
    "${repo_root}/Sources/ExperimentReplicationMaterialization.cpp" \
    "${repo_root}/Sources/ExperimentReplicationMaterializationCommand.cpp" \
    "${repo_root}/Sources/ExperimentReplicationMaterializationRepository.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanning.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanningService.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanningCommand.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluation.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluationRepository.cpp" \
    "${repo_root}/Sources/PairedTrainingObjectiveEvaluation.cpp" \
    "${repo_root}/Sources/PairedTrainingObjectiveEvaluationRepository.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    -L/opt/homebrew/opt/libpqxx@7.10.1/lib -L/opt/homebrew/opt/libpq/lib \
    -lpqxx -lpq -pthread -o "${binary}"

EA_REPLICATION_MATERIALIZER_DB_HOST="${socket_dir}" \
EA_REPLICATION_MATERIALIZER_DB_PORT="${port}" \
EA_REPLICATION_MATERIALIZER_DB_NAME="${database_name}" \
    "${binary}"

printf '%s\n' 'Experiment replication materialization repository tests passed'
