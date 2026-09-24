#include "ExperimentReplicationMaterializationRepository.hpp"

#include <stdexcept>

namespace EA::ExperimentReplicationMaterialization
{

long long InsertFreshPausedReplicationExperiment(
    pqxx::transaction_base& transaction,
    const ExperimentReplicationPlanning::ProposedExperimentSpecification&
        specification)
{
    if (specification.authoritativeSourceExperimentId <= 0 ||
        specification.freshInitializationSeed == 0 ||
        specification.experimentId !=
            specification.authoritativeSourceExperimentId)
        throw std::invalid_argument(
            "invalid_proposed_experiment_persistence_specification");

    // Scientific/configured columns come from the locked and revalidated
    // source row; only fresh_initialization_seed changes. Operational,
    // progress, result, worker, timestamp, pause/resume, and execution-lineage
    // columns are absent and receive fresh defaults (or paused/train below).
    const pqxx::result inserted = transaction.exec(
        "INSERT INTO experiment ("
        "symbol,prediction_horizon,c_next_threshold,core_lr_mult,"
        "head_lr_mult,target_epochs,checkpoint_interval,train_start,"
        "train_end,infer_start,infer_end,resume_model_id,duplicate_nonce,"
        "status,phase,invocation_mode,donchian20_mode,feature_warmup_scope,"
        "donchian_lookback,feature_ablation_mask,fresh_initialization_seed,"
        "resume_expand_input_width,training_objective_canonical,"
        "training_objective_hash,training_objective_id,"
        "training_objective_version,loss_definition_version,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,"
        "regression_target_definition,regression_normalization_identity,"
        "robust_loss_definition,robust_loss_delta,"
        "target_clipping_definition,objective_normalization_identity,"
        "model_input_width,model_input_semantic_layout_version,"
        "economic_calendar_snapshot_id,economic_calendar_snapshot_hash,"
        "opportunistic_checkpoint_infer,checkpoint_infer_enabled,"
        "checkpoint_infer_min_epoch,checkpoint_infer_interval,"
        "checkpoint_policy_enabled,checkpoint_policy_min_leader_score,"
        "checkpoint_policy_min_infer_accuracy,checkpoint_policy_top_n,"
        "checkpoint_policy_scope,checkpoint_policy_stop_mode,"
        "checkpoint_policy_grace_evals,checkpoint_policy_revision,"
        "checkpoint_policy_hash,continuation_policy_enabled,"
        "continuation_policy_target_epochs,continuation_policy_min_evals,"
        "continuation_policy_patience,continuation_policy_min_leader_score,"
        "continuation_policy_min_infer_accuracy,"
        "continuation_policy_min_profit_actionable_count,"
        "continuation_policy_min_profit_aggregate_log_return_sum,"
        "continuation_policy_min_profit_average_log_return,"
        "continuation_policy_min_improvement,"
        "continuation_policy_max_degradation,continuation_policy_top_n,"
        "continuation_policy_scope,continuation_policy_trend_mode,"
        "continuation_policy_source_mode,"
        "continuation_policy_include_excluded,continuation_candidate_excluded,"
        "continuation_policy_inherit_to_child,"
        "continuation_policy_progression_mode,"
        "continuation_policy_target_sequence,"
        "continuation_policy_target_increment,"
        "continuation_policy_max_target_epochs,continuation_policy_inherited,"
        "continuation_policy_inherited_from_experiment_id,"
        "continuation_policy_inherited_from_revision,"
        "continuation_policy_inherited_from_hash,"
        "continuation_policy_inheritance_status,"
        "continuation_policy_revision"
        ") SELECT "
        "symbol,prediction_horizon,c_next_threshold,core_lr_mult,"
        "head_lr_mult,target_epochs,checkpoint_interval,train_start,"
        "train_end,infer_start,infer_end,NULL,"
        "(SELECT COALESCE(max(duplicate_nonce),0) + 1 FROM experiment),"
        "'paused','train',"
        "'controlled_replication_materialization',donchian20_mode,"
        "feature_warmup_scope,donchian_lookback,feature_ablation_mask,$2,"
        "false,training_objective_canonical,training_objective_hash,"
        "training_objective_id,training_objective_version,"
        "loss_definition_version,auxiliary_loss_mode,"
        "auxiliary_loss_coefficient,regression_target_definition,"
        "regression_normalization_identity,robust_loss_definition,"
        "robust_loss_delta,target_clipping_definition,"
        "objective_normalization_identity,model_input_width,"
        "model_input_semantic_layout_version,economic_calendar_snapshot_id,"
        "economic_calendar_snapshot_hash,opportunistic_checkpoint_infer,"
        "checkpoint_infer_enabled,checkpoint_infer_min_epoch,"
        "checkpoint_infer_interval,checkpoint_policy_enabled,"
        "checkpoint_policy_min_leader_score,"
        "checkpoint_policy_min_infer_accuracy,checkpoint_policy_top_n,"
        "checkpoint_policy_scope,checkpoint_policy_stop_mode,"
        "checkpoint_policy_grace_evals,checkpoint_policy_revision,"
        "checkpoint_policy_hash,continuation_policy_enabled,"
        "continuation_policy_target_epochs,continuation_policy_min_evals,"
        "continuation_policy_patience,continuation_policy_min_leader_score,"
        "continuation_policy_min_infer_accuracy,"
        "continuation_policy_min_profit_actionable_count,"
        "continuation_policy_min_profit_aggregate_log_return_sum,"
        "continuation_policy_min_profit_average_log_return,"
        "continuation_policy_min_improvement,"
        "continuation_policy_max_degradation,continuation_policy_top_n,"
        "continuation_policy_scope,continuation_policy_trend_mode,"
        "continuation_policy_source_mode,"
        "continuation_policy_include_excluded,continuation_candidate_excluded,"
        "continuation_policy_inherit_to_child,"
        "continuation_policy_progression_mode,"
        "continuation_policy_target_sequence,"
        "continuation_policy_target_increment,"
        "continuation_policy_max_target_epochs,continuation_policy_inherited,"
        "continuation_policy_inherited_from_experiment_id,"
        "continuation_policy_inherited_from_revision,"
        "continuation_policy_inherited_from_hash,"
        "continuation_policy_inheritance_status,"
        "continuation_policy_revision "
        "FROM experiment WHERE experiment_id=$1 "
        "RETURNING experiment_id;",
        pqxx::params{specification.authoritativeSourceExperimentId,
                     specification.freshInitializationSeed});
    if (inserted.size() != 1)
        throw std::runtime_error(
            "authoritative_source_experiment_disappeared_during_insert");
    return inserted.one_row()[0].as<long long>();
}

} // namespace EA::ExperimentReplicationMaterialization
