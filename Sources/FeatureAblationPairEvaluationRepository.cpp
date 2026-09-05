#include "FeatureAblationPairEvaluationRepository.hpp"

#include "InferenceProfitabilityRepository.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"

#include <optional>
#include <string>

namespace EA::FeatureAblationPairEvaluation
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row,
                                   const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

void LoadResumeCheckpointProvenance(
    pqxx::transaction_base& transaction,
    ArmEvidence& arm)
{
    const auto& configuration = arm.authoritative.configuration;
    if (!configuration.resumeModelId) return;

    const long long experimentId = configuration.experimentId;
    const long long resumeModelId = *configuration.resumeModelId;

    const pqxx::result ownership = transaction.exec(
        "SELECT experiment_id FROM model WHERE model_id=$1;",
        pqxx::params{resumeModelId});
    if (ownership.size() != 1 ||
        ownership[0]["experiment_id"].is_null())
        throw PairedTrainingObjectiveEvaluation::EvidenceLoadError(
            PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                ProvenanceContractFailure,
            "experiment_" + std::to_string(experimentId) +
                "_resume_model_identity_invalid");

    ResumeCheckpointProvenance provenance;
    provenance.resumeModelId = resumeModelId;
    provenance.modelExperimentId =
        ownership.one_row()["experiment_id"].as<long long>();

    // Checkpoint-stop/resume records the authoritative continuation point
    // directly on experiment. Prefer that identity: unlike
    // experiment_checkpoint_eval, it exists even when checkpoint inference was
    // disabled and the checkpoint was used only for stop/resume.
    const pqxx::result stopped = transaction.exec(
        "SELECT stopped_at_checkpoint_epoch,stopped_at_checkpoint_model_id "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});

    if (stopped.size() != 1)
        throw PairedTrainingObjectiveEvaluation::EvidenceLoadError(
            PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                ProvenanceContractFailure,
            "experiment_" + std::to_string(experimentId) +
                "_resume_checkpoint_experiment_identity_invalid");

    const pqxx::row stoppedRow = stopped.one_row();
    const auto stoppedEpoch = OptionalValue<int>(
        stoppedRow, "stopped_at_checkpoint_epoch");
    const auto stoppedModelId = OptionalValue<long long>(
        stoppedRow, "stopped_at_checkpoint_model_id");

    if (stoppedEpoch && stoppedModelId &&
        *stoppedModelId == resumeModelId &&
        *stoppedEpoch > 0)
    {
        provenance.checkpointEpoch = *stoppedEpoch;
        provenance.ownExperimentCheckpoint =
            provenance.modelExperimentId == experimentId;
    }
    else
    {
        // Retain checkpoint-evaluation provenance as a fail-closed fallback
        // for workflows whose resume model was persisted through that path.
        const pqxx::result checkpoints = transaction.exec(
            "SELECT checkpoint_epoch FROM experiment_checkpoint_eval "
            "WHERE parent_experiment_id=$1 AND checkpoint_model_id=$2 "
            "ORDER BY checkpoint_eval_id;",
            pqxx::params{experimentId, resumeModelId});

        if (checkpoints.size() == 1 &&
            !checkpoints.one_row()["checkpoint_epoch"].is_null())
        {
            provenance.checkpointEpoch =
                checkpoints.one_row()["checkpoint_epoch"].as<int>();
            provenance.ownExperimentCheckpoint =
                provenance.modelExperimentId == experimentId &&
                *provenance.checkpointEpoch > 0;
        }
    }

    arm.resumeCheckpointProvenance = provenance;
}

} // namespace

ArmEvidence LoadAuthoritativeArmEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    ArmEvidence arm;
    arm.authoritative =
        PairedTrainingObjectiveEvaluation::LoadAuthoritativeArmEvidence(
            transaction, experimentId);

    LoadResumeCheckpointProvenance(transaction, arm);

    const pqxx::result rows = transaction.exec(
        "SELECT training_objective_version,loss_definition_version,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,"
        "regression_target_definition,regression_normalization_identity,"
        "robust_loss_definition,robust_loss_delta,target_clipping_definition,"
        "objective_normalization_identity,checkpoint_infer_enabled,"
        "checkpoint_infer_min_epoch,checkpoint_infer_interval,"
        "checkpoint_policy_enabled,checkpoint_policy_min_leader_score,"
        "checkpoint_policy_min_infer_accuracy,checkpoint_policy_top_n,"
        "checkpoint_policy_scope,checkpoint_policy_stop_mode,"
        "checkpoint_policy_grace_evals,checkpoint_policy_revision,"
        "checkpoint_policy_hash,model_input_width,"
        "model_input_semantic_layout_version,scheduler_priority,worker_pid,"
        "continuation_policy_enabled,"
        "CASE WHEN continuation_policy_enabled THEN ROW("
        "continuation_policy_target_epochs,continuation_policy_min_evals,"
        "continuation_policy_patience,continuation_policy_min_leader_score,"
        "continuation_policy_min_infer_accuracy,"
        "continuation_policy_min_profit_actionable_count,"
        "continuation_policy_min_profit_aggregate_log_return_sum,"
        "continuation_policy_min_profit_average_log_return,"
        "continuation_policy_min_improvement,"
        "continuation_policy_max_degradation,continuation_policy_top_n,"
        "continuation_policy_scope,continuation_policy_trend_mode,"
        "continuation_policy_source_mode,continuation_policy_include_excluded,"
        "continuation_candidate_excluded,continuation_policy_inherit_to_child,"
        "continuation_policy_progression_mode,"
        "array_to_string(continuation_policy_target_sequence,':'),"
        "continuation_policy_target_increment,"
        "continuation_policy_max_target_epochs,continuation_policy_inherited,"
        "continuation_policy_inherited_from_experiment_id,"
        "continuation_policy_inherited_from_revision,"
        "continuation_policy_inherited_from_hash,"
        "continuation_policy_inheritance_status,continuation_policy_revision"
        ")::text ELSE 'disabled' END AS continuation_policy_scientific_identity "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.size() != 1)
        throw PairedTrainingObjectiveEvaluation::EvidenceLoadError(
            PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                ProvenanceContractFailure,
            "experiment_" + std::to_string(experimentId) +
                "_extended_configuration_missing");
    const pqxx::row row = rows.one_row();
    auto& extended = arm.extended;
    extended.configuredModelInputWidth = OptionalValue<int>(
        row, "model_input_width");
    extended.configuredModelInputLayoutVersion = OptionalValue<int>(
        row, "model_input_semantic_layout_version");
    // Fresh initialization is deterministic in the current executable. A
    // resumed arm is governed by its validated model ancestry instead.
    if (arm.authoritative.configuration.resumeModelId)
        extended.freshInitializationSeed.reset();
    extended.trainingObjectiveVersion =
        row["training_objective_version"].as<int>();
    extended.lossDefinitionVersion = row["loss_definition_version"].as<int>();
    extended.auxiliaryLossMode = row["auxiliary_loss_mode"].as<std::string>();
    extended.auxiliaryLossCoefficient =
        row["auxiliary_loss_coefficient"].as<double>();
    extended.regressionTargetDefinition = OptionalValue<std::string>(
        row, "regression_target_definition");
    extended.regressionNormalizationIdentity = OptionalValue<std::string>(
        row, "regression_normalization_identity");
    extended.robustLossDefinition = OptionalValue<std::string>(
        row, "robust_loss_definition");
    extended.robustLossDelta = OptionalValue<double>(row, "robust_loss_delta");
    extended.targetClippingDefinition =
        row["target_clipping_definition"].as<std::string>();
    extended.objectiveNormalizationIdentity =
        row["objective_normalization_identity"].as<std::string>();
    extended.checkpointInferenceEnabled =
        row["checkpoint_infer_enabled"].as<bool>();
    extended.checkpointInferenceMinimumEpoch = OptionalValue<int>(
        row, "checkpoint_infer_min_epoch");
    extended.checkpointInferenceInterval = OptionalValue<int>(
        row, "checkpoint_infer_interval");
    extended.checkpointPolicyEnabled =
        row["checkpoint_policy_enabled"].as<bool>();
    extended.checkpointPolicyMinimumLeaderScore = OptionalValue<double>(
        row, "checkpoint_policy_min_leader_score");
    extended.checkpointPolicyMinimumInferenceAccuracy = OptionalValue<double>(
        row, "checkpoint_policy_min_infer_accuracy");
    extended.checkpointPolicyTopN = OptionalValue<int>(
        row, "checkpoint_policy_top_n");
    extended.checkpointPolicyScope =
        row["checkpoint_policy_scope"].as<std::string>();
    extended.checkpointPolicyStopMode =
        row["checkpoint_policy_stop_mode"].as<std::string>();
    extended.checkpointPolicyGraceEvaluations =
        row["checkpoint_policy_grace_evals"].as<int>();
    extended.checkpointPolicyRevision =
        row["checkpoint_policy_revision"].as<long long>();
    extended.checkpointPolicyHash = OptionalValue<std::string>(
        row, "checkpoint_policy_hash");
    extended.continuationPolicyEnabled =
        row["continuation_policy_enabled"].as<bool>();
    extended.continuationPolicyScientificIdentity =
        row["continuation_policy_scientific_identity"].as<std::string>();
    arm.operational.schedulerPriority =
        row["scheduler_priority"].as<std::string>();
    arm.operational.workerPid = OptionalValue<int>(row, "worker_pid");

    if (arm.authoritative.finalModelId)
    {
        const auto exact =
            InferenceProfitability::ResolveExactFinalInferenceResult(
                transaction, experimentId,
                *arm.authoritative.finalModelId);
        using Status = InferenceProfitability::ExactFinalInferenceResultStatus;
        if (exact.status == Status::ambiguousFinalInferenceResult)
            throw PairedTrainingObjectiveEvaluation::EvidenceLoadError(
                PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                    AmbiguousEvidence,
                "experiment_" + std::to_string(experimentId) +
                    "_final_inference_ambiguous");
        if (exact.status == Status::finalInferenceContextMismatch)
            throw PairedTrainingObjectiveEvaluation::EvidenceLoadError(
                PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                    ProvenanceContractFailure,
                "experiment_" + std::to_string(experimentId) +
                    "_final_inference_context_mismatch");
        arm.exactFinalInferenceResultId = exact.inferenceEvalResultId;
    }
    return arm;
}

} // namespace EA::FeatureAblationPairEvaluation
