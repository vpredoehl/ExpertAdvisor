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

} // namespace

ArmEvidence LoadAuthoritativeArmEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    ArmEvidence arm;
    arm.authoritative =
        PairedTrainingObjectiveEvaluation::LoadAuthoritativeArmEvidence(
            transaction, experimentId);

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
        "checkpoint_policy_hash FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.size() != 1)
        throw PairedTrainingObjectiveEvaluation::EvidenceLoadError(
            PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                ProvenanceContractFailure,
            "experiment_" + std::to_string(experimentId) +
                "_extended_configuration_missing");
    const pqxx::row row = rows.one_row();
    auto& extended = arm.extended;
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
