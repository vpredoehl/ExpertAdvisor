#include "ContinuationPolicyPersistence.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace EA::ExperimentScheduler
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(
    const pqxx::row& row,
    const char* column)
{
    const pqxx::field field = row[column];
    if (field.is_null())
        return std::nullopt;
    return field.as<Value>();
}

bool TableExists(pqxx::work& transaction, const std::string& table)
{
    return !transaction.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        table).empty();
}

bool ColumnExists(
    pqxx::work& transaction,
    const std::string& table,
    const std::string& column)
{
    return !transaction.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 "
        "AND column_name = $2 LIMIT 1;",
        table,
        column).empty();
}

ContinuationPolicyConfig MapContinuationPolicyConfig(
    const pqxx::row& row)
{
    ContinuationPolicyConfig config;
    config.sourceExperimentId = row["source_experiment_id"].as<long long>();
    config.source.symbol = row["source_symbol"].as<std::string>();
    config.source.predictionHorizon = row["source_prediction_horizon"].as<int>();
    config.source.cNextThreshold = row["source_c_next_threshold"].as<double>();
    config.source.checkpointInterval = row["source_checkpoint_interval"].as<int>();
    config.source.targetEpochs = row["source_target_epochs"].as<int>();
    config.source.trainStart = row["source_train_start"].as<std::string>();
    config.source.trainEnd = row["source_train_end"].as<std::string>();
    config.source.inferStart = OptionalValue<std::string>(row, "source_infer_start");
    config.source.inferEnd = OptionalValue<std::string>(row, "source_infer_end");
    config.source.lastModelId = OptionalValue<long long>(row, "source_last_model_id");
    config.status = row["source_status"].as<std::string>();
    config.phase = row["source_phase"].as<std::string>();
    config.enabled = row["policy_enabled"].as<bool>();
    config.targetEpochs = OptionalValue<int>(row, "policy_target_epochs");
    config.minEvals = row["policy_min_evals"].as<int>();
    config.patience = row["policy_patience"].as<int>();
    config.minLeaderScore = OptionalValue<double>(row, "policy_min_leader_score");
    config.minInferAccuracy = OptionalValue<double>(row, "policy_min_infer_accuracy");
    config.minImprovement = OptionalValue<double>(row, "policy_min_improvement");
    config.maxDegradation = OptionalValue<double>(row, "policy_max_degradation");
    config.topN = OptionalValue<int>(row, "policy_top_n");
    config.scope = row["policy_scope"].as<std::string>();
    config.trendMode = row["policy_trend_mode"].as<std::string>();
    config.sourceMode = row["policy_source_mode"].as<std::string>();
    config.includeExcluded = row["policy_include_excluded"].as<bool>();
    config.candidateExcluded = row["candidate_excluded"].as<bool>();
    config.policyRevision = row["policy_revision"].as<long long>();
    config.lastDecision = OptionalValue<std::string>(row, "policy_last_decision");
    config.lastReason = OptionalValue<std::string>(row, "policy_last_reason");
    config.selectedModelId = OptionalValue<long long>(row, "policy_selected_model_id");
    config.queuedExperimentId = OptionalValue<long long>(row, "policy_queued_experiment_id");
    config.continuationSourceExperimentId =
        OptionalValue<long long>(row, "lineage_source_experiment_id");
    config.continuationSourceModelId =
        OptionalValue<long long>(row, "lineage_source_model_id");
    config.continuationSourceEpoch = OptionalValue<int>(row, "lineage_source_epoch");
    config.continuationDecisionId = OptionalValue<long long>(row, "lineage_decision_id");
    config.continuationGeneration = row["lineage_generation"].as<int>();
    config.inheritToChild = row["policy_inherit_to_child"].as<bool>();
    config.targetIncrement = OptionalValue<int>(row, "policy_target_increment");
    config.policyInherited = row["policy_inherited"].as<bool>();
    config.inheritedFromExperimentId =
        OptionalValue<long long>(row, "policy_inherited_from_experiment_id");
    config.inheritanceStatus = row["policy_inheritance_status"].as<std::string>();
    config.maxTargetEpochs = OptionalValue<int>(row, "policy_max_target_epochs");
    config.inheritedFromRevision =
        OptionalValue<long long>(row, "policy_inherited_from_revision");
    config.inheritedFromHash =
        OptionalValue<std::string>(row, "policy_inherited_from_hash");
    config.progressionMode = OptionalValue<std::string>(row, "policy_progression_mode");
    const std::optional<std::string> sequence =
        OptionalValue<std::string>(row, "policy_target_sequence");
    if (sequence.has_value())
        config.targetSequence = ParseContinuationTargetSequence(*sequence);
    return config;
}

PersistedContinuationIdentity MapPersistedContinuationIdentity(
    const pqxx::row& row)
{
    PersistedContinuationIdentity persisted;
    persisted.decisionId = row["decision_id"].as<long long>();
    persisted.sourceExperimentId = row["decision_source_experiment_id"].as<long long>();
    persisted.sourceModelId = row["decision_source_model_id"].as<long long>();
    persisted.sourceAnalysisId = row["decision_source_analysis_id"].as<long long>();
    persisted.sourceCheckpointEvalId =
        OptionalValue<long long>(row, "decision_source_checkpoint_eval_id");
    persisted.sourceEpoch = row["decision_source_epoch"].as<int>();
    persisted.targetEpochs = row["decision_target_epochs"].as<int>();
    persisted.decision = row["decision_value"].as<std::string>();
    persisted.leaderScore = OptionalValue<double>(row, "decision_leader_score");
    persisted.inferAccuracy = OptionalValue<double>(row, "decision_infer_accuracy");
    persisted.rankValue = OptionalValue<int>(row, "decision_rank_value");
    persisted.observedEvalCount = row["decision_observed_eval_count"].as<int>();
    persisted.patienceWindow = row["decision_patience_window"].as<int>();
    persisted.trendMetric = OptionalValue<std::string>(row, "decision_trend_metric");
    persisted.trendValue = OptionalValue<double>(row, "decision_trend_value");
    persisted.policyRevision = row["decision_policy_revision"].as<long long>();
    persisted.policyHash = row["decision_policy_hash"].as<std::string>();
    persisted.evidenceWatermark = row["decision_evidence_watermark"].as<std::string>();
    persisted.queuedExperimentId = OptionalValue<long long>(row, "queued_experiment_id");
    persisted.queuedChildExists = !row["child_experiment_id"].is_null();
    if (persisted.queuedChildExists)
    {
        persisted.queuedChildStatus = row["child_status"].as<std::string>();
        persisted.childParentExperimentId =
            OptionalValue<long long>(row, "child_parent_experiment_id");
        persisted.childSourceExperimentId =
            OptionalValue<long long>(row, "child_source_experiment_id");
        persisted.childResumeModelId =
            OptionalValue<long long>(row, "child_resume_model_id");
        persisted.childSourceModelId =
            OptionalValue<long long>(row, "child_source_model_id");
        persisted.childSourceEpoch = OptionalValue<int>(row, "child_source_epoch");
        persisted.childTargetEpochs = row["child_target_epochs"].as<int>();
        persisted.childGeneration = row["child_generation"].as<int>();
        persisted.childPolicyInherited = row["child_policy_inherited"].as<bool>();
        persisted.childPolicySourceMode = row["child_policy_source_mode"].as<std::string>();
    }
    persisted.sourceAnalysisScope = row["source_analysis_scope"].as<std::string>();
    persisted.sourceAnalysisValid = row["source_analysis_valid"].as<bool>();
    persisted.sourceModelOwnedBySource = row["source_model_owned_by_source"].as<bool>();
    persisted.evidenceChangedAfterDecision =
        row["evidence_changed_after_decision"].as<bool>();
    return persisted;
}

} // namespace

bool ContinuationPolicySchemaExists(pqxx::work& transaction)
{
    if (!TableExists(transaction, "experiment_continuation_decision"))
        return false;
    const std::vector<std::pair<std::string, std::string>> requiredColumns = {
        {"experiment", "continuation_policy_enabled"},
        {"experiment", "continuation_policy_target_epochs"},
        {"experiment", "continuation_policy_revision"},
        {"experiment", "continuation_policy_inherit_to_child"},
        {"experiment", "continuation_policy_target_increment"},
        {"experiment", "continuation_policy_max_target_epochs"},
        {"experiment", "continuation_policy_progression_mode"},
        {"experiment", "continuation_policy_target_sequence"},
        {"experiment", "continuation_policy_inherited"},
        {"experiment", "continuation_policy_inherited_from_experiment_id"},
        {"experiment", "continuation_policy_inherited_from_revision"},
        {"experiment", "continuation_policy_inherited_from_hash"},
        {"experiment", "continuation_policy_inheritance_status"},
        {"experiment", "continuation_candidate_excluded"},
        {"experiment", "continuation_source_experiment_id"},
        {"experiment", "continuation_decision_id"},
        {"model", "experiment_id"},
        {"experiment_analysis_result", "analysis_scope"},
        {"experiment_checkpoint_eval", "analysis_id"},
    };
    for (const auto& [table, column] : requiredColumns)
    {
        if (!ColumnExists(transaction, table, column))
            return false;
    }
    return true;
}

std::optional<ContinuationPolicyConfig> LoadContinuationPolicyConfig(
    pqxx::work& transaction,
    long long sourceExperimentId,
    bool lockRow)
{
    if (!ContinuationPolicySchemaExists(transaction))
        return std::nullopt;
    std::string sql =
        "SELECT e.experiment_id AS source_experiment_id, e.symbol AS source_symbol, "
        "e.prediction_horizon AS source_prediction_horizon, "
        "e.c_next_threshold AS source_c_next_threshold, "
        "e.checkpoint_interval AS source_checkpoint_interval, "
        "e.target_epochs AS source_target_epochs, e.train_start::text AS source_train_start, "
        "e.train_end::text AS source_train_end, e.infer_start::text AS source_infer_start, "
        "e.infer_end::text AS source_infer_end, e.last_model_id AS source_last_model_id, "
        "e.status AS source_status, e.phase AS source_phase, "
        "e.continuation_policy_enabled AS policy_enabled, "
        "e.continuation_policy_target_epochs AS policy_target_epochs, "
        "e.continuation_policy_min_evals AS policy_min_evals, "
        "e.continuation_policy_patience AS policy_patience, "
        "e.continuation_policy_min_leader_score AS policy_min_leader_score, "
        "e.continuation_policy_min_infer_accuracy AS policy_min_infer_accuracy, "
        "e.continuation_policy_min_improvement AS policy_min_improvement, "
        "e.continuation_policy_max_degradation AS policy_max_degradation, "
        "e.continuation_policy_top_n AS policy_top_n, "
        "e.continuation_policy_scope AS policy_scope, "
        "e.continuation_policy_trend_mode AS policy_trend_mode, "
        "e.continuation_policy_source_mode AS policy_source_mode, "
        "e.continuation_policy_include_excluded AS policy_include_excluded, "
        "e.continuation_candidate_excluded AS candidate_excluded, "
        "e.continuation_policy_revision AS policy_revision, "
        "e.continuation_policy_last_decision AS policy_last_decision, "
        "e.continuation_policy_last_reason AS policy_last_reason, "
        "e.continuation_policy_selected_model_id AS policy_selected_model_id, "
        "e.continuation_policy_queued_experiment_id AS policy_queued_experiment_id, "
        "e.continuation_source_experiment_id AS lineage_source_experiment_id, "
        "e.continuation_source_model_id AS lineage_source_model_id, "
        "e.continuation_source_epoch AS lineage_source_epoch, "
        "e.continuation_decision_id AS lineage_decision_id, "
        "e.continuation_generation AS lineage_generation, "
        "e.continuation_policy_inherit_to_child AS policy_inherit_to_child, "
        "e.continuation_policy_target_increment AS policy_target_increment, "
        "e.continuation_policy_inherited AS policy_inherited, "
        "e.continuation_policy_inherited_from_experiment_id AS policy_inherited_from_experiment_id, "
        "e.continuation_policy_inheritance_status AS policy_inheritance_status, "
        "e.continuation_policy_max_target_epochs AS policy_max_target_epochs, "
        "e.continuation_policy_inherited_from_revision AS policy_inherited_from_revision, "
        "e.continuation_policy_inherited_from_hash AS policy_inherited_from_hash, "
        "e.continuation_policy_progression_mode AS policy_progression_mode, "
        "array_to_string(e.continuation_policy_target_sequence, ':') AS policy_target_sequence "
        "FROM experiment e WHERE e.experiment_id = $1";
    if (lockRow)
        sql += " FOR UPDATE";
    sql += ";";
    const pqxx::result rows = transaction.exec_params(sql, sourceExperimentId);
    if (rows.empty())
        return std::nullopt;
    return MapContinuationPolicyConfig(rows[0]);
}

ContinuationAutoPreflightLookup LoadAutomaticContinuationPreflight(
    pqxx::work& transaction,
    long long sourceExperimentId)
{
    ContinuationAutoPreflightLookup preflight;
    const std::optional<ContinuationPolicyConfig> loaded =
        LoadContinuationPolicyConfig(transaction, sourceExperimentId, false);
    if (!loaded.has_value())
        throw std::runtime_error("source_experiment_not_found");
    preflight.currentPolicy = *loaded;
    preflight.satisfaction.currentPolicyHash =
        ContinuationPolicySemanticHash(preflight.currentPolicy);
    preflight.satisfaction.reason = "no_persisted_decision_for_current_target";
    if (!preflight.currentPolicy.targetEpochs.has_value())
        return preflight;

    const pqxx::result rows = transaction.exec_params(
        "SELECT d.continuation_decision_id AS decision_id, "
        "d.source_experiment_id AS decision_source_experiment_id, "
        "d.source_model_id AS decision_source_model_id, "
        "d.source_analysis_id AS decision_source_analysis_id, "
        "d.source_checkpoint_eval_id AS decision_source_checkpoint_eval_id, "
        "d.source_epoch AS decision_source_epoch, d.target_epochs AS decision_target_epochs, "
        "d.decision AS decision_value, d.leader_score AS decision_leader_score, "
        "d.infer_accuracy AS decision_infer_accuracy, d.rank_value AS decision_rank_value, "
        "d.observed_eval_count AS decision_observed_eval_count, "
        "d.patience_window AS decision_patience_window, "
        "d.trend_metric AS decision_trend_metric, d.trend_value AS decision_trend_value, "
        "d.policy_revision AS decision_policy_revision, d.policy_hash AS decision_policy_hash, "
        "d.evidence_watermark AS decision_evidence_watermark, "
        "d.queued_experiment_id AS queued_experiment_id, "
        "child.experiment_id AS child_experiment_id, child.status AS child_status, "
        "child.parent_experiment_id AS child_parent_experiment_id, "
        "child.continuation_source_experiment_id AS child_source_experiment_id, "
        "child.resume_model_id AS child_resume_model_id, "
        "child.continuation_source_model_id AS child_source_model_id, "
        "child.continuation_source_epoch AS child_source_epoch, "
        "child.target_epochs AS child_target_epochs, "
        "child.continuation_generation AS child_generation, "
        "COALESCE(child.continuation_policy_inherited, false) AS child_policy_inherited, "
        "COALESCE(child.continuation_policy_source_mode, '') AS child_policy_source_mode, "
        "COALESCE(source_analysis.analysis_scope, '') AS source_analysis_scope, "
        "(source_analysis.analysis_id IS NOT NULL "
        " AND source_analysis.experiment_id = d.source_experiment_id "
        " AND source_analysis.model_id = d.source_model_id "
        " AND source_analysis.completed_epochs = d.source_epoch "
        " AND source_analysis.analysis_status = 'completed' "
        " AND source_analysis.checkpoint_eval_id IS NOT DISTINCT FROM d.source_checkpoint_eval_id) "
        "AS source_analysis_valid, "
        "EXISTS (SELECT 1 FROM model source_model "
        "        WHERE source_model.model_id = d.source_model_id "
        "        AND source_model.experiment_id = d.source_experiment_id) "
        "AS source_model_owned_by_source, "
        "(EXISTS (SELECT 1 FROM experiment_analysis_result changed_analysis "
        "         WHERE changed_analysis.experiment_id = d.source_experiment_id "
        "         AND changed_analysis.analysis_status = 'completed' "
        "         AND changed_analysis.updated_at > d.updated_at) "
        " OR EXISTS (SELECT 1 FROM experiment_checkpoint_eval changed_checkpoint "
        "            WHERE changed_checkpoint.parent_experiment_id = d.source_experiment_id "
        "            AND changed_checkpoint.updated_at > d.updated_at)) "
        "AS evidence_changed_after_decision "
        "FROM experiment_continuation_decision d "
        "LEFT JOIN experiment child ON child.experiment_id = d.queued_experiment_id "
        "LEFT JOIN experiment_analysis_result source_analysis "
        "  ON source_analysis.analysis_id = d.source_analysis_id "
        "WHERE d.source_experiment_id = $1 AND d.target_epochs = $2;",
        sourceExperimentId,
        *preflight.currentPolicy.targetEpochs);
    if (rows.empty())
        return preflight;

    preflight.persistedIdentity = MapPersistedContinuationIdentity(rows[0]);
    preflight.satisfaction = CheckAutomaticContinuationSatisfaction(
        preflight.currentPolicy,
        *preflight.persistedIdentity);
    return preflight;
}

std::string FormatAutomaticContinuationSatisfiedFields(
    const ContinuationAutoPreflightLookup& preflight,
    bool dryRun)
{
    if (!preflight.persistedIdentity.has_value() ||
        !preflight.satisfaction.alreadySatisfied)
    {
        throw std::invalid_argument("automatic_continuation_not_satisfied");
    }
    const PersistedContinuationIdentity& persisted = *preflight.persistedIdentity;
    std::ostringstream out;
    out << ",source_experiment_id=" << preflight.currentPolicy.sourceExperimentId
        << ",source_model_id=" << persisted.sourceModelId
        << ",source_epoch=" << persisted.sourceEpoch
        << ",target_epochs=" << persisted.targetEpochs
        << ",decision_id=" << persisted.decisionId
        << ",queued_experiment_id=" << *persisted.queuedExperimentId
        << ",reason=" << preflight.satisfaction.reason
        << ",current_policy_revision=" << preflight.currentPolicy.policyRevision
        << ",persisted_decision_policy_revision=" << persisted.policyRevision
        << ",current_policy_hash=" << preflight.satisfaction.currentPolicyHash
        << ",persisted_decision_policy_hash="
        << preflight.satisfaction.persistedDecisionPolicyHash
        << ",policy_hash_changed="
        << (preflight.satisfaction.currentPolicyHash ==
                    preflight.satisfaction.persistedDecisionPolicyHash
                ? "0"
                : "1")
        << ",persisted_evidence_watermark=" << persisted.evidenceWatermark
        << ",dry_run=" << (dryRun ? "1" : "0");
    return out.str();
}

} // namespace EA::ExperimentScheduler
