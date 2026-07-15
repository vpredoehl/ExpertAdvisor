#include "ExperimentRecommendationEvaluationRepository.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

constexpr int kMaximumEvaluationLimit = 1000;

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* name)
{
    return row[name].is_null() ? std::nullopt
                               : std::optional<Value>{row[name].as<Value>()};
}

void ValidateFilters(const RecommendationEvaluationFilters& filters)
{
    if (filters.limit <= 0 || filters.limit > kMaximumEvaluationLimit)
        throw std::invalid_argument("recommendation_evaluation_limit_invalid");
    if (filters.recommendationId && *filters.recommendationId <= 0)
        throw std::invalid_argument("recommendation_evaluation_id_invalid");
    if (filters.recommendationScanId && *filters.recommendationScanId <= 0)
        throw std::invalid_argument("recommendation_evaluation_scan_id_invalid");
}

bool TableExists(pqxx::transaction_base& transaction, const std::string& table)
{
    return transaction.exec(
        "SELECT to_regclass(current_schema() || '.' || $1) IS NOT NULL;",
        pqxx::params{table}).one_row()[0].as<bool>();
}

EffectiveExperimentConfiguration MapExperimentConfiguration(
    const pqxx::row& row)
{
    EffectiveExperimentConfiguration value;
    value.symbol = row["candidate_symbol"].as<std::string>();
    value.predictionHorizon = row["candidate_prediction_horizon"].as<int>();
    value.labelThreshold = row["candidate_label_threshold"].as<double>();
    value.coreLrMult = OptionalValue<double>(row, "candidate_core_lr_mult");
    value.headLrMult = OptionalValue<double>(row, "candidate_head_lr_mult");
    value.targetEpochs = row["candidate_target_epochs"].as<int>();
    value.trainStartDate = row["candidate_train_start_date"].as<std::string>();
    value.trainEndDate = row["candidate_train_end_date"].as<std::string>();
    value.inferStartDate = OptionalValue<std::string>(
        row, "candidate_infer_start_date");
    value.inferEndDate = OptionalValue<std::string>(
        row, "candidate_infer_end_date");
    return value;
}

std::vector<RecommendationEvaluationExperimentConflict> FindConflicts(
    pqxx::transaction_base& transaction,
    const std::string& symbol,
    int horizon,
    const std::string& semanticCanonical)
{
    const pqxx::result rows = transaction.exec(
        "SELECT experiment_id,candidate_status,candidate_symbol,"
        "candidate_prediction_horizon,candidate_label_threshold,"
        "candidate_core_lr_mult,candidate_head_lr_mult,candidate_target_epochs,"
        "candidate_train_start_date,candidate_train_end_date,"
        "candidate_infer_start_date,candidate_infer_end_date FROM ("
        "SELECT e.experiment_id,e.status AS candidate_status,"
        "e.symbol AS candidate_symbol,e.prediction_horizon AS candidate_prediction_horizon,"
        "e.c_next_threshold AS candidate_label_threshold,"
        "e.core_lr_mult AS candidate_core_lr_mult,e.head_lr_mult AS candidate_head_lr_mult,"
        "e.target_epochs AS candidate_target_epochs,"
        "to_char(e.train_start AT TIME ZONE 'America/Chicago','YYYY-MM-DD') AS candidate_train_start_date,"
        "to_char(e.train_end AT TIME ZONE 'America/Chicago','YYYY-MM-DD') AS candidate_train_end_date,"
        "CASE WHEN e.infer_start IS NULL THEN NULL ELSE to_char(e.infer_start AT TIME ZONE 'America/Chicago','YYYY-MM-DD') END AS candidate_infer_start_date,"
        "CASE WHEN e.infer_end IS NULL THEN NULL ELSE to_char(e.infer_end AT TIME ZONE 'America/Chicago','YYYY-MM-DD') END AS candidate_infer_end_date "
        "FROM experiment e WHERE lower(btrim(e.symbol))=lower(btrim($1)) "
        "AND e.prediction_horizon=$2 AND e.status IN ('pending','running','paused','completed')"
        ") candidates ORDER BY experiment_id ASC;",
        pqxx::params{symbol, horizon});
    std::vector<RecommendationEvaluationExperimentConflict> conflicts;
    for (const pqxx::row& row : rows)
    {
        const RecommendationCandidateIdentity identity =
            BuildRecommendationCandidateIdentity(MapExperimentConfiguration(row));
        if (identity.canonicalText == semanticCanonical)
            conflicts.push_back({row["experiment_id"].as<long long>(),
                                 row["candidate_status"].as<std::string>()});
    }
    return conflicts;
}

RecommendationEvaluationInput MapEvaluationInput(
    pqxx::transaction_base& transaction,
    const pqxx::row& row)
{
    RecommendationEvaluationInput input;
    RecommendationScoringInput& score = input.scoringInput;
    score.recommendationId = row["recommendation_id"].as<long long>();
    score.recommendationStatus = row["recommendation_status"].as<std::string>();
    score.sourceExperimentId = row["source_experiment_id"].as<long long>();
    score.sourcePredictionHorizon = row["source_prediction_horizon"].as<int>();
    score.sourceRankWithinGroup = row["source_rank"].is_null()
        ? 0 : row["source_rank"].as<int>();
    score.sourceLeaderScore = row["source_leader_score"].as<double>();
    score.sourceInferenceAccuracy = row["source_infer_accuracy"].as<double>();
    score.sourcePredictedNeutralProportion = OptionalValue<double>(
        row, "source_predicted_neutral_proportion");
    score.sourceEvidenceCount = row["source_evidence_count"].as<long long>();
    score.changedParameter = row["changed_parameter"].as<std::string>();
    score.sourceValueCanonical = row["source_value_canonical"].as<std::string>();
    score.proposedValueCanonical = row["proposed_value_canonical"].as<std::string>();
    score.absoluteDelta = row["absolute_delta"].as<double>();
    score.relativeDelta = OptionalValue<double>(row, "relative_delta");
    score.horizonDelta = OptionalValue<int>(row, "horizon_delta");
    score.generationOrdinal = row["generation_ordinal"].as<int>();
    score.structuralRank = row["structural_rank"].as<int>();
    score.semanticCanonicalText =
        row["semantic_configuration_canonical"].as<std::string>();
    input.recommendationSemanticHash = row["semantic_hash"].as<std::string>();
    score.invocationCanonicalText =
        row["invocation_configuration_canonical"].as<std::string>();
    score.recommendationPolicyCanonicalText =
        row["policy_canonical"].as<std::string>();
    score.duplicateType = row["duplicate_type"].as<std::string>();
    input.recommendationScanId = row["recommendation_scan_id"].as<long long>();
    input.sourceModelId = OptionalValue<long long>(row, "source_model_id");
    input.sourceAnalysisId = OptionalValue<long long>(row, "source_analysis_id");
    input.sourceSymbol = row["source_symbol"].as<std::string>();
    input.scanStatus = row["scan_status"].as<std::string>();
    input.sourceExperimentStatus = row["source_experiment_status"].as<std::string>();
    input.sourceExperimentPhase = row["source_experiment_phase"].as<std::string>();
    input.currentSourceModelId = OptionalValue<long long>(row, "current_source_model_id");
    input.currentSourceAnalysisId = OptionalValue<long long>(
        row, "current_source_analysis_id");
    input.currentSourceAnalysisStatus = row["current_source_analysis_status"].is_null()
        ? "" : row["current_source_analysis_status"].as<std::string>();
    input.currentSourceAnalysisScope = row["current_source_analysis_scope"].is_null()
        ? "" : row["current_source_analysis_scope"].as<std::string>();
    input.exactExperimentConflicts = FindConflicts(
        transaction, input.sourceSymbol, score.sourcePredictionHorizon,
        score.semanticCanonicalText);
    return input;
}

RecommendationEligibility ParseEligibility(const std::string& value)
{
    if (value == "eligible") return RecommendationEligibility::eligible;
    if (value == "ineligible") return RecommendationEligibility::ineligible;
    throw std::runtime_error("invalid_persisted_recommendation_eligibility");
}

PersistedRecommendationEvaluationSummary MapSummary(const pqxx::row& row)
{
    PersistedRecommendationEvaluationSummary value;
    value.evaluationResultId = row["recommendation_evaluation_result_id"].as<long long>();
    value.evaluationRunId = row["recommendation_evaluation_run_id"].as<long long>();
    value.recommendationId = row["recommendation_id"].as<long long>();
    value.recommendationScanId = row["recommendation_scan_id"].as<long long>();
    value.sourceExperimentId = row["source_experiment_id"].as<long long>();
    value.sourceSymbol = row["source_symbol"].as<std::string>();
    value.sourcePredictionHorizon = row["source_prediction_horizon"].as<int>();
    value.changedParameter = row["changed_parameter"].as<std::string>();
    value.recommendationSemanticHash =
        row["recommendation_semantic_hash"].as<std::string>();
    value.evaluationIdentityHash = row["evaluation_identity_hash"].as<std::string>();
    value.evaluationPolicyHash = row["evaluation_policy_hash"].as<std::string>();
    value.evaluationVersion = row["evaluation_version"].as<int>();
    value.evaluatorVersion = row["evaluator_version"].as<int>();
    value.scoringPolicyHash = row["scoring_policy_hash"].as<std::string>();
    value.scoringVersion = row["scoring_version"].as<int>();
    value.eligibility = ParseEligibility(row["eligibility"].as<std::string>());
    const auto disposition = ParseRecommendationEvaluationDisposition(
        row["disposition"].as<std::string>());
    if (!disposition)
        throw std::runtime_error("invalid_persisted_evaluation_disposition");
    value.disposition = *disposition;
    value.reasonCode = row["reason_code"].as<std::string>();
    value.explanation = row["explanation"].as<std::string>();
    value.finalScore = OptionalValue<double>(row, "final_score");
    value.componentCount = row["component_count"].as<int>();
    value.missingEvidenceCount = row["missing_evidence_count"].as<int>();
    value.rankingOrdinal = row["ranking_ordinal"].as<int>();
    value.createdAt = row["created_at"].as<std::string>();
    return value;
}

std::string SummaryColumns()
{
    return "er.recommendation_evaluation_result_id,"
        "er.recommendation_evaluation_run_id,er.recommendation_id,"
        "er.recommendation_scan_id,er.source_experiment_id,r.source_symbol,"
        "r.source_prediction_horizon,r.changed_parameter,"
        "er.recommendation_semantic_hash,er.evaluation_identity_hash,"
        "er.eligibility,er.disposition,"
        "run.evaluation_policy_hash,run.evaluation_version,"
        "run.evaluator_version,run.scoring_policy_hash,run.scoring_version,"
        "er.reason_code,er.explanation,er.final_score,er.component_count,"
        "er.missing_evidence_count,er.ranking_ordinal,"
        "er.created_at::text AS created_at";
}

bool ComponentEqual(const pqxx::row& row,
                    const RecommendationScoreComponent& component,
                    int ordinal)
{
    return row["component_ordinal"].as<int>() == ordinal &&
        row["component_name"].as<std::string>() == component.componentName &&
        row["reason_code"].as<std::string>() == component.reasonCode &&
        row["input_canonical"].as<std::string>() == component.inputCanonical &&
        row["normalized_value"].as<double>() == component.normalizedValue &&
        row["weight"].as<double>() == component.weight &&
        row["weighted_contribution"].as<double>() == component.weightedContribution &&
        row["is_penalty"].as<bool>() == component.penalty &&
        !row["is_missing"].as<bool>() &&
        row["explanation"].as<std::string>() == component.explanation;
}

PersistedRecommendationEvaluationRun MapRun(const pqxx::row& row)
{
    PersistedRecommendationEvaluationRun value;
    value.evaluationRunId = row["recommendation_evaluation_run_id"].as<long long>();
    value.status = row["status"].as<std::string>();
    value.runIdentityHash = row["evaluation_run_identity_hash"].as<std::string>();
    value.evaluationPolicyHash = row["evaluation_policy_hash"].as<std::string>();
    value.evaluationVersion = row["evaluation_version"].as<int>();
    value.evaluatorVersion = row["evaluator_version"].as<int>();
    value.scoringPolicyHash = row["scoring_policy_hash"].as<std::string>();
    value.scoringVersion = row["scoring_version"].as<int>();
    value.recommendationScanFilter = OptionalValue<long long>(
        row, "recommendation_scan_filter");
    value.recommendationIdFilter = OptionalValue<long long>(
        row, "recommendation_id_filter");
    value.requestedLimit = OptionalValue<int>(row, "requested_limit");
    value.counters.recommendationsConsidered =
        row["recommendations_considered"].as<int>();
    value.counters.recommendationsEvaluated =
        row["recommendations_evaluated"].as<int>();
    value.counters.recommendationsEligible =
        row["recommendations_eligible"].as<int>();
    value.counters.recommendationsBlocked =
        row["recommendations_blocked"].as<int>();
    value.counters.evaluationErrors = row["evaluation_errors"].as<int>();
    value.startedAt = row["started_at"].as<std::string>();
    value.completedAt = OptionalValue<std::string>(row, "completed_at");
    value.errorMessage = OptionalValue<std::string>(row, "error_message");
    return value;
}

std::string RunColumns()
{
    return "recommendation_evaluation_run_id,status,"
        "evaluation_run_identity_hash,evaluation_policy_hash,"
        "evaluation_version,evaluator_version,scoring_policy_hash,"
        "scoring_version,recommendation_scan_filter,recommendation_id_filter,"
        "requested_limit,recommendations_considered,recommendations_evaluated,"
        "recommendations_eligible,recommendations_blocked,evaluation_errors,"
        "started_at::text AS started_at,completed_at::text AS completed_at,"
        "error_message";
}

} // namespace

bool RecommendationEvaluationSchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return TableExists(transaction, "experiment_recommendation_evaluation_run") &&
           TableExists(transaction, "experiment_recommendation_evaluation_result") &&
           TableExists(transaction, "experiment_recommendation_evaluation_component");
}

std::vector<RecommendationEvaluationLoadResult>
LoadRecommendationsForEvaluation(
    pqxx::connection& connection,
    const RecommendationEvaluationFilters& filters)
{
    ValidateFilters(filters);
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT r.recommendation_id,r.status AS recommendation_status,"
        "r.recommendation_scan_id,r.source_experiment_id,r.source_model_id,"
        "r.source_analysis_id,r.source_symbol,r.source_prediction_horizon,"
        "r.source_rank,r.source_leader_score,r.source_infer_accuracy,"
        "r.source_predicted_neutral_proportion,r.source_evidence_count,"
        "r.changed_parameter,r.source_value_canonical,r.proposed_value_canonical,"
        "r.absolute_delta,r.relative_delta,r.horizon_delta,r.generation_ordinal,"
        "r.structural_rank,r.semantic_configuration_canonical,r.semantic_hash,"
        "r.invocation_configuration_canonical,r.policy_canonical,r.duplicate_type,"
        "s.status AS scan_status,e.status AS source_experiment_status,"
        "e.phase AS source_experiment_phase,e.last_model_id AS current_source_model_id,"
        "a.analysis_id AS current_source_analysis_id,"
        "a.analysis_status AS current_source_analysis_status,"
        "a.analysis_scope AS current_source_analysis_scope "
        "FROM experiment_recommendation r "
        "JOIN experiment_recommendation_scan s ON s.recommendation_scan_id=r.recommendation_scan_id "
        "JOIN experiment e ON e.experiment_id=r.source_experiment_id "
        "LEFT JOIN experiment_analysis_result a ON a.experiment_id=e.experiment_id "
        "AND a.model_id=e.last_model_id AND COALESCE(a.analysis_scope,'final')='final' "
        "WHERE r.status='proposed' "
        "AND ($1::bigint IS NULL OR r.recommendation_scan_id=$1) "
        "AND ($2::bigint IS NULL OR r.recommendation_id=$2) "
        "ORDER BY r.recommendation_id ASC LIMIT $3;",
        pqxx::params{filters.recommendationScanId,
                     filters.recommendationId, filters.limit});
    std::vector<RecommendationEvaluationLoadResult> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        RecommendationEvaluationInput input = MapEvaluationInput(transaction, row);
        values.push_back({input.scoringInput.recommendationId, std::move(input)});
    }
    return values;
}

RecommendationEvaluationRunBeginResult BeginOrFindRecommendationEvaluationRun(
    pqxx::connection& connection,
    const RecommendationEvaluationRunRequest& request)
{
    ValidateFilters(request.filters);
    if (request.runIdentityCanonical.empty() || request.runIdentityHash.empty() ||
        request.evidenceSnapshotCanonical.empty() || request.evidenceSnapshotHash.empty())
        throw std::invalid_argument("invalid_recommendation_evaluation_run_request");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::result inserted = transaction.exec(
        "INSERT INTO experiment_recommendation_evaluation_run (status,"
        "evaluation_run_identity_canonical,evaluation_run_identity_hash,"
        "evaluation_policy_canonical,evaluation_policy_hash,evaluation_version,"
        "evaluator_version,scoring_policy_canonical,scoring_policy_hash,"
        "scoring_version,recommendation_scan_filter,recommendation_id_filter,"
        "requested_limit,evidence_snapshot_canonical,evidence_snapshot_hash) "
        "VALUES ('running',$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) "
        "ON CONFLICT (evaluation_run_identity_canonical) DO NOTHING "
        "RETURNING recommendation_evaluation_run_id,status;",
        pqxx::params{request.runIdentityCanonical, request.runIdentityHash,
            RecommendationEvaluationPolicyCanonicalText(request.policy),
            RecommendationEvaluationPolicyHash(request.policy),
            request.policy.evaluationVersion, request.policy.evaluatorVersion,
            RecommendationScoringPolicyCanonicalText(request.policy.scoringPolicy),
            RecommendationScoringPolicyHash(request.policy.scoringPolicy),
            request.policy.scoringPolicy.scoringVersion,
            request.filters.recommendationScanId,
            request.filters.recommendationId, request.filters.limit,
            request.evidenceSnapshotCanonical, request.evidenceSnapshotHash});
    RecommendationEvaluationRunBeginResult result;
    if (!inserted.empty())
    {
        result.evaluationRunId = inserted.one_row()[0].as<long long>();
        result.status = inserted.one_row()[1].as<std::string>();
        result.created = true;
    }
    else
    {
        const pqxx::row row = transaction.exec(
            "SELECT recommendation_evaluation_run_id,status,"
            "evaluation_run_identity_hash,evidence_snapshot_canonical "
            "FROM experiment_recommendation_evaluation_run "
            "WHERE evaluation_run_identity_canonical=$1;",
            pqxx::params{request.runIdentityCanonical}).one_row();
        if (row["evaluation_run_identity_hash"].as<std::string>() !=
                request.runIdentityHash ||
            row["evidence_snapshot_canonical"].as<std::string>() !=
                request.evidenceSnapshotCanonical)
            throw std::runtime_error("recommendation_evaluation_run_retry_mismatch");
        result.evaluationRunId = row["recommendation_evaluation_run_id"].as<long long>();
        result.status = row["status"].as<std::string>();
    }
    transaction.commit();
    return result;
}

RecommendationEvaluationPersistResult PersistRecommendationEvaluation(
    pqxx::connection& connection,
    const RecommendationEvaluationPersistenceRequest& request)
{
    if (request.evaluationRunId <= 0 || request.result.recommendationId <= 0 ||
        request.result.rankingOrdinal <= 0 ||
        request.result.recommendationId != request.input.scoringInput.recommendationId)
        throw std::invalid_argument("invalid_recommendation_evaluation_persistence_request");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const RecommendationEvaluationResult& value = request.result;
    const RecommendationEvaluationInput& input = request.input;
    const pqxx::result identity = transaction.exec(
        "SELECT semantic_hash,policy_hash FROM experiment_recommendation "
        "WHERE recommendation_id=$1;",
        pqxx::params{value.recommendationId});
    if (identity.empty())
        throw std::runtime_error("recommendation_evaluation_recommendation_not_found");
    const std::string persistedSemanticHash =
        identity.one_row()["semantic_hash"].as<std::string>();
    if (input.recommendationSemanticHash != persistedSemanticHash ||
        value.recommendationSemanticHash != persistedSemanticHash)
        throw std::runtime_error(
            "recommendation_evaluation_semantic_identity_mismatch");
    const pqxx::result runStatusRows = transaction.exec(
        "SELECT status FROM experiment_recommendation_evaluation_run "
        "WHERE recommendation_evaluation_run_id=$1 FOR SHARE;",
        pqxx::params{request.evaluationRunId});
    if (runStatusRows.empty())
        throw std::runtime_error("recommendation_evaluation_run_not_found");
    const std::string runStatus = runStatusRows.one_row()[0].as<std::string>();
    const pqxx::result inserted = transaction.exec(
        "INSERT INTO experiment_recommendation_evaluation_result ("
        "recommendation_evaluation_run_id,recommendation_id,"
        "evaluation_identity_canonical,evaluation_identity_hash,"
        "recommendation_semantic_canonical,recommendation_semantic_hash,"
        "recommendation_policy_canonical,recommendation_policy_hash,"
        "recommendation_scan_id,source_experiment_id,source_model_id,"
        "source_analysis_id,evidence_canonical,evidence_hash,eligibility,"
        "disposition,reason_code,explanation,final_score,raw_positive_score,"
        "raw_penalty_score,raw_total_score,component_count,missing_evidence_count,"
        "ranking_ordinal) SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,"
        "$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25 "
        "FROM experiment_recommendation_evaluation_run run_guard "
        "WHERE run_guard.recommendation_evaluation_run_id=$1 "
        "AND run_guard.status='running' "
        "ON CONFLICT (recommendation_evaluation_run_id,recommendation_id) "
        "DO NOTHING RETURNING recommendation_evaluation_result_id;",
        pqxx::params{request.evaluationRunId, value.recommendationId,
            value.evaluationIdentityCanonical, value.evaluationIdentityHash,
            input.scoringInput.semanticCanonicalText,
            persistedSemanticHash,
            input.scoringInput.recommendationPolicyCanonicalText,
            identity.one_row()["policy_hash"].as<std::string>(),
            input.recommendationScanId, input.scoringInput.sourceExperimentId,
            input.sourceModelId, input.sourceAnalysisId,
            value.evidenceCanonical, value.evidenceHash,
            RecommendationEligibilityText(value.eligibility),
            RecommendationEvaluationDispositionText(value.disposition),
            value.reasonCode, value.explanation, value.finalScore,
            value.rawPositiveScore, value.rawPenaltyScore, value.rawTotalScore,
            static_cast<int>(value.components.size()), value.missingEvidenceCount,
            value.rankingOrdinal});
    RecommendationEvaluationPersistResult result;
    if (!inserted.empty())
    {
        result.evaluationResultId = inserted.one_row()[0].as<long long>();
        result.created = true;
        int ordinal = 0;
        for (const RecommendationScoreComponent& component : value.components)
        {
            transaction.exec(
                "INSERT INTO experiment_recommendation_evaluation_component ("
                "recommendation_evaluation_result_id,component_ordinal,"
                "component_name,reason_code,input_canonical,normalized_value,"
                "weight,weighted_contribution,is_penalty,is_missing,explanation) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,false,$10);",
                pqxx::params{result.evaluationResultId, ++ordinal,
                    component.componentName, component.reasonCode,
                    component.inputCanonical, component.normalizedValue,
                    component.weight, component.weightedContribution,
                    component.penalty, component.explanation});
        }
    }
    else
    {
        const pqxx::result existing = transaction.exec(
            "SELECT recommendation_evaluation_result_id,"
            "evaluation_identity_canonical,evaluation_identity_hash,"
            "recommendation_semantic_canonical,recommendation_semantic_hash,"
            "recommendation_policy_canonical,recommendation_policy_hash,"
            "recommendation_scan_id,source_experiment_id,source_model_id,"
            "source_analysis_id,"
            "evidence_canonical,evidence_hash,eligibility,disposition,reason_code,"
            "explanation,final_score,raw_positive_score,raw_penalty_score,"
            "raw_total_score,component_count,missing_evidence_count,ranking_ordinal "
            "FROM experiment_recommendation_evaluation_result "
            "WHERE recommendation_evaluation_run_id=$1 AND recommendation_id=$2;",
            pqxx::params{request.evaluationRunId, value.recommendationId});
        if (existing.empty())
        {
            if (runStatus != "running")
                throw std::runtime_error(
                    "recommendation_evaluation_run_not_running");
            throw std::runtime_error(
                "recommendation_evaluation_conflict_without_row");
        }
        const pqxx::row row = existing.one_row();
        const bool equal =
            row["evaluation_identity_canonical"].as<std::string>() == value.evaluationIdentityCanonical &&
            row["evaluation_identity_hash"].as<std::string>() == value.evaluationIdentityHash &&
            row["recommendation_semantic_canonical"].as<std::string>() == input.scoringInput.semanticCanonicalText &&
            row["recommendation_semantic_hash"].as<std::string>() == persistedSemanticHash &&
            row["recommendation_policy_canonical"].as<std::string>() == input.scoringInput.recommendationPolicyCanonicalText &&
            row["recommendation_policy_hash"].as<std::string>() == identity.one_row()["policy_hash"].as<std::string>() &&
            row["recommendation_scan_id"].as<long long>() == input.recommendationScanId &&
            row["source_experiment_id"].as<long long>() == input.scoringInput.sourceExperimentId &&
            OptionalValue<long long>(row, "source_model_id") == input.sourceModelId &&
            OptionalValue<long long>(row, "source_analysis_id") == input.sourceAnalysisId &&
            row["evidence_canonical"].as<std::string>() == value.evidenceCanonical &&
            row["evidence_hash"].as<std::string>() == value.evidenceHash &&
            row["eligibility"].as<std::string>() == RecommendationEligibilityText(value.eligibility) &&
            row["disposition"].as<std::string>() == RecommendationEvaluationDispositionText(value.disposition) &&
            row["reason_code"].as<std::string>() == value.reasonCode &&
            row["explanation"].as<std::string>() == value.explanation &&
            OptionalValue<double>(row, "final_score") == value.finalScore &&
            OptionalValue<double>(row, "raw_positive_score") == value.rawPositiveScore &&
            OptionalValue<double>(row, "raw_penalty_score") == value.rawPenaltyScore &&
            OptionalValue<double>(row, "raw_total_score") == value.rawTotalScore &&
            row["component_count"].as<int>() == static_cast<int>(value.components.size()) &&
            row["missing_evidence_count"].as<int>() == value.missingEvidenceCount &&
            row["ranking_ordinal"].as<int>() == value.rankingOrdinal;
        result.evaluationResultId = row["recommendation_evaluation_result_id"].as<long long>();
        const pqxx::result components = transaction.exec(
            "SELECT component_ordinal,component_name,reason_code,input_canonical,"
            "normalized_value,weight,weighted_contribution,is_penalty,is_missing,"
            "explanation FROM experiment_recommendation_evaluation_component "
            "WHERE recommendation_evaluation_result_id=$1 ORDER BY component_ordinal;",
            pqxx::params{result.evaluationResultId});
        const int componentCount = static_cast<int>(value.components.size());
        bool componentsEqual = components.size() == componentCount;
        for (int index = 0; componentsEqual && index < components.size(); ++index)
            componentsEqual = ComponentEqual(
                components[index], value.components[static_cast<std::size_t>(index)],
                index + 1);
        if (!equal || !componentsEqual)
            throw std::runtime_error("recommendation_evaluation_retry_mismatch");
    }
    transaction.commit();
    return result;
}

void CompleteRecommendationEvaluationRun(
    pqxx::connection& connection,
    long long evaluationRunId,
    const RecommendationEvaluationRunCounters& counters)
{
    if (evaluationRunId <= 0)
        throw std::invalid_argument("recommendation_evaluation_run_id_invalid");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::result updated = transaction.exec(
        "UPDATE experiment_recommendation_evaluation_run SET status='completed',"
        "completed_at=now(),error_message=NULL,updated_at=now(),"
        "recommendations_considered=$2,recommendations_evaluated=$3,"
        "recommendations_eligible=$4,recommendations_blocked=$5,evaluation_errors=$6 "
        "WHERE recommendation_evaluation_run_id=$1 AND status='running';",
        pqxx::params{evaluationRunId, counters.recommendationsConsidered,
            counters.recommendationsEvaluated, counters.recommendationsEligible,
            counters.recommendationsBlocked, counters.evaluationErrors});
    if (updated.affected_rows() == 0)
    {
        const pqxx::row row = transaction.exec(
            "SELECT status,recommendations_considered,recommendations_evaluated,"
            "recommendations_eligible,recommendations_blocked,evaluation_errors "
            "FROM experiment_recommendation_evaluation_run "
            "WHERE recommendation_evaluation_run_id=$1;",
            pqxx::params{evaluationRunId}).one_row();
        if (row["status"].as<std::string>() != "completed" ||
            row[1].as<int>() != counters.recommendationsConsidered ||
            row[2].as<int>() != counters.recommendationsEvaluated ||
            row[3].as<int>() != counters.recommendationsEligible ||
            row[4].as<int>() != counters.recommendationsBlocked ||
            row[5].as<int>() != counters.evaluationErrors)
            throw std::runtime_error("recommendation_evaluation_run_not_running");
    }
    transaction.commit();
}

void FailRecommendationEvaluationRun(
    pqxx::connection& connection,
    long long evaluationRunId,
    const RecommendationEvaluationRunCounters& counters,
    const std::string& errorMessage)
{
    if (evaluationRunId <= 0)
        throw std::invalid_argument("recommendation_evaluation_run_id_invalid");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const std::string persisted = errorMessage.empty()
        ? "unknown_recommendation_evaluation_failure" : errorMessage;
    const pqxx::result updated = transaction.exec(
        "UPDATE experiment_recommendation_evaluation_run SET status='failed',"
        "completed_at=now(),error_message=$2,updated_at=now(),"
        "recommendations_considered=$3,recommendations_evaluated=$4,"
        "recommendations_eligible=$5,recommendations_blocked=$6,evaluation_errors=$7 "
        "WHERE recommendation_evaluation_run_id=$1 AND status='running';",
        pqxx::params{evaluationRunId, persisted, counters.recommendationsConsidered,
            counters.recommendationsEvaluated, counters.recommendationsEligible,
            counters.recommendationsBlocked, counters.evaluationErrors});
    if (updated.affected_rows() == 0)
        throw std::runtime_error("recommendation_evaluation_run_not_running");
    transaction.commit();
}

std::vector<PersistedRecommendationEvaluationSummary>
ListRecommendationEvaluations(
    pqxx::connection& connection,
    const RecommendationEvaluationFilters& filters)
{
    ValidateFilters(filters);
    pqxx::read_transaction transaction{connection};
    const std::optional<std::string> disposition = filters.disposition
        ? std::optional<std::string>{RecommendationEvaluationDispositionText(
              *filters.disposition)} : std::nullopt;
    const pqxx::result rows = transaction.exec(
        "SELECT " + SummaryColumns() +
        " FROM experiment_recommendation_evaluation_result er "
        "JOIN experiment_recommendation_evaluation_run run ON run.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment_recommendation r ON r.recommendation_id=er.recommendation_id "
        "WHERE ($1::bigint IS NULL OR er.recommendation_scan_id=$1) "
        "AND ($2::bigint IS NULL OR er.recommendation_id=$2) "
        "AND ($3::text IS NULL OR er.disposition=$3) "
        "ORDER BY er.recommendation_evaluation_run_id DESC,er.ranking_ordinal ASC "
        "LIMIT $4;",
        pqxx::params{filters.recommendationScanId, filters.recommendationId,
                     disposition, filters.limit});
    std::vector<PersistedRecommendationEvaluationSummary> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows) values.push_back(MapSummary(row));
    return values;
}

std::optional<PersistedRecommendationEvaluationDetail>
FindRecommendationEvaluation(
    pqxx::connection& connection,
    long long evaluationResultId)
{
    if (evaluationResultId <= 0)
        throw std::invalid_argument("recommendation_evaluation_result_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + SummaryColumns() +
        ",er.source_model_id,er.source_analysis_id,"
        "er.evaluation_identity_canonical,er.recommendation_semantic_canonical,"
        "er.recommendation_policy_canonical,"
        "er.recommendation_policy_hash,er.evidence_canonical,er.evidence_hash,"
        "er.raw_positive_score,er.raw_penalty_score,er.raw_total_score "
        "FROM experiment_recommendation_evaluation_result er "
        "JOIN experiment_recommendation_evaluation_run run ON run.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment_recommendation r ON r.recommendation_id=er.recommendation_id "
        "WHERE er.recommendation_evaluation_result_id=$1;",
        pqxx::params{evaluationResultId});
    if (rows.empty()) return std::nullopt;
    PersistedRecommendationEvaluationDetail detail;
    static_cast<PersistedRecommendationEvaluationSummary&>(detail) =
        MapSummary(rows.one_row());
    const pqxx::row row = rows.one_row();
    detail.sourceModelId = OptionalValue<long long>(row, "source_model_id");
    detail.sourceAnalysisId = OptionalValue<long long>(row, "source_analysis_id");
    detail.evaluationIdentityCanonical = row["evaluation_identity_canonical"].as<std::string>();
    detail.recommendationSemanticCanonical = row["recommendation_semantic_canonical"].as<std::string>();
    detail.recommendationPolicyCanonical = row["recommendation_policy_canonical"].as<std::string>();
    detail.recommendationPolicyHash = row["recommendation_policy_hash"].as<std::string>();
    detail.evidenceCanonical = row["evidence_canonical"].as<std::string>();
    detail.evidenceHash = row["evidence_hash"].as<std::string>();
    detail.rawPositiveScore = OptionalValue<double>(row, "raw_positive_score");
    detail.rawPenaltyScore = OptionalValue<double>(row, "raw_penalty_score");
    detail.rawTotalScore = OptionalValue<double>(row, "raw_total_score");
    const pqxx::result components = transaction.exec(
        "SELECT component_name,reason_code,input_canonical,normalized_value,"
        "weight,weighted_contribution,is_penalty,explanation "
        "FROM experiment_recommendation_evaluation_component "
        "WHERE recommendation_evaluation_result_id=$1 ORDER BY component_ordinal;",
        pqxx::params{evaluationResultId});
    for (const pqxx::row& component : components)
        detail.components.push_back({component["component_name"].as<std::string>(),
            component["reason_code"].as<std::string>(),
            component["input_canonical"].as<std::string>(),
            component["normalized_value"].as<double>(),
            component["weight"].as<double>(),
            component["weighted_contribution"].as<double>(),
            component["is_penalty"].as<bool>(),
            component["explanation"].as<std::string>()});
    return detail;
}

std::vector<PersistedRecommendationEvaluationRun>
ListRecommendationEvaluationRuns(pqxx::connection& connection, int limit)
{
    if (limit <= 0 || limit > kMaximumEvaluationLimit)
        throw std::invalid_argument("recommendation_evaluation_limit_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + RunColumns() +
        " FROM experiment_recommendation_evaluation_run "
        "ORDER BY recommendation_evaluation_run_id DESC LIMIT $1;",
        pqxx::params{limit});
    std::vector<PersistedRecommendationEvaluationRun> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows) values.push_back(MapRun(row));
    return values;
}

std::optional<PersistedRecommendationEvaluationRun>
FindRecommendationEvaluationRun(
    pqxx::connection& connection,
    long long evaluationRunId)
{
    if (evaluationRunId <= 0)
        throw std::invalid_argument("recommendation_evaluation_run_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + RunColumns() +
        " FROM experiment_recommendation_evaluation_run "
        "WHERE recommendation_evaluation_run_id=$1;",
        pqxx::params{evaluationRunId});
    if (rows.empty()) return std::nullopt;
    return MapRun(rows.one_row());
}

} // namespace EA::ExperimentRecommendation
