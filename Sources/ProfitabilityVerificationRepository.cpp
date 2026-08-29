#include "ProfitabilityVerificationRepository.hpp"

#include "ExperimentRecommendationRanking.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace EA::ProfitabilityVerification
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row,
                                   const std::string& column)
{
    return row[column].is_null()
        ? std::nullopt
        : std::optional<Value>{row[column].as<Value>()};
}

std::optional<ExperimentRecommendation::RecommendationSource::
    FinalProfitabilityEvidence> MapFrozenEvidence(
        const pqxx::row& row, const std::string& prefix)
{
    const auto version = OptionalValue<int>(
        row, prefix + "final_profitability_provenance_version");
    if (!version) return std::nullopt;
    ExperimentRecommendation::RecommendationSource::FinalProfitabilityEvidence
        evidence;
    evidence.provenanceVersion = *version;
    evidence.finalInferenceEvalResultId = OptionalValue<long long>(
        row, prefix + "source_final_inference_eval_result_id");
    evidence.profitabilityObservationId = OptionalValue<long long>(
        row, prefix + "source_final_profitability_observation_id");
    evidence.unavailableReason = OptionalValue<std::string>(
        row, prefix + "source_final_profitability_unavailable_reason")
            .value_or("");
    evidence.inferenceScope = row[
        prefix + "source_final_profitability_inference_scope"].as<std::string>();
    evidence.inferenceStart = OptionalValue<std::string>(
        row, prefix + "source_final_profitability_inference_start");
    evidence.inferenceEnd = OptionalValue<std::string>(
        row, prefix + "source_final_profitability_inference_end");
    evidence.actionablePredictionCount = OptionalValue<long long>(
        row, prefix + "source_final_profitability_actionable_count");
    evidence.aggregateTerminalHorizonLogReturnSum = OptionalValue<double>(
        row, prefix + "source_final_profitability_aggregate_return");
    evidence.averageTerminalHorizonLogReturnPerActionablePrediction =
        OptionalValue<double>(
            row, prefix + "source_final_profitability_average_return");
    evidence.metricDefinitionHash = OptionalValue<std::string>(
        row, prefix + "source_final_profitability_metric_definition_hash");
    evidence.sourceContentHash = OptionalValue<std::string>(
        row, prefix + "source_final_profitability_source_content_hash");
    evidence.observationIdentityHash = OptionalValue<std::string>(
        row, prefix + "source_final_profitability_observation_identity_hash");
    if (const auto error = ExperimentRecommendation::
            ValidateRecommendationFinalProfitabilityEvidence(evidence))
        throw std::runtime_error(*error);
    return evidence;
}

std::string RankingMembershipCanonical(
    std::vector<std::pair<std::string, long long>> identities)
{
    std::sort(identities.begin(), identities.end());
    std::ostringstream output;
    output << "experiment_recommendation_ranking_membership_v1;count="
           << identities.size();
    for (std::size_t index = 0; index < identities.size(); ++index)
        output << ";member[" << index << "].evaluation_identity="
               << identities[index].first.size() << ':'
               << identities[index].first
               << ";member[" << index << "].evaluation_result_id="
               << identities[index].second;
    return output.str();
}

EvidenceResult StateOnly(long long experimentId,
                         const std::optional<long long>& modelId,
                         const std::optional<long long>& inferenceResultId,
                         const std::string& inferenceStart,
                         const std::string& inferenceEnd,
                         EvidenceState state,
                         const std::string& reason)
{
    ExpectedFinalEvidence expected;
    expected.experimentId = experimentId;
    expected.modelId = modelId.value_or(-1);
    expected.inferenceEvalResultId = inferenceResultId.value_or(-1);
    expected.inferenceStart = inferenceStart;
    expected.inferenceEnd = inferenceEnd;
    EvidenceResult result = ValidateExactFinalObservation(expected, std::nullopt);
    result.state = state;
    result.reason = reason;

    // Rebuild the deterministic identity with the requested state/reason by
    // validating an intentionally incomplete expectation, then binding the
    // public diagnostic fields below. State-only results never claim a valid
    // observation identity.
    std::string canonical = "profitability_exact_final_evidence_state_v1;";
    canonical += "experiment_id=" + std::to_string(experimentId) + ";";
    canonical += "model_id=" +
        (modelId ? std::to_string(*modelId) : "NULL") + ";";
    canonical += "inference_eval_result_id=" +
        (inferenceResultId ? std::to_string(*inferenceResultId) : "NULL") + ";";
    canonical += "inference_start=" +
        (inferenceStart.empty() ? "NULL" : inferenceStart) + ";";
    canonical += "inference_end=" +
        (inferenceEnd.empty() ? "NULL" : inferenceEnd) + ";";
    canonical += "state=" + EvidenceStateText(state) + ";reason=" + reason;
    result.evidenceIdentityCanonical = std::move(canonical);
    result.evidenceIdentityHash = InferenceProfitability::DeterministicHash(
        result.evidenceIdentityCanonical);
    result.experimentId = experimentId;
    result.finalModelId = modelId;
    result.finalInferenceEvalResultId = inferenceResultId;
    return result;
}

} // namespace

EvidenceResult LoadAndVerifyExactFinalEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    if (experimentId <= 0)
        throw std::invalid_argument("invalid_profitability_experiment_id");

    const pqxx::result experiments = transaction.exec(R"SQL(
SELECT e.experiment_id,e.last_model_id,
       e.infer_start::date::text AS inference_start,
       e.infer_end::date::text AS inference_end
FROM experiment e
WHERE e.experiment_id=$1
)SQL", pqxx::params{experimentId});
    if (experiments.empty())
        return StateOnly(experimentId, std::nullopt, std::nullopt, {}, {},
                         EvidenceState::incomplete, "experiment_not_found");
    if (experiments.size() != 1)
        return StateOnly(experimentId, std::nullopt, std::nullopt, {}, {},
                         EvidenceState::ambiguous,
                         "ambiguous_experiment_identity");

    const pqxx::row row = experiments.one_row();
    const std::optional<long long> modelId = row["last_model_id"].is_null()
        ? std::nullopt
        : std::optional<long long>{row["last_model_id"].as<long long>()};
    const std::string inferenceStart = row["inference_start"].is_null()
        ? "" : row["inference_start"].as<std::string>();
    const std::string inferenceEnd = row["inference_end"].is_null()
        ? "" : row["inference_end"].as<std::string>();
    if (!modelId || inferenceStart.empty() || inferenceEnd.empty())
        return StateOnly(experimentId, modelId, std::nullopt,
                         inferenceStart, inferenceEnd,
                         EvidenceState::incomplete,
                         "final_inference_context_incomplete");

    const auto finalResult =
        InferenceProfitability::ResolveExactFinalInferenceResult(
            transaction, experimentId, *modelId);
    using FinalStatus =
        InferenceProfitability::ExactFinalInferenceResultStatus;
    if (finalResult.status != FinalStatus::available)
    {
        const EvidenceState state =
            finalResult.status == FinalStatus::ambiguousFinalInferenceResult
                ? EvidenceState::ambiguous
            : finalResult.status == FinalStatus::finalInferenceContextMismatch
                ? EvidenceState::invalidProvenance
                : EvidenceState::incomplete;
        return StateOnly(
            experimentId, modelId, finalResult.inferenceEvalResultId,
            inferenceStart, inferenceEnd, state,
            InferenceProfitability::ExactFinalInferenceResultStatusText(
                finalResult.status));
    }

    if (!InferenceProfitability::SchemaExists(transaction))
        throw std::runtime_error("profitability_schema_unavailable");

    InferenceProfitability::AuthoritativeObservationSelector selector;
    selector.experimentId = experimentId;
    selector.modelId = *modelId;
    selector.inferenceEvalResultId = *finalResult.inferenceEvalResultId;
    selector.scope = InferenceProfitability::Scope::finalInference;
    selector.checkpointEvalId.reset();
    selector.metricDefinitionCanonical =
        InferenceProfitability::kMetricDefinitionCanonical;
    selector.metricDefinitionHash =
        InferenceProfitability::MetricDefinitionHash();
    const auto selected =
        InferenceProfitability::SelectAuthoritativeObservation(
            transaction, selector);
    using ObservationStatus =
        InferenceProfitability::AuthoritativeObservationStatus;
    if (selected.status != ObservationStatus::available)
    {
        const EvidenceState state =
            selected.status == ObservationStatus::noObservation
                ? EvidenceState::unavailable
            : selected.status == ObservationStatus::ambiguousObservation
                ? EvidenceState::ambiguous
            : selected.status == ObservationStatus::metricDefinitionMismatch
                ? EvidenceState::invalidMetricDefinition
                : EvidenceState::invalidProvenance;
        return StateOnly(
            experimentId, modelId, finalResult.inferenceEvalResultId,
            inferenceStart, inferenceEnd, state,
            InferenceProfitability::AuthoritativeObservationStatusText(
                selected.status));
    }

    ExpectedFinalEvidence expected;
    expected.experimentId = experimentId;
    expected.modelId = *modelId;
    expected.inferenceEvalResultId = *finalResult.inferenceEvalResultId;
    expected.inferenceStart = inferenceStart;
    expected.inferenceEnd = inferenceEnd;
    return ValidateExactFinalObservation(expected, selected.observation);
}

CampaignProfitabilityShadowSource LoadCampaignProfitabilityShadowSource(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId)
{
    if (rankingSnapshotId <= 0)
        throw std::invalid_argument("invalid_campaign_shadow_snapshot_id");
    const pqxx::result snapshots = transaction.exec(R"SQL(
SELECT snapshot.status,snapshot.ranking_snapshot_identity_canonical,
       snapshot.ranking_snapshot_identity_hash,
       snapshot.ranking_policy_canonical,snapshot.ranking_policy_hash,
       snapshot.ranking_version,snapshot.scope_type,snapshot.scope_canonical,
       snapshot.scope_hash,snapshot.evaluation_run_filter,
       snapshot.requested_limit,snapshot.source_membership_canonical,
       snapshot.source_membership_hash,
       snapshot.ranking_snapshot_identity_version,
       snapshot.member_count,snapshot.advisory_ready_count,
       snapshot.blocked_count,snapshot.non_actionable_count,
       snapshot.population_semantic_state,
       snapshot.scoring_semantic_canonical,snapshot.scoring_semantic_hash,
       snapshot.scoring_semantic_version,
       snapshot.evaluation_semantic_canonical,
       snapshot.evaluation_semantic_hash,
       snapshot.evaluation_semantic_version,
       snapshot.distinct_scoring_semantic_count,
       snapshot.distinct_evaluation_semantic_count,
       snapshot.homogeneity_validation_result,
       run.status AS evaluation_run_status,
       run.evaluation_run_identity_canonical,
       run.evaluation_run_identity_hash,
       run.recommendations_evaluated,run.recommendations_eligible,
       run.recommendations_blocked,run.evaluation_errors
FROM experiment_recommendation_ranking_snapshot snapshot
JOIN experiment_recommendation_evaluation_run run
  ON run.recommendation_evaluation_run_id=snapshot.evaluation_run_filter
WHERE snapshot.recommendation_ranking_snapshot_id=$1
)SQL", pqxx::params{rankingSnapshotId});
    if (snapshots.empty())
        throw std::runtime_error("campaign_shadow_snapshot_not_found");
    const pqxx::row snapshot = snapshots.one_row();
    CampaignProfitabilityShadowSource source;
    source.controlSnapshotId = rankingSnapshotId;
    source.controlSnapshotIdentityCanonical =
        snapshot["ranking_snapshot_identity_canonical"].as<std::string>();
    source.controlSnapshotIdentityHash =
        snapshot["ranking_snapshot_identity_hash"].as<std::string>();
    source.controlRankingPolicyCanonical =
        snapshot["ranking_policy_canonical"].as<std::string>();
    source.controlRankingPolicyHash =
        snapshot["ranking_policy_hash"].as<std::string>();
    source.sourceEvaluationRunId =
        snapshot["evaluation_run_filter"].as<long long>();
    source.persistedMemberCount = snapshot["member_count"].as<int>();
    const ExperimentRecommendation::RecommendationRankingPolicy controlPolicy;
    const std::string expectedPolicy = ExperimentRecommendation::
        RecommendationRankingPolicyCanonicalText(controlPolicy);
    ExperimentRecommendation::RecommendationRankingScope scope;
    scope.type = ExperimentRecommendation::
        RecommendationRankingScopeType::evaluationRun;
    scope.evaluationRunId = source.sourceEvaluationRunId;
    const std::string membership =
        snapshot["source_membership_canonical"].as<std::string>();
    ExperimentRecommendation::RecommendationRankingPopulationSemanticValidation
        semantics;
    semantics.state = ExperimentRecommendation::
        RecommendationRankingPopulationSemanticState::verifiedHomogeneous;
    semantics.scoringIdentity =
        ExperimentRecommendation::RecommendationScoringSemanticIdentity{
            snapshot["scoring_semantic_canonical"].as<std::string>(),
            snapshot["scoring_semantic_hash"].as<std::string>(),
            snapshot["scoring_semantic_version"].as<int>()};
    semantics.evaluationIdentity =
        ExperimentRecommendation::RecommendationEvaluationSemanticIdentity{
            snapshot["evaluation_semantic_canonical"].as<std::string>(),
            snapshot["evaluation_semantic_hash"].as<std::string>(),
            snapshot["evaluation_semantic_version"].as<int>()};
    semantics.distinctScoringIdentityCount =
        snapshot["distinct_scoring_semantic_count"].as<int>();
    semantics.distinctEvaluationIdentityCount =
        snapshot["distinct_evaluation_semantic_count"].as<int>();
    const int requestedLimit = snapshot["requested_limit"].as<int>();
    const std::string expectedSnapshotIdentity = ExperimentRecommendation::
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            controlPolicy, scope, requestedLimit, membership, semantics);
    if (snapshot["status"].as<std::string>() != "completed" ||
        source.sourceEvaluationRunId <= 0 || source.persistedMemberCount <= 0 ||
        snapshot["ranking_version"].as<int>() != 1 ||
        snapshot["scope_type"].as<std::string>() != "evaluation_run" ||
        snapshot["scope_canonical"].as<std::string>() !=
            ExperimentRecommendation::RecommendationRankingScopeCanonicalText(
                scope) ||
        snapshot["scope_hash"].as<std::string>() !=
            ExperimentRecommendation::RecommendationRankingCanonicalHash(
                snapshot["scope_canonical"].as<std::string>()) ||
        requestedLimit < source.persistedMemberCount ||
        snapshot["ranking_snapshot_identity_version"].as<int>() != 2 ||
        snapshot["advisory_ready_count"].as<int>() !=
            source.persistedMemberCount ||
        snapshot["blocked_count"].as<int>() != 0 ||
        snapshot["non_actionable_count"].as<int>() != 0 ||
        snapshot["population_semantic_state"].as<std::string>() !=
            "verified_homogeneous" ||
        snapshot["homogeneity_validation_result"].as<std::string>() !=
            "verified_homogeneous" ||
        source.controlRankingPolicyCanonical != expectedPolicy ||
        source.controlRankingPolicyHash != ExperimentRecommendation::
            RecommendationRankingCanonicalHash(expectedPolicy) ||
        snapshot["source_membership_hash"].as<std::string>() !=
            ExperimentRecommendation::RecommendationRankingCanonicalHash(
                membership) ||
        semantics.distinctScoringIdentityCount != 1 ||
        semantics.distinctEvaluationIdentityCount != 1 ||
        semantics.scoringIdentity->version != 1 ||
        semantics.evaluationIdentity->version != 1 ||
        semantics.scoringIdentity->hash != ExperimentRecommendation::
            RecommendationRankingCanonicalHash(
                semantics.scoringIdentity->canonical) ||
        semantics.evaluationIdentity->hash != ExperimentRecommendation::
            RecommendationRankingCanonicalHash(
                semantics.evaluationIdentity->canonical) ||
        snapshot["evaluation_run_status"].as<std::string>() != "completed" ||
        ExperimentRecommendation::RecommendationCanonicalHash(
            snapshot["evaluation_run_identity_canonical"].as<std::string>()) !=
            snapshot["evaluation_run_identity_hash"].as<std::string>() ||
        snapshot["recommendations_evaluated"].as<int>() !=
            source.persistedMemberCount ||
        snapshot["recommendations_eligible"].as<int>() !=
            source.persistedMemberCount ||
        snapshot["recommendations_blocked"].as<int>() != 0 ||
        snapshot["evaluation_errors"].as<int>() != 0 ||
        source.controlSnapshotIdentityCanonical != expectedSnapshotIdentity ||
        ExperimentRecommendation::RecommendationCanonicalHash(
            source.controlSnapshotIdentityCanonical) !=
            source.controlSnapshotIdentityHash ||
        ExperimentRecommendation::RecommendationCanonicalHash(
            source.controlRankingPolicyCanonical) !=
            source.controlRankingPolicyHash)
        throw std::runtime_error("campaign_shadow_control_snapshot_invalid");

    const pqxx::result rows = transaction.exec(R"SQL(
SELECT rm.recommendation_ranking_member_id,rm.global_ordinal,rm.bucket,
       rm.final_score,rm.recommendation_id,rm.recommendation_evaluation_result_id,
       rm.source_experiment_id,rm.source_model_id,rm.symbol,rm.horizon,
       er.recommendation_id AS er_recommendation_id,
       er.evaluation_identity_canonical,er.evaluation_identity_hash,
       er.recommendation_evaluation_run_id AS er_evaluation_run_id,
       er.source_experiment_id AS er_source_experiment_id,
       er.source_model_id AS er_source_model_id,
       er.final_score AS er_final_score,er.eligibility,er.disposition,
       er.result_status,
       er.final_profitability_provenance_version AS e_final_profitability_provenance_version,
       er.source_final_inference_eval_result_id AS e_source_final_inference_eval_result_id,
       er.source_final_profitability_observation_id AS e_source_final_profitability_observation_id,
       er.source_final_profitability_unavailable_reason AS e_source_final_profitability_unavailable_reason,
       er.source_final_profitability_inference_scope AS e_source_final_profitability_inference_scope,
       er.source_final_profitability_inference_start AS e_source_final_profitability_inference_start,
       er.source_final_profitability_inference_end AS e_source_final_profitability_inference_end,
       er.source_final_profitability_actionable_count AS e_source_final_profitability_actionable_count,
       er.source_final_profitability_aggregate_return AS e_source_final_profitability_aggregate_return,
       er.source_final_profitability_average_return AS e_source_final_profitability_average_return,
       er.source_final_profitability_metric_definition_hash AS e_source_final_profitability_metric_definition_hash,
       er.source_final_profitability_source_content_hash AS e_source_final_profitability_source_content_hash,
       er.source_final_profitability_observation_identity_hash AS e_source_final_profitability_observation_identity_hash,
       r.source_experiment_id AS r_source_experiment_id,
       r.source_model_id AS r_source_model_id,
       r.source_symbol AS r_source_symbol,
       r.source_prediction_horizon AS r_source_prediction_horizon,
       r.final_profitability_provenance_version AS r_final_profitability_provenance_version,
       r.source_final_inference_eval_result_id AS r_source_final_inference_eval_result_id,
       r.source_final_profitability_observation_id AS r_source_final_profitability_observation_id,
       r.source_final_profitability_unavailable_reason AS r_source_final_profitability_unavailable_reason,
       r.source_final_profitability_inference_scope AS r_source_final_profitability_inference_scope,
       r.source_final_profitability_inference_start AS r_source_final_profitability_inference_start,
       r.source_final_profitability_inference_end AS r_source_final_profitability_inference_end,
       r.source_final_profitability_actionable_count AS r_source_final_profitability_actionable_count,
       r.source_final_profitability_aggregate_return AS r_source_final_profitability_aggregate_return,
       r.source_final_profitability_average_return AS r_source_final_profitability_average_return,
       r.source_final_profitability_metric_definition_hash AS r_source_final_profitability_metric_definition_hash,
       r.source_final_profitability_source_content_hash AS r_source_final_profitability_source_content_hash,
       r.source_final_profitability_observation_identity_hash AS r_source_final_profitability_observation_identity_hash
FROM experiment_recommendation_ranking_member rm
JOIN experiment_recommendation_evaluation_result er
  ON er.recommendation_evaluation_result_id=rm.recommendation_evaluation_result_id
JOIN experiment_recommendation r
  ON r.recommendation_id=rm.recommendation_id
WHERE rm.recommendation_ranking_snapshot_id=$1
ORDER BY rm.global_ordinal,rm.recommendation_ranking_member_id
)SQL", pqxx::params{rankingSnapshotId});
    if (rows.size() != source.persistedMemberCount)
        throw std::runtime_error("campaign_shadow_member_count_mismatch");
    source.candidates.reserve(rows.size());
    std::vector<std::pair<std::string, long long>> memberIdentities;
    memberIdentities.reserve(rows.size());
    int expectedRank = 0;
    for (const pqxx::row& row : rows)
    {
        ++expectedRank;
        const auto evaluationEvidence = MapFrozenEvidence(row, "e_");
        const auto recommendationEvidence = MapFrozenEvidence(row, "r_");
        if (!evaluationEvidence || !recommendationEvidence ||
            *evaluationEvidence != *recommendationEvidence ||
            row["global_ordinal"].as<int>() != expectedRank ||
            row["bucket"].as<std::string>() != "advisory_ready" ||
            row["eligibility"].as<std::string>() != "eligible" ||
            row["disposition"].as<std::string>() != "advisory_ready" ||
            row["result_status"].as<std::string>() != "evaluated" ||
            ExperimentRecommendation::RecommendationCanonicalHash(
                row["evaluation_identity_canonical"].as<std::string>()) !=
                row["evaluation_identity_hash"].as<std::string>() ||
            row["recommendation_id"].as<long long>() !=
                row["er_recommendation_id"].as<long long>() ||
            row["er_evaluation_run_id"].as<long long>() !=
                source.sourceEvaluationRunId ||
            row["source_experiment_id"].as<long long>() !=
                row["er_source_experiment_id"].as<long long>() ||
            row["source_experiment_id"].as<long long>() !=
                row["r_source_experiment_id"].as<long long>() ||
            OptionalValue<long long>(row, "source_model_id") !=
                OptionalValue<long long>(row, "er_source_model_id") ||
            OptionalValue<long long>(row, "source_model_id") !=
                OptionalValue<long long>(row, "r_source_model_id") ||
            row["symbol"].as<std::string>() !=
                row["r_source_symbol"].as<std::string>() ||
            row["horizon"].as<int>() !=
                row["r_source_prediction_horizon"].as<int>() ||
            row["final_score"].is_null() || row["er_final_score"].is_null() ||
            row["final_score"].as<double>() !=
                row["er_final_score"].as<double>())
            throw std::runtime_error("campaign_shadow_member_provenance_mismatch");

        memberIdentities.emplace_back(
            row["evaluation_identity_canonical"].as<std::string>(),
            row["recommendation_evaluation_result_id"].as<long long>());

        std::optional<InferenceProfitability::Observation> observation;
        if (evaluationEvidence->profitabilityObservationId)
            observation = InferenceProfitability::LoadObservationById(
                transaction, *evaluationEvidence->profitabilityObservationId);
        ShadowCandidate candidate;
        candidate.rankingMemberId = row[
            "recommendation_ranking_member_id"].as<long long>();
        candidate.recommendationId = row["recommendation_id"].as<long long>();
        candidate.recommendationEvaluationResultId = row[
            "recommendation_evaluation_result_id"].as<long long>();
        candidate.recommendationEvaluationRunId = source.sourceEvaluationRunId;
        candidate.sourceExperimentId =
            row["source_experiment_id"].as<long long>();
        candidate.sourceModelId = OptionalValue<long long>(row, "source_model_id");
        candidate.symbol = row["symbol"].as<std::string>();
        candidate.horizon = row["horizon"].as<int>();
        candidate.currentRank = expectedRank;
        candidate.currentScore = row["final_score"].as<double>();
        candidate.profitability = ValidateFrozenCampaignEvidence(
            candidate.sourceExperimentId, candidate.sourceModelId,
            *evaluationEvidence, observation);
        source.candidates.push_back(std::move(candidate));
    }
    if (RankingMembershipCanonical(std::move(memberIdentities)) != membership)
        throw std::runtime_error("campaign_shadow_source_membership_mismatch");
    return source;
}

CampaignProfitabilityCoverageAudit LoadCampaignProfitabilityCoverageAudit(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId)
{
    const CampaignProfitabilityShadowSource source =
        LoadCampaignProfitabilityShadowSource(transaction, rankingSnapshotId);
    const pqxx::result rows = transaction.exec(R"SQL(
SELECT rm.recommendation_ranking_member_id,
       er.final_profitability_provenance_version,
       er.source_final_inference_eval_result_id,
       er.source_final_profitability_observation_id,
       er.source_final_profitability_unavailable_reason,
       er.source_final_profitability_inference_scope,
       er.source_final_profitability_inference_start,
       er.source_final_profitability_inference_end,
       er.source_final_profitability_actionable_count,
       er.source_final_profitability_aggregate_return,
       er.source_final_profitability_average_return,
       er.source_final_profitability_metric_definition_hash,
       er.source_final_profitability_source_content_hash,
       er.source_final_profitability_observation_identity_hash,
       EXISTS (
           SELECT 1 FROM inference_eval_result ier
           WHERE ier.id=er.source_final_inference_eval_result_id
             AND ier.status='completed' AND ier.inference_scope='final'
             AND ier.checkpoint_eval_id IS NULL
             AND ier.parent_experiment_id IS NULL
       ) AS exact_final_result_exists,
       EXISTS (
           SELECT 1 FROM inference_eval_result ier
           WHERE ier.model_id=rm.source_model_id
             AND ier.status='completed' AND ier.inference_scope='final'
             AND ier.checkpoint_eval_id IS NULL
             AND ier.parent_experiment_id IS NULL
       ) AS any_final_result_exists,
       EXISTS (
           SELECT 1 FROM inference_profitability_observation ipo
           WHERE ipo.inference_eval_result_id=
                     er.source_final_inference_eval_result_id
             AND ipo.experiment_id=rm.source_experiment_id
             AND ipo.model_id=rm.source_model_id
             AND ipo.inference_scope='final'
             AND ipo.checkpoint_eval_id IS NULL
       ) AS exact_final_observation_exists,
       EXISTS (
           SELECT 1 FROM inference_profitability_observation ipo
           WHERE ipo.model_id=rm.source_model_id
       ) AS any_inference_observation_exists
FROM experiment_recommendation_ranking_member rm
JOIN experiment_recommendation_evaluation_result er
  ON er.recommendation_evaluation_result_id=
     rm.recommendation_evaluation_result_id
WHERE rm.recommendation_ranking_snapshot_id=$1
ORDER BY rm.global_ordinal,rm.recommendation_ranking_member_id
)SQL", pqxx::params{rankingSnapshotId});
    if (static_cast<std::size_t>(rows.size()) != source.candidates.size())
        throw std::runtime_error("campaign_coverage_member_count_mismatch");

    CampaignProfitabilityCoverageAudit audit;
    audit.controlSnapshotId = source.controlSnapshotId;
    audit.sourceEvaluationRunId = source.sourceEvaluationRunId;
    audit.members.reserve(rows.size());
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(rows.size()); ++index)
    {
        const auto& candidate = source.candidates[index];
        const pqxx::row row = rows[index];
        if (row["recommendation_ranking_member_id"].as<long long>() !=
            candidate.rankingMemberId)
            throw std::runtime_error("campaign_coverage_member_order_mismatch");
        const auto frozen = MapFrozenEvidence(row, "");
        if (!frozen)
            throw std::runtime_error(
                "campaign_coverage_legacy_incomplete_provenance");

        CampaignProfitabilityCoverageMember member;
        member.rankingMemberId = candidate.rankingMemberId;
        member.recommendationId = candidate.recommendationId;
        member.recommendationEvaluationResultId =
            candidate.recommendationEvaluationResultId;
        member.sourceExperimentId = candidate.sourceExperimentId;
        member.sourceModelId = candidate.sourceModelId;
        member.symbol = candidate.symbol;
        member.horizon = candidate.horizon;
        member.controlRank = candidate.currentRank;
        member.frozenEvidence = *frozen;
        member.validatedEvidence = candidate.profitability;
        member.exactFinalInferenceResultExists =
            row["exact_final_result_exists"].as<bool>();
        member.anyFinalInferenceResultExists =
            row["any_final_result_exists"].as<bool>();
        member.exactFinalProfitabilityObservationExists =
            row["exact_final_observation_exists"].as<bool>();
        member.anyInferenceProfitabilityObservationExists =
            row["any_inference_observation_exists"].as<bool>();
        member.frozenSnapshotBackfillPermitted = false;

        const std::string reason = member.validatedEvidence.state ==
                EvidenceState::valid
            ? "valid_profitability_observation"
            : member.validatedEvidence.reason;
        if (member.validatedEvidence.state == EvidenceState::valid)
        {
            member.recoveryClass = CoverageRecoveryClass::available;
            member.reconstructionAssessment = "not_required";
        }
        else if (EvidenceStateIsInvalid(member.validatedEvidence.state) ||
                 member.validatedEvidence.state == EvidenceState::incomplete)
        {
            member.recoveryClass =
                CoverageRecoveryClass::invalidIncompleteProvenance;
            member.reconstructionAssessment =
                "fail_closed_no_substitution_or_backfill";
        }
        else if (reason == "no_profitability_observation" &&
                 member.exactFinalInferenceResultExists)
        {
            member.recoveryClass =
                CoverageRecoveryClass::recoverableHistoricalAbsence;
            member.reconstructionAssessment =
                "exact_final_replay_requires_separate_lifecycle_authority_"
                "and_new_artifact_not_frozen_snapshot_backfill";
        }
        else if (reason == "final_inference_context_mismatch")
        {
            member.recoveryClass = CoverageRecoveryClass::contextMismatch;
            member.reconstructionAssessment =
                "not_recoverable_without_crossing_frozen_final_context";
        }
        else if (reason == "no_exact_final_inference_result")
        {
            member.recoveryClass = CoverageRecoveryClass::noExactFinalInference;
            member.reconstructionAssessment =
                "requires_separately_authorized_final_inference_lifecycle";
        }
        else
        {
            member.recoveryClass = CoverageRecoveryClass::otherUnavailable;
            member.reconstructionAssessment =
                "not_recoverable_without_reason_specific_authority";
        }

        const std::string frozenCanonical = ExperimentRecommendation::
            RecommendationFinalProfitabilityEvidenceCanonicalText(
                member.frozenEvidence);
        member.canonical = "campaign_profitability_coverage_member_v1;";
        member.canonical += "ranking_member_id=" +
            std::to_string(member.rankingMemberId) + ";";
        member.canonical += "recommendation_id=" +
            std::to_string(member.recommendationId) + ";";
        member.canonical += "evaluation_result_id=" +
            std::to_string(member.recommendationEvaluationResultId) + ";";
        member.canonical += "control_rank=" +
            std::to_string(member.controlRank) + ";";
        member.canonical += "frozen_evidence_hash=" +
            ExperimentRecommendation::RecommendationCanonicalHash(
                frozenCanonical) + ";";
        member.canonical += "validated_evidence_hash=" +
            member.validatedEvidence.evidenceIdentityHash + ";";
        member.canonical += "reason=" + reason + ";recovery_class=" +
            CoverageRecoveryClassText(member.recoveryClass) + ";";
        member.canonical += "exact_final_result_exists=" +
            std::string(member.exactFinalInferenceResultExists ? "true" : "false") +
            ";any_final_result_exists=" +
            std::string(member.anyFinalInferenceResultExists ? "true" : "false") +
            ";exact_final_observation_exists=" +
            std::string(member.exactFinalProfitabilityObservationExists
                            ? "true" : "false") +
            ";any_inference_observation_exists=" +
            std::string(member.anyInferenceProfitabilityObservationExists
                            ? "true" : "false") +
            ";frozen_snapshot_backfill_permitted=false";
        member.hash = InferenceProfitability::DeterministicHash(member.canonical);
        ++audit.reasonCounts[reason];
        ++audit.recoveryClassCounts[
            CoverageRecoveryClassText(member.recoveryClass)];
        audit.members.push_back(std::move(member));
    }
    audit.canonical = "campaign_profitability_coverage_audit_v1;";
    audit.canonical += "control_snapshot_id=" +
        std::to_string(audit.controlSnapshotId) + ";source_evaluation_run_id=" +
        std::to_string(audit.sourceEvaluationRunId) + ";member_count=" +
        std::to_string(audit.members.size()) + ";";
    for (std::size_t index = 0; index < audit.members.size(); ++index)
        audit.canonical += "member[" + std::to_string(index) + "]=" +
            audit.members[index].hash + ";";
    audit.hash = InferenceProfitability::DeterministicHash(audit.canonical);
    return audit;
}

} // namespace EA::ProfitabilityVerification
