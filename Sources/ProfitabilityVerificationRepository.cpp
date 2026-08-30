#include "ProfitabilityVerificationRepository.hpp"

#include "ExperimentRecommendationRanking.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <set>
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

std::string Number(double value)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << (value == 0.0 ? 0.0 : value);
    return output.str();
}

std::string Sha256File(const std::string& path, std::string& content)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        throw std::runtime_error("phase12_committed_artifact_missing:" + path);
    std::ostringstream bytes;
    bytes << input.rdbuf();
    if (!input.eof() && input.fail())
        throw std::runtime_error("phase12_committed_artifact_read_failed");
    content = bytes.str();
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(content.data(), static_cast<CC_LONG>(content.size()),
              digest.data());
    std::ostringstream rendered;
    rendered << std::hex << std::setfill('0');
    for (unsigned char byte : digest)
        rendered << std::setw(2) << static_cast<unsigned int>(byte);
    return rendered.str();
}

void VerifyPhase12Artifact(const std::string& path,
                           std::string& sha256,
                           std::string& content)
{
    sha256 = Sha256File(path, content);
    if (sha256 != kPhase12ArtifactSha256)
        throw std::runtime_error("phase12_committed_artifact_sha256_mismatch");
    const std::string header =
        std::string{"CAMPAIGN_PROFITABILITY_FORWARD_VALIDATION_PRECOMMIT,"}
        + "protocol_version=1,validation_cohort_identity_hash=" +
        kPhase12ValidationCohortIdentityHash + ",ranking_snapshot_id=5," +
        "source_evaluation_run_id=6,";
    if (!content.starts_with(header) ||
        content.find(std::string{"expected_outcome_start="} +
                     kPhase12OutcomeStart + ",expected_outcome_end=" +
                     kPhase12OutcomeEnd) == std::string::npos ||
        content.find(std::string{"control_ranking_hash="} +
                     kPhase12ControlRankingHash + ",candidate_ranking_hash=" +
                     kPhase12CandidateRankingHash + ",member_count=79") ==
            std::string::npos)
        throw std::runtime_error("phase12_committed_artifact_contract_mismatch");
    std::size_t memberCount = 0;
    std::size_t topNCount = 0;
    std::istringstream lines{content};
    for (std::string line; std::getline(lines, line);)
    {
        if (line.starts_with("CAMPAIGN_PROFITABILITY_ASOF_MEMBER,"))
            ++memberCount;
        if (line.starts_with("CAMPAIGN_PROFITABILITY_TEMPORAL_TOP_N,"))
            ++topNCount;
    }
    if (memberCount != kPhase12MemberCount || topNCount != 3)
        throw std::runtime_error("phase12_committed_artifact_record_count_mismatch");
}

std::string VerifyPhase12PreparationArtifact(const std::string& path)
{
    std::string content;
    const std::string sha256 = Sha256File(path, content);
    if (sha256 != kPhase12PreparationArtifactSha256)
        throw std::runtime_error(
            "phase13_phase12_preparation_artifact_sha256_mismatch");
    const std::string expectedHeader =
        std::string{"CAMPAIGN_PROFITABILITY_OUTCOME_SUMMARY,"}
        + "validation_cohort_identity_hash=" +
        kPhase12ValidationCohortIdentityHash +
        ",ranking_snapshot_id=5,source_evaluation_run_id=6,";
    if (!content.starts_with(expectedHeader) ||
        content.find(std::string{"artifact_sha256="} +
                     kPhase12ArtifactSha256) == std::string::npos ||
        content.find(std::string{"preparation_hash="} +
                     kPhase12PreparationIdentityHash) == std::string::npos ||
        content.find(std::string{"outcome_start="} + kPhase12OutcomeStart +
                     ",outcome_end=" + kPhase12OutcomeEnd) ==
            std::string::npos ||
        content.find(std::string{"metric_hash="} +
                     InferenceProfitability::MetricDefinitionHash()) ==
            std::string::npos)
        throw std::runtime_error(
            "phase13_phase12_preparation_artifact_contract_mismatch");
    std::size_t jobCount = 0;
    std::size_t selectionCount = 0;
    std::istringstream lines{content};
    for (std::string line; std::getline(lines, line);)
    {
        if (line.starts_with("CAMPAIGN_PROFITABILITY_OUTCOME_JOB,"))
            ++jobCount;
        if (line.starts_with(
                "CAMPAIGN_PROFITABILITY_OUTCOME_SELECTION_MAPPING,"))
            ++selectionCount;
    }
    if (jobCount != 23 || selectionCount != 3)
        throw std::runtime_error(
            "phase13_phase12_preparation_artifact_record_count_mismatch");
    return sha256;
}

bool TaggedHash(const std::string& value)
{
    if (value.size() != 24 || value.rfind("fnv1a64:", 0) != 0) return false;
    return std::all_of(value.begin() + 8, value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string IdList(const std::vector<long long>& values)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index) output << ':';
        output << values[index];
    }
    return output.str();
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
        const pqxx::row row = rows[static_cast<int>(index)];
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

CampaignProfitabilityTemporalCohort LoadCampaignProfitabilityTemporalCohort(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId)
{
    if (rankingSnapshotId <= 0)
        throw std::invalid_argument("invalid_temporal_validation_snapshot_id");
    const pqxx::result snapshots = transaction.exec(R"SQL(
SELECT recommendation_ranking_snapshot_id,evaluation_run_filter,member_count,
       to_char(completed_at AT TIME ZONE 'UTC',
               'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') AS as_of_timestamp
FROM experiment_recommendation_ranking_snapshot
WHERE recommendation_ranking_snapshot_id=$1 AND status='completed'
  AND completed_at IS NOT NULL
)SQL", pqxx::params{rankingSnapshotId});
    if (snapshots.empty())
        throw std::runtime_error("temporal_validation_snapshot_not_found");
    const pqxx::row snapshot = snapshots.one_row();

    CampaignProfitabilityTemporalCohort cohort;
    cohort.rankingSnapshotId = rankingSnapshotId;
    cohort.sourceEvaluationRunId = OptionalValue<long long>(
        snapshot, "evaluation_run_filter").value_or(-1);
    cohort.totalCandidateCount = snapshot["member_count"].as<int>();
    cohort.asOfTimestamp = snapshot["as_of_timestamp"].as<std::string>();

    try
    {
        const CampaignProfitabilityCoverageAudit coverage =
            LoadCampaignProfitabilityCoverageAudit(transaction,
                                                    rankingSnapshotId);
        const CampaignProfitabilityShadowSource source =
            LoadCampaignProfitabilityShadowSource(transaction,
                                                  rankingSnapshotId);
        cohort.rankingPopulationReconstructable =
            source.persistedMemberCount == cohort.totalCandidateCount &&
            source.sourceEvaluationRunId == cohort.sourceEvaluationRunId;
        cohort.rankingTimeUnavailableReasonCounts = coverage.reasonCounts;
        for (const auto& member : coverage.members)
        {
            if (member.validatedEvidence.state == EvidenceState::valid)
                ++cohort.validRankingTimeProfitabilityEvidenceCount;
            else
                ++cohort.unavailableRankingTimeEvidenceCount;
            if (member.frozenEvidence.inferenceStart &&
                (!cohort.rankingInputStart ||
                 *member.frozenEvidence.inferenceStart <
                     *cohort.rankingInputStart))
                cohort.rankingInputStart =
                    *member.frozenEvidence.inferenceStart;
            if (member.frozenEvidence.inferenceEnd &&
                (!cohort.rankingInputEnd ||
                 *member.frozenEvidence.inferenceEnd > *cohort.rankingInputEnd))
                cohort.rankingInputEnd = *member.frozenEvidence.inferenceEnd;
        }
        const WeightedShadowRanking control = BuildWeightedShadowRanking(
            source.candidates, source.controlSnapshotId,
            source.sourceEvaluationRunId, source.controlSnapshotIdentityHash,
            0.0);
        cohort.exactControlReconstruction =
            control.candidates.size() == source.candidates.size();
        for (const auto& member : control.candidates)
            cohort.exactControlReconstruction =
                cohort.exactControlReconstruction &&
                member.shadowRank == member.source.currentRank &&
                member.rankDelta == 0;
    }
    catch (const std::exception& error)
    {
        cohort.rankingPopulationReconstructable = false;
        cohort.exactControlReconstruction = false;
        cohort.validRankingTimeProfitabilityEvidenceCount = 0;
        cohort.unavailableRankingTimeEvidenceCount =
            cohort.totalCandidateCount;
        cohort.reason = std::string{"ranking_reconstruction_failed:"} +
            error.what();
    }

    const pqxx::row temporal = transaction.exec(R"SQL(
WITH members AS (
    SELECT s.completed_at AS cutoff,rm.recommendation_ranking_member_id,
           rm.recommendation_id,rm.source_experiment_id,rm.source_model_id,
           rm.symbol,rm.horizon,rm.created_at AS member_created_at,
           r.created_at AS recommendation_created_at,
           er.created_at AS evaluation_created_at,
           er.source_final_inference_eval_result_id AS frozen_result_id,
           er.source_final_profitability_observation_id AS frozen_observation_id,
           er.source_final_profitability_inference_end AS ranking_input_end
    FROM experiment_recommendation_ranking_snapshot s
    JOIN experiment_recommendation_ranking_member rm
      ON rm.recommendation_ranking_snapshot_id=
         s.recommendation_ranking_snapshot_id
    JOIN experiment_recommendation r
      ON r.recommendation_id=rm.recommendation_id
    JOIN experiment_recommendation_evaluation_result er
      ON er.recommendation_evaluation_result_id=
         rm.recommendation_evaluation_result_id
    WHERE s.recommendation_ranking_snapshot_id=$1
), observations AS (
    SELECT m.*,o.profitability_observation_id,o.created_at AS outcome_created_at,
           o.inference_start AS outcome_start,o.inference_end AS outcome_end,
           o.experiment_id AS outcome_experiment_id,
           o.model_id AS outcome_model_id,o.inference_scope AS outcome_scope,
           o.checkpoint_eval_id AS outcome_checkpoint_eval_id,
           o.metric_definition_hash,o.source_content_hash,
           o.observation_identity_hash,ier.id AS outcome_result_id,
           ier.completed_at AS outcome_result_completed_at,
           ier.status AS outcome_result_status,
           ier.inference_scope AS outcome_result_scope,
           ier.checkpoint_eval_id AS outcome_result_checkpoint_eval_id,
           ier.parent_experiment_id AS outcome_parent_experiment_id,
           ier.model_id AS outcome_result_model_id,ier.symbol AS outcome_symbol,
           ier.prediction_horizon AS outcome_horizon,
           ier.from_date AS outcome_result_start,
           ier.to_date AS outcome_result_end
    FROM members m
    LEFT JOIN inference_profitability_observation o
      ON o.model_id=m.source_model_id
    LEFT JOIN inference_eval_result ier
      ON ier.id=o.inference_eval_result_id
), marked AS (
    SELECT *,
      (outcome_created_at > cutoff) AS arrived_after_cutoff,
      (outcome_experiment_id=source_experiment_id AND
       outcome_model_id=source_model_id AND
       outcome_result_model_id=source_model_id AND
       outcome_scope='final' AND outcome_result_scope='final' AND
       outcome_checkpoint_eval_id IS NULL AND
       outcome_result_checkpoint_eval_id IS NULL AND
       outcome_parent_experiment_id IS NULL AND
       outcome_result_status='completed' AND outcome_symbol=symbol AND
       outcome_horizon=horizon AND
       outcome_result_start=outcome_start AND
       outcome_result_end=outcome_end AND
       metric_definition_hash=$2 AND source_content_hash IS NOT NULL AND
       source_content_hash<>'' AND observation_identity_hash IS NOT NULL AND
       observation_identity_hash<>'') AS exact_identity,
      (outcome_start::date > cutoff::date AND
       outcome_end::date > outcome_start::date) AS strictly_subsequent_period
    FROM observations
), per_member AS (
    SELECT recommendation_ranking_member_id,
      bool_or(arrived_after_cutoff AND exact_identity AND
              strictly_subsequent_period AND
              outcome_result_completed_at > cutoff) AS legitimate,
      bool_or(arrived_after_cutoff AND exact_identity AND
              (NOT strictly_subsequent_period OR
               outcome_result_completed_at <= cutoff)) AS future_leakage,
      bool_or(arrived_after_cutoff AND exact_identity AND ranking_input_end IS NOT NULL
              AND outcome_start::date <= ranking_input_end::date) AS overlap,
      bool_or(arrived_after_cutoff AND NOT exact_identity) AS identity_mismatch,
      min(outcome_start) FILTER (
          WHERE arrived_after_cutoff AND exact_identity AND
                strictly_subsequent_period AND
                outcome_result_completed_at > cutoff) AS valid_outcome_start,
      max(outcome_end) FILTER (
          WHERE arrived_after_cutoff AND exact_identity AND
                strictly_subsequent_period AND
                outcome_result_completed_at > cutoff) AS valid_outcome_end
    FROM marked GROUP BY recommendation_ranking_member_id
), provenance AS (
    SELECT count(*) FILTER (
        WHERE recommendation_created_at > cutoff OR
              evaluation_created_at > cutoff OR member_created_at > cutoff OR
              EXISTS (SELECT 1 FROM inference_eval_result ier
                      WHERE ier.id=frozen_result_id AND
                            ier.completed_at > cutoff) OR
              EXISTS (SELECT 1 FROM inference_profitability_observation ipo
                      WHERE ipo.profitability_observation_id=
                            frozen_observation_id AND ipo.created_at > cutoff)
    ) AS violation_count
    FROM members
)
SELECT (SELECT violation_count FROM provenance) AS violation_count,
       count(*) FILTER (WHERE legitimate) AS legitimate_count,
       count(*) FILTER (WHERE future_leakage) AS future_leakage_count,
       count(*) FILTER (WHERE overlap) AS overlap_count,
       count(*) FILTER (WHERE identity_mismatch) AS mismatch_count,
       min(valid_outcome_start) AS outcome_start,
       max(valid_outcome_end) AS outcome_end
FROM per_member
)SQL", pqxx::params{rankingSnapshotId,
                     InferenceProfitability::MetricDefinitionHash()}).one_row();
    cohort.pointInTimeProvenanceViolationCount =
        temporal["violation_count"].as<int>();
    cohort.legitimateSubsequentOutcomeCount =
        temporal["legitimate_count"].as<int>();
    cohort.futureInformationLeakageCount =
        temporal["future_leakage_count"].as<int>();
    cohort.overlappingInputAndOutcomeCount =
        temporal["overlap_count"].as<int>();
    cohort.contextOrIdentityMismatchCount =
        temporal["mismatch_count"].as<int>();
    cohort.outcomeStart = OptionalValue<std::string>(temporal, "outcome_start");
    cohort.outcomeEnd = OptionalValue<std::string>(temporal, "outcome_end");

    if (!cohort.rankingPopulationReconstructable ||
        !cohort.exactControlReconstruction)
    {
        cohort.classification = TemporalCohortClassification::
            insufficientRankingTimeProvenance;
        if (cohort.reason.empty())
            cohort.reason = "authoritative_point_in_time_control_not_reconstructable";
    }
    else if (cohort.pointInTimeProvenanceViolationCount > 0)
    {
        cohort.classification =
            TemporalCohortClassification::futureInformationLeakage;
        cohort.reason = "ranking_population_contains_post_cutoff_evidence";
    }
    else if (cohort.legitimateSubsequentOutcomeCount == 0)
    {
        if (cohort.futureInformationLeakageCount > 0)
        {
            cohort.classification =
                TemporalCohortClassification::futureInformationLeakage;
            cohort.reason =
                "later_arriving_record_does_not_have_subsequent_outcome_period";
        }
        else if (cohort.contextOrIdentityMismatchCount > 0)
        {
            cohort.classification =
                TemporalCohortClassification::contextOrIdentityMismatch;
            cohort.reason = "only_later_outcome_records_mismatch_exact_identity";
        }
        else
        {
            cohort.classification = TemporalCohortClassification::
                insufficientSubsequentOutcome;
            cohort.reason = "no_exact_strictly_subsequent_outcome_evidence";
        }
    }
    else if (cohort.overlappingInputAndOutcomeCount > 0)
    {
        cohort.classification = TemporalCohortClassification::
            overlappingInputAndOutcomePeriod;
        cohort.reason = "ranking_input_and_outcome_period_overlap";
    }
    else if (cohort.legitimateSubsequentOutcomeCount <
             cohort.totalCandidateCount)
    {
        cohort.classification = TemporalCohortClassification::
            insufficientSubsequentOutcome;
        cohort.reason = "subsequent_outcome_population_incomplete";
    }
    else
    {
        cohort.classification =
            TemporalCohortClassification::admissibleTemporalHoldout;
        cohort.reason = "exact_point_in_time_population_and_outcomes_available";
    }

    cohort.canonical = "campaign_profitability_temporal_cohort_v1;";
    cohort.canonical += "snapshot_id=" + std::to_string(rankingSnapshotId) +
        ";evaluation_run_id=" + std::to_string(cohort.sourceEvaluationRunId) +
        ";as_of=" + cohort.asOfTimestamp +
        ";candidate_count=" + std::to_string(cohort.totalCandidateCount) +
        ";valid_ranking_evidence=" +
        std::to_string(cohort.validRankingTimeProfitabilityEvidenceCount) +
        ";unavailable_ranking_evidence=" +
        std::to_string(cohort.unavailableRankingTimeEvidenceCount) +
        ";subsequent_outcome_count=" +
        std::to_string(cohort.legitimateSubsequentOutcomeCount) +
        ";point_in_time_violations=" +
        std::to_string(cohort.pointInTimeProvenanceViolationCount) +
        ";overlap_count=" +
        std::to_string(cohort.overlappingInputAndOutcomeCount) +
        ";future_leakage_count=" +
        std::to_string(cohort.futureInformationLeakageCount) +
        ";context_mismatch_count=" +
        std::to_string(cohort.contextOrIdentityMismatchCount) +
        ";control_reconstruction=" +
        (cohort.exactControlReconstruction ? "true" : "false") +
        ";classification=" +
        TemporalCohortClassificationText(cohort.classification) +
        ";reason=" + cohort.reason;
    cohort.hash = InferenceProfitability::DeterministicHash(cohort.canonical);
    return cohort;
}

CampaignProfitabilityTemporalFeasibilityAudit
LoadCampaignProfitabilityTemporalFeasibilityAudit(
    pqxx::transaction_base& transaction)
{
    const pqxx::result snapshots = transaction.exec(R"SQL(
SELECT recommendation_ranking_snapshot_id
FROM experiment_recommendation_ranking_snapshot
WHERE status='completed' AND completed_at IS NOT NULL
ORDER BY recommendation_ranking_snapshot_id
)SQL");
    CampaignProfitabilityTemporalFeasibilityAudit audit;
    audit.cohorts.reserve(snapshots.size());
    for (const pqxx::row& row : snapshots)
    {
        auto cohort = LoadCampaignProfitabilityTemporalCohort(
            transaction,
            row["recommendation_ranking_snapshot_id"].as<long long>());
        ++audit.classificationCounts[
            TemporalCohortClassificationText(cohort.classification)];
        audit.cohorts.push_back(std::move(cohort));
    }
    audit.canonical = "campaign_profitability_temporal_feasibility_audit_v1;";
    audit.canonical += "cohort_count=" +
        std::to_string(audit.cohorts.size()) + ";";
    for (std::size_t index = 0; index < audit.cohorts.size(); ++index)
        audit.canonical += "cohort[" + std::to_string(index) + "]=" +
            audit.cohorts[index].hash + ";";
    audit.hash = InferenceProfitability::DeterministicHash(audit.canonical);
    return audit;
}

CampaignProfitabilityOutcomePreparation
LoadCampaignProfitabilityOutcomePreparation(
    pqxx::transaction_base& transaction,
    const std::string& currentDate,
    const std::string& artifactPath)
{
    if (currentDate.size() != 10)
        throw std::invalid_argument("phase12_current_date_invalid");

    CampaignProfitabilityOutcomePreparation preparation;
    preparation.artifactPath = artifactPath;
    preparation.currentDate = currentDate;
    std::string artifactContent;
    VerifyPhase12Artifact(artifactPath, preparation.artifactSha256,
                          artifactContent);
    preparation.artifactIdentityVerified = true;

    const CampaignProfitabilityTemporalCohort temporal =
        LoadCampaignProfitabilityTemporalCohort(
            transaction, kPhase12RankingSnapshotId);
    if (!temporal.rankingPopulationReconstructable ||
        !temporal.exactControlReconstruction ||
        temporal.pointInTimeProvenanceViolationCount != 0 ||
        temporal.sourceEvaluationRunId != kPhase12SourceEvaluationRunId)
        throw std::runtime_error("phase12_snapshot5_reconstruction_mismatch");
    const CampaignProfitabilityShadowSource source =
        LoadCampaignProfitabilityShadowSource(
            transaction, kPhase12RankingSnapshotId);
    const auto precommit = BuildCampaignProfitabilityForwardValidationPrecommit(
        source, temporal.asOfTimestamp, kPhase12OutcomeStart,
        kPhase12OutcomeEnd);
    if (precommit.hash != kPhase12ValidationCohortIdentityHash ||
        precommit.controlRankingHash != kPhase12ControlRankingHash ||
        precommit.candidateRankingHash != kPhase12CandidateRankingHash ||
        precommit.sourceEvaluationRunId != kPhase12SourceEvaluationRunId ||
        precommit.members.size() != kPhase12MemberCount)
        throw std::runtime_error("phase12_frozen_cohort_identity_mismatch");
    preparation.topN = precommit.topN;

    std::map<std::pair<long long, long long>, CampaignProfitabilityOutcomeJob>
        grouped;
    for (const auto& member : precommit.members)
    {
        if (!member.source.sourceModelId)
            throw std::runtime_error("phase12_source_model_missing");
        const auto key = std::pair{member.source.sourceExperimentId,
                                   *member.source.sourceModelId};
        auto& job = grouped[key];
        job.validationCohortIdentityHash = precommit.hash;
        job.rankingSnapshotId = precommit.rankingSnapshotId;
        job.sourceEvaluationRunId = precommit.sourceEvaluationRunId;
        job.sourceExperimentId = key.first;
        job.sourceModelId = key.second;
        job.symbol = member.source.symbol;
        job.horizon = member.source.horizon;
        job.outcomeStart = precommit.expectedOutcomeStart;
        job.outcomeEnd = precommit.expectedOutcomeEnd;
        job.recommendationIds.push_back(member.source.recommendationId);
    }

    for (const auto& top : precommit.topN)
    {
        const std::set<long long> control(top.controlRecommendationIds.begin(),
                                          top.controlRecommendationIds.end());
        const std::set<long long> candidate(
            top.candidateRecommendationIds.begin(),
            top.candidateRecommendationIds.end());
        for (auto& [key, job] : grouped)
        {
            (void)key;
            for (long long recommendationId : job.recommendationIds)
            {
                const bool inControl = control.contains(recommendationId);
                const bool inCandidate = candidate.contains(recommendationId);
                std::string role = "lower_ranked";
                if (inControl && inCandidate) role = "retained";
                else if (inCandidate) role = "candidate_entrant";
                else if (inControl) role = "control_exit";
                job.topNRoleByRecommendation[static_cast<int>(
                    top.n * 1000000 + recommendationId)] = role;
            }
        }
    }

    for (auto& [key, job] : grouped)
    {
        const pqxx::result rows = transaction.exec(R"SQL(
SELECT e.symbol,e.prediction_horizon,e.train_start::date::text AS train_start,
       e.train_end::date::text AS train_end,
       e.infer_start::date::text AS infer_start,
       e.infer_end::date::text AS infer_end,e.c_next_threshold,
       e.last_model_id,e.donchian20_mode,e.feature_warmup_scope,
       e.donchian_lookback,e.feature_ablation_mask,
       m.model_id,m.experiment_id AS model_experiment_id,m.parent_model_id,
       (SELECT max(value) FROM matrix WHERE model_id=$2 AND
            param_name='train_config_meta' AND row_idx=0 AND col_idx=1) AS model_horizon,
       (SELECT max(value) FROM matrix WHERE model_id=$2 AND
            param_name='train_config_meta' AND row_idx=0 AND col_idx=2) AS model_threshold,
       (SELECT max(value) FROM matrix WHERE model_id=$2 AND
            param_name='train_config_meta' AND row_idx=0 AND col_idx=3) AS window_size,
       (SELECT max(value) FROM matrix WHERE model_id=$2 AND
            param_name='train_config_meta' AND row_idx=0 AND col_idx=4) AS label_rule,
       (SELECT max(value) FROM matrix WHERE model_id=$2 AND
            param_name='model_meta' AND row_idx=0 AND col_idx=1) AS input_width,
       (SELECT max(value) FROM matrix WHERE model_id=$2 AND
            param_name='target_meta' AND row_idx=0 AND col_idx=0) AS target_type,
       (SELECT count(*) FROM matrix WHERE model_id=$2) AS parameter_rows,
       (SELECT count(*) FROM inference_eval_result ier
         WHERE ier.model_id=$2 AND ier.status='completed'
           AND ier.inference_scope='final' AND ier.checkpoint_eval_id IS NULL
           AND ier.parent_experiment_id IS NULL
           AND ier.symbol=e.symbol
           AND ier.prediction_horizon=e.prediction_horizon
           AND abs(ier.threshold_logret-e.c_next_threshold) <= 0.0000001
           AND ier.from_date=e.infer_start::date::text
           AND ier.to_date=e.infer_end::date::text) AS exact_final_count
FROM experiment e LEFT JOIN model m ON m.model_id=$2
WHERE e.experiment_id=$1
)SQL", pqxx::params{key.first, key.second});
        if (rows.size() != 1)
            throw std::runtime_error("phase12_source_experiment_not_found");
        const pqxx::row row = rows.one_row();
        job.modelExists = !row["model_id"].is_null();
        job.originalTrainStart = row["train_start"].as<std::string>();
        job.originalTrainEnd = row["train_end"].as<std::string>();
        job.originalInferenceStart = row["infer_start"].as<std::string>();
        job.originalInferenceEnd = row["infer_end"].as<std::string>();
        job.threshold = row["c_next_threshold"].as<double>();
        job.exactModelExperimentLink = job.modelExists &&
            !row["model_experiment_id"].is_null() &&
            row["model_experiment_id"].as<long long>() == key.first;
        job.exactFinalSourceModel = job.modelExists &&
            !row["last_model_id"].is_null() &&
            row["last_model_id"].as<long long>() == key.second;
        job.exactOriginalFinalInference =
            row["exact_final_count"].as<long long>() == 1;
        job.checkpointSubstitution = !job.exactFinalSourceModel;

        const auto requiredInt = [&](const char* column) {
            if (row[column].is_null())
                throw std::runtime_error(std::string{"phase12_model_metadata_missing:"} + column);
            return static_cast<int>(std::llround(row[column].as<double>()));
        };
        const int modelHorizon = requiredInt("model_horizon");
        const double modelThreshold = row["model_threshold"].is_null()
            ? std::numeric_limits<double>::quiet_NaN()
            : row["model_threshold"].as<double>();
        job.windowSize = requiredInt("window_size");
        job.labelRuleId = requiredInt("label_rule");
        job.inputWidth = requiredInt("input_width");
        job.targetType = requiredInt("target_type");
        job.modelParameterRowCount = row["parameter_rows"].as<long long>();

        std::string parameterCanonical =
            "campaign_profitability_frozen_model_artifact_v1;model_id=" +
            std::to_string(job.sourceModelId) + ";";
        const pqxx::result parameters = transaction.exec(R"SQL(
SELECT param_name,row_idx,col_idx,encode(float8send(value),'hex') AS value_bits
FROM matrix WHERE model_id=$1
ORDER BY param_name COLLATE "C",row_idx,col_idx
)SQL", pqxx::params{job.sourceModelId});
        for (const pqxx::row& parameter : parameters)
            parameterCanonical += parameter[0].as<std::string>() + ":" +
                parameter[1].as<std::string>() + ":" +
                parameter[2].as<std::string>() + ":" +
                parameter[3].as<std::string>() + ";";
        job.modelArtifactContentHash =
            InferenceProfitability::DeterministicHash(parameterCanonical);

        job.featureSemanticCanonical =
            "campaign_profitability_frozen_feature_semantics_v1;symbol=" +
            job.symbol + ";horizon=" + std::to_string(job.horizon) +
            ";threshold=" + Number(job.threshold) + ";window_size=" +
            std::to_string(job.windowSize) + ";label_rule_id=" +
            std::to_string(job.labelRuleId) + ";target_type=" +
            std::to_string(job.targetType) + ";input_width=" +
            std::to_string(job.inputWidth) + ";donchian20_mode=" +
            row["donchian20_mode"].as<std::string>() +
            ";feature_warmup_scope=" +
            row["feature_warmup_scope"].as<std::string>() +
            ";donchian_lookback=" +
            row["donchian_lookback"].as<std::string>() +
            ";feature_ablation_mask=" +
            row["feature_ablation_mask"].as<std::string>();
        job.featureSemanticHash = InferenceProfitability::DeterministicHash(
            job.featureSemanticCanonical);
        job.modelLineageCanonical =
            "campaign_profitability_frozen_model_lineage_v1;source_experiment_id=" +
            std::to_string(job.sourceExperimentId) + ";source_model_id=" +
            std::to_string(job.sourceModelId) + ";model_experiment_id=" +
            (row["model_experiment_id"].is_null() ? "NULL" :
                row["model_experiment_id"].as<std::string>()) +
            ";parent_model_id=" +
            (row["parent_model_id"].is_null() ? "NULL" :
                row["parent_model_id"].as<std::string>()) +
            ";last_model_id=" + row["last_model_id"].as<std::string>() +
            ";model_artifact_content_hash=" + job.modelArtifactContentHash;
        job.modelLineageHash = InferenceProfitability::DeterministicHash(
            job.modelLineageCanonical);
        job.metricDefinitionCanonical =
            InferenceProfitability::kMetricDefinitionCanonical;
        job.metricDefinitionHash = InferenceProfitability::MetricDefinitionHash();

        const bool contextMatches =
            row["symbol"].as<std::string>() == job.symbol &&
            row["prediction_horizon"].as<int>() == job.horizon &&
            modelHorizon == job.horizon && std::isfinite(modelThreshold) &&
            std::abs(modelThreshold - job.threshold) <= 0.0000001 &&
            job.modelParameterRowCount > 0;
        if (!job.modelExists) job.compatibilityState = "model_artifact_missing";
        else if (!job.exactModelExperimentLink)
            job.compatibilityState = "model_lineage_ambiguous";
        else if (!job.exactFinalSourceModel)
            job.compatibilityState = "checkpoint_or_nonfinal_substitution_rejected";
        else if (!contextMatches)
            job.compatibilityState = "model_semantics_mismatch";
        else if (job.outcomeStart <= job.originalInferenceEnd)
            job.compatibilityState = "outcome_window_overlaps_original_inference";
        else
            job.compatibilityState = job.exactOriginalFinalInference
                ? "compatible"
                : "compatible_historical_final_inference_absent";
        job.compatible = job.compatibilityState == "compatible" ||
            job.compatibilityState ==
                "compatible_historical_final_inference_absent";
        job.readiness = !job.compatible
            ? OutcomeJobReadiness::incompatibleSource
            : (currentDate <= job.outcomeEnd
                   ? OutcomeJobReadiness::waitingForOutcomeData
                   : OutcomeJobReadiness::readyToExecute);

        std::ostringstream participation;
        bool first = true;
        for (const auto& [encoded, role] : job.topNRoleByRecommendation)
        {
            if (!first) participation << '|';
            first = false;
            participation << encoded / 1000000 << ':'
                          << encoded % 1000000 << ':' << role;
        }
        job.topNParticipation = participation.str();
        job.canonical =
            "campaign_profitability_outcome_job_v1;cohort_hash=" +
            job.validationCohortIdentityHash + ";snapshot_id=" +
            std::to_string(job.rankingSnapshotId) + ";evaluation_run_id=" +
            std::to_string(job.sourceEvaluationRunId) +
            ";source_experiment_id=" + std::to_string(job.sourceExperimentId) +
            ";source_model_id=" + std::to_string(job.sourceModelId) +
            ";symbol=" + job.symbol + ";horizon=" +
            std::to_string(job.horizon) + ";train_range=" +
            job.originalTrainStart + ":" + job.originalTrainEnd +
            ";original_inference_range=" + job.originalInferenceStart + ":" +
            job.originalInferenceEnd + ";outcome_range=" + job.outcomeStart +
            ":" + job.outcomeEnd + ";feature_semantic_hash=" +
            job.featureSemanticHash + ";model_lineage_hash=" +
            job.modelLineageHash + ";metric_definition_hash=" +
            job.metricDefinitionHash + ";recommendation_ids=" +
            IdList(job.recommendationIds) + ";selection_roles=" +
            job.topNParticipation;
        job.hash = InferenceProfitability::DeterministicHash(job.canonical);
        preparation.jobs.push_back(std::move(job));
    }
    std::sort(preparation.jobs.begin(), preparation.jobs.end(),
              [](const auto& left, const auto& right) {
                  return std::tie(left.sourceModelId, left.sourceExperimentId) <
                         std::tie(right.sourceModelId, right.sourceExperimentId);
              });
    preparation.canonical =
        "campaign_profitability_outcome_preparation_v1;cohort_hash=" +
        std::string{kPhase12ValidationCohortIdentityHash} +
        ";artifact_sha256=" + preparation.artifactSha256 + ";job_count=" +
        std::to_string(preparation.jobs.size()) + ";";
    for (std::size_t index = 0; index < preparation.jobs.size(); ++index)
        preparation.canonical += "job[" + std::to_string(index) + "]=" +
            preparation.jobs[index].hash + ";";
    preparation.hash = InferenceProfitability::DeterministicHash(
        preparation.canonical);
    return preparation;
}

CampaignProfitabilityProspectiveComparison
LoadCampaignProfitabilityProspectiveComparison(
    pqxx::transaction_base& transaction,
    const std::string& currentDate,
    const std::string& phase11ArtifactPath,
    const std::string& phase12PreparationArtifactPath)
{
    const std::string phase12PreparationSha256 =
        VerifyPhase12PreparationArtifact(phase12PreparationArtifactPath);
    CampaignProfitabilityProspectiveComparisonRequest request;
    request.validationCohortIdentityHash =
        kPhase12ValidationCohortIdentityHash;
    request.phase11ArtifactSha256 = kPhase12ArtifactSha256;
    request.phase12PreparationArtifactSha256 = phase12PreparationSha256;
    request.phase12PreparationIdentityHash =
        kPhase12PreparationIdentityHash;
    request.metricDefinitionCanonical =
        InferenceProfitability::kMetricDefinitionCanonical;
    request.metricDefinitionHash = InferenceProfitability::MetricDefinitionHash();
    request.outcomeStart = kPhase12OutcomeStart;
    request.outcomeEnd = kPhase12OutcomeEnd;
    request.currentDate = currentDate;
    request.preparation = LoadCampaignProfitabilityOutcomePreparation(
        transaction, currentDate, phase11ArtifactPath);

    const bool outcomeTableExists = !transaction.exec(R"SQL(
SELECT to_regclass('public.campaign_profitability_prospective_outcome_result')
       AS outcome_table
)SQL").one_row()["outcome_table"].is_null();
    const pqxx::result rows = outcomeTableExists ? transaction.exec(R"SQL(
SELECT prospective_outcome_result_id,validation_cohort_identity_hash,
       ranking_snapshot_id,source_evaluation_run_id,source_experiment_id,
       source_model_id,outcome_start::text,outcome_end::text,
       job_identity_hash,feature_semantic_hash,model_lineage_hash,
       model_artifact_content_hash,metric_definition_canonical,
       metric_definition_hash,source_content_hash,prediction_count,
       actionable_count,winning_actionable_count,losing_actionable_count,
       gross_positive_terminal_horizon_log_return_sum,
       gross_negative_terminal_horizon_log_return_sum,
       aggregate_terminal_horizon_log_return_sum,
       average_terminal_horizon_log_return_per_actionable_prediction,
       outcome_identity_canonical,outcome_identity_hash
FROM campaign_profitability_prospective_outcome_result
WHERE validation_cohort_identity_hash=$1
ORDER BY source_model_id,source_experiment_id,
         prospective_outcome_result_id
)SQL", pqxx::params{kPhase12ValidationCohortIdentityHash}) : pqxx::result{};
    request.outcomes.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        CampaignProfitabilityProspectiveOutcome outcome;
        outcome.resultId =
            row["prospective_outcome_result_id"].as<long long>();
        outcome.validationCohortIdentityHash =
            row["validation_cohort_identity_hash"].as<std::string>();
        outcome.rankingSnapshotId = row["ranking_snapshot_id"].as<long long>();
        outcome.sourceEvaluationRunId =
            row["source_evaluation_run_id"].as<long long>();
        outcome.sourceExperimentId =
            row["source_experiment_id"].as<long long>();
        outcome.sourceModelId = row["source_model_id"].as<long long>();
        outcome.outcomeStart = row["outcome_start"].as<std::string>();
        outcome.outcomeEnd = row["outcome_end"].as<std::string>();
        outcome.jobIdentityHash =
            row["job_identity_hash"].as<std::string>();
        outcome.featureSemanticHash =
            row["feature_semantic_hash"].as<std::string>();
        outcome.modelLineageHash =
            row["model_lineage_hash"].as<std::string>();
        outcome.modelArtifactContentHash =
            row["model_artifact_content_hash"].as<std::string>();
        outcome.metricDefinitionCanonical =
            row["metric_definition_canonical"].as<std::string>();
        outcome.metricDefinitionHash =
            row["metric_definition_hash"].as<std::string>();
        outcome.sourceContentHash =
            row["source_content_hash"].as<std::string>();
        outcome.predictionCount = static_cast<std::uint64_t>(
            row["prediction_count"].as<long long>());
        outcome.actionableCount = static_cast<std::uint64_t>(
            row["actionable_count"].as<long long>());
        outcome.winningActionableCount = static_cast<std::uint64_t>(
            row["winning_actionable_count"].as<long long>());
        outcome.losingActionableCount = static_cast<std::uint64_t>(
            row["losing_actionable_count"].as<long long>());
        outcome.grossPositiveReturn =
            row["gross_positive_terminal_horizon_log_return_sum"].as<double>();
        outcome.grossNegativeReturn =
            row["gross_negative_terminal_horizon_log_return_sum"].as<double>();
        outcome.aggregateReturn =
            row["aggregate_terminal_horizon_log_return_sum"].as<double>();
        outcome.averageReturn = OptionalValue<double>(
            row,
            "average_terminal_horizon_log_return_per_actionable_prediction");
        outcome.outcomeIdentityCanonical =
            row["outcome_identity_canonical"].as<std::string>();
        outcome.outcomeIdentityHash =
            row["outcome_identity_hash"].as<std::string>();
        request.outcomes.push_back(std::move(outcome));
    }
    return BuildCampaignProfitabilityProspectiveComparison(std::move(request));
}

CampaignProfitabilityOutcomeJob LoadCampaignProfitabilityOutcomeExecutionJob(
    pqxx::transaction_base& transaction,
    const std::string& cohortHash,
    long long sourceExperimentId,
    long long sourceModelId,
    const std::string& outcomeStart,
    const std::string& outcomeEnd,
    const std::string& jobHash,
    const std::string& currentDate,
    const std::string& artifactPath)
{
    if (cohortHash != kPhase12ValidationCohortIdentityHash ||
        outcomeStart != kPhase12OutcomeStart || outcomeEnd != kPhase12OutcomeEnd)
        throw std::invalid_argument("phase12_execution_cohort_or_window_mismatch");
    const auto preparation = LoadCampaignProfitabilityOutcomePreparation(
        transaction, currentDate, artifactPath);
    const auto found = std::find_if(
        preparation.jobs.begin(), preparation.jobs.end(), [&](const auto& job) {
            return job.sourceExperimentId == sourceExperimentId &&
                job.sourceModelId == sourceModelId;
        });
    if (found == preparation.jobs.end())
        throw std::invalid_argument("phase12_execution_job_not_in_frozen_cohort");
    if (found->hash != jobHash)
        throw std::invalid_argument("phase12_execution_job_hash_mismatch");
    if (!found->compatible)
        throw std::runtime_error("phase12_execution_source_incompatible:" +
                                 found->compatibilityState);
    if (found->readiness != OutcomeJobReadiness::readyToExecute)
        throw std::runtime_error("phase12_execution_waiting_for_outcome_data");
    return *found;
}

CampaignProfitabilityOutcomePersistResult
PersistCampaignProfitabilityOutcomeIdempotently(
    pqxx::transaction_base& transaction,
    const CampaignProfitabilityOutcomePersistRequest& request)
{
    const auto& job = request.job;
    if (!job.compatible || job.hash.empty() ||
        job.metricDefinitionCanonical !=
            InferenceProfitability::kMetricDefinitionCanonical ||
        job.metricDefinitionHash != InferenceProfitability::MetricDefinitionHash() ||
        !TaggedHash(request.sourceContentHash) ||
        request.actionableCount > request.predictionCount ||
        request.winningActionableCount + request.losingActionableCount >
            request.actionableCount ||
        (request.actionableCount == 0) != !request.averageReturn.has_value())
        throw std::invalid_argument("phase12_outcome_persist_contract_invalid");
    const std::string statistics =
        "prediction_count=" + std::to_string(request.predictionCount) +
        ";actionable_count=" + std::to_string(request.actionableCount) +
        ";winning_actionable_count=" +
        std::to_string(request.winningActionableCount) +
        ";losing_actionable_count=" +
        std::to_string(request.losingActionableCount) +
        ";aggregate_return=" + Number(request.aggregateReturn) +
        ";average_return=" +
        (request.averageReturn ? Number(*request.averageReturn) : "NULL");
    CampaignProfitabilityOutcomePersistResult result;
    result.outcomeIdentityCanonical =
        "campaign_profitability_prospective_outcome_v1;job_hash=" + job.hash +
        ";cohort_hash=" + job.validationCohortIdentityHash +
        ";source_experiment_id=" + std::to_string(job.sourceExperimentId) +
        ";source_model_id=" + std::to_string(job.sourceModelId) +
        ";outcome_start=" + job.outcomeStart + ";outcome_end=" +
        job.outcomeEnd + ";metric_definition_hash=" +
        job.metricDefinitionHash + ";source_content_hash=" +
        request.sourceContentHash + ";" + statistics;
    result.outcomeIdentityHash = InferenceProfitability::DeterministicHash(
        result.outcomeIdentityCanonical);
    pqxx::result inserted = transaction.exec(R"SQL(
INSERT INTO campaign_profitability_prospective_outcome_result(
 validation_cohort_identity_hash,ranking_snapshot_id,source_evaluation_run_id,
 source_experiment_id,source_model_id,symbol,prediction_horizon,threshold_logret,
 window_size,label_rule_id,target_type,input_width,outcome_start,outcome_end,
 job_identity_hash,feature_semantic_hash,model_lineage_hash,
 model_artifact_content_hash,metric_definition_canonical,
 metric_definition_hash,source_content_hash,prediction_count,actionable_count,
 winning_actionable_count,losing_actionable_count,
 gross_positive_terminal_horizon_log_return_sum,
 gross_negative_terminal_horizon_log_return_sum,
 aggregate_terminal_horizon_log_return_sum,
 average_terminal_horizon_log_return_per_actionable_prediction,
 inference_accuracy,outcome_identity_canonical,outcome_identity_hash)
VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
       $19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32)
ON CONFLICT (outcome_identity_canonical) DO NOTHING
RETURNING prospective_outcome_result_id
)SQL", pqxx::params{
        job.validationCohortIdentityHash, job.rankingSnapshotId,
        job.sourceEvaluationRunId, job.sourceExperimentId, job.sourceModelId,
        job.symbol, job.horizon, job.threshold, job.windowSize,
        job.labelRuleId, job.targetType, job.inputWidth, job.outcomeStart,
        job.outcomeEnd, job.hash, job.featureSemanticHash,
        job.modelLineageHash, job.modelArtifactContentHash,
        job.metricDefinitionCanonical, job.metricDefinitionHash,
        request.sourceContentHash, request.predictionCount,
        request.actionableCount, request.winningActionableCount,
        request.losingActionableCount, request.grossPositiveReturn,
        request.grossNegativeReturn, request.aggregateReturn,
        request.averageReturn, request.inferenceAccuracy,
        result.outcomeIdentityCanonical, result.outcomeIdentityHash});
    result.created = !inserted.empty();
    if (inserted.empty())
        inserted = transaction.exec(
            "SELECT prospective_outcome_result_id,outcome_identity_hash FROM "
            "campaign_profitability_prospective_outcome_result WHERE "
            "outcome_identity_canonical=$1", pqxx::params{
                result.outcomeIdentityCanonical});
    if (inserted.size() != 1)
        throw std::runtime_error("phase12_outcome_idempotency_failure");
    result.resultId = inserted.one_row()[0].as<long long>();
    if (inserted.one_row().size() > 1 &&
        inserted.one_row()[1].as<std::string>() != result.outcomeIdentityHash)
        throw std::runtime_error("phase12_outcome_identity_hash_collision");
    return result;
}

} // namespace EA::ProfitabilityVerification
