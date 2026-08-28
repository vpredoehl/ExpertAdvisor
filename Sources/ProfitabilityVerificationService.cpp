#include "ProfitabilityVerificationService.hpp"

#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ProfitabilityVerificationRepository.hpp"

#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <pqxx/pqxx>

namespace EA::ProfitabilityVerification
{
namespace
{

namespace Recommendation = EA::ExperimentRecommendation;

std::string Boolean(bool value)
{
    return value ? "true" : "false";
}

std::string OptionalId(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalCount(const std::optional<std::uint64_t>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string Number(double value)
{
    if (!std::isfinite(value)) return "NONFINITE";
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << (value == 0.0 ? 0.0 : value);
    return output.str();
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const unsigned char byte = static_cast<unsigned char>(character);
        const bool safe = (byte >= 'a' && byte <= 'z') ||
            (byte >= 'A' && byte <= 'Z') ||
            (byte >= '0' && byte <= '9') || character == '_' ||
            character == '-' || character == '.' || character == ':';
        if (!safe) character = '_';
    }
    return value.empty() ? "NULL" : value;
}

void RenderEvidence(std::ostream& output, const EvidenceResult& result)
{
    const auto& observation = result.observation;
    const std::optional<std::uint64_t> predictionCount = observation
        ? std::optional<std::uint64_t>{observation->statistics.predictionCount}
        : std::nullopt;
    const std::optional<std::uint64_t> actionableCount = observation
        ? std::optional<std::uint64_t>{observation->statistics.actionableCount}
        : std::nullopt;
    output << "PROFITABILITY_EVIDENCE"
           << ",experiment_id=" << result.experimentId
           << ",final_model_id=" << OptionalId(result.finalModelId)
           << ",final_inference_result_id="
           << OptionalId(result.finalInferenceEvalResultId)
           << ",profitability_observation_id="
           << (observation ? std::to_string(observation->observationId) : "NULL")
           << ",inference_scope="
           << (observation
                   ? InferenceProfitability::ScopeText(
                         observation->provenance.scope) : "final")
           << ",checkpoint_eval_id="
           << (observation
                   ? OptionalId(observation->provenance.checkpointEvalId) : "NULL")
           << ",metric_definition_canonical="
           << InferenceProfitability::kMetricDefinitionCanonical
           << ",metric_definition_hash="
           << InferenceProfitability::MetricDefinitionHash()
           << ",inference_start="
           << (observation ? observation->provenance.inferenceStart : "NULL")
           << ",inference_end="
           << (observation ? observation->provenance.inferenceEnd : "NULL")
           << ",prediction_count=" << OptionalCount(predictionCount)
           << ",actionable_count=" << OptionalCount(actionableCount)
           << ",aggregate_terminal_horizon_log_return_sum="
           << (observation
                   ? Number(observation->statistics
                         .aggregateTerminalHorizonLogReturnSum) : "NULL")
           << ",average_terminal_horizon_log_return_per_actionable_prediction="
           << (observation
                   ? OptionalNumber(observation
                         ->averageTerminalHorizonLogReturnPerActionablePrediction)
                   : "NULL")
           << ",observation_identity_hash="
           << (observation ? observation->observationIdentityHash : "NULL")
           << ",evidence_identity_hash=" << result.evidenceIdentityHash
           << ",evidence_status=" << EvidenceStateText(result.state)
           << ",reason=" << MachineText(result.reason)
           << ",read_only=true\n";
}

struct Coverage
{
    std::size_t candidateCount = 0;
    std::size_t validCount = 0;
    std::size_t unavailableCount = 0;
    std::size_t incompleteCount = 0;
    std::size_t invalidCount = 0;
    std::size_t zeroActionableCount = 0;
    std::size_t positiveAggregateCount = 0;
    std::size_t negativeAggregateCount = 0;
    std::size_t zeroAggregateCount = 0;
    std::size_t positiveAverageCount = 0;
    std::size_t negativeAverageCount = 0;
    std::size_t zeroAverageCount = 0;
    std::map<std::string, std::size_t> metricDefinitions;
};

Coverage Summarize(const ShadowRanking& shadow)
{
    Coverage coverage;
    coverage.candidateCount = shadow.candidates.size();
    for (const auto& candidate : shadow.candidates)
    {
        const auto& result = candidate.profitability;
        if (result.state == EvidenceState::valid)
            ++coverage.validCount;
        else if (result.state == EvidenceState::unavailable)
            ++coverage.unavailableCount;
        else if (result.state == EvidenceState::incomplete)
            ++coverage.incompleteCount;
        else
            ++coverage.invalidCount;
        if (!result.observation) continue;
        const auto& observation = *result.observation;
        ++coverage.metricDefinitions[observation.metricDefinitionHash];
        if (observation.statistics.actionableCount == 0)
            ++coverage.zeroActionableCount;
        const double aggregate =
            observation.statistics.aggregateTerminalHorizonLogReturnSum;
        aggregate > 0.0 ? ++coverage.positiveAggregateCount
            : aggregate < 0.0 ? ++coverage.negativeAggregateCount
                              : ++coverage.zeroAggregateCount;
        if (observation
                .averageTerminalHorizonLogReturnPerActionablePrediction)
        {
            const double average = *observation
                .averageTerminalHorizonLogReturnPerActionablePrediction;
            average > 0.0 ? ++coverage.positiveAverageCount
                : average < 0.0 ? ++coverage.negativeAverageCount
                                : ++coverage.zeroAverageCount;
        }
    }
    return coverage;
}

} // namespace

int RunVerificationCommand(const std::string& connectionString,
                           const std::vector<long long>& experimentIds,
                           std::ostream& output,
                           std::ostream& errors)
{
    (void)errors;
    if (experimentIds.empty())
        throw std::invalid_argument("profitability_experiment_set_empty");
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    std::vector<EvidenceResult> results;
    results.reserve(experimentIds.size());
    for (const long long experimentId : experimentIds)
    {
        results.push_back(
            LoadAndVerifyExactFinalEvidence(transaction, experimentId));
        RenderEvidence(output, results.back());
    }
    const int exitCode = ExitCode(results);
    output << "PROFITABILITY_VERIFICATION_SUMMARY"
           << ",declared_experiment_count=" << results.size()
           << ",valid_count="
           << std::count_if(results.begin(), results.end(), [](const auto& r) {
                  return r.state == EvidenceState::valid; })
           << ",unavailable_or_incomplete_count="
           << std::count_if(results.begin(), results.end(), [](const auto& r) {
                  return r.state == EvidenceState::unavailable ||
                      r.state == EvidenceState::incomplete; })
           << ",invalid_or_ambiguous_count="
           << std::count_if(results.begin(), results.end(), [](const auto& r) {
                  return EvidenceStateIsInvalid(r.state); })
           << ",software_success=true,read_only=true,exit_code="
           << exitCode << '\n';
    return exitCode;
}

int RunCampaignReadinessCommand(const std::string& connectionString,
                                long long rankingSnapshotId,
                                std::ostream& output,
                                std::ostream& errors)
{
    (void)errors;
    if (rankingSnapshotId <= 0)
        throw std::invalid_argument("invalid_campaign_ranking_snapshot_id");
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    if (!Recommendation::RecommendationCampaignPlanningSchemasExist(transaction) ||
        !InferenceProfitability::SchemaExists(transaction))
        throw std::runtime_error("profitability_readiness_schema_unavailable");

    Recommendation::RecommendationCampaignPlanningPolicy planningPolicy;
    planningPolicy.maximumCandidatesConsidered =
        Recommendation::kMaximumRecommendationCampaignCandidates;
    Recommendation::RecommendationCampaignPlanningScope scope;
    scope.rankingSnapshotId = rankingSnapshotId;
    const auto input = Recommendation::LoadRecommendationCampaignPlanInput(
        transaction, planningPolicy, scope);

    std::vector<ShadowCandidate> candidates;
    candidates.reserve(input.candidates.size());
    for (const auto& source : input.candidates)
    {
        EvidenceResult evidence = LoadAndVerifyExactFinalEvidence(
            transaction, source.sourceExperimentId);
        evidence = EnforceFrozenCampaignEvidence(
            source.sourceModelId,
            source.finalProfitabilityEvidence,
            std::move(evidence));

        ShadowCandidate candidate;
        candidate.rankingMemberId = source.rankingMemberId;
        candidate.recommendationId = source.recommendationId;
        candidate.sourceExperimentId = source.sourceExperimentId;
        candidate.sourceModelId = source.sourceModelId;
        candidate.currentRank = source.rankingPosition;
        candidate.currentScore = source.rankingScore;
        candidate.leaderScore = source.leaderScore;
        candidate.inferenceAccuracy = source.inferenceAccuracy;
        candidate.predictedNeutralProportion =
            source.predictedNeutralProportion;
        candidate.profitability = std::move(evidence);
        candidates.push_back(std::move(candidate));
    }

    const ShadowRanking shadow = BuildShadowRanking(std::move(candidates));
    for (const auto& candidate : shadow.candidates)
    {
        const auto& observation = candidate.profitability.observation;
        output << "CAMPAIGN_PROFITABILITY_SHADOW_CANDIDATE"
               << ",ranking_snapshot_id=" << rankingSnapshotId
               << ",ranking_member_id=" << candidate.rankingMemberId
               << ",recommendation_id=" << candidate.recommendationId
               << ",source_experiment_id="
               << candidate.sourceExperimentId
               << ",source_model_id=" << OptionalId(candidate.sourceModelId)
               << ",current_rank=" << candidate.currentRank
               << ",current_score=" << OptionalNumber(candidate.currentScore)
               << ",leader_score=" << Number(candidate.leaderScore)
               << ",inference_accuracy="
               << Number(candidate.inferenceAccuracy)
               << ",predicted_neutral_proportion="
               << OptionalNumber(candidate.predictedNeutralProportion)
               << ",profitability_evidence_available="
               << Boolean(candidate.profitability.state == EvidenceState::valid)
               << ",profitability_evidence_status="
               << EvidenceStateText(candidate.profitability.state)
               << ",profitability_sign="
               << ProfitabilitySignText(candidate.profitabilitySign)
               << ",actionable_count="
               << (observation
                       ? std::to_string(
                             observation->statistics.actionableCount) : "NULL")
               << ",aggregate_terminal_horizon_log_return_sum="
               << (observation
                       ? Number(observation->statistics
                             .aggregateTerminalHorizonLogReturnSum) : "NULL")
               << ",average_terminal_horizon_log_return_per_actionable_prediction="
               << (observation
                       ? OptionalNumber(observation
                             ->averageTerminalHorizonLogReturnPerActionablePrediction)
                       : "NULL")
               << ",profitability_evidence_identity="
               << candidate.profitability.evidenceIdentityHash
               << ",profitability_shadow_rank="
               << candidate.profitabilityShadowRank
               << ",rank_delta=" << candidate.rankDelta
               << ",live_rank_authoritative=true"
               << ",candidate_hash=" << candidate.hash << '\n';
    }

    const Coverage coverage = Summarize(shadow);
    output << "CAMPAIGN_PROFITABILITY_COVERAGE"
           << ",ranking_snapshot_id=" << rankingSnapshotId
           << ",candidate_count=" << coverage.candidateCount
           << ",exact_final_profitability_count=" << coverage.validCount
           << ",missing_profitability_count=" << coverage.unavailableCount
           << ",incomplete_final_evidence_count=" << coverage.incompleteCount
           << ",invalid_or_ambiguous_count=" << coverage.invalidCount
           << ",zero_actionable_count=" << coverage.zeroActionableCount
           << ",positive_aggregate_count=" << coverage.positiveAggregateCount
           << ",negative_aggregate_count=" << coverage.negativeAggregateCount
           << ",zero_aggregate_count=" << coverage.zeroAggregateCount
           << ",positive_average_count=" << coverage.positiveAverageCount
           << ",negative_average_count=" << coverage.negativeAverageCount
           << ",zero_average_count=" << coverage.zeroAverageCount
           << ",final_evidence_count=" << coverage.validCount
           << ",checkpoint_evidence_count=0"
           << ",current_rank_changed=false,read_only=true\n";
    for (const auto& [hash, count] : coverage.metricDefinitions)
        output << "CAMPAIGN_PROFITABILITY_METRIC_DEFINITION"
               << ",metric_definition_hash=" << hash
               << ",candidate_count=" << count << '\n';

    const ReadinessGate gate = EvaluateReadinessGate(true, true, shadow);
    output << "CAMPAIGN_PROFITABILITY_SHADOW_POLICY"
           << ",version=" << shadow.policyVersion
           << ",policy_hash=" << shadow.policyHash
           << ",canonical=" << shadow.policyCanonical
           << ",shadow_ranking_hash=" << shadow.hash
           << ",live_profitability_weight=0"
           << ",live_profitability_score_contribution=0\n";
    output << "CAMPAIGN_PROFITABILITY_READINESS_GATE"
           << ",profitability_software_ready="
           << Boolean(gate.profitabilitySoftwareReady)
           << ",campaign_profitability_contract_ready="
           << Boolean(gate.campaignProfitabilityContractReady)
           << ",profitability_shadow_ranking_ready="
           << Boolean(gate.profitabilityShadowRankingReady)
           << ",campaign_profitability_activation_performed=false"
           << ",campaign_profitability_action="
           << ReadinessActionText(gate.action)
           << ",live_ranking_changed=false,read_only=true\n";

    std::vector<EvidenceResult> evidence;
    evidence.reserve(shadow.candidates.size());
    for (const auto& candidate : shadow.candidates)
        evidence.push_back(candidate.profitability);
    return ExitCode(evidence);
}

} // namespace EA::ProfitabilityVerification
