#pragma once

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationScoring.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

// Phase 4B Step 1 is an advisory classification envelope around the existing
// Phase 4A Step 4 scoring policy. It deliberately does not define another
// scoring formula.
struct RecommendationEvaluationPolicy
{
    int evaluationVersion = 1;
    int evaluatorVersion = 1;
    RecommendationScoringPolicy scoringPolicy;
};

std::optional<std::string> ValidateRecommendationEvaluationPolicy(
    const RecommendationEvaluationPolicy& policy);
std::string RecommendationEvaluationPolicyCanonicalText(
    const RecommendationEvaluationPolicy& policy);
std::string RecommendationEvaluationPolicyHash(
    const RecommendationEvaluationPolicy& policy);
// Tagged deterministic accelerator for evaluation-owned canonical text.
// Callers must persist and compare the canonical text as authoritative.
std::string RecommendationEvaluationCanonicalHash(const std::string& canonical);

struct RecommendationEvaluationSemanticIdentity
{
    std::string canonical;
    std::string hash;
    int version = 0;

    bool operator==(const RecommendationEvaluationSemanticIdentity&) const = default;
};

std::optional<std::string> ValidateRecommendationEvaluationPolicyProvenance(
    const std::string& evaluationPolicyCanonical,
    const std::string& evaluationPolicyHash,
    int evaluationVersion,
    int evaluatorVersion,
    const std::string& scoringPolicyCanonical,
    const std::string& scoringPolicyHash,
    int scoringVersion);
RecommendationEvaluationSemanticIdentity
RecommendationEvaluationSemanticIdentityForPolicy(
    const RecommendationEvaluationPolicy& policy);
RecommendationEvaluationSemanticIdentity
RecommendationEvaluationSemanticIdentityFromPolicyProvenance(
    const std::string& evaluationPolicyCanonical,
    const std::string& evaluationPolicyHash,
    int evaluationVersion,
    int evaluatorVersion,
    const RecommendationScoringSemanticIdentity& scoringSemanticIdentity,
    const std::string& scoringPolicyCanonical,
    const std::string& scoringPolicyHash,
    int scoringVersion);
std::optional<std::string> ValidateRecommendationEvaluationSemanticIdentity(
    const RecommendationEvaluationSemanticIdentity& identity,
    const std::string& evaluationPolicyCanonical,
    const std::string& evaluationPolicyHash,
    int evaluationVersion,
    int evaluatorVersion,
    const RecommendationScoringSemanticIdentity& scoringSemanticIdentity,
    const std::string& scoringPolicyCanonical,
    const std::string& scoringPolicyHash,
    int scoringVersion);

enum class RecommendationEligibility
{
    eligible,
    ineligible
};

enum class RecommendationEvaluationDisposition
{
    advisoryReady,
    insufficientEvidence,
    blockedPendingDuplicate,
    blockedActiveDuplicate,
    completedDuplicate,
    staleSourceEvidence,
    unsupportedRecommendationFamily,
    invalidPersistedEvidence
};

std::string RecommendationEligibilityText(RecommendationEligibility value);
std::string RecommendationEvaluationDispositionText(
    RecommendationEvaluationDisposition value);
std::optional<RecommendationEvaluationDisposition>
ParseRecommendationEvaluationDisposition(const std::string& value);

struct RecommendationEvaluationExperimentConflict
{
    long long experimentId = -1;
    std::string status;
};

struct RecommendationEvaluationInput
{
    RecommendationScoringInput scoringInput;
    std::string recommendationSemanticHash;
    long long recommendationScanId = -1;
    std::optional<long long> sourceModelId;
    std::optional<long long> sourceAnalysisId;
    std::string sourceSymbol;
    std::string scanStatus;
    std::string sourceExperimentStatus;
    std::string sourceExperimentPhase;
    std::optional<long long> currentSourceModelId;
    std::optional<long long> currentSourceAnalysisId;
    std::string currentSourceAnalysisStatus;
    std::string currentSourceAnalysisScope;
    std::vector<RecommendationEvaluationExperimentConflict>
        exactExperimentConflicts;
    std::optional<RecommendationSource::FinalProfitabilityEvidence>
        finalProfitabilityEvidence;
};

struct RecommendationEvaluationResult
{
    long long recommendationId = -1;
    RecommendationEligibility eligibility = RecommendationEligibility::ineligible;
    RecommendationEvaluationDisposition disposition =
        RecommendationEvaluationDisposition::invalidPersistedEvidence;
    std::string reasonCode;
    std::string explanation;
    std::string evaluationPolicyCanonical;
    std::string evaluationPolicyHash;
    int evaluationVersion = 0;
    int evaluatorVersion = 0;
    std::string scoringPolicyCanonical;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    RecommendationScoringSemanticIdentity scoringSemanticIdentity;
    RecommendationEvaluationSemanticIdentity evaluationSemanticIdentity;
    std::string evidenceCanonical;
    std::string evidenceHash;
    std::string evaluationIdentityCanonical;
    std::string evaluationIdentityHash;
    std::string recommendationSemanticCanonical;
    std::string recommendationSemanticHash;
    std::optional<double> finalScore;
    std::optional<double> rawPositiveScore;
    std::optional<double> rawPenaltyScore;
    std::optional<double> rawTotalScore;
    int missingEvidenceCount = 0;
    std::vector<RecommendationScoreComponent> components;
    int rankingOrdinal = 0;
    std::optional<RecommendationSource::FinalProfitabilityEvidence>
        finalProfitabilityEvidence;
    std::string profitabilityEvidenceCanonical;
    std::string profitabilityEvidenceHash;
};

std::string RecommendationEvaluationEvidenceCanonicalText(
    const RecommendationEvaluationInput& input);
std::string RecommendationEvaluationEvidenceHash(
    const RecommendationEvaluationInput& input);
RecommendationEvaluationResult EvaluateExperimentRecommendation(
    const RecommendationEvaluationPolicy& policy,
    const RecommendationEvaluationInput& input);
std::vector<RecommendationEvaluationResult> RankRecommendationEvaluations(
    std::vector<RecommendationEvaluationResult> results);

} // namespace EA::ExperimentRecommendation
