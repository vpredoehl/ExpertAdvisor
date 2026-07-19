#pragma once

#include <string>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationConversionActivationContractVersion = 1;
inline constexpr const char* kRecommendationConversionActivationPreviousStatus =
    "paused";
inline constexpr const char* kRecommendationConversionActivationPreviousPhase =
    "train";
inline constexpr const char* kRecommendationConversionActivationResultingStatus =
    "pending";
inline constexpr const char* kRecommendationConversionActivationResultingPhase =
    "train";

struct RecommendationConversionActivationIdentityInput
{
    long long executionId = -1;
    long long proposalId = -1;
    long long reviewDecisionId = -1;
    long long experimentId = -1;
    std::string executionIdentityHash;
};

struct RecommendationConversionActivationIdentity
{
    std::string canonicalText;
    std::string hash;
};

RecommendationConversionActivationIdentity
BuildRecommendationConversionActivationIdentity(
    const RecommendationConversionActivationIdentityInput& input);

} // namespace EA::ExperimentRecommendation
