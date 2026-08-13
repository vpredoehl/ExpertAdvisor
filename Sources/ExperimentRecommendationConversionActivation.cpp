#include "ExperimentRecommendationConversionActivation.hpp"

#include "ExperimentRecommendation.hpp"

#include <locale>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

void ValidateIdentityText(const std::string& value)
{
    if (value.empty() || value.size() > 256 ||
        value.find('\0') != std::string::npos)
        throw std::invalid_argument(
            "recommendation_conversion_activation_execution_identity_invalid");
}

} // namespace

RecommendationConversionActivationIdentity
BuildRecommendationConversionActivationIdentity(
    const RecommendationConversionActivationIdentityInput& input)
{
    if (input.executionId <= 0 || input.proposalId <= 0 ||
        input.reviewDecisionId <= 0 || input.experimentId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_activation_identity_id_invalid");
    ValidateIdentityText(input.executionIdentityHash);

    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical
        << "experiment_recommendation_conversion_activation_v1"
        << ";contract_version="
        << kRecommendationConversionActivationContractVersion
        << ";execution_id=" << input.executionId
        << ";proposal_id=" << input.proposalId
        << ";review_decision_id=" << input.reviewDecisionId
        << ";experiment_id=" << input.experimentId
        << ";previous_status="
        << kRecommendationConversionActivationPreviousStatus
        << ";previous_phase="
        << kRecommendationConversionActivationPreviousPhase
        << ";resulting_status="
        << kRecommendationConversionActivationResultingStatus
        << ";resulting_phase="
        << kRecommendationConversionActivationResultingPhase
        << ";execution_identity_hash="
        << LengthText(input.executionIdentityHash);

    RecommendationConversionActivationIdentity result;
    result.canonicalText = canonical.str();
    result.hash = RecommendationCanonicalHash(result.canonicalText);
    return result;
}

} // namespace EA::ExperimentRecommendation
