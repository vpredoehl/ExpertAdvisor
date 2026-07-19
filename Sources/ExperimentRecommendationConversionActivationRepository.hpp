#pragma once

#include "ExperimentRecommendationConversionActivation.hpp"

#include <optional>
#include <string>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

enum class RecommendationConversionActivationOutcome
{
    activated,
    existingIdentical,
    notFound,
    invalidState,
    conflict
};

struct PersistedRecommendationConversionActivation
{
    long long activationId = -1;
    long long executionId = -1;
    long long proposalId = -1;
    long long reviewDecisionId = -1;
    long long experimentId = -1;
    int activationContractVersion = 0;
    std::string previousStatus;
    std::string previousPhase;
    std::string resultingStatus;
    std::string resultingPhase;
    std::string activationIdentityCanonical;
    std::string activationIdentityHash;
    std::string createdAt;
};

struct RecommendationConversionActivationResult
{
    RecommendationConversionActivationOutcome outcome =
        RecommendationConversionActivationOutcome::activated;
    std::optional<PersistedRecommendationConversionActivation> activation;
    std::string reason;
};

std::string RecommendationConversionActivationOutcomeText(
    RecommendationConversionActivationOutcome outcome);

bool RecommendationConversionActivationSchemaExists(
    pqxx::connection& connection);

RecommendationConversionActivationResult
ActivateRecommendationConversionExecution(
    pqxx::connection& connection,
    long long executionId);

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivation(
    pqxx::connection& connection,
    long long activationId);

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivationByExecution(
    pqxx::connection& connection,
    long long executionId);

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivationByExperiment(
    pqxx::connection& connection,
    long long experimentId);

} // namespace EA::ExperimentRecommendation
