#pragma once

#include "ExperimentRecommendationConversionActivation.hpp"
#include "ExperimentRecommendationConversionExecutionRepository.hpp"

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

enum class RecommendationConversionActivationAssessmentState
{
    eligible,
    existingIdentical,
    notFound,
    invalidState,
    conflict
};

struct RecommendationConversionActivationAssessment
{
    RecommendationConversionActivationAssessmentState state =
        RecommendationConversionActivationAssessmentState::eligible;
    std::optional<PersistedRecommendationConversionExecution> execution;
    std::optional<PersistedRecommendationConversionActivation> activation;
    std::string experimentStatus;
    std::string experimentPhase;
    std::string reason;
};

std::string RecommendationConversionActivationOutcomeText(
    RecommendationConversionActivationOutcome outcome);

bool RecommendationConversionActivationSchemaExists(
    pqxx::connection& connection);

// Direct and aggregate activation callers share these exact lock domains.
void LockRecommendationConversionActivationSequence(
    pqxx::transaction_base& transaction,
    long long executionId);
void LockRecommendationConversionActivationExperiment(
    pqxx::transaction_base& transaction,
    long long experimentId);

// Read-only assessment of the authoritative Phase 4C activation contract.
// lockExperiment must be false for a dry run. A write caller may use true or
// hold the same experiment row lock already.
RecommendationConversionActivationAssessment
AssessRecommendationConversionActivation(
    pqxx::transaction_base& transaction,
    long long executionId,
    bool lockExperiment);

// The caller owns the transaction and must hold the execution activation lock.
// This primitive never starts, commits, or aborts a transaction.
RecommendationConversionActivationResult
ActivateRecommendationConversionExecutionInTransaction(
    pqxx::transaction_base& transaction,
    long long executionId);

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

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivationByExecution(
    pqxx::transaction_base& transaction,
    long long executionId);

} // namespace EA::ExperimentRecommendation
