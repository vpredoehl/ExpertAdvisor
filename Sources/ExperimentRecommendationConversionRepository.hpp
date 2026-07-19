#pragma once

#include "ExperimentRecommendationConversion.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int kMaximumRecommendationConversionProposalListLimit = 1000;

enum class RecommendationConversionProposalPersistOutcome
{
    created,
    existingIdentical,
    createdWithHashCollision
};

struct PersistedRecommendationConversionProposal
{
    long long proposalId = -1;
    int conversionContractVersion = 0;
    int hashCollisionOrdinal = 0;
    ProposedExperimentSpecification proposal;
    std::string createdAt;
};

struct RecommendationConversionProposalPersistResult
{
    RecommendationConversionProposalPersistOutcome outcome =
        RecommendationConversionProposalPersistOutcome::created;
    PersistedRecommendationConversionProposal persisted;
};

std::string RecommendationConversionProposalPersistOutcomeText(
    RecommendationConversionProposalPersistOutcome value);

bool RecommendationConversionProposalSchemaExists(pqxx::connection& connection);

RecommendationConversionProposalPersistResult
PersistRecommendationConversionProposal(
    pqxx::connection& connection,
    const ProposedExperimentSpecification& proposal);

std::optional<PersistedRecommendationConversionProposal>
FindRecommendationConversionProposal(
    pqxx::connection& connection,
    long long proposalId);

// Transaction-bound overload for callers that must keep proposal validation
// and a dependent write in one database transaction.
std::optional<PersistedRecommendationConversionProposal>
FindRecommendationConversionProposal(
    pqxx::transaction_base& transaction,
    long long proposalId);

std::optional<PersistedRecommendationConversionProposal>
FindRecommendationConversionProposalByIdentity(
    pqxx::connection& connection,
    const std::string& conversionIdentityCanonical,
    const std::string& conversionIdentityHash);

std::vector<PersistedRecommendationConversionProposal>
ListRecommendationConversionProposalsByRecommendation(
    pqxx::connection& connection,
    long long recommendationId,
    int limit = 100);

std::vector<PersistedRecommendationConversionProposal>
ListRecommendationConversionProposalsBySourceExperiment(
    pqxx::connection& connection,
    long long sourceExperimentId,
    int limit = 100);

} // namespace EA::ExperimentRecommendation
