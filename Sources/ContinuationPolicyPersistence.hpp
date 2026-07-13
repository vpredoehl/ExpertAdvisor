#pragma once

#include <optional>
#include <string>

#include <pqxx/pqxx>

#include "ContinuationPolicyInheritance.hpp"

namespace EA::ExperimentScheduler
{

struct ContinuationAutoPreflightLookup
{
    ContinuationPolicyConfig currentPolicy;
    std::optional<PersistedContinuationIdentity> persistedIdentity;
    ContinuationAutoSatisfactionResult satisfaction;
};

bool ContinuationPolicySchemaExists(pqxx::work& transaction);

std::optional<ContinuationPolicyConfig> LoadContinuationPolicyConfig(
    pqxx::work& transaction,
    long long sourceExperimentId,
    bool lockRow);

ContinuationAutoPreflightLookup LoadAutomaticContinuationPreflight(
    pqxx::work& transaction,
    long long sourceExperimentId);

std::string FormatAutomaticContinuationSatisfiedFields(
    const ContinuationAutoPreflightLookup& preflight,
    bool dryRun);

} // namespace EA::ExperimentScheduler
