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

// Non-locking policy lookup for status, ranking, dry-run, and other reads.
std::optional<ContinuationPolicyConfig> FindContinuationPolicyConfig(
    pqxx::work& transaction,
    long long sourceExperimentId);

// Explicit row-locking lookup for policy mutation, decision persistence, and
// the authoritative transactional queue path.
std::optional<ContinuationPolicyConfig> LockContinuationPolicyConfigForUpdate(
    pqxx::work& transaction,
    long long sourceExperimentId);

// Optimization-only lookup. The supplied transaction must already be READ
// ONLY. This function never locks rows for update or performs database writes.
ContinuationAutoPreflightLookup LoadAutomaticContinuationPreflightReadOnly(
    pqxx::work& transaction,
    long long sourceExperimentId);

std::string FormatAutomaticContinuationSatisfiedFields(
    const ContinuationAutoPreflightLookup& preflight,
    bool dryRun);

} // namespace EA::ExperimentScheduler
