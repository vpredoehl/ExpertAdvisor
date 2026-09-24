#pragma once

#include "ExperimentPairComparison.hpp"
#include "ExperimentReplicationComparison.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace EA::ExperimentReplicationPlanning
{

namespace Pair = ExperimentPairComparison;

inline constexpr int kPlannerVersion = 1;

enum class PlanState
{
    Valid,
    Invalid,
    UndeterminedDueToMissingEvidence
};

enum class EquivalentExperimentState
{
    NoEquivalentExperimentFound,
    EquivalentExperimentFound,
    EquivalentExperimentAmbiguous
};

struct EquivalentExperimentResult
{
    EquivalentExperimentState state =
        EquivalentExperimentState::NoEquivalentExperimentFound;
    std::vector<long long> experimentIds;
    std::string reason;
    bool operator==(const EquivalentExperimentResult&) const = default;
};

// This is the single authoritative proposed-experiment representation shared
// by planning, equivalence validation, rendering, and persistence.  The base
// value is the exact configured scientific identity copied from the source;
// the added fields are the authoritative persistence recipe.  Persistence
// clones that source experiment's configured columns and applies exactly this
// seed -- it never reconstructs configuration from rendered output.
struct ProposedExperimentSpecification : Pair::ArmResultSet
{
    long long authoritativeSourceExperimentId = 0;
    unsigned int freshInitializationSeed = 0;
};

class EquivalentExperimentSource
{
public:
    virtual ~EquivalentExperimentSource() = default;
    virtual EquivalentExperimentResult FindEquivalent(
        const Pair::ArmResultSet& proposedArm) const = 0;
};

struct ArmPlan
{
    std::string role;
    long long sourceExperimentId = 0;
    ProposedExperimentSpecification proposed;
    std::vector<Pair::IdentityDifference> changedFromSource;
    EquivalentExperimentResult equivalent;
};

struct PlannedPair
{
    std::size_t ordinal = 0;
    unsigned int requestedSeed = 0;
    bool sourceSeedReuse = false;
    std::string replicationClassification;
    ArmPlan armA;
    ArmPlan armB;
    PlanState preflightState = PlanState::UndeterminedDueToMissingEvidence;
    std::vector<std::string> reasons;
    std::vector<Pair::IdentityDifference> intentionalDifferences;
    std::vector<Pair::IdentityDifference> unexpectedDifferences;
};

struct Plan
{
    int plannerVersion = kPlannerVersion;
    long long sourceExperimentAId = 0;
    long long sourceExperimentBId = 0;
    unsigned int sourceSeed = 0;
    bool sourceSeedAvailable = false;
    std::vector<unsigned int> requestedSeeds;
    Pair::ArmResultSet sourceArmA;
    Pair::ArmResultSet sourceArmB;
    Pair::Request comparisonRequest;
    std::vector<Pair::IdentityDifference> sourceIntentionalDifferences;
    std::vector<Pair::IdentityDifference> sourceUnexpectedDifferences;
    PlanState state = PlanState::UndeterminedDueToMissingEvidence;
    std::vector<std::string> reasons;
    std::vector<PlannedPair> pairs;
};

// Strict positive uint32 parsing. Input order is retained and duplicates are
// rejected rather than collapsed.
std::vector<unsigned int> ParseReplicationSeeds(std::string_view text);

Plan MakePlan(const Pair::ArmResultSet& sourceArmA,
              const Pair::ArmResultSet& sourceArmB,
              const Pair::Request& comparisonRequest,
              const std::vector<unsigned int>& requestedSeeds,
              const EquivalentExperimentSource* equivalents = nullptr);

// Re-evaluates every exact identity invariant. This is public so future
// materialization code can validate a transported plan before acting on it.
void RecomputePreflight(Plan& plan);

std::string Render(const Plan& plan);
std::string PlanStateText(PlanState state);
std::string EquivalentExperimentStateText(EquivalentExperimentState state);

} // namespace EA::ExperimentReplicationPlanning
