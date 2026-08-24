#pragma once

#include "PostPairReplicationPolicy.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace EA::ReplicationCandidatePlanning
{

namespace Policy = PostPairReplicationPolicy;
namespace Pair = PairedTrainingObjectiveEvaluation;

enum class PlanStatus
{
    NoActionWait,
    NoActionRepair,
    NoActionStop,
    NoActionCoefficientEligible,
    PlanReplication,
    PlanBroaderReplication,
    InsufficientCandidateDiversity
};

enum class DiversityClass
{
    AddNewSymbolAndHorizon,
    AddNewSymbol,
    AddNewHorizon,
    NoSymbolOrHorizonDiversity
};

enum class CandidateReason
{
    AddNewSymbolAndHorizon,
    AddNewSymbol,
    AddNewHorizon,
    NoSymbolOrHorizonDiversity,
    ConfirmNegativeOnDiverseUnit,
    ReplicateClassificationConcern,
    ReplicateProfitabilityDisagreement,
    ReplicateWeakMateriality,
    ConservativeGenericReplication
};

enum class ExclusionReason
{
    DuplicatesExistingReplicationUnit,
    DuplicateCandidateIdentity,
    CandidateIdentityConflict,
    InvalidSymbol,
    InvalidHorizon,
    InvalidReplicationUnitCanonical,
    InvalidReplicationUnitIdentity,
    SymbolNotAllowed,
    HorizonNotAllowed,
    DoesNotAddRequiredDiversity
};

// replicationUnitCanonical is the complete, objective-neutral deterministic
// scientific configuration defined by the Phase 4E contract. It must include
// symbol/horizon and initialization lineage, but no experiment ID. The caller
// supplies the candidate universe; the planner never discovers availability.
struct CandidateUnit
{
    std::string symbol;
    int predictionHorizon = 0;
    std::string replicationUnitCanonical;
    std::string replicationUnitIdentity;
    bool operator==(const CandidateUnit&) const = default;
};

CandidateUnit MakeCandidateUnit(
    std::string symbol,
    int predictionHorizon,
    std::string replicationUnitCanonical);

// This Phase 4F envelope binds the Phase 4E objective-neutral replication unit
// to symbol, horizon, control objective, and objective under review.
std::string ScientificIdentityCanonical(
    const CandidateUnit& candidate,
    const Pair::ObjectiveProvenance& controlObjective,
    const Pair::ObjectiveProvenance& objectiveUnderReview);
std::string ScientificIdentity(
    const CandidateUnit& candidate,
    const Pair::ObjectiveProvenance& controlObjective,
    const Pair::ObjectiveProvenance& objectiveUnderReview);

struct CandidateConstraints
{
    // Empty vectors mean unconstrained. Non-empty vectors are exact allowlists.
    std::vector<std::string> allowedSymbols;
    std::vector<int> allowedHorizons;
};

struct PlannedCandidate
{
    CandidateUnit unit;
    std::string scientificIdentityCanonical;
    std::string scientificIdentity;
    DiversityClass diversityClass =
        DiversityClass::NoSymbolOrHorizonDiversity;
    std::vector<CandidateReason> reasons;
};

struct ExcludedCandidate
{
    std::string replicationUnitIdentity;
    ExclusionReason reason = ExclusionReason::InvalidReplicationUnitIdentity;
    bool operator==(const ExcludedCandidate&) const = default;
};

struct Plan
{
    Policy::NextAction phase4ENextAction =
        Policy::NextAction::WaitForValidResult;
    PlanStatus status = PlanStatus::NoActionWait;
    Pair::ObjectiveProvenance objectiveUnderReview;
    Pair::ObjectiveProvenance controlObjective;
    std::size_t existingValidReplicationCount = 0;
    std::size_t existingDistinctSymbolCount = 0;
    std::size_t existingDistinctHorizonCount = 0;
    std::size_t requestedCandidateLimit = 0;
    bool requiredDiversityCanBeSatisfied = false;
    std::vector<PlannedCandidate> selectedCandidates;
    std::vector<ExcludedCandidate> excludedCandidates;
    std::vector<std::string> reasons;
};

Plan MakePlan(
    const Policy::Evaluation& evaluation,
    std::vector<CandidateUnit> candidateUniverse,
    std::size_t requestedCandidateLimit,
    CandidateConstraints constraints = {});

std::string PlanStatusText(PlanStatus value);
std::string DiversityClassText(DiversityClass value);
std::string CandidateReasonText(CandidateReason value);
std::string ExclusionReasonText(ExclusionReason value);
std::string RenderMachineReadable(const Plan& plan);

} // namespace EA::ReplicationCandidatePlanning
