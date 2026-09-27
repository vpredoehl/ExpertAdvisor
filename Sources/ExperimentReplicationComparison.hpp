#pragma once

#include "ExperimentPairComparison.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentReplicationComparison
{

namespace Pair = ExperimentPairComparison;

enum class Compatibility
{
    Compatible,
    Incompatible,
    UndeterminedDueToMissingEvidence
};

// These assessments are deliberately separate from the existing strict
// Compatibility result. A configured identity establishes only that the
// intended controlled comparison was configured; it does not establish that
// completed observations were produced by homogeneous executables.
enum class EvidenceCompatibility
{
    Compatible,
    Incompatible,
    UndeterminedDueToMissingEvidence,
    InvalidEvidence
};

enum class SeedReplicationMode
{
    DifferentSeeds,
    SameSeedRepeatedPairs,
    UnavailableOrNotApplicable
};

struct ReplicationDimension
{
    std::size_t pairOrdinal = 0;
    std::optional<std::string> armASeed;
    std::optional<std::string> armBSeed;
};

struct MetricAggregate
{
    std::string name;
    std::size_t pairCount = 0;
    std::size_t availableCount = 0;
    std::vector<std::optional<double>> pairDeltas;
    std::optional<double> descriptiveMean;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::size_t positiveCount = 0;
    std::size_t zeroCount = 0;
    std::size_t negativeCount = 0;
};

struct PairCompatibilityAssessment
{
    EvidenceCompatibility configuredScientificIdentity =
        EvidenceCompatibility::UndeterminedDueToMissingEvidence;
    EvidenceCompatibility executionProvenance =
        EvidenceCompatibility::UndeterminedDueToMissingEvidence;
    std::vector<std::string> configuredScientificIdentityReasons;
    std::vector<std::string> executionProvenanceReasons;
    bool strictCompletedCompatible = false;
};

struct StrictSubsetExclusion
{
    std::size_t pairOrdinal = 0;
    long long experimentAId = 0;
    long long experimentBId = 0;
    std::optional<std::string> armASeed;
    std::optional<std::string> armBSeed;
    std::vector<std::string> reasons;
};

// The strict subset is an explicitly disclosed descriptive aggregate. It is
// available only for at least two completed pairs with homogeneous configured
// identity and producing execution provenance. It never silently discards a
// configured pair.
struct StrictCompletedSubset
{
    std::size_t totalConfiguredPairCount = 0;
    std::size_t strictCompletedCompatiblePairCount = 0;
    std::vector<StrictSubsetExclusion> exclusions;
    Compatibility aggregateCompatibility =
        Compatibility::UndeterminedDueToMissingEvidence;
    std::vector<std::string> aggregateReasons;
    std::vector<MetricAggregate> metrics;
};

struct Result
{
    std::vector<Pair::ComparisonResult> pairs;
    std::vector<PairCompatibilityAssessment> pairCompatibility;
    Compatibility compatibility =
        Compatibility::UndeterminedDueToMissingEvidence;
    std::vector<std::string> compatibilityReasons;
    EvidenceCompatibility configuredScientificIdentity =
        EvidenceCompatibility::UndeterminedDueToMissingEvidence;
    std::vector<std::string> configuredScientificIdentityReasons;
    StrictCompletedSubset strictCompletedSubset;
    std::vector<ReplicationDimension> replicationDimensions;
    SeedReplicationMode seedReplicationMode =
        SeedReplicationMode::UnavailableOrNotApplicable;
    std::vector<MetricAggregate> metrics;
};

// Several replication families may be rendered together, but each family is
// evaluated independently. This deliberately has no cross-family metric
// aggregate: a caller must not pool symbol-specific paired runs as one
// homogeneous population.
struct FamilyResult
{
    Result replication;
    std::optional<std::string> homogeneousSymbol;
};

struct FamilyReport
{
    std::vector<FamilyResult> families;
    std::size_t distinctHomogeneousSymbolCount = 0;
};

Result Compare(std::vector<Pair::ComparisonResult> pairs);
FamilyReport CompareFamilies(
    std::vector<std::vector<Pair::ComparisonResult>> families);
std::string Render(const Result& result);
std::string RenderFamilyReport(const FamilyReport& report);
std::string CompatibilityText(Compatibility value);
std::string EvidenceCompatibilityText(EvidenceCompatibility value);
std::string SeedReplicationModeText(SeedReplicationMode value);

} // namespace EA::ExperimentReplicationComparison
