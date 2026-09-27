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

struct Result
{
    std::vector<Pair::ComparisonResult> pairs;
    Compatibility compatibility =
        Compatibility::UndeterminedDueToMissingEvidence;
    std::vector<std::string> compatibilityReasons;
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
std::string SeedReplicationModeText(SeedReplicationMode value);

} // namespace EA::ExperimentReplicationComparison
