#pragma once

#include "ExperimentPairComparisonService.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::FeatureAblationFactorialComparison
{

// The generic calculation has two declared binary factors.  The public CLI
// gives them the study-specific labels counts and ages, but neither the
// compatibility checks nor the arithmetic rely on those labels.
struct Design
{
    std::string factorALabel = "counts";
    std::string factorBLabel = "ages";
    // Y11, Y01, Y10, Y00, in that order.  These are canonical requested
    // feature-ablation expressions and must be distinct.
    std::array<std::string, 4> expectedMasks;
};

struct SeedCells
{
    // Y11 (A on, B on), Y01 (A off, B on), Y10 (A on, B off), Y00.
    std::array<long long, 4> experimentIds;
};

struct Command
{
    Design design;
    std::vector<SeedCells> seedCells;
};

struct Effect
{
    std::optional<double> factorAMainEffect;
    std::optional<double> factorBMainEffect;
    std::optional<double> interaction;
};

struct SeedResult
{
    std::size_t ordinal = 0;
    std::array<long long, 4> experimentIds;
    std::optional<std::string> freshInitializationSeed;
    bool eligible = false;
    std::vector<std::string> reasons;
    // The three comparisons retain the existing pair-comparison evidence for
    // Y11:Y01, Y11:Y10, and Y11:Y00 respectively.
    std::array<ExperimentPairComparison::ComparisonResult, 3> comparisons;
    std::array<std::optional<double>, 14> cell11Metrics;
    std::array<std::optional<double>, 14> cell01Metrics;
    std::array<std::optional<double>, 14> cell10Metrics;
    std::array<std::optional<double>, 14> cell00Metrics;
    std::array<Effect, 14> effects;
};

struct MetricSummary
{
    std::string name;
    std::size_t eligibleSeedCount = 0;
    std::size_t availableSeedCount = 0;
    std::optional<double> descriptiveFactorAMainEffectMean;
    std::optional<double> descriptiveFactorBMainEffectMean;
    std::optional<double> descriptiveInteractionMean;
};

struct Report
{
    Design design;
    std::vector<SeedResult> seeds;
    std::size_t declaredSeedCount = 0;
    std::size_t eligibleSeedCount = 0;
    std::vector<MetricSummary> metrics;
};

// Argument grammar:
//   [FACTOR_A:FACTOR_B|]Y11:Y01:Y10:Y00;Y11:Y01:Y10:Y00@MASK11/MASK01/MASK10/MASK00
// FACTOR_A and FACTOR_B are optional lowercase machine identifiers. Omitting
// them retains the original counts/ages labels for existing invocations.
// The four masks are canonicalized before evidence is loaded.  ';', '@', and
// '/' are deliberately not feature-ablation token characters, while commas
// remain available inside an individual mask expression.
Command ParseCommand(std::string_view text);

Report Evaluate(const Command& command,
                const ExperimentPairComparison::EvidenceSource& source);
std::string Render(const Report& report);

int RunCommand(const Command& command,
               const ExperimentPairComparison::EvidenceSource& source,
               std::ostream& output,
               std::ostream& errors);
int RunCommand(const std::string& connectionString,
               const Command& command,
               std::ostream& output,
               std::ostream& errors);

} // namespace EA::FeatureAblationFactorialComparison
