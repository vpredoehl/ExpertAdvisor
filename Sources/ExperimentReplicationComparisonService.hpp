#pragma once

#include "ExperimentPairComparisonService.hpp"
#include "ExperimentReplicationComparison.hpp"

#include <iosfwd>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace EA::ExperimentReplicationComparison
{

struct ComparisonCommand
{
    std::vector<std::pair<long long, long long>> experimentPairs;
};

std::vector<std::pair<long long, long long>> ParseExperimentIdPairs(
    std::string_view text);

// Fixture-friendly read-only service seam. A successful report is exit 0,
// including incomplete, incompatible, and undetermined reports. Authoritative
// evidence load failures are exit 3.
int RunComparisonCommand(
    const ComparisonCommand& command,
    const ExperimentPairComparison::EvidenceSource& source,
    std::ostream& output,
    std::ostream& errors);

// PostgreSQL adapter used by the CLI. All members are loaded through the
// existing authoritative selector inside one repeatable-read transaction.
int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::ExperimentReplicationComparison
