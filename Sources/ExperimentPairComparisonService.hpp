#pragma once

#include "ExperimentPairComparison.hpp"

#include <iosfwd>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace EA::ExperimentPairComparison
{

struct ComparisonCommand
{
    std::pair<long long, long long> experimentIds;
    bool summary = false;
};

class EvidenceSource
{
public:
    virtual ~EvidenceSource() = default;
    virtual FeatureAblationPairEvaluation::ArmEvidence Load(
        long long experimentId) const = 0;
};

class EvidenceUnavailableError final : public std::runtime_error
{
public:
    explicit EvidenceUnavailableError(std::string reason);
    const std::string& reason() const noexcept { return reason_; }

private:
    std::string reason_;
};

std::pair<long long, long long> ParseExperimentIdPair(std::string_view text);

// The initial generic policy permits only the unambiguous control-versus-
// ablation case: exactly one arm has an empty feature-ablation mask. Every
// other scientific difference remains visible to Compare as unexpected.
Request MakeComparisonRequest(
    const FeatureAblationPairEvaluation::ArmEvidence& armA,
    const FeatureAblationPairEvaluation::ArmEvidence& armB);

// Fixture-friendly read-only service seam. A successful report is exit 0,
// including incomplete or scientifically incompatible reports. Evidence load
// failures are exit 3.
int RunComparisonCommand(const ComparisonCommand& command,
                         const EvidenceSource& source,
                         std::ostream& output,
                         std::ostream& errors);

// PostgreSQL adapter used by the CLI. Database failures intentionally
// propagate to the scheduler CLI's established exit-2 handler.
int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::ExperimentPairComparison
