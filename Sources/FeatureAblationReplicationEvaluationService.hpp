#pragma once

#include "FeatureAblationReplicationEvaluation.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace EA::FeatureAblationReplicationEvaluation
{

struct ComparisonCommand
{
    std::vector<std::pair<long long, long long>> experimentIdPairs;
    ReplicationPolicy policy;
    SoftwareReadinessAudit softwareAudit;
    // Present: modern CONTROL_ID:ABLATION_ID semantics for the exact mask.
    // Absent: legacy consensus ABLATED_ID:ENABLED_ID compatibility mode.
    std::optional<std::string> expectedAblationMask;
};

std::string RenderComparisonOutput(const ReplicationEvaluation& result);

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::FeatureAblationReplicationEvaluation
