#pragma once

#include "FeatureAblationReplicationEvaluation.hpp"

#include <iosfwd>
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
};

std::string RenderComparisonOutput(const ReplicationEvaluation& result);

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::FeatureAblationReplicationEvaluation
