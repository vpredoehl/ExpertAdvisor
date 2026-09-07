#pragma once

#include "FeatureAblationReplicationEvaluation.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pqxx/pqxx>

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

// Shared read-only evaluation primitive. The caller owns the transaction;
// this function performs no persistence and permits a larger workflow to bind
// continuation gating to the exact same authoritative aggregate evaluator.
ReplicationEvaluation EvaluateComparison(
    pqxx::transaction_base& transaction,
    const ComparisonCommand& command);

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::FeatureAblationReplicationEvaluation
