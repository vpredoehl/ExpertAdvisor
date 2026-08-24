#pragma once

#include "PairedTrainingObjectiveEvaluation.hpp"

#include <iosfwd>
#include <string>
#include <utility>

namespace EA::PairedTrainingObjectiveEvaluation
{

struct ComparisonCommand
{
    std::pair<long long, long long> experimentIds;
    MaterialityPolicy policy;
};

std::string RenderComparisonOutput(
    const ArmEvidence& control,
    const ArmEvidence& treatment,
    const MaterialityPolicy& policy,
    const ComparisonResult& result);

// Exit codes: 0=evaluator disposition (including INVALID_COMPARISON or
// INCOMPLETE), 3=not found/ambiguous/invalid persisted evidence. Database
// exceptions intentionally propagate to the repository-consistent CLI's
// database-error handler (exit 2).
int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::PairedTrainingObjectiveEvaluation
