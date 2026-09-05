#pragma once

#include "FeatureAblationPairEvaluation.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <utility>

namespace EA::FeatureAblationPairEvaluation
{

struct ComparisonCommand
{
    std::pair<long long, long long> experimentIds;
    // When present, experimentIds are CONTROL_ID:ABLATION_ID. When absent,
    // the historical consensus ABLATED_CONTROL_ID:ENABLED_TREATMENT_ID command
    // is retained as an explicit compatibility mode.
    std::optional<std::string> expectedAblationMask = std::nullopt;
};

std::string RenderComparisonOutput(const ArmEvidence& control,
                                   const ArmEvidence& ablation,
                                   const ComparisonResult& result);

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::FeatureAblationPairEvaluation
