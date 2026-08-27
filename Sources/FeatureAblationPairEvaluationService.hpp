#pragma once

#include "FeatureAblationPairEvaluation.hpp"

#include <iosfwd>
#include <string>
#include <utility>

namespace EA::FeatureAblationPairEvaluation
{

struct ComparisonCommand
{
    std::pair<long long, long long> experimentIds;
};

std::string RenderComparisonOutput(const ArmEvidence& control,
                                   const ArmEvidence& treatment,
                                   const ComparisonResult& result);

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors);

} // namespace EA::FeatureAblationPairEvaluation
