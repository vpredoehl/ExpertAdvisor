#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"
#include "LSTM.hpp"
#include "ModelInputExpansion.hpp"
#include "LaunchArguments.hpp"

namespace EA::CheckpointModelPersistence
{

std::optional<long long> SavePeriodicCheckpointIfDue(
    const EA::LaunchArgs& launchArgs,
    const std::optional<long long>& parentModelId,
    const std::optional<EA::InputWidthExpansionProvenance>&
        inputWidthExpansionProvenance,
    const std::string& rawPriceTableName,
    const std::string& fromDate,
    const std::string& toDate,
    EA::LSTM& lstm,
    Donchian20Mode donchian20Mode,
    EA::FeatureWarmupScope featureWarmupScope,
    std::size_t donchianLookback,
    const EA::TrainingObjective::Configuration& trainingObjective);

} // namespace EA::CheckpointModelPersistence

