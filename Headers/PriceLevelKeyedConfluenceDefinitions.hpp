#pragma once

#include "PriceLevelMarketStructureObservationAdapter.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

// Frozen, descriptive-only Price-Level relationships.  This header merely
// declares definitions for explicit callers; it does not invoke a detector,
// publish a worker, project into Tensor, or alter raw observations.
namespace EA::MarketStructure::PriceLevelBridge
{
inline constexpr std::string_view kReinforcedThenRetestDefinitionId =
    "price-level-reinforced-then-retest-cooccurrence";
inline constexpr std::string_view kReinforcedThenRetestDefinitionVersion = "v1";
inline constexpr std::size_t kMaxCorrelationKeysPerEvaluation = 64;

// The output says only that the exact frozen level identity had a reinforced
// observation available before a retest observation.  It does not say why the
// retest occurred or imply support, resistance, direction, continuation,
// reversal, strength, confidence, or a trading result.
inline std::array<ConfluenceDefinition, 1> FrozenKeyedConfluenceDefinitions()
{
    return {{
        {std::string{kReinforcedThenRetestDefinitionId},
         std::string{kReinforcedThenRetestDefinitionVersion},
         RelationKind::co_occurrence,
         {std::string{kFamilyId}, "level_reinforced"},
         {std::string{kFamilyId}, "retest"},
         1,
         CorrelationConstraint::exact_key_equality,
         TemporalPredicate::left_available_at_before_right_available_at,
         kMaxCorrelationKeysPerEvaluation},
    }};
}

} // namespace EA::MarketStructure::PriceLevelBridge
