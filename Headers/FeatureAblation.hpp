#ifndef FeatureAblation_hpp
#define FeatureAblation_hpp

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "FeatureLayout.hpp"

namespace EA
{

// Persist names, never physical Tensor offsets.  Offsets remain an implementation
// detail and are checked against the persisted model-input contract at use time.
struct AblatableFeature
{
    std::string_view name;
    std::size_t tensorColumn;
};

inline constexpr std::array<AblatableFeature, 19> kAblatableFeatures{{
    {"relative_tick_volume", relativeTickVolumeCol},
    {"rms_return_surprise", causalReturnSurpriseCol},
    {"volatility_regime", causalVolatilityRegimeCol},
    {"directional_range", causalDirectionalRangeCol},
    {"close_location", causalCloseLocationCol},
    {"return_sign_persistence", causalReturnSignPersistenceCol},
    {"return_direction_imbalance", causalReturnDirectionImbalanceCol},
    {"multi_bar_range_pressure", causalMultiBarRangePressureCol},
    {"rolling_range_expansion", causalRollingRangeExpansionCol},
    {"historical_level_proximity", historicalLevelProximityCol},
    {"return_autocorrelation", returnAutocorrelationCol},
    {"relevant_event_has_consensus", relevantEventHasConsensusCol},
    {"relevant_event_consensus_low", relevantEventConsensusLowCol},
    {"relevant_event_consensus_high", relevantEventConsensusHighCol},
    {"relevant_event_consensus_is_range", relevantEventConsensusIsRangeCol},
    {"authoritative_initial_has_surprise",
     authoritativeInitialHasSurpriseCol},
    {"authoritative_initial_surprise", authoritativeInitialSurpriseCol},
    {"authoritative_initial_surprise_abs",
     authoritativeInitialSurpriseAbsCol},
    {"authoritative_initial_surprise_direction",
     authoritativeInitialSurpriseDirectionCol},
    // directional_efficiency is the historic semantic name for the column
    // introduced as causalDirectionalPersistenceCol.
    // Kept in registry order at its physical location.
}};

// The controlled consensus experiment uses all four active channels together.
// Each remains individually named in persisted provenance; this constant keeps
// the scientific control arm deterministic without adding group/alias parsing.
inline constexpr std::string_view kEconomicEventConsensusAblationMaskText =
    "relevant_event_has_consensus,relevant_event_consensus_low,"
    "relevant_event_consensus_high,relevant_event_consensus_is_range";

inline constexpr std::string_view
    kEconomicEventReleaseActualAblationMaskText =
        "authoritative_initial_has_surprise,authoritative_initial_surprise,"
        "authoritative_initial_surprise_abs,"
        "authoritative_initial_surprise_direction";

// The registry is deliberately split because the existing layout uses a legacy
// implementation identifier for directional efficiency.
inline constexpr AblatableFeature kDirectionalEfficiencyFeature{
    "directional_efficiency", causalDirectionalPersistenceCol};
inline constexpr AblatableFeature kDirectionalAdverseExcursionFeature{
    "directional_adverse_excursion", causalDirectionalAdverseExcursionCol};

inline std::string TrimFeatureAblationToken(std::string value)
{
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

inline const AblatableFeature* FindAblatableFeature(std::string_view name)
{
    for (const auto& feature : kAblatableFeatures)
        if (feature.name == name) return &feature;
    if (kDirectionalEfficiencyFeature.name == name) return &kDirectionalEfficiencyFeature;
    if (kDirectionalAdverseExcursionFeature.name == name) return &kDirectionalAdverseExcursionFeature;
    return nullptr;
}

inline std::size_t AblatableFeatureRegistryOrder(const AblatableFeature& feature)
{
    return feature.tensorColumn;
}

class FeatureAblationMask
{
public:
    FeatureAblationMask() = default;

    static FeatureAblationMask Parse(const std::string& persistedText)
    {
        FeatureAblationMask result;
        if (persistedText.empty()) return result;
        std::size_t start = 0;
        while (start <= persistedText.size())
        {
            const std::size_t end = persistedText.find(',', start);
            const std::string token = TrimFeatureAblationToken(
                persistedText.substr(start, end == std::string::npos ? std::string::npos : end - start));
            if (token.empty())
                throw std::invalid_argument("FEATURE_ABLATION_MASK_INVALID: empty feature name");
            const AblatableFeature* feature = FindAblatableFeature(token);
            if (feature == nullptr)
                throw std::invalid_argument("FEATURE_ABLATION_MASK_UNKNOWN_FEATURE:" + token);
            if (std::find(result.columns_.begin(), result.columns_.end(), feature->tensorColumn) == result.columns_.end())
                result.columns_.push_back(feature->tensorColumn);
            if (end == std::string::npos) break;
            start = end + 1;
        }
        std::sort(result.columns_.begin(), result.columns_.end());
        return result;
    }

    bool empty() const { return columns_.empty(); }
    const std::vector<std::size_t>& tensorColumns() const { return columns_; }

    std::string CanonicalText() const
    {
        std::string result;
        for (const std::size_t column : columns_)
        {
            const AblatableFeature* found = nullptr;
            for (const auto& feature : kAblatableFeatures)
                if (feature.tensorColumn == column) found = &feature;
            if (column == kDirectionalEfficiencyFeature.tensorColumn) found = &kDirectionalEfficiencyFeature;
            if (column == kDirectionalAdverseExcursionFeature.tensorColumn) found = &kDirectionalAdverseExcursionFeature;
            if (found == nullptr) throw std::logic_error("FEATURE_ABLATION_MASK_REGISTRY_CORRUPT");
            if (!result.empty()) result += ',';
            result += found->name;
        }
        return result;
    }

    void ValidateForTensorFeatureCount(std::size_t tensorFeatureCount) const
    {
        for (const std::size_t column : columns_)
            if (column >= tensorFeatureCount)
                throw std::runtime_error("FEATURE_ABLATION_MASK_FEATURE_ABSENT_FROM_MODEL_INPUT: " + CanonicalText());
    }

    template <typename T>
    void ApplyToProjectedTensorFeatures(T* destination,
                                        std::size_t tensorFeatureCount) const
    {
        ValidateForTensorFeatureCount(tensorFeatureCount);
        for (const std::size_t column : columns_)
            destination[column] = static_cast<T>(0);
    }

private:
    std::vector<std::size_t> columns_;
};

} // namespace EA

#endif
