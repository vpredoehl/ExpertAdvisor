#ifndef MarketStructureRegistry_hpp
#define MarketStructureRegistry_hpp

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "FeatureLayout.hpp"

namespace EA::MarketStructure
{

// This catalog intentionally contains only producers which already emit real
// Tensor channels.  Adding a family is a semantic-layout change, not a way to
// reserve names for speculative detectors.
struct Family
{
    std::string_view id;
    int semanticVersion;
    std::string_view detectorVersion;
    std::string_view causalAvailability;
};

struct Channel
{
    // Hierarchical identity used in configuration and documentation.
    std::string_view featureId;
    // Immutable pre-registry experiment identity.  Historical layouts retain
    // this concrete name in feature_ablation_mask.
    std::string_view persistedFeatureId;
    std::string_view familyId;
    std::size_t tensorColumn;
    int introducedSemanticLayout;
};

inline constexpr std::array<Family, 3> kFamilies{{
    {"tg_structure", 1, "tg4-production-pulse-v1",
     "completed canonical bar; never retrospectively rewritten"},
    // Layout 9 is retained as an immutable legacy producer. It composes the
    // pre-existing TG1/TG3 research primitives and therefore is not evidence
    // of the independent detector contract required for future layouts.
    {"fibonacci", 1, "layout9-legacy-tg1-tg3-composed-fibonacci-v1",
     "confirmed state at the completed decision bar"},
    // Phase Pocket 2 already has a causal detector and explicit information
    // cutoff. Layout 10 appends its frozen Tensor channels; layouts 8 and 9
    // remain unavailable by the explicit channel-layout contract below.
    {"pockets", 1, "causal-pocket-detector-phase2-v1",
     "confirmation bar / information cutoff timestamp"},
}};

// These identities describe actual layout-8, layout-9, and layout-10 channels. They do
// not claim that TG3's historical research confluence is a generic confluence
// feature family: no generic explicit-confluence Tensor channels exist yet.
inline constexpr std::array<Channel, 37> kChannels{{
    {"tg_structure.tg4.inner_break.any", "tg4_inner_break_any",
     "tg_structure", tg4InnerBreakAnyCol, 8},
    {"tg_structure.tg4.source_tg3.structurally_eligible",
     "tg4_source_tg3_structurally_eligible", "tg_structure",
     tg4SourceTg3StructurallyEligibleCol, 8},
    {"tg_structure.tg4.source_tg3.confluent", "tg4_source_tg3_confluent",
     "tg_structure", tg4SourceTg3ConfluentCol, 8},
    {"fibonacci.recent.price_scale_valid", "fib_recent_price_scale_valid",
     "fibonacci", fibRecentPriceScaleValidCol, 9},
    {"fibonacci.up.recent.union_count_log", "fib_up_recent_union_count_log",
     "fibonacci", fibUpRecentUnionCountLogCol, 9},
    {"fibonacci.up.recent.h1_count_log", "fib_up_recent_h1_count_log",
     "fibonacci", fibUpRecentH1CountLogCol, 9},
    {"fibonacci.up.recent.h2_count_log", "fib_up_recent_h2_count_log",
     "fibonacci", fibUpRecentH2CountLogCol, 9},
    {"fibonacci.up.recent.h1_h2_both_count_log",
     "fib_up_recent_h1_h2_both_count_log", "fibonacci",
     fibUpRecentH1H2BothCountLogCol, 9},
    {"fibonacci.up.recent.h1_youngest_age_20",
     "fib_up_recent_h1_youngest_age_20", "fibonacci",
     fibUpRecentH1YoungestAge20Col, 9},
    {"fibonacci.up.recent.h2_youngest_age_20",
     "fib_up_recent_h2_youngest_age_20", "fibonacci",
     fibUpRecentH2YoungestAge20Col, 9},
    {"fibonacci.up.recent.median_extension_1272_signed_atr",
     "fib_up_recent_median_1272_signed_atr", "fibonacci",
     fibUpRecentMedian1272Col, 9},
    {"fibonacci.up.recent.median_extension_1618_signed_atr",
     "fib_up_recent_median_1618_signed_atr", "fibonacci",
     fibUpRecentMedian1618Col, 9},
    {"fibonacci.up.recent.median_pullback_0382_signed_atr",
     "fib_up_recent_median_pullback_0382_signed_atr", "fibonacci",
     fibUpRecentMedianPullback0382Col, 9},
    {"fibonacci.up.recent.median_pullback_0500_signed_atr",
     "fib_up_recent_median_pullback_0500_signed_atr", "fibonacci",
     fibUpRecentMedianPullback0500Col, 9},
    {"fibonacci.up.recent.median_pullback_0618_signed_atr",
     "fib_up_recent_median_pullback_0618_signed_atr", "fibonacci",
     fibUpRecentMedianPullback0618Col, 9},
    {"fibonacci.down.recent.union_count_log", "fib_down_recent_union_count_log",
     "fibonacci", fibDownRecentUnionCountLogCol, 9},
    {"fibonacci.down.recent.h1_count_log", "fib_down_recent_h1_count_log",
     "fibonacci", fibDownRecentH1CountLogCol, 9},
    {"fibonacci.down.recent.h2_count_log", "fib_down_recent_h2_count_log",
     "fibonacci", fibDownRecentH2CountLogCol, 9},
    {"fibonacci.down.recent.h1_h2_both_count_log",
     "fib_down_recent_h1_h2_both_count_log", "fibonacci",
     fibDownRecentH1H2BothCountLogCol, 9},
    {"fibonacci.down.recent.h1_youngest_age_20",
     "fib_down_recent_h1_youngest_age_20", "fibonacci",
     fibDownRecentH1YoungestAge20Col, 9},
    {"fibonacci.down.recent.h2_youngest_age_20",
     "fib_down_recent_h2_youngest_age_20", "fibonacci",
     fibDownRecentH2YoungestAge20Col, 9},
    {"fibonacci.down.recent.median_extension_1272_signed_atr",
     "fib_down_recent_median_1272_signed_atr", "fibonacci",
     fibDownRecentMedian1272Col, 9},
    {"fibonacci.down.recent.median_extension_1618_signed_atr",
     "fib_down_recent_median_1618_signed_atr", "fibonacci",
     fibDownRecentMedian1618Col, 9},
    {"fibonacci.down.recent.median_pullback_0382_signed_atr",
     "fib_down_recent_median_pullback_0382_signed_atr", "fibonacci",
     fibDownRecentMedianPullback0382Col, 9},
    {"fibonacci.down.recent.median_pullback_0500_signed_atr",
     "fib_down_recent_median_pullback_0500_signed_atr", "fibonacci",
     fibDownRecentMedianPullback0500Col, 9},
    {"fibonacci.down.recent.median_pullback_0618_signed_atr",
     "fib_down_recent_median_pullback_0618_signed_atr", "fibonacci",
     fibDownRecentMedianPullback0618Col, 9},
    {"pockets.recent.price_scale_valid", "pocket_recent_price_scale_valid",
     "pockets", pocketRecentPriceScaleValidCol, 10},
    {"pockets.bull.recent.count_log", "pocket_bull_recent_count_log",
     "pockets", pocketBullRecentCountLogCol, 10},
    {"pockets.bull.recent.youngest_age_20", "pocket_bull_youngest_age20",
     "pockets", pocketBullYoungestAge20Col, 10},
    {"pockets.bull.recent.median_touch_distance",
     "pocket_bull_median_touch_distance", "pockets",
     pocketBullMedianTouchDistanceCol, 10},
    {"pockets.bull.recent.median_close_distance",
     "pocket_bull_median_close_distance", "pockets",
     pocketBullMedianCloseDistanceCol, 10},
    {"pockets.bull.recent.median_width", "pocket_bull_median_width",
     "pockets", pocketBullMedianWidthCol, 10},
    {"pockets.bear.recent.count_log", "pocket_bear_recent_count_log",
     "pockets", pocketBearRecentCountLogCol, 10},
    {"pockets.bear.recent.youngest_age_20", "pocket_bear_youngest_age20",
     "pockets", pocketBearYoungestAge20Col, 10},
    {"pockets.bear.recent.median_touch_distance",
     "pocket_bear_median_touch_distance", "pockets",
     pocketBearMedianTouchDistanceCol, 10},
    {"pockets.bear.recent.median_close_distance",
     "pocket_bear_median_close_distance", "pockets",
     pocketBearMedianCloseDistanceCol, 10},
    {"pockets.bear.recent.median_width", "pocket_bear_median_width",
     "pockets", pocketBearMedianWidthCol, 10},
}};

// Keep catalog validation separate from lookup so focused tests can exercise
// malformed future catalogs without modifying the production registry.  The
// lookup namespace is deliberately shared by hierarchical and historical
// persisted names: a collision between either form would make identity
// resolution ambiguous and must never depend on declaration order.
inline void ValidateCatalog(std::span<const Family> families,
                            std::span<const Channel> channels)
{
    for (std::size_t left = 0; left < families.size(); ++left)
    {
        const Family& family = families[left];
        if (family.id.empty() || family.detectorVersion.empty() ||
            family.causalAvailability.empty())
        {
            throw std::invalid_argument("MARKET_STRUCTURE_FAMILY_INVALID");
        }
        for (std::size_t right = left + 1; right < families.size(); ++right)
        {
            if (family.id == families[right].id)
                throw std::invalid_argument("MARKET_STRUCTURE_DUPLICATE_FAMILY");
        }
    }

    for (std::size_t left = 0; left < channels.size(); ++left)
    {
        const Channel& channel = channels[left];
        if (channel.featureId.empty() || channel.persistedFeatureId.empty() ||
            channel.familyId.empty() || channel.introducedSemanticLayout <= 0)
        {
            throw std::invalid_argument("MARKET_STRUCTURE_CHANNEL_INVALID");
        }
        const bool familyExists = std::any_of(
            families.begin(), families.end(), [&channel](const Family& family) {
                return family.id == channel.familyId;
            });
        if (!familyExists)
            throw std::invalid_argument("MARKET_STRUCTURE_CHANNEL_UNKNOWN_FAMILY");

        for (std::size_t right = left + 1; right < channels.size(); ++right)
        {
            const Channel& other = channels[right];
            if (channel.tensorColumn == other.tensorColumn)
                throw std::invalid_argument("MARKET_STRUCTURE_DUPLICATE_TENSOR_COLUMN");
            if (channel.featureId == other.featureId ||
                channel.persistedFeatureId == other.persistedFeatureId ||
                channel.featureId == other.persistedFeatureId ||
                channel.persistedFeatureId == other.featureId)
            {
                throw std::invalid_argument(
                    "MARKET_STRUCTURE_AMBIGUOUS_CHANNEL_IDENTITY");
            }
        }
    }
}

inline void ValidateRegistry()
{
    ValidateCatalog(kFamilies, kChannels);
}

inline const Channel* FindChannel(std::string_view id)
{
    ValidateRegistry();
    const auto found = std::find_if(kChannels.begin(), kChannels.end(),
        [id](const Channel& channel) {
            return channel.featureId == id || channel.persistedFeatureId == id;
        });
    return found == kChannels.end() ? nullptr : &*found;
}

inline const Family* FindFamily(std::string_view id)
{
    ValidateRegistry();
    const auto found = std::find_if(kFamilies.begin(), kFamilies.end(),
        [id](const Family& family) { return family.id == id; });
    return found == kFamilies.end() ? nullptr : &*found;
}

inline bool ChannelAvailableForSemanticLayout(const Channel& channel,
                                               int semanticLayoutVersion)
{
    // Layouts 8 and 9 are append-only descendants for these channels.  A
    // future branch must be deliberately added here rather than inheriting a
    // family wildcard merely because it happens to have the same width.
    switch (semanticLayoutVersion)
    {
        case 8: return channel.introducedSemanticLayout == 8;
        case 9: return channel.introducedSemanticLayout == 8 ||
                    channel.introducedSemanticLayout == 9;
        case 10: return channel.introducedSemanticLayout == 8 ||
                     channel.introducedSemanticLayout == 9 ||
                     channel.introducedSemanticLayout == 10;
        default: return false;
    }
}

inline std::vector<const Channel*> ResolvePrefix(std::string_view prefix,
                                                  int semanticLayoutVersion = 10)
{
    ValidateRegistry();
    std::vector<const Channel*> result;
    const std::string prefixWithSeparator = std::string{prefix} + ".";
    for (const Channel& channel : kChannels)
    {
        if (channel.featureId.starts_with(prefixWithSeparator) &&
            ChannelAvailableForSemanticLayout(channel, semanticLayoutVersion))
        {
            result.push_back(&channel);
        }
    }
    return result;
}

inline std::vector<std::string_view> RegisteredFamilyIdsForSemanticLayout(
    int semanticLayoutVersion)
{
    ValidateRegistry();
    std::vector<std::string_view> result;
    for (const Family& family : kFamilies)
    {
        const bool hasChannel = std::any_of(kChannels.begin(), kChannels.end(),
            [&family, semanticLayoutVersion](const Channel& channel) {
                return channel.familyId == family.id &&
                    ChannelAvailableForSemanticLayout(
                        channel, semanticLayoutVersion);
            });
        if (hasChannel) result.push_back(family.id);
    }
    return result;
}

// Observations deliberately carry availability independently from occurrence.
// Detectors produce these objects; confluence may only read a filtered copy.
struct Observation
{
    std::string_view familyId;
    std::string detectorVersion;
    std::chrono::sys_seconds observedAt{};
    std::chrono::sys_seconds availableAt{};
    std::string sourceProvenance;

    bool operator==(const Observation&) const = default;
};

inline void ValidateObservation(const Observation& observation)
{
    if (FindFamily(observation.familyId) == nullptr)
        throw std::invalid_argument("MARKET_STRUCTURE_UNKNOWN_FAMILY");
    if (observation.detectorVersion.empty() ||
        observation.sourceProvenance.empty())
        throw std::invalid_argument("MARKET_STRUCTURE_PROVENANCE_INCOMPLETE");
    if (observation.availableAt < observation.observedAt)
        throw std::invalid_argument("MARKET_STRUCTURE_CAUSAL_AVAILABILITY_INVALID");
}

inline std::vector<Observation> CausallyAvailableObservations(
    const std::vector<Observation>& observations,
    std::chrono::sys_seconds decisionTime)
{
    std::vector<Observation> result;
    for (const Observation& observation : observations)
    {
        ValidateObservation(observation);
        if (observation.availableAt <= decisionTime)
            result.push_back(observation);
    }
    return result;
}

// A future confluence engine returns separate descriptive observations.  It
// receives immutable detector observations and has no facility to alter them.
struct ConfluenceObservation
{
    std::string confluenceVersion;
    std::chrono::sys_seconds availableAt{};
    std::vector<Observation> components;
};

class ConfluenceEngine
{
public:
    virtual ~ConfluenceEngine() = default;
    virtual std::vector<ConfluenceObservation> Describe(
        const std::vector<Observation>& availableObservations,
        std::chrono::sys_seconds decisionTime) const = 0;
};

} // namespace EA::MarketStructure

#endif
