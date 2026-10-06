#ifndef MarketStructureRegistry_hpp
#define MarketStructureRegistry_hpp

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <map>
#include <optional>
#include <stdexcept>
#include <span>
#include <sstream>
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

inline constexpr std::array<Family, 6> kFamilies{{
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
    {"confluence", 1, "phase2b-fixed-descriptive-confluence-v1",
     "completed canonical bar; no earlier than every selected component"},
    {"price_level_structure", 1, "causal-price-level/v2",
     "post-AddCompletedBar completed decision bar"},
    {"fibonacci_lifecycle", 1, "causal-fibonacci-retracement-lifecycle-1272-d-v1",
     "completed decision bar; retained independently until 1.272 D terminal"},
}};

// These identities describe actual layout-8 through layout-11 channels.
// TG3's historical research confluence remains distinct from the generic
// descriptive confluence family introduced by the fixed layout-11 projection.
inline constexpr std::array<Channel, 94> kChannels{{
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
    {"confluence.tg4.structural_fibonacci_retracement.support_available",
     "confluence_tg4_structural_fibonacci_retracement_support_available",
     "confluence", confluenceTg4StructuralFibonacciRetracementSupportAvailableCol,
     11},
    {"confluence.tg4.structural_fibonacci_retracement.contradiction_available",
     "confluence_tg4_structural_fibonacci_retracement_contradiction_available",
     "confluence", confluenceTg4StructuralFibonacciRetracementContradictionAvailableCol,
     11},
    {"price_level_structure.available", "available", "price_level_structure",
     priceLevelAvailableCol, 12},
    {"price_level_structure.zone_scale_valid", "zone_scale_valid", "price_level_structure",
     priceLevelZoneScaleValidCol, 12},
    {"price_level_structure.zone_gap_signed_clipped", "zone_gap_signed_clipped", "price_level_structure",
     priceLevelZoneGapSignedClippedCol, 12},
    {"price_level_structure.zone_relation", "zone_relation", "price_level_structure",
     priceLevelZoneRelationCol, 12},
    {"price_level_structure.current_role", "current_role", "price_level_structure",
     priceLevelCurrentRoleCol, 12},
    {"price_level_structure.age_fraction", "age_fraction", "price_level_structure",
     priceLevelAgeFractionCol, 12},
    {"price_level_structure.prior_evidence_saturation", "prior_evidence_saturation", "price_level_structure",
     priceLevelPriorEvidenceSaturationCol, 12},
    {"price_level_structure.touch_now", "touch_now", "price_level_structure",
     priceLevelTouchNowCol, 12},
    {"price_level_structure.cross_direction_now", "cross_direction_now", "price_level_structure",
     priceLevelCrossDirectionNowCol, 12},
    {"price_level_structure.retest_now", "retest_now", "price_level_structure",
     priceLevelRetestNowCol, 12},
    {"price_level_structure.role_reversal_now", "role_reversal_now", "price_level_structure",
     priceLevelRoleReversalNowCol, 12},
    {"fibonacci_lifecycle.up.0382.reached_count_log", "fib_lifecycle_up_0382_reached_count_log", "fibonacci_lifecycle", fibLifecycleUp0382ReachedCountLogCol, 13},
    {"fibonacci_lifecycle.up.0382.directional_close_count_log", "fib_lifecycle_up_0382_directional_close_count_log", "fibonacci_lifecycle", fibLifecycleUp0382DirectionalCloseCountLogCol, 13},
    {"fibonacci_lifecycle.up.0382.directional_break_count_log", "fib_lifecycle_up_0382_directional_break_count_log", "fibonacci_lifecycle", fibLifecycleUp0382DirectionalBreakCountLogCol, 13},
    {"fibonacci_lifecycle.up.0382.close_back_through_count_log", "fib_lifecycle_up_0382_close_back_through_count_log", "fibonacci_lifecycle", fibLifecycleUp0382CloseBackThroughCountLogCol, 13},
    {"fibonacci_lifecycle.up.0382.youngest_reach_age_log1p", "fib_lifecycle_up_0382_youngest_reach_age_log1p", "fibonacci_lifecycle", fibLifecycleUp0382ReachedYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.0382.youngest_directional_close_age_log1p", "fib_lifecycle_up_0382_youngest_directional_close_age_log1p", "fibonacci_lifecycle", fibLifecycleUp0382DirectionalCloseYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.0500.reached_count_log", "fib_lifecycle_up_0500_reached_count_log", "fibonacci_lifecycle", fibLifecycleUp0500ReachedCountLogCol, 13},
    {"fibonacci_lifecycle.up.0500.directional_close_count_log", "fib_lifecycle_up_0500_directional_close_count_log", "fibonacci_lifecycle", fibLifecycleUp0500DirectionalCloseCountLogCol, 13},
    {"fibonacci_lifecycle.up.0500.directional_break_count_log", "fib_lifecycle_up_0500_directional_break_count_log", "fibonacci_lifecycle", fibLifecycleUp0500DirectionalBreakCountLogCol, 13},
    {"fibonacci_lifecycle.up.0500.close_back_through_count_log", "fib_lifecycle_up_0500_close_back_through_count_log", "fibonacci_lifecycle", fibLifecycleUp0500CloseBackThroughCountLogCol, 13},
    {"fibonacci_lifecycle.up.0500.youngest_reach_age_log1p", "fib_lifecycle_up_0500_youngest_reach_age_log1p", "fibonacci_lifecycle", fibLifecycleUp0500ReachedYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.0500.youngest_directional_close_age_log1p", "fib_lifecycle_up_0500_youngest_directional_close_age_log1p", "fibonacci_lifecycle", fibLifecycleUp0500DirectionalCloseYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.0618.reached_count_log", "fib_lifecycle_up_0618_reached_count_log", "fibonacci_lifecycle", fibLifecycleUp0618ReachedCountLogCol, 13},
    {"fibonacci_lifecycle.up.0618.directional_close_count_log", "fib_lifecycle_up_0618_directional_close_count_log", "fibonacci_lifecycle", fibLifecycleUp0618DirectionalCloseCountLogCol, 13},
    {"fibonacci_lifecycle.up.0618.directional_break_count_log", "fib_lifecycle_up_0618_directional_break_count_log", "fibonacci_lifecycle", fibLifecycleUp0618DirectionalBreakCountLogCol, 13},
    {"fibonacci_lifecycle.up.0618.close_back_through_count_log", "fib_lifecycle_up_0618_close_back_through_count_log", "fibonacci_lifecycle", fibLifecycleUp0618CloseBackThroughCountLogCol, 13},
    {"fibonacci_lifecycle.up.0618.youngest_reach_age_log1p", "fib_lifecycle_up_0618_youngest_reach_age_log1p", "fibonacci_lifecycle", fibLifecycleUp0618ReachedYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.0618.youngest_directional_close_age_log1p", "fib_lifecycle_up_0618_youngest_directional_close_age_log1p", "fibonacci_lifecycle", fibLifecycleUp0618DirectionalCloseYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.a_penetration_count_log", "fib_lifecycle_up_a_penetration_count_log", "fibonacci_lifecycle", fibLifecycleUpAPenetrationCountLogCol, 13},
    {"fibonacci_lifecycle.up.a_close_beyond_count_log", "fib_lifecycle_up_a_close_beyond_count_log", "fibonacci_lifecycle", fibLifecycleUpACloseBeyondCountLogCol, 13},
    {"fibonacci_lifecycle.up.youngest_a_penetration_age_log1p", "fib_lifecycle_up_youngest_a_penetration_age_log1p", "fibonacci_lifecycle", fibLifecycleUpAPenetrationYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.up.youngest_a_close_beyond_age_log1p", "fib_lifecycle_up_youngest_a_close_beyond_age_log1p", "fibonacci_lifecycle", fibLifecycleUpACloseBeyondYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.0382.reached_count_log", "fib_lifecycle_down_0382_reached_count_log", "fibonacci_lifecycle", fibLifecycleDown0382ReachedCountLogCol, 13},
    {"fibonacci_lifecycle.down.0382.directional_close_count_log", "fib_lifecycle_down_0382_directional_close_count_log", "fibonacci_lifecycle", fibLifecycleDown0382DirectionalCloseCountLogCol, 13},
    {"fibonacci_lifecycle.down.0382.directional_break_count_log", "fib_lifecycle_down_0382_directional_break_count_log", "fibonacci_lifecycle", fibLifecycleDown0382DirectionalBreakCountLogCol, 13},
    {"fibonacci_lifecycle.down.0382.close_back_through_count_log", "fib_lifecycle_down_0382_close_back_through_count_log", "fibonacci_lifecycle", fibLifecycleDown0382CloseBackThroughCountLogCol, 13},
    {"fibonacci_lifecycle.down.0382.youngest_reach_age_log1p", "fib_lifecycle_down_0382_youngest_reach_age_log1p", "fibonacci_lifecycle", fibLifecycleDown0382ReachedYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.0382.youngest_directional_close_age_log1p", "fib_lifecycle_down_0382_youngest_directional_close_age_log1p", "fibonacci_lifecycle", fibLifecycleDown0382DirectionalCloseYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.0500.reached_count_log", "fib_lifecycle_down_0500_reached_count_log", "fibonacci_lifecycle", fibLifecycleDown0500ReachedCountLogCol, 13},
    {"fibonacci_lifecycle.down.0500.directional_close_count_log", "fib_lifecycle_down_0500_directional_close_count_log", "fibonacci_lifecycle", fibLifecycleDown0500DirectionalCloseCountLogCol, 13},
    {"fibonacci_lifecycle.down.0500.directional_break_count_log", "fib_lifecycle_down_0500_directional_break_count_log", "fibonacci_lifecycle", fibLifecycleDown0500DirectionalBreakCountLogCol, 13},
    {"fibonacci_lifecycle.down.0500.close_back_through_count_log", "fib_lifecycle_down_0500_close_back_through_count_log", "fibonacci_lifecycle", fibLifecycleDown0500CloseBackThroughCountLogCol, 13},
    {"fibonacci_lifecycle.down.0500.youngest_reach_age_log1p", "fib_lifecycle_down_0500_youngest_reach_age_log1p", "fibonacci_lifecycle", fibLifecycleDown0500ReachedYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.0500.youngest_directional_close_age_log1p", "fib_lifecycle_down_0500_youngest_directional_close_age_log1p", "fibonacci_lifecycle", fibLifecycleDown0500DirectionalCloseYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.0618.reached_count_log", "fib_lifecycle_down_0618_reached_count_log", "fibonacci_lifecycle", fibLifecycleDown0618ReachedCountLogCol, 13},
    {"fibonacci_lifecycle.down.0618.directional_close_count_log", "fib_lifecycle_down_0618_directional_close_count_log", "fibonacci_lifecycle", fibLifecycleDown0618DirectionalCloseCountLogCol, 13},
    {"fibonacci_lifecycle.down.0618.directional_break_count_log", "fib_lifecycle_down_0618_directional_break_count_log", "fibonacci_lifecycle", fibLifecycleDown0618DirectionalBreakCountLogCol, 13},
    {"fibonacci_lifecycle.down.0618.close_back_through_count_log", "fib_lifecycle_down_0618_close_back_through_count_log", "fibonacci_lifecycle", fibLifecycleDown0618CloseBackThroughCountLogCol, 13},
    {"fibonacci_lifecycle.down.0618.youngest_reach_age_log1p", "fib_lifecycle_down_0618_youngest_reach_age_log1p", "fibonacci_lifecycle", fibLifecycleDown0618ReachedYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.0618.youngest_directional_close_age_log1p", "fib_lifecycle_down_0618_youngest_directional_close_age_log1p", "fibonacci_lifecycle", fibLifecycleDown0618DirectionalCloseYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.a_penetration_count_log", "fib_lifecycle_down_a_penetration_count_log", "fibonacci_lifecycle", fibLifecycleDownAPenetrationCountLogCol, 13},
    {"fibonacci_lifecycle.down.a_close_beyond_count_log", "fib_lifecycle_down_a_close_beyond_count_log", "fibonacci_lifecycle", fibLifecycleDownACloseBeyondCountLogCol, 13},
    {"fibonacci_lifecycle.down.youngest_a_penetration_age_log1p", "fib_lifecycle_down_youngest_a_penetration_age_log1p", "fibonacci_lifecycle", fibLifecycleDownAPenetrationYoungestAgeLogCol, 13},
    {"fibonacci_lifecycle.down.youngest_a_close_beyond_age_log1p", "fib_lifecycle_down_youngest_a_close_beyond_age_log1p", "fibonacci_lifecycle", fibLifecycleDownACloseBeyondYoungestAgeLogCol, 13},
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
        case 11: return channel.introducedSemanticLayout == 8 ||
                     channel.introducedSemanticLayout == 9 ||
                     channel.introducedSemanticLayout == 10 ||
                     channel.introducedSemanticLayout == 11;
        case 12: return channel.introducedSemanticLayout == 8 ||
                     channel.introducedSemanticLayout == 9 ||
                     channel.introducedSemanticLayout == 10 ||
                     channel.introducedSemanticLayout == 11 ||
                     channel.introducedSemanticLayout == 12;
        case 13: return channel.introducedSemanticLayout == 8 ||
                     channel.introducedSemanticLayout == 9 ||
                     channel.introducedSemanticLayout == 10 ||
                     channel.introducedSemanticLayout == 11 ||
                     channel.introducedSemanticLayout == 12 ||
                     channel.introducedSemanticLayout == 13;
        default: return false;
    }
}

inline std::vector<const Channel*> ResolvePrefix(std::string_view prefix,
                                                  int semanticLayoutVersion = 13)
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
// The descriptor is the detector-independent boundary: a producer declares a
// role and an explicit polarity, never a detector-specific payload.  The
// optional confidence is valid only when the producer has normalized it to
// this contract's invariant [0, 1] meaning.  The descriptive engine does not
// rank or compare confidence values.
enum class DescriptorPolarity
{
    positive,
    negative,
    neutral,
};

inline std::string_view CanonicalDescriptorPolarity(DescriptorPolarity value)
{
    switch (value)
    {
        case DescriptorPolarity::positive: return "positive";
        case DescriptorPolarity::negative: return "negative";
        case DescriptorPolarity::neutral: return "neutral";
    }
    throw std::invalid_argument("MARKET_STRUCTURE_DESCRIPTOR_POLARITY_INVALID");
}

struct Descriptor
{
    std::string schemaVersion;
    std::string role;
    DescriptorPolarity polarity = DescriptorPolarity::neutral;
    std::optional<double> normalizedConfidence;

    bool operator==(const Descriptor&) const = default;
};

inline void ValidateDescriptor(const Descriptor& descriptor)
{
    if (descriptor.schemaVersion.empty() || descriptor.role.empty())
        throw std::invalid_argument("MARKET_STRUCTURE_DESCRIPTOR_INCOMPLETE");
    if (descriptor.normalizedConfidence &&
        (!std::isfinite(*descriptor.normalizedConfidence) ||
         *descriptor.normalizedConfidence < 0.0 ||
         *descriptor.normalizedConfidence > 1.0))
    {
        throw std::invalid_argument("MARKET_STRUCTURE_DESCRIPTOR_CONFIDENCE_INVALID");
    }
    (void)CanonicalDescriptorPolarity(descriptor.polarity);
}

inline std::string CanonicalDouble(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("MARKET_STRUCTURE_NONFINITE_VALUE");
    if (value == 0.0) value = 0.0; // Canonicalize negative zero.
    std::ostringstream result;
    result.imbue(std::locale::classic());
    result << std::setprecision(17) << value;
    return result.str();
}

inline std::string LengthPrefixed(std::string_view value)
{
    return std::to_string(value.size()) + ":" + std::string{value};
}

inline std::string CanonicalTimestamp(std::chrono::sys_seconds value)
{
    return std::to_string(value.time_since_epoch().count());
}

// Correlation is explicit relationship metadata supplied by a producer.  It
// is neither a descriptor (which states an observation's meaning) nor source
// provenance (which records where it came from).  Values are opaque to the
// generic engine: it can only validate and compare the complete typed value.
struct CorrelationKey
{
    std::string type;
    std::string value;

    bool operator==(const CorrelationKey&) const = default;
};

inline void ValidateCorrelationKey(const CorrelationKey& key)
{
    constexpr std::size_t kMaximumTypeLength = 64;
    constexpr std::size_t kMaximumValueLength = 4096;
    if (key.type.empty() || key.type.size() > kMaximumTypeLength ||
        key.value.empty() || key.value.size() > kMaximumValueLength ||
        key.type.front() < 'a' || key.type.front() > 'z')
    {
        throw std::invalid_argument("MARKET_STRUCTURE_CORRELATION_KEY_INVALID");
    }
    for (const unsigned char character : key.type)
    {
        if (!((character >= 'a' && character <= 'z') ||
              (character >= '0' && character <= '9') || character == '.' ||
              character == '_' || character == '-'))
        {
            throw std::invalid_argument("MARKET_STRUCTURE_CORRELATION_KEY_INVALID");
        }
    }
    for (const unsigned char character : key.value)
    {
        if (character < 0x21 || character > 0x7e)
            throw std::invalid_argument("MARKET_STRUCTURE_CORRELATION_KEY_INVALID");
    }
}

inline std::string CanonicalCorrelationKey(const CorrelationKey& key)
{
    ValidateCorrelationKey(key);
    return "type=" + LengthPrefixed(key.type) + ";value=" +
        LengthPrefixed(key.value);
}

// A family ID at this boundary is an opaque, producer-owned identity.  It is
// intentionally not limited to kFamilies, which is a catalog of Tensor
// channels only; descriptive observations may exist before Tensor integration.
struct Observation
{
    std::string familyId;
    std::string detectorVersion;
    std::chrono::sys_seconds observedAt{};
    std::chrono::sys_seconds availableAt{};
    std::string sourceProvenance;
    std::string sourceObservationId;
    Descriptor descriptor;
    std::optional<CorrelationKey> correlationKey;

    bool operator==(const Observation&) const = default;
};

inline void ValidateObservation(const Observation& observation)
{
    if (observation.familyId.empty() || observation.detectorVersion.empty() ||
        observation.sourceProvenance.empty() ||
        observation.sourceObservationId.empty())
        throw std::invalid_argument("MARKET_STRUCTURE_PROVENANCE_INCOMPLETE");
    if (observation.availableAt < observation.observedAt)
        throw std::invalid_argument("MARKET_STRUCTURE_CAUSAL_AVAILABILITY_INVALID");
    ValidateDescriptor(observation.descriptor);
    if (observation.correlationKey)
        ValidateCorrelationKey(*observation.correlationKey);
}

inline std::string CanonicalObservationIdentity(const Observation& observation)
{
    ValidateObservation(observation);
    const std::string confidence = observation.descriptor.normalizedConfidence
        ? CanonicalDouble(*observation.descriptor.normalizedConfidence) : "absent";
    const std::string fields = "family=" + LengthPrefixed(observation.familyId) +
        ";detector=" + LengthPrefixed(observation.detectorVersion) +
        ";observed_at=" + CanonicalTimestamp(observation.observedAt) +
        ";available_at=" + CanonicalTimestamp(observation.availableAt) +
        ";provenance=" + LengthPrefixed(observation.sourceProvenance) +
        ";source_id=" + LengthPrefixed(observation.sourceObservationId) +
        ";descriptor_schema=" + LengthPrefixed(observation.descriptor.schemaVersion) +
        ";role=" + LengthPrefixed(observation.descriptor.role) +
        ";polarity=" + std::string{CanonicalDescriptorPolarity(
            observation.descriptor.polarity)} +
        ";confidence=" + confidence;
    if (!observation.correlationKey)
        return "observation-v1;" + fields;
    return "observation-v2;" + fields + ";correlation_key=" +
        LengthPrefixed(CanonicalCorrelationKey(*observation.correlationKey));
}

inline std::vector<Observation> CausallyAvailableObservations(
    const std::vector<Observation>& observations,
    std::chrono::sys_seconds decisionTime)
{
    std::vector<Observation> result;
    std::vector<std::string> identities;
    for (const Observation& observation : observations)
    {
        // Filter before validation.  A future observation, even a malformed
        // one, is outside this decision prefix and cannot alter it.
        if (observation.availableAt > decisionTime) continue;
        ValidateObservation(observation);
        const std::string identity = CanonicalObservationIdentity(observation);
        if (std::find(identities.begin(), identities.end(), identity) !=
            identities.end())
        {
            throw std::invalid_argument(
                "MARKET_STRUCTURE_DUPLICATE_OBSERVATION_IDENTITY");
        }
        identities.push_back(identity);
        result.push_back(observation);
    }
    std::sort(result.begin(), result.end(),
        [](const Observation& left, const Observation& right) {
            return CanonicalObservationIdentity(left) <
                CanonicalObservationIdentity(right);
        });
    return result;
}

enum class RelationKind
{
    support,
    contradiction,
    co_occurrence,
};

inline std::string_view CanonicalRelationKind(RelationKind value)
{
    switch (value)
    {
        case RelationKind::support: return "support";
        case RelationKind::contradiction: return "contradiction";
        case RelationKind::co_occurrence: return "co_occurrence";
    }
    throw std::invalid_argument("MARKET_STRUCTURE_RELATION_KIND_INVALID");
}

struct RoleSelector
{
    std::string familyId;
    std::string role;

    bool operator==(const RoleSelector&) const = default;
};

enum class CorrelationConstraint
{
    none,
    exact_key_equality,
};

inline std::string_view CanonicalCorrelationConstraint(
    CorrelationConstraint value)
{
    switch (value)
    {
        case CorrelationConstraint::none: return "none";
        case CorrelationConstraint::exact_key_equality:
            return "exact_key_equality";
    }
    throw std::invalid_argument("MARKET_STRUCTURE_CORRELATION_CONSTRAINT_INVALID");
}

enum class TemporalPredicate
{
    none,
    left_available_at_before_right_available_at,
};

inline std::string_view CanonicalTemporalPredicate(TemporalPredicate value)
{
    switch (value)
    {
        case TemporalPredicate::none: return "none";
        case TemporalPredicate::left_available_at_before_right_available_at:
            return "left_available_at_before_right_available_at";
    }
    throw std::invalid_argument("MARKET_STRUCTURE_TEMPORAL_PREDICATE_INVALID");
}

// The engine makes one bounded aggregate for each qualifying polarity, never
// a Cartesian product of components.  Engine construction copies and freezes
// this definition; callers cannot subsequently alter its semantics.
struct ConfluenceDefinition
{
    std::string definitionId;
    std::string definitionVersion;
    RelationKind relation = RelationKind::support;
    RoleSelector left;
    RoleSelector right;
    std::size_t maxCandidatesPerSelector = 0;
    CorrelationConstraint correlationConstraint = CorrelationConstraint::none;
    TemporalPredicate temporalPredicate = TemporalPredicate::none;
    std::size_t maxCorrelationKeys = 0;

    bool operator==(const ConfluenceDefinition&) const = default;
};

inline void ValidateConfluenceDefinition(const ConfluenceDefinition& definition)
{
    if (definition.definitionId.empty() || definition.definitionVersion.empty() ||
        definition.left.familyId.empty() || definition.left.role.empty() ||
        definition.right.familyId.empty() || definition.right.role.empty() ||
        definition.maxCandidatesPerSelector == 0 ||
        definition.left == definition.right)
    {
        throw std::invalid_argument("MARKET_STRUCTURE_CONFLUENCE_DEFINITION_INVALID");
    }
    (void)CanonicalRelationKind(definition.relation);
    (void)CanonicalCorrelationConstraint(definition.correlationConstraint);
    (void)CanonicalTemporalPredicate(definition.temporalPredicate);
    if (definition.correlationConstraint == CorrelationConstraint::none)
    {
        if (definition.relation == RelationKind::co_occurrence ||
            definition.temporalPredicate != TemporalPredicate::none ||
            definition.maxCorrelationKeys != 0)
        {
            throw std::invalid_argument("MARKET_STRUCTURE_CONFLUENCE_DEFINITION_INVALID");
        }
    }
    else
    {
        if (definition.maxCorrelationKeys == 0 ||
            (definition.temporalPredicate != TemporalPredicate::none &&
             definition.maxCandidatesPerSelector != 1))
        {
            throw std::invalid_argument("MARKET_STRUCTURE_CONFLUENCE_DEFINITION_INVALID");
        }
    }
}

inline std::string CanonicalConfluenceDefinitionIdentity(
    const ConfluenceDefinition& definition)
{
    ValidateConfluenceDefinition(definition);
    const std::string fields = "id=" + LengthPrefixed(definition.definitionId) +
        ";version=" + LengthPrefixed(definition.definitionVersion) +
        ";relation=" + std::string{CanonicalRelationKind(definition.relation)} +
        ";left_family=" + LengthPrefixed(definition.left.familyId) +
        ";left_role=" + LengthPrefixed(definition.left.role) +
        ";right_family=" + LengthPrefixed(definition.right.familyId) +
        ";right_role=" + LengthPrefixed(definition.right.role) +
        ";candidate_cap=" + std::to_string(definition.maxCandidatesPerSelector);
    if (definition.correlationConstraint == CorrelationConstraint::none)
        return "confluence-definition-v1;" + fields;
    return "confluence-definition-v2;" + fields +
        ";correlation_constraint=" + std::string{CanonicalCorrelationConstraint(
            definition.correlationConstraint)} +
        ";temporal_predicate=" + std::string{CanonicalTemporalPredicate(
            definition.temporalPredicate)} +
        ";correlation_key_cap=" + std::to_string(definition.maxCorrelationKeys);
}

struct ConfluenceResultDescriptor
{
    RelationKind relation = RelationKind::support;
    DescriptorPolarity leftPolarity = DescriptorPolarity::neutral;
    DescriptorPolarity rightPolarity = DescriptorPolarity::neutral;
    std::size_t leftComponentCount = 0;
    std::size_t rightComponentCount = 0;
    std::optional<CorrelationKey> correlationKey;
    TemporalPredicate temporalPredicate = TemporalPredicate::none;

    bool operator==(const ConfluenceResultDescriptor&) const = default;
};

// A confluence observation is derived evidence.  Components are value copies
// solely for provenance/replay; the engine never mutates or removes source
// observations.
struct ConfluenceObservation
{
    std::string outputIdentity;
    std::string definitionIdentity;
    std::string definitionId;
    std::string definitionVersion;
    std::chrono::sys_seconds decisionTime{};
    std::chrono::sys_seconds availableAt{};
    std::vector<Observation> components;
    ConfluenceResultDescriptor descriptor;

    bool operator==(const ConfluenceObservation&) const = default;
};

struct SelectorDiagnostic
{
    std::string selectorName;
    std::size_t matchingCandidateCount = 0;
    std::size_t retainedCandidateCount = 0;
    std::size_t overflowCandidateCount = 0;
    std::vector<std::string> retainedComponentIdentities;

    bool operator==(const SelectorDiagnostic&) const = default;
};

struct ConfluenceReplay
{
    std::string definitionIdentity;
    std::chrono::sys_seconds decisionTime{};
    SelectorDiagnostic left;
    SelectorDiagnostic right;
    std::vector<ConfluenceObservation> outputs;

    std::string CanonicalRepresentation() const
    {
        auto selectorText = [](const SelectorDiagnostic& selector) {
            std::string result = selector.selectorName + ";matching=" +
                std::to_string(selector.matchingCandidateCount) + ";retained=" +
                std::to_string(selector.retainedCandidateCount) + ";overflow=" +
                std::to_string(selector.overflowCandidateCount);
            for (const std::string& identity : selector.retainedComponentIdentities)
                result += ";selected=" + LengthPrefixed(identity);
            return result;
        };
        const bool legacy = definitionIdentity.starts_with("confluence-definition-v1;");
        std::string result = std::string{legacy ? "confluence-replay-v1;definition=" :
            "confluence-replay-v2;definition="} +
            LengthPrefixed(definitionIdentity) + ";decision_time=" +
            CanonicalTimestamp(decisionTime) + ";left=" +
            LengthPrefixed(selectorText(left)) + ";right=" +
            LengthPrefixed(selectorText(right)) + ";output_count=" +
            std::to_string(outputs.size());
        for (const ConfluenceObservation& output : outputs)
        {
            result += ";output=" + LengthPrefixed(output.outputIdentity) +
                ";available_at=" + CanonicalTimestamp(output.availableAt) +
                ";relation=" + std::string{CanonicalRelationKind(output.descriptor.relation)} +
                ";left_polarity=" + std::string{CanonicalDescriptorPolarity(
                    output.descriptor.leftPolarity)} +
                ";right_polarity=" + std::string{CanonicalDescriptorPolarity(
                    output.descriptor.rightPolarity)};
            if (!legacy)
            {
                result += ";left_count=" +
                    std::to_string(output.descriptor.leftComponentCount) +
                    ";right_count=" +
                    std::to_string(output.descriptor.rightComponentCount) +
                    ";temporal_predicate=" + std::string{CanonicalTemporalPredicate(
                        output.descriptor.temporalPredicate)} +
                    ";correlation_key=" + LengthPrefixed(
                        output.descriptor.correlationKey
                            ? CanonicalCorrelationKey(*output.descriptor.correlationKey)
                            : "absent");
            }
            for (const Observation& component : output.components)
                result += ";component=" + LengthPrefixed(
                    CanonicalObservationIdentity(component));
        }
        return result;
    }
};

class ConfluenceEngine
{
public:
    virtual ~ConfluenceEngine() = default;
    virtual std::vector<ConfluenceObservation> Describe(
        const std::vector<Observation>& availableObservations,
        std::chrono::sys_seconds decisionTime) const = 0;
};

class DescriptiveConfluenceEngine final : public ConfluenceEngine
{
public:
    explicit DescriptiveConfluenceEngine(ConfluenceDefinition definition)
        : definition_(std::move(definition))
    {
        ValidateConfluenceDefinition(definition_);
        definitionIdentity_ = CanonicalConfluenceDefinitionIdentity(definition_);
    }

    const ConfluenceDefinition& definition() const { return definition_; }
    const std::string& definitionIdentity() const { return definitionIdentity_; }

    std::vector<ConfluenceObservation> Describe(
        const std::vector<Observation>& observations,
        std::chrono::sys_seconds decisionTime) const override
    {
        return Evaluate(observations, decisionTime).outputs;
    }

    ConfluenceReplay Evaluate(const std::vector<Observation>& observations,
                              std::chrono::sys_seconds decisionTime) const
    {
        if (definition_.correlationConstraint ==
            CorrelationConstraint::exact_key_equality)
        {
            return EvaluateKeyed(observations, decisionTime);
        }
        const std::vector<Observation> causal = CausallyAvailableObservations(
            observations, decisionTime);
        const auto select = [&](const RoleSelector& selector,
                                std::string_view name) {
            std::vector<Observation> matching;
            for (const Observation& observation : causal)
            {
                if (observation.familyId == selector.familyId &&
                    observation.descriptor.role == selector.role)
                {
                    matching.push_back(observation);
                }
            }
            std::sort(matching.begin(), matching.end(),
                [](const Observation& left, const Observation& right) {
                    return CanonicalObservationIdentity(left) <
                        CanonicalObservationIdentity(right);
                });
            SelectorDiagnostic diagnostic;
            diagnostic.selectorName = std::string{name};
            diagnostic.matchingCandidateCount = matching.size();
            if (matching.size() > definition_.maxCandidatesPerSelector)
                matching.resize(definition_.maxCandidatesPerSelector);
            diagnostic.retainedCandidateCount = matching.size();
            diagnostic.overflowCandidateCount =
                diagnostic.matchingCandidateCount - diagnostic.retainedCandidateCount;
            for (const Observation& observation : matching)
                diagnostic.retainedComponentIdentities.push_back(
                    CanonicalObservationIdentity(observation));
            return std::pair{std::move(matching), std::move(diagnostic)};
        };

        auto [leftCandidates, leftDiagnostic] = select(definition_.left, "left");
        auto [rightCandidates, rightDiagnostic] = select(definition_.right, "right");
        ConfluenceReplay replay{definitionIdentity_, decisionTime,
                                 std::move(leftDiagnostic), std::move(rightDiagnostic), {}};

        // There are exactly two non-neutral polarities.  This fixed aggregate
        // bound replaces all component-pair expansion.
        for (const DescriptorPolarity leftPolarity :
             {DescriptorPolarity::positive, DescriptorPolarity::negative})
        {
            const DescriptorPolarity rightPolarity =
                definition_.relation == RelationKind::support ? leftPolarity :
                (leftPolarity == DescriptorPolarity::positive ?
                    DescriptorPolarity::negative : DescriptorPolarity::positive);
            std::vector<Observation> components;
            for (const Observation& observation : leftCandidates)
                if (observation.descriptor.polarity == leftPolarity)
                    components.push_back(observation);
            const std::size_t leftCount = components.size();
            for (const Observation& observation : rightCandidates)
                if (observation.descriptor.polarity == rightPolarity)
                    components.push_back(observation);
            const std::size_t rightCount = components.size() - leftCount;
            if (leftCount == 0 || rightCount == 0) continue;

            std::sort(components.begin(), components.end(),
                [](const Observation& left, const Observation& right) {
                    return CanonicalObservationIdentity(left) <
                        CanonicalObservationIdentity(right);
                });
            const auto availability = std::max_element(components.begin(), components.end(),
                [](const Observation& left, const Observation& right) {
                    return left.availableAt < right.availableAt;
                })->availableAt;
            ConfluenceResultDescriptor descriptor{definition_.relation, leftPolarity,
                rightPolarity, leftCount, rightCount, std::nullopt,
                TemporalPredicate::none};
            std::string outputIdentity = "confluence-output-v1;definition=" +
                LengthPrefixed(definitionIdentity_) + ";decision_time=" +
                CanonicalTimestamp(decisionTime) + ";available_at=" +
                CanonicalTimestamp(availability) + ";relation=" +
                std::string{CanonicalRelationKind(descriptor.relation)} +
                ";left_polarity=" + std::string{CanonicalDescriptorPolarity(leftPolarity)} +
                ";right_polarity=" + std::string{CanonicalDescriptorPolarity(rightPolarity)};
            for (const Observation& component : components)
                outputIdentity += ";component=" + LengthPrefixed(
                    CanonicalObservationIdentity(component));
            replay.outputs.push_back({std::move(outputIdentity), definitionIdentity_,
                definition_.definitionId, definition_.definitionVersion, decisionTime,
                availability, std::move(components), descriptor});
        }
        return replay;
    }

private:
    ConfluenceReplay EvaluateKeyed(const std::vector<Observation>& observations,
                                   std::chrono::sys_seconds decisionTime) const
    {
        const std::vector<Observation> causal = CausallyAvailableObservations(
            observations, decisionTime);
        struct KeyCandidates
        {
            CorrelationKey key;
            std::vector<Observation> left;
            std::vector<Observation> right;
        };
        std::map<std::string, KeyCandidates> groups;
        SelectorDiagnostic leftDiagnostic{"left", 0, 0, 0, {}};
        SelectorDiagnostic rightDiagnostic{"right", 0, 0, 0, {}};
        for (const Observation& observation : causal)
        {
            const bool isLeft = observation.familyId == definition_.left.familyId &&
                observation.descriptor.role == definition_.left.role;
            const bool isRight = observation.familyId == definition_.right.familyId &&
                observation.descriptor.role == definition_.right.role;
            if (isLeft) ++leftDiagnostic.matchingCandidateCount;
            if (isRight) ++rightDiagnostic.matchingCandidateCount;
            if ((!isLeft && !isRight) || !observation.correlationKey) continue;

            const std::string keyIdentity = CanonicalCorrelationKey(
                *observation.correlationKey);
            KeyCandidates& group = groups[keyIdentity];
            group.key = *observation.correlationKey;
            if (isLeft) group.left.push_back(observation);
            if (isRight) group.right.push_back(observation);
        }

        ConfluenceReplay replay{definitionIdentity_, decisionTime,
                                 std::move(leftDiagnostic), std::move(rightDiagnostic), {}};
        const auto canonicalLess = [](const Observation& left,
                                      const Observation& right) {
            return CanonicalObservationIdentity(left) <
                CanonicalObservationIdentity(right);
        };
        const auto temporalLess = [&canonicalLess](const Observation& left,
                                                   const Observation& right) {
            if (left.availableAt != right.availableAt)
                return left.availableAt < right.availableAt;
            return canonicalLess(left, right);
        };
        const auto retain = [&](std::vector<Observation> candidates) {
            std::sort(candidates.begin(), candidates.end(), canonicalLess);
            if (candidates.size() > definition_.maxCandidatesPerSelector)
                candidates.resize(definition_.maxCandidatesPerSelector);
            return candidates;
        };
        const auto recordRetained = [&replay](const std::vector<Observation>& left,
                                               const std::vector<Observation>& right) {
            for (const Observation& observation : left)
            {
                ++replay.left.retainedCandidateCount;
                replay.left.retainedComponentIdentities.push_back(
                    CanonicalObservationIdentity(observation));
            }
            for (const Observation& observation : right)
            {
                ++replay.right.retainedCandidateCount;
                replay.right.retainedComponentIdentities.push_back(
                    CanonicalObservationIdentity(observation));
            }
        };
        const auto appendOutput = [&](const CorrelationKey& key,
                                      DescriptorPolarity leftPolarity,
                                      DescriptorPolarity rightPolarity,
                                      std::vector<Observation> left,
                                      std::vector<Observation> right) {
            if (left.empty() || right.empty()) return;
            recordRetained(left, right);
            std::vector<Observation> components = std::move(left);
            components.insert(components.end(), right.begin(), right.end());
            std::sort(components.begin(), components.end(), canonicalLess);
            const auto availability = std::max_element(components.begin(), components.end(),
                [](const Observation& first, const Observation& second) {
                    return first.availableAt < second.availableAt;
                })->availableAt;
            ConfluenceResultDescriptor descriptor{definition_.relation, leftPolarity,
                rightPolarity, components.size() - right.size(), right.size(), key,
                definition_.temporalPredicate};
            std::string outputIdentity = "confluence-output-v2;definition=" +
                LengthPrefixed(definitionIdentity_) + ";decision_time=" +
                CanonicalTimestamp(decisionTime) + ";available_at=" +
                CanonicalTimestamp(availability) + ";relation=" +
                std::string{CanonicalRelationKind(descriptor.relation)} +
                ";left_polarity=" + std::string{CanonicalDescriptorPolarity(leftPolarity)} +
                ";right_polarity=" + std::string{CanonicalDescriptorPolarity(rightPolarity)} +
                ";correlation_key=" + LengthPrefixed(CanonicalCorrelationKey(key)) +
                ";temporal_predicate=" + std::string{CanonicalTemporalPredicate(
                    definition_.temporalPredicate)};
            for (const Observation& component : components)
                outputIdentity += ";component=" + LengthPrefixed(
                    CanonicalObservationIdentity(component));
            replay.outputs.push_back({std::move(outputIdentity), definitionIdentity_,
                definition_.definitionId, definition_.definitionVersion, decisionTime,
                availability, std::move(components), std::move(descriptor)});
        };
        const auto hasTemporalPair = [&temporalLess](
            std::vector<Observation> left, std::vector<Observation> right) {
            std::sort(left.begin(), left.end(), temporalLess);
            std::sort(right.begin(), right.end(), temporalLess);
            for (const Observation& rightCandidate : right)
                if (std::any_of(left.begin(), left.end(),
                    [&rightCandidate](const Observation& candidate) {
                        return candidate.availableAt < rightCandidate.availableAt;
                    }))
                    return true;
            return false;
        };
        const auto groupQualifies = [&](const KeyCandidates& group) {
            const auto hasRelation = [&](DescriptorPolarity leftPolarity,
                                         DescriptorPolarity rightPolarity) {
                std::vector<Observation> left;
                std::vector<Observation> right;
                for (const Observation& candidate : group.left)
                    if (candidate.descriptor.polarity == leftPolarity)
                        left.push_back(candidate);
                for (const Observation& candidate : group.right)
                    if (candidate.descriptor.polarity == rightPolarity)
                        right.push_back(candidate);
                if (left.empty() || right.empty()) return false;
                return definition_.temporalPredicate == TemporalPredicate::none ||
                    hasTemporalPair(std::move(left), std::move(right));
            };
            if (definition_.relation == RelationKind::co_occurrence)
                return hasRelation(DescriptorPolarity::neutral,
                                   DescriptorPolarity::neutral);
            return hasRelation(DescriptorPolarity::positive,
                definition_.relation == RelationKind::support
                    ? DescriptorPolarity::positive : DescriptorPolarity::negative) ||
                hasRelation(DescriptorPolarity::negative,
                definition_.relation == RelationKind::support
                    ? DescriptorPolarity::negative : DescriptorPolarity::positive);
        };

        std::size_t retainedKeys = 0;
        for (const auto& [keyIdentity, group] : groups)
        {
            (void)keyIdentity;
            if (!groupQualifies(group)) continue;
            if (retainedKeys++ == definition_.maxCorrelationKeys) break;

            const auto byPolarity = [](const std::vector<Observation>& candidates,
                                       DescriptorPolarity polarity) {
                std::vector<Observation> result;
                for (const Observation& candidate : candidates)
                    if (candidate.descriptor.polarity == polarity)
                        result.push_back(candidate);
                return result;
            };
            const auto emit = [&](DescriptorPolarity leftPolarity,
                                  DescriptorPolarity rightPolarity) {
                std::vector<Observation> left = byPolarity(group.left, leftPolarity);
                std::vector<Observation> right = byPolarity(group.right, rightPolarity);
                if (definition_.temporalPredicate == TemporalPredicate::none)
                {
                    appendOutput(group.key, leftPolarity, rightPolarity,
                                 retain(std::move(left)), retain(std::move(right)));
                    return;
                }
                std::sort(left.begin(), left.end(), temporalLess);
                std::sort(right.begin(), right.end(), temporalLess);
                for (const Observation& rightCandidate : right)
                {
                    const auto leftCandidate = std::find_if(left.begin(), left.end(),
                        [&rightCandidate](const Observation& candidate) {
                            return candidate.availableAt < rightCandidate.availableAt;
                        });
                    if (leftCandidate != left.end())
                    {
                        appendOutput(group.key, leftPolarity, rightPolarity,
                                     {*leftCandidate}, {rightCandidate});
                        return;
                    }
                }
            };

            if (definition_.relation == RelationKind::co_occurrence)
            {
                emit(DescriptorPolarity::neutral, DescriptorPolarity::neutral);
            }
            else
            {
                emit(DescriptorPolarity::positive,
                    definition_.relation == RelationKind::support
                        ? DescriptorPolarity::positive : DescriptorPolarity::negative);
                emit(DescriptorPolarity::negative,
                    definition_.relation == RelationKind::support
                        ? DescriptorPolarity::negative : DescriptorPolarity::positive);
            }
        }
        replay.left.overflowCandidateCount = replay.left.matchingCandidateCount -
            replay.left.retainedCandidateCount;
        replay.right.overflowCandidateCount = replay.right.matchingCandidateCount -
            replay.right.retainedCandidateCount;
        std::sort(replay.left.retainedComponentIdentities.begin(),
                  replay.left.retainedComponentIdentities.end());
        std::sort(replay.right.retainedComponentIdentities.begin(),
                  replay.right.retainedComponentIdentities.end());
        return replay;
    }

    ConfluenceDefinition definition_;
    std::string definitionIdentity_;
};

} // namespace EA::MarketStructure

#endif
