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
#include "MarketStructureRegistry.hpp"

namespace EA
{

// Persist names, never physical Tensor offsets.  Offsets remain an implementation
// detail and are checked against the persisted model-input contract at use time.
struct AblatableFeature
{
    std::string_view name;
    std::size_t tensorColumn;
};

inline constexpr std::array<AblatableFeature, 58> kAblatableFeatures{{
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
    {"causal_first_release_surprise_available",
     causalFirstReleaseSurpriseAvailableCol},
    {"causal_first_release_surprise", causalFirstReleaseSurpriseCol},
    {"tg4_inner_break_any", tg4InnerBreakAnyCol},
    {"tg4_source_tg3_structurally_eligible",
     tg4SourceTg3StructurallyEligibleCol},
    {"tg4_source_tg3_confluent", tg4SourceTg3ConfluentCol},
    {"fib_recent_price_scale_valid", fibRecentPriceScaleValidCol},
    {"fib_up_recent_union_count_log", fibUpRecentUnionCountLogCol},
    {"fib_up_recent_h1_count_log", fibUpRecentH1CountLogCol},
    {"fib_up_recent_h2_count_log", fibUpRecentH2CountLogCol},
    {"fib_up_recent_h1_h2_both_count_log", fibUpRecentH1H2BothCountLogCol},
    {"fib_up_recent_h1_youngest_age_20", fibUpRecentH1YoungestAge20Col},
    {"fib_up_recent_h2_youngest_age_20", fibUpRecentH2YoungestAge20Col},
    {"fib_up_recent_median_1272_signed_atr", fibUpRecentMedian1272Col},
    {"fib_up_recent_median_1618_signed_atr", fibUpRecentMedian1618Col},
    {"fib_up_recent_median_pullback_0382_signed_atr",
     fibUpRecentMedianPullback0382Col},
    {"fib_up_recent_median_pullback_0500_signed_atr",
     fibUpRecentMedianPullback0500Col},
    {"fib_up_recent_median_pullback_0618_signed_atr",
     fibUpRecentMedianPullback0618Col},
    {"fib_down_recent_union_count_log", fibDownRecentUnionCountLogCol},
    {"fib_down_recent_h1_count_log", fibDownRecentH1CountLogCol},
    {"fib_down_recent_h2_count_log", fibDownRecentH2CountLogCol},
    {"fib_down_recent_h1_h2_both_count_log", fibDownRecentH1H2BothCountLogCol},
    {"fib_down_recent_h1_youngest_age_20", fibDownRecentH1YoungestAge20Col},
    {"fib_down_recent_h2_youngest_age_20", fibDownRecentH2YoungestAge20Col},
    {"fib_down_recent_median_1272_signed_atr", fibDownRecentMedian1272Col},
    {"fib_down_recent_median_1618_signed_atr", fibDownRecentMedian1618Col},
    {"fib_down_recent_median_pullback_0382_signed_atr",
     fibDownRecentMedianPullback0382Col},
    {"fib_down_recent_median_pullback_0500_signed_atr",
     fibDownRecentMedianPullback0500Col},
    {"fib_down_recent_median_pullback_0618_signed_atr",
     fibDownRecentMedianPullback0618Col},
    {"pocket_recent_price_scale_valid", pocketRecentPriceScaleValidCol},
    {"pocket_bull_recent_count_log", pocketBullRecentCountLogCol},
    {"pocket_bull_youngest_age20", pocketBullYoungestAge20Col},
    {"pocket_bull_median_touch_distance", pocketBullMedianTouchDistanceCol},
    {"pocket_bull_median_close_distance", pocketBullMedianCloseDistanceCol},
    {"pocket_bull_median_width", pocketBullMedianWidthCol},
    {"pocket_bear_recent_count_log", pocketBearRecentCountLogCol},
    {"pocket_bear_youngest_age20", pocketBearYoungestAge20Col},
    {"pocket_bear_median_touch_distance", pocketBearMedianTouchDistanceCol},
    {"pocket_bear_median_close_distance", pocketBearMedianCloseDistanceCol},
    {"pocket_bear_median_width", pocketBearMedianWidthCol},
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

inline constexpr std::string_view
    kCausalEconomicEventSurpriseAblationMaskText =
        "causal_first_release_surprise_available,"
        "causal_first_release_surprise";

// The TG4 control arm uses the three individually persisted feature names.
// This is intentionally not a parser alias: canonical experiment provenance
// always records the concrete channels that were ablated.
inline constexpr std::string_view kTG4AblationMaskText =
    "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,"
    "tg4_source_tg3_confluent";

// Layout-9 causal Fibonacci structural control. This is intentionally a
// concrete persisted feature list, not a parser alias, so pair provenance
// remains canonical and independently auditable.
inline constexpr std::string_view kCausalFibonacciStructuralAblationMaskText =
    "fib_recent_price_scale_valid,"
    "fib_up_recent_union_count_log,fib_up_recent_h1_count_log,"
    "fib_up_recent_h2_count_log,fib_up_recent_h1_h2_both_count_log,"
    "fib_up_recent_h1_youngest_age_20,fib_up_recent_h2_youngest_age_20,"
    "fib_up_recent_median_1272_signed_atr,"
    "fib_up_recent_median_1618_signed_atr,"
    "fib_up_recent_median_pullback_0382_signed_atr,"
    "fib_up_recent_median_pullback_0500_signed_atr,"
    "fib_up_recent_median_pullback_0618_signed_atr,"
    "fib_down_recent_union_count_log,fib_down_recent_h1_count_log,"
    "fib_down_recent_h2_count_log,fib_down_recent_h1_h2_both_count_log,"
    "fib_down_recent_h1_youngest_age_20,fib_down_recent_h2_youngest_age_20,"
    "fib_down_recent_median_1272_signed_atr,"
    "fib_down_recent_median_1618_signed_atr,"
    "fib_down_recent_median_pullback_0382_signed_atr,"
    "fib_down_recent_median_pullback_0500_signed_atr,"
    "fib_down_recent_median_pullback_0618_signed_atr";

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
    if (const auto* channel = MarketStructure::FindChannel(name))
    {
        for (const auto& feature : kAblatableFeatures)
            if (feature.tensorColumn == channel->tensorColumn) return &feature;
    }
    return nullptr;
}

inline std::size_t AblatableFeatureRegistryOrder(const AblatableFeature& feature)
{
    return feature.tensorColumn;
}

struct FeatureAblationResolution;

class FeatureAblationMask
{
public:
    FeatureAblationMask() = default;

    // Parse accepts the historical flat identities and the new hierarchical
    // identities.  Wildcards are immediately resolved into concrete columns;
    // CanonicalText is therefore safe to persist as immutable experiment
    // identity without giving a later registry addition new meaning.
    static FeatureAblationMask Parse(const std::string& text);
    // Queue parsing uses this syntax-only step. In particular, it preserves a
    // wildcard until the experiment's persisted semantic layout is known.
    static std::string CanonicalizeRequestedExpression(
        const std::string& text);
    static FeatureAblationMask ParseForSemanticLayout(
        const std::string& text, int semanticLayoutVersion);
    static FeatureAblationResolution Resolve(
        const std::string& requestedText, int semanticLayoutVersion = 10);

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
    static FeatureAblationMask FromTensorColumns(std::vector<std::size_t> columns)
    {
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
        FeatureAblationMask result;
        result.columns_ = std::move(columns);
        return result;
    }

    std::vector<std::size_t> columns_;

    friend struct FeatureAblationResolution;
};

// Requested text is an input-side diagnostic.  ResolvedMask::CanonicalText()
// is the persisted representation and the only identity used by historical
// model loading, duplicate detection, continuation, and inference.
struct FeatureAblationResolution
{
    std::string requestedCanonicalText;
    FeatureAblationMask resolvedMask;
};

inline std::vector<std::string> ParseFeatureAblationTokens(
    const std::string& requestedText)
{
    std::vector<std::string> result;
    if (requestedText.empty()) return result;
    std::size_t start = 0;
    while (start <= requestedText.size())
    {
        const std::size_t end = requestedText.find(',', start);
        std::string token = TrimFeatureAblationToken(
            requestedText.substr(start, end == std::string::npos
                                           ? std::string::npos : end - start));
        if (token.empty())
            throw std::invalid_argument(
                "FEATURE_ABLATION_MASK_INVALID: empty feature name");
        result.push_back(std::move(token));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

inline std::string CanonicalRequestedFeatureAblationText(
    std::vector<std::string> tokens)
{
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    std::string result;
    for (const std::string& token : tokens)
    {
        if (!result.empty()) result += ',';
        result += token;
    }
    return result;
}

inline std::string FeatureAblationMask::CanonicalizeRequestedExpression(
    const std::string& requestedText)
{
    const std::vector<std::string> tokens =
        ParseFeatureAblationTokens(requestedText);
    for (const std::string& token : tokens)
    {
        const std::size_t wildcard = token.find('*');
        if (wildcard != std::string::npos)
        {
            if (wildcard != token.size() - 1 || wildcard == 0 ||
                token[wildcard - 1] != '.' ||
                token.find('*', wildcard + 1) != std::string::npos)
            {
                throw std::invalid_argument(
                    "FEATURE_ABLATION_MASK_INVALID_WILDCARD:" + token);
            }
            const std::string_view prefix{token.data(), token.size() - 2};
            const std::size_t separator = prefix.find('.');
            const std::string_view family = prefix.substr(
                0, separator == std::string_view::npos ? prefix.size()
                                                        : separator);
            if (MarketStructure::FindFamily(family) == nullptr)
                throw std::invalid_argument(
                    "FEATURE_ABLATION_MASK_UNKNOWN_FAMILY_OR_SUBFAMILY:" +
                    token);
            continue;
        }
        if (FindAblatableFeature(token) == nullptr)
            throw std::invalid_argument(
                "FEATURE_ABLATION_MASK_UNKNOWN_FEATURE:" + token);
    }
    return CanonicalRequestedFeatureAblationText(tokens);
}

inline FeatureAblationResolution FeatureAblationMask::Resolve(
    const std::string& requestedText, int semanticLayoutVersion)
{
    const std::string canonicalRequestedText =
        CanonicalizeRequestedExpression(requestedText);
    const std::vector<std::string> tokens =
        ParseFeatureAblationTokens(canonicalRequestedText);
    std::vector<std::size_t> columns;
    for (const std::string& token : tokens)
    {
        const std::size_t wildcard = token.find('*');
        if (wildcard != std::string::npos)
        {
            if (wildcard != token.size() - 1 || wildcard == 0 ||
                token[wildcard - 1] != '.' ||
                token.find('*', wildcard + 1) != std::string::npos)
            {
                throw std::invalid_argument(
                    "FEATURE_ABLATION_MASK_INVALID_WILDCARD:" + token);
            }
            const std::string_view prefix{token.data(), token.size() - 2};
            const auto matches = MarketStructure::ResolvePrefix(
                prefix, semanticLayoutVersion);
            if (matches.empty())
                throw std::invalid_argument(
                    "FEATURE_ABLATION_MASK_UNKNOWN_FAMILY_OR_SUBFAMILY:" +
                    token);
            for (const MarketStructure::Channel* channel : matches)
                columns.push_back(channel->tensorColumn);
            continue;
        }

        const AblatableFeature* feature = FindAblatableFeature(token);
        if (feature == nullptr)
            throw std::invalid_argument(
                "FEATURE_ABLATION_MASK_UNKNOWN_FEATURE:" + token);
        if (const auto* channel = MarketStructure::FindChannel(token);
            channel != nullptr &&
            !MarketStructure::ChannelAvailableForSemanticLayout(
                *channel, semanticLayoutVersion))
        {
            throw std::invalid_argument(
                "FEATURE_ABLATION_MASK_FEATURE_UNAVAILABLE_IN_SEMANTIC_LAYOUT:" +
                token);
        }
        columns.push_back(feature->tensorColumn);
    }
    return {canonicalRequestedText,
            FromTensorColumns(std::move(columns))};
}

inline FeatureAblationMask FeatureAblationMask::Parse(const std::string& text)
{
    return Resolve(text).resolvedMask;
}

inline FeatureAblationMask FeatureAblationMask::ParseForSemanticLayout(
    const std::string& text, int semanticLayoutVersion)
{
    return Resolve(text, semanticLayoutVersion).resolvedMask;
}

} // namespace EA

#endif
