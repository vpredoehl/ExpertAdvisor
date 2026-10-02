#pragma once

#include "CausalPriceLevelEngine.hpp"
#include "MarketStructureRegistry.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// A one-way, stateless bridge for immutable causal-price-level/v1 observations.
// It neither owns nor invokes the detector and has no Tensor dependency.
namespace EA::MarketStructure::PriceLevelBridge
{
inline constexpr std::string_view kFamilyId = "price_level";
inline constexpr std::string_view kBridgeDefinitionId =
    "price-level-market-structure-observation-bridge";
// v2 adds immutable correlation metadata to the copied observation.  The
// detector evidence and descriptor remain the frozen v1 values.
inline constexpr std::string_view kBridgeDefinitionVersion = "v2";
inline constexpr std::string_view kDescriptorSchemaVersion =
    "price-level-market-structure-observation-v1";
// The value is the complete frozen Phase-1 Level::identity.  It is opaque to
// MarketStructure and deliberately does not encode a price, proximity, role,
// insertion order, or mutable active-level state.
inline constexpr std::string_view kLevelIdentityCorrelationKeyType =
    "price_level.level_identity.v1";

class ObservationAdapter final
{
public:
    explicit ObservationAdapter(std::string symbol)
        : symbol_(CanonicalSymbol::Normalize(symbol)),
          bridgeProvenancePrefix_(std::string{kBridgeDefinitionId} + "-" +
              std::string{kBridgeDefinitionVersion} + ";symbol=" + symbol_ +
              ";source_provenance=")
    {
    }

    // The level snapshot is validated only; it is not transformed into a
    // descriptor because this boundary has no price-level numeric semantics.
    std::vector<Observation> Adapt(const EA::PriceLevel::Update& update) const
    {
        ValidateActiveLevels(update.activeLevels);
        return Adapt(std::span<const EA::PriceLevel::Observation>{
            update.observations});
    }

    std::vector<Observation> Adapt(
        std::span<const EA::PriceLevel::Observation> source) const
    {
        std::vector<std::string> sourceIdentities;
        sourceIdentities.reserve(source.size());
        std::vector<Observation> result;
        result.reserve(source.size());
        for (const EA::PriceLevel::Observation& value : source)
        {
            ValidateSource(value);
            sourceIdentities.push_back(value.identity);
            if (const auto mapped = Map(value)) result.push_back(*mapped);
        }

        std::sort(sourceIdentities.begin(), sourceIdentities.end());
        if (std::adjacent_find(sourceIdentities.begin(), sourceIdentities.end()) !=
            sourceIdentities.end())
        {
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_DUPLICATE_SOURCE_OBSERVATION");
        }

        std::sort(result.begin(), result.end(),
            [](const Observation& left, const Observation& right) {
                return CanonicalObservationIdentity(left) <
                    CanonicalObservationIdentity(right);
            });
        const auto duplicate = std::adjacent_find(result.begin(), result.end(),
            [](const Observation& left, const Observation& right) {
                return CanonicalObservationIdentity(left) ==
                    CanonicalObservationIdentity(right);
            });
        if (duplicate != result.end())
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_DUPLICATE_OBSERVATION");
        return result;
    }

private:
    static std::string DetectorVersion()
    {
        return std::string{EA::PriceLevel::kDefinitionId} + "/" +
            std::string{EA::PriceLevel::kDefinitionVersion};
    }

    std::string ExpectedSourceProvenancePrefix() const
    {
        return "causal-price-level-engine-v1;symbol=" + symbol_ +
            ";definition=";
    }

    void ValidateSource(const EA::PriceLevel::Observation& source) const
    {
        if (source.identity.empty() || source.levelIdentity.empty() ||
            source.sourceProvenance.empty() || source.sourceObservationId.empty())
        {
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_PROVENANCE_INCOMPLETE");
        }
        if (source.availableAt < source.observedAt)
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_CAUSAL_AVAILABILITY_INVALID");

        const std::string provenancePrefix = ExpectedSourceProvenancePrefix();
        if (!source.sourceProvenance.starts_with(provenancePrefix))
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_PROVENANCE_INVALID");
        const std::string definitionIdentity = source.sourceProvenance.substr(
            provenancePrefix.size());
        const std::string requiredDefinitionPrefix =
            "price-level-definition-v1;id=causal-price-level;version=v1;";
        if (!definitionIdentity.starts_with(requiredDefinitionPrefix))
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_PROVENANCE_INVALID");

        const std::string levelPrefix = "price-level-v1;symbol=" + symbol_ +
            ";definition=" + definitionIdentity + ";origin=";
        if (!source.levelIdentity.starts_with(levelPrefix) ||
            source.levelIdentity.size() == levelPrefix.size())
        {
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_LEVEL_IDENTITY_INVALID");
        }

        const std::string expectedIdentity = "price-level-observation-v1;level=" +
            source.levelIdentity + ";kind=" + std::string{
                EA::PriceLevel::CanonicalInteractionKind(source.kind)} +
            ";observed=" + EA::PriceLevel::CanonicalTimestamp(source.observedAt) +
            ";available=" + EA::PriceLevel::CanonicalTimestamp(source.availableAt) +
            ";source=" + source.sourceObservationId + ";before=" +
            (source.roleBefore ? std::string{EA::PriceLevel::CanonicalRole(
                *source.roleBefore)} : "absent") + ";after=" +
            (source.roleAfter ? std::string{EA::PriceLevel::CanonicalRole(
                *source.roleAfter)} : "absent");
        if (source.identity != expectedIdentity)
            throw std::invalid_argument(
                "PRICE_LEVEL_MARKET_STRUCTURE_SOURCE_IDENTITY_INVALID");

        switch (source.kind)
        {
            case EA::PriceLevel::InteractionKind::cross_up:
            case EA::PriceLevel::InteractionKind::role_reversal:
                if (source.roleBefore != EA::PriceLevel::Role::resistance_like ||
                    source.roleAfter != EA::PriceLevel::Role::support_like)
                {
                    throw std::invalid_argument(
                        "PRICE_LEVEL_MARKET_STRUCTURE_ROLE_TRANSITION_INVALID");
                }
                break;
            case EA::PriceLevel::InteractionKind::cross_down:
                if (source.roleBefore != EA::PriceLevel::Role::support_like ||
                    source.roleAfter != EA::PriceLevel::Role::resistance_like)
                {
                    throw std::invalid_argument(
                        "PRICE_LEVEL_MARKET_STRUCTURE_ROLE_TRANSITION_INVALID");
                }
                break;
            case EA::PriceLevel::InteractionKind::level_established:
            case EA::PriceLevel::InteractionKind::level_reinforced:
            case EA::PriceLevel::InteractionKind::touch:
            case EA::PriceLevel::InteractionKind::retest:
            case EA::PriceLevel::InteractionKind::level_expired:
            case EA::PriceLevel::InteractionKind::level_evicted:
                if (source.roleBefore || source.roleAfter)
                {
                    throw std::invalid_argument(
                        "PRICE_LEVEL_MARKET_STRUCTURE_ROLE_TRANSITION_INVALID");
                }
                break;
        }
    }

    void ValidateActiveLevels(std::span<const EA::PriceLevel::Level> levels) const
    {
        for (const EA::PriceLevel::Level& level : levels)
        {
            if (level.identity.empty() || !std::isfinite(level.anchorPrice) ||
                !std::isfinite(level.lower) || !std::isfinite(level.upper) ||
                level.lower > level.anchorPrice || level.anchorPrice > level.upper ||
                level.availableAt < level.originObservedAt ||
                level.availableBar < level.originBar || level.pivotObservationCount == 0)
            {
                throw std::invalid_argument(
                    "PRICE_LEVEL_MARKET_STRUCTURE_LEVEL_INVALID");
            }
            const std::string levelPrefix = "price-level-v1;symbol=" + symbol_ +
                ";definition=price-level-definition-v1;id=causal-price-level;version=v1;";
            if (!level.identity.starts_with(levelPrefix))
                throw std::invalid_argument(
                    "PRICE_LEVEL_MARKET_STRUCTURE_LEVEL_IDENTITY_INVALID");
            (void)EA::PriceLevel::CanonicalPivotKind(level.originatingPivot);
            (void)EA::PriceLevel::CanonicalRole(level.currentRole);
            for (const std::string& evidence : level.retainedPivotEvidence)
                if (evidence.empty())
                    throw std::invalid_argument(
                        "PRICE_LEVEL_MARKET_STRUCTURE_LEVEL_INVALID");
        }
    }

    static bool Included(EA::PriceLevel::InteractionKind kind)
    {
        switch (kind)
        {
            case EA::PriceLevel::InteractionKind::level_established:
            case EA::PriceLevel::InteractionKind::level_reinforced:
            case EA::PriceLevel::InteractionKind::touch:
            case EA::PriceLevel::InteractionKind::cross_up:
            case EA::PriceLevel::InteractionKind::cross_down:
            case EA::PriceLevel::InteractionKind::retest:
            case EA::PriceLevel::InteractionKind::role_reversal:
                return true;
            case EA::PriceLevel::InteractionKind::level_expired:
            case EA::PriceLevel::InteractionKind::level_evicted:
                return false;
        }
        throw std::invalid_argument("PRICE_LEVEL_MARKET_STRUCTURE_KIND_INVALID");
    }

    std::optional<Observation> Map(
        const EA::PriceLevel::Observation& source) const
    {
        if (!Included(source.kind)) return std::nullopt;
        const CorrelationKey correlationKey{
            std::string{kLevelIdentityCorrelationKeyType}, source.levelIdentity};
        // The generic boundary owns correlation-key validation.  Validate
        // before producing an observation so malformed/oversized frozen
        // identities fail closed rather than becoming a partial bridge output.
        ValidateCorrelationKey(correlationKey);
        return Observation{std::string{kFamilyId}, DetectorVersion(),
            source.observedAt, source.availableAt,
            bridgeProvenancePrefix_ + LengthPrefixed(source.sourceProvenance),
            source.identity,
            {std::string{kDescriptorSchemaVersion}, std::string{
                EA::PriceLevel::CanonicalInteractionKind(source.kind)},
             DescriptorPolarity::neutral, std::nullopt}, correlationKey};
    }

    std::string symbol_;
    std::string bridgeProvenancePrefix_;
};

} // namespace EA::MarketStructure::PriceLevelBridge
