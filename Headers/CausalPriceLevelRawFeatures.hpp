#pragma once

#include "CausalPriceLevelV2Engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>

// This is the model-facing, observational projection of the frozen
// causal-price-level/v2 update.  It deliberately owns no detector state: the
// caller supplies the post-AddCompletedBar update, the completed close, and
// that bar's zero-based coordinate.
namespace EA::PriceLevel::Raw
{
inline constexpr std::size_t kFeatureCount = 11;
inline constexpr std::array<std::string_view, kFeatureCount> kFeatureNames{{
    "available",
    "zone_scale_valid",
    "zone_gap_signed_clipped",
    "zone_relation",
    "current_role",
    "age_fraction",
    "prior_evidence_saturation",
    "touch_now",
    "cross_direction_now",
    "retest_now",
    "role_reversal_now",
}};

using FeatureVector = std::array<float, kFeatureCount>;

class Producer final
{
public:
    explicit Producer(V2::Configuration configuration = V2::ProductionConfiguration())
        : configuration_(configuration)
    {
        V2::ValidateConfiguration(configuration_);
        if (configuration_.pivotRadiusBars != 3 ||
            configuration_.scaleLookbackBars != 64 ||
            configuration_.scaleMultiplier != 1.0 ||
            configuration_.maxAgeBars != 512 ||
            configuration_.maxActiveLevels != 48 ||
            configuration_.maxRetainedPivotEvidence != 32 ||
            configuration_.completedBarDuration != std::chrono::seconds{900})
        {
            throw std::invalid_argument("PRICE_LEVEL_RAW_FEATURES_REQUIRES_PRODUCTION_V2");
        }
    }

    FeatureVector Project(const V2::Update& update,
                          double close,
                          std::size_t currentBar) const
    {
        if (!std::isfinite(close))
            ThrowMalformed();
        ValidateSnapshot(update, currentBar);

        const V2::Level* selected = Select(update, close);
        FeatureVector result{};
        if (selected == nullptr) return result;

        result[0] = 1.0f;
        const bool scaleValid = std::isfinite(selected->zoneHalfWidth) &&
            selected->zoneHalfWidth > 0.0;
        result[1] = scaleValid ? 1.0f : 0.0f;

        if (close < selected->lower)
        {
            result[3] = -1.0f;
            if (scaleValid)
                result[2] = static_cast<float>(std::clamp(
                    (close - selected->lower) / selected->zoneHalfWidth,
                    -1.0, 1.0));
        }
        else if (close > selected->upper)
        {
            result[3] = 1.0f;
            if (scaleValid)
                result[2] = static_cast<float>(std::clamp(
                    (close - selected->upper) / selected->zoneHalfWidth,
                    -1.0, 1.0));
        }

        result[4] = RoleValue(selected->currentRole);
        result[5] = static_cast<float>(std::clamp(
            static_cast<double>(currentBar - selected->availableBar) /
                static_cast<double>(configuration_.maxAgeBars),
            0.0, 1.0));

        std::size_t addedNow = 0;
        bool touch = false;
        bool crossUp = false;
        bool crossDown = false;
        bool retest = false;
        bool roleReversal = false;
        for (const V2::Observation& observation : update.observations)
        {
            if (observation.levelIdentity != selected->identity ||
                observation.availableAt != update.decisionTime)
                continue;
            switch (observation.kind)
            {
                case InteractionKind::level_established:
                    ++addedNow;
                    break;
                case InteractionKind::level_reinforced:
                    if (!observation.retainedEvidenceAlreadySaturated) ++addedNow;
                    break;
                case InteractionKind::touch: touch = true; break;
                case InteractionKind::cross_up: crossUp = true; break;
                case InteractionKind::cross_down: crossDown = true; break;
                case InteractionKind::retest: retest = true; break;
                case InteractionKind::role_reversal: roleReversal = true; break;
                case InteractionKind::level_expired:
                case InteractionKind::level_evicted:
                    break;
            }
        }
        if (addedNow > selected->retainedPivotEvidence.size()) ThrowMalformed();
        const std::size_t prior = selected->retainedPivotEvidence.size() - addedNow;
        result[6] = static_cast<float>(std::clamp(
            static_cast<double>(prior) /
                static_cast<double>(configuration_.maxRetainedPivotEvidence),
            0.0, 1.0));
        result[7] = touch ? 1.0f : 0.0f;
        if (crossUp && crossDown) ThrowMalformed();
        result[8] = crossUp ? 1.0f : (crossDown ? -1.0f : 0.0f);
        result[9] = retest ? 1.0f : 0.0f;
        result[10] = roleReversal ? 1.0f : 0.0f;
        return result;
    }

private:
    V2::Configuration configuration_;

    [[noreturn]] static void ThrowMalformed()
    {
        throw std::runtime_error("PRICE_LEVEL_RAW_FEATURES_MALFORMED_V2_SNAPSHOT");
    }

    static const V2::Level* FindActive(const V2::Update& update,
                                        std::string_view identity)
    {
        const auto found = std::find_if(update.activeLevels.begin(),
                                        update.activeLevels.end(),
            [identity](const V2::Level& level) { return level.identity == identity; });
        return found == update.activeLevels.end() ? nullptr : &*found;
    }

    void ValidateSnapshot(const V2::Update& update, std::size_t currentBar) const
    {
        for (std::size_t index = 0; index < update.activeLevels.size(); ++index)
        {
            const V2::Level& level = update.activeLevels[index];
            if (level.identity.empty() || !std::isfinite(level.anchorPrice) ||
                !std::isfinite(level.zoneHalfWidth) || !std::isfinite(level.lower) ||
                !std::isfinite(level.upper) || level.zoneHalfWidth < 0.0 ||
                level.lower > level.upper || level.anchorPrice < level.lower ||
                level.anchorPrice > level.upper || level.availableBar > currentBar ||
                currentBar - level.availableBar > configuration_.maxAgeBars ||
                level.retainedPivotEvidence.size() >
                    configuration_.maxRetainedPivotEvidence)
            {
                ThrowMalformed();
            }
            for (std::size_t prior = 0; prior < index; ++prior)
                if (update.activeLevels[prior].identity == level.identity)
                    ThrowMalformed();

            std::size_t addedNow = 0;
            for (const V2::Observation& observation : update.observations)
            {
                if (observation.levelIdentity != level.identity ||
                    observation.availableAt != update.decisionTime)
                    continue;
                if (observation.kind == InteractionKind::level_established ||
                    (observation.kind == InteractionKind::level_reinforced &&
                     !observation.retainedEvidenceAlreadySaturated))
                    ++addedNow;
            }
            if (addedNow > level.retainedPivotEvidence.size()) ThrowMalformed();
        }

        for (const V2::Observation& observation : update.observations)
        {
            const V2::Level* level = FindActive(update, observation.levelIdentity);
            if (level == nullptr || observation.availableAt != update.decisionTime)
                continue;
            if (observation.kind == InteractionKind::cross_up ||
                observation.kind == InteractionKind::cross_down)
            {
                for (const V2::Observation& other : update.observations)
                {
                    if (&observation == &other ||
                        other.levelIdentity != level->identity ||
                        other.availableAt != update.decisionTime)
                        continue;
                    if ((observation.kind == InteractionKind::cross_up &&
                         other.kind == InteractionKind::cross_down) ||
                        (observation.kind == InteractionKind::cross_down &&
                         other.kind == InteractionKind::cross_up))
                        ThrowMalformed();
                }
            }
        }
    }

    static double ZoneDistance(double close, const V2::Level& level)
    {
        if (close < level.lower) return level.lower - close;
        if (close > level.upper) return close - level.upper;
        return 0.0;
    }

    static float RoleValue(Role role)
    {
        switch (role)
        {
            case Role::resistance_like: return -1.0f;
            case Role::support_like: return 1.0f;
        }
        ThrowMalformed();
    }

    static bool IsBetter(const V2::Level& candidate,
                         const V2::Level& incumbent,
                         double close)
    {
        const double candidateDistance = ZoneDistance(close, candidate);
        const double incumbentDistance = ZoneDistance(close, incumbent);
        if (candidateDistance != incumbentDistance)
            return candidateDistance < incumbentDistance;
        const double candidateAnchorDistance = std::abs(close - candidate.anchorPrice);
        const double incumbentAnchorDistance = std::abs(close - incumbent.anchorPrice);
        if (candidateAnchorDistance != incumbentAnchorDistance)
            return candidateAnchorDistance < incumbentAnchorDistance;
        if (candidate.anchorPrice != incumbent.anchorPrice)
            return candidate.anchorPrice < incumbent.anchorPrice;
        return candidate.identity < incumbent.identity;
    }

    static const V2::Level* Select(const V2::Update& update, double close)
    {
        const V2::Level* selected = nullptr;
        for (const V2::Observation& observation : update.observations)
        {
            if (observation.kind != InteractionKind::retest ||
                observation.availableAt != update.decisionTime)
                continue;
            const V2::Level* candidate = FindActive(update, observation.levelIdentity);
            if (candidate != nullptr &&
                (selected == nullptr || IsBetter(*candidate, *selected, close)))
                selected = candidate;
        }
        if (selected != nullptr) return selected;
        for (const V2::Level& candidate : update.activeLevels)
            if (selected == nullptr || IsBetter(candidate, *selected, close))
                selected = &candidate;
        return selected;
    }
};
} // namespace EA::PriceLevel::Raw
