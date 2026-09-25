#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

// Shared, outcome-blind causal H1/H2 primitives.  They intentionally contain
// no reporting, target, resolution, or persistence dependency.
namespace EA::CausalFibonacciStructural {

struct NormalizedDistance {
    double value = 0.0;
    bool usedPipFallback = false;
};

inline std::optional<NormalizedDistance> NormalizeDistance(
    bool upAB, double level, double close, const std::optional<double>& atr,
    double canonicalPipSize)
{
    if (!std::isfinite(level) || !std::isfinite(close) ||
        !std::isfinite(canonicalPipSize) || canonicalPipSize <= 0.0)
        return {};
    const bool useAtr = atr && std::isfinite(*atr) && *atr > canonicalPipSize;
    const double denominator = useAtr ? *atr : canonicalPipSize;
    const double value = (upAB ? 1.0 : -1.0) * (level - close) / denominator;
    if (!std::isfinite(value)) return {};
    return NormalizedDistance{value, !useAtr};
}

inline bool IsEventRelevantAtBar(std::size_t eventBar, std::size_t currentBar,
                                 std::size_t horizonBars = 20)
{
    return currentBar >= eventBar && currentBar - eventBar <= horizonBars;
}

inline std::size_t EventAgeAtBar(std::size_t eventBar, std::size_t currentBar)
{
    if (currentBar < eventBar)
        throw std::invalid_argument("event bar is after current bar");
    return currentBar - eventBar;
}

enum class EventRelevance { None, H1Only, H2Only, Both };

inline EventRelevance ClassifyEventRelevance(
    const std::optional<std::size_t>& h1Bar,
    const std::optional<std::size_t>& h2Bar, std::size_t currentBar,
    std::size_t horizonBars = 20)
{
    const bool h1 = h1Bar && IsEventRelevantAtBar(*h1Bar, currentBar, horizonBars);
    const bool h2 = h2Bar && IsEventRelevantAtBar(*h2Bar, currentBar, horizonBars);
    return h1 ? (h2 ? EventRelevance::Both : EventRelevance::H1Only)
              : (h2 ? EventRelevance::H2Only : EventRelevance::None);
}

struct H1H2EventState {
    std::optional<std::size_t> touchBar;
    std::optional<std::size_t> h1BeyondBar;
    std::optional<std::size_t> h2RejectionBar;
};

// A rejection needs a touch from an earlier completed bar; a same-bar touch
// cannot create H2.  H1 may become available on the current completed bar.
inline void AdvanceH1H2EventState(H1H2EventState& state, std::size_t bar,
                                  bool touched, bool beyond, bool rejected)
{
    const bool earlierTouch = state.touchBar && bar > *state.touchBar;
    if (!state.touchBar && touched) state.touchBar = bar;
    if (!state.h1BeyondBar && beyond) state.h1BeyondBar = bar;
    if (!state.h2RejectionBar && earlierTouch && rejected)
        state.h2RejectionBar = bar;
}

} // namespace EA::CausalFibonacciStructural
