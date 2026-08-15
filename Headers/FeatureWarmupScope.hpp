// Stateful Tensor features include recursive EMA/ATR state. Exact values at a
// requested logical boundary therefore require every preceding source row.
#ifndef FeatureWarmupScope_hpp
#define FeatureWarmupScope_hpp

#include <stdexcept>
#include <string>

namespace EA
{
enum class FeatureWarmupScope
{
    LegacyColdBoundary,
    FullHistoryWarmup
};

inline constexpr FeatureWarmupScope kDefaultFeatureWarmupScope =
    FeatureWarmupScope::FullHistoryWarmup;

inline const char* FeatureWarmupScopeText(FeatureWarmupScope scope)
{
    switch (scope)
    {
        case FeatureWarmupScope::LegacyColdBoundary:
            return "legacy_cold_boundary";
        case FeatureWarmupScope::FullHistoryWarmup:
            return "full_history_warmup";
    }
    throw std::invalid_argument("unsupported feature warmup scope");
}

inline FeatureWarmupScope ParseFeatureWarmupScope(const std::string& text)
{
    if (text == "legacy_cold_boundary")
        return FeatureWarmupScope::LegacyColdBoundary;
    if (text == "full_history_warmup")
        return FeatureWarmupScope::FullHistoryWarmup;
    throw std::invalid_argument(
        "invalid feature warmup scope; expected legacy_cold_boundary or "
        "full_history_warmup");
}

inline constexpr const char* kTensorFeatureHistoryQueryStart = "-infinity";
} // namespace EA

#endif /* FeatureWarmupScope_hpp */
