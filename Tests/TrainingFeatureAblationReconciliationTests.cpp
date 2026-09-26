#include "TrainingFeatureAblationReconciliation.hpp"

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
EA::FeatureAblationMask Mask(std::string_view value)
{
    return EA::FeatureAblationMask::Parse(std::string{value});
}

void ExpectFailure(const std::optional<EA::FeatureAblationMask>& cli,
                   const EA::FeatureAblationMask& persisted,
                   const std::optional<EA::FeatureAblationMask>& source,
                   std::string_view expected)
{
    try
    {
        (void)EA::ReconcileSchedulerTrainFeatureAblationMask(
            cli, persisted, source);
        assert(false && "expected reconciliation failure");
    }
    catch (const std::runtime_error& error)
    {
        assert(std::string_view{error.what()}.find(expected) !=
               std::string_view::npos);
    }
}
} // namespace

int main()
{
    const auto target = Mask(
        "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,"
        "tg4_source_tg3_confluent");
    const auto reordered = Mask(
        "tg4_source_tg3_confluent,tg4_inner_break_any,"
        "tg4_source_tg3_structurally_eligible");

    const auto effective = EA::ReconcileSchedulerTrainFeatureAblationMask(
        reordered, target, target);
    assert(effective.CanonicalText() == target.CanonicalText());

    ExpectFailure(Mask("tg4_inner_break_any"), target, std::nullopt,
                  "TRAIN_FEATURE_ABLATION_CLI_PERSISTED_MISMATCH");
    ExpectFailure(std::nullopt, target, std::nullopt,
                  "TRAIN_FEATURE_ABLATION_CLI_REQUIRED");
    ExpectFailure(Mask("tg4_inner_break_any"), Mask(""), std::nullopt,
                  "TRAIN_FEATURE_ABLATION_CLI_UNEXPECTED_FOR_CONTROL");
    ExpectFailure(reordered, target, Mask("tg4_inner_break_any"),
                  "TRAIN_FEATURE_ABLATION_RESUME_LINEAGE_MISMATCH");
    return 0;
}
