#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// This is the factual, configured portion of experiment-pair scientific
// identity.  It deliberately excludes disposition, ranking, analysis, and
// producing-worker judgments.  The generic comparison workflow and the
// operational observer share this catalog so an observer comparison cannot
// silently acquire a different configured identity definition.
namespace EA::ExperimentComparisonIdentity
{

inline constexpr std::array<std::string_view, 50>
    kConfiguredScientificIdentityFields {{
        "symbol", "prediction_horizon", "train_start", "train_end",
        "inference_start", "inference_end", "target_epochs", "threshold",
        "core_lr_mult", "head_lr_mult", "checkpoint_interval",
        "feature_warmup_scope", "donchian_mode", "donchian_lookback",
        "feature_ablation_mask", "initialization_lineage",
        "resume_expand_input_width", "training_objective_canonical",
        "training_objective_hash", "configured_model_input_width",
        "configured_model_input_semantic_layout_version",
        "economic_calendar_snapshot_id", "economic_calendar_snapshot_hash",
        "base_learning_rate", "batch_size", "fresh_initialization_seed",
        "training_objective_version", "loss_definition_version",
        "auxiliary_loss_mode", "auxiliary_loss_coefficient",
        "regression_target_definition", "regression_normalization_identity",
        "robust_loss_definition", "robust_loss_delta",
        "target_clipping_definition", "objective_normalization_identity",
        "checkpoint_inference_enabled", "checkpoint_inference_minimum_epoch",
        "checkpoint_inference_interval", "checkpoint_policy_enabled",
        "checkpoint_policy_minimum_leader_score",
        "checkpoint_policy_minimum_inference_accuracy", "checkpoint_policy_top_n",
        "checkpoint_policy_scope", "checkpoint_policy_stop_mode",
        "checkpoint_policy_grace_evaluations", "checkpoint_policy_revision",
        "checkpoint_policy_hash", "continuation_policy_enabled",
        "continuation_policy_scientific_identity",
    }};

struct Field
{
    std::string name;
    std::optional<std::string> value;
};

inline bool HasConfiguredScientificIdentityFields(
    const std::vector<Field>& fields)
{
    for (const std::string_view name : kConfiguredScientificIdentityFields)
    {
        bool found = false;
        for (const Field& field : fields)
            if (field.name == name)
            {
                found = true;
                break;
            }
        if (!found) return false;
    }
    return true;
}

} // namespace EA::ExperimentComparisonIdentity
