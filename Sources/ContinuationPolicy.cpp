#include "ContinuationPolicy.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ContinuationPolicyInheritance.hpp"

namespace EA::ExperimentScheduler
{
namespace
{

std::vector<std::string> SplitCommaSeparated(const std::string& value)
{
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ','))
    {
        if (!part.empty())
            parts.push_back(part);
    }
    return parts;
}

int ParsePositiveInt(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    return static_cast<int>(parsed);
}

double ParseFiniteDouble(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    double parsed = 0.0;
    try
    {
        parsed = std::stod(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || !std::isfinite(parsed))
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

double ParseNonNegativeFiniteDouble(
    const std::string& optionName,
    const std::string& value)
{
    const double parsed = ParseFiniteDouble(optionName, value);
    if (parsed < 0.0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

bool ParseBoolean(const std::string& optionName, const std::string& value)
{
    if (value == "true" || value == "1")
        return true;
    if (value == "false" || value == "0")
        return false;
    throw std::invalid_argument(
        "invalid " + optionName + " value '" + value + "'; expected true or false");
}

std::string InvalidTargetSequenceValue(const std::string& value)
{
    return "invalid target_sequence value '" + value + "'";
}

} // namespace

bool ValidContinuationScope(const std::string& value)
{
    return value == "symbol_horizon" || value == "horizon" || value == "global";
}

bool ValidContinuationTrendMode(const std::string& value)
{
    return value == "none" || value == "non_degrading" || value == "improving";
}

bool ValidContinuationSourceMode(const std::string& value)
{
    return value == "best_checkpoint" ||
           value == "latest_checkpoint" ||
           value == "final_model";
}

bool ValidContinuationProgressionMode(const std::string& value)
{
    return value == "fixed_increment" || value == "target_sequence";
}

std::optional<std::string> EffectiveContinuationProgressionMode(
    const ContinuationPolicyConfig& config)
{
    if (config.progressionMode.has_value())
        return config.progressionMode;
    if (config.targetIncrement.has_value())
        return "fixed_increment";
    return std::nullopt;
}

std::vector<int> ParseContinuationTargetSequence(const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("target_sequence_required");

    std::vector<int> sequence;
    std::set<int> seen;
    size_t begin = 0;
    while (begin <= value.size())
    {
        const size_t delimiter = value.find(':', begin);
        const size_t end = delimiter == std::string::npos ? value.size() : delimiter;
        if (end == begin)
            throw std::invalid_argument(InvalidTargetSequenceValue(value));
        const std::string item = value.substr(begin, end - begin);
        size_t consumed = 0;
        long long parsed = 0;
        try
        {
            parsed = std::stoll(item, &consumed, 10);
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument(InvalidTargetSequenceValue(value));
        }
        if (consumed != item.size() || parsed > std::numeric_limits<int>::max() ||
            parsed < std::numeric_limits<int>::min())
            throw std::invalid_argument(InvalidTargetSequenceValue(value));
        const int target = static_cast<int>(parsed);
        if (target <= 0)
            throw std::invalid_argument("target_sequence_contains_nonpositive_value");
        if (!seen.insert(target).second)
            throw std::invalid_argument("target_sequence_contains_duplicate");
        if (!sequence.empty() && target < sequence.back())
            throw std::invalid_argument("target_sequence_must_be_strictly_increasing");
        sequence.push_back(target);
        if (delimiter == std::string::npos)
            break;
        begin = delimiter + 1;
    }
    return sequence;
}

std::string ContinuationTargetSequenceText(
    const std::optional<std::vector<int>>& sequence,
    const std::string& nullText)
{
    if (!sequence.has_value())
        return nullText;
    std::ostringstream out;
    for (size_t index = 0; index < sequence->size(); ++index)
    {
        if (index != 0)
            out << ':';
        out << (*sequence)[index];
    }
    return out.str();
}

ContinuationPolicyUpdate ParseContinuationPolicyUpdate(const std::string& text)
{
    ContinuationPolicyUpdate update;
    for (const std::string& item : SplitCommaSeparated(text))
    {
        const size_t equals = item.find('=');
        if (equals == std::string::npos || equals == 0)
            throw std::invalid_argument("--set-continuation-policy requires key=value pairs");
        const std::string key = item.substr(0, equals);
        const std::string value = item.substr(equals + 1);
        if (value.empty())
        {
            if (key == "target_sequence")
                throw std::invalid_argument("target_sequence_required");
            throw std::invalid_argument("--set-continuation-policy requires key=value pairs");
        }
        update.keys.insert(key);
        const bool clearValue = value == "null";

        if (key == "target_epochs")
        {
            if (!clearValue)
                update.targetEpochs = ParsePositiveInt(key, value);
        }
        else if (key == "min_evals")
            update.minEvals = ParsePositiveInt(key, value);
        else if (key == "patience")
            update.patience = ParsePositiveInt(key, value);
        else if (key == "min_leader_score")
        {
            if (!clearValue)
                update.minLeaderScore = ParseFiniteDouble(key, value);
        }
        else if (key == "min_infer_accuracy")
        {
            if (!clearValue)
                update.minInferAccuracy = ParseFiniteDouble(key, value);
        }
        else if (key == "min_improvement")
        {
            if (!clearValue)
                update.minImprovement = ParseNonNegativeFiniteDouble(key, value);
        }
        else if (key == "max_degradation")
        {
            if (!clearValue)
                update.maxDegradation = ParseNonNegativeFiniteDouble(key, value);
        }
        else if (key == "top_n")
        {
            if (!clearValue)
                update.topN = ParsePositiveInt(key, value);
        }
        else if (key == "scope")
            update.scope = value;
        else if (key == "trend_mode")
            update.trendMode = value;
        else if (key == "source_mode")
            update.sourceMode = value;
        else if (key == "include_excluded")
            update.includeExcluded = ParseBoolean(key, value);
        else if (key == "candidate_excluded")
            update.candidateExcluded = ParseBoolean(key, value);
        else if (key == "inherit_to_child")
            update.inheritToChild = ParseBoolean(key, value);
        else if (key == "progression_mode")
        {
            if (!clearValue)
                update.progressionMode = value;
        }
        else if (key == "target_increment")
        {
            if (!clearValue)
                update.targetIncrement = ParsePositiveInt(key, value);
        }
        else if (key == "max_target_epochs")
        {
            if (!clearValue)
                update.maxTargetEpochs = ParsePositiveInt(key, value);
        }
        else if (key == "target_sequence")
        {
            if (!clearValue)
                update.targetSequence = ParseContinuationTargetSequence(value);
        }
        else
            throw std::invalid_argument("unsupported continuation policy key '" + key + "'");
    }
    if (update.keys.empty())
        throw std::invalid_argument("--set-continuation-policy requires at least one key=value pair");
    return update;
}

void ApplyContinuationPolicyUpdate(
    ContinuationPolicyConfig& config,
    const ContinuationPolicyUpdate& update)
{
    if (update.keys.count("target_epochs")) config.targetEpochs = update.targetEpochs;
    if (update.keys.count("min_evals")) config.minEvals = update.minEvals;
    if (update.keys.count("patience")) config.patience = update.patience;
    if (update.keys.count("min_leader_score")) config.minLeaderScore = update.minLeaderScore;
    if (update.keys.count("min_infer_accuracy")) config.minInferAccuracy = update.minInferAccuracy;
    if (update.keys.count("min_improvement")) config.minImprovement = update.minImprovement;
    if (update.keys.count("max_degradation")) config.maxDegradation = update.maxDegradation;
    if (update.keys.count("top_n")) config.topN = update.topN;
    if (update.keys.count("scope")) config.scope = update.scope;
    if (update.keys.count("trend_mode")) config.trendMode = update.trendMode;
    if (update.keys.count("source_mode")) config.sourceMode = update.sourceMode;
    if (update.keys.count("include_excluded")) config.includeExcluded = update.includeExcluded;
    if (update.keys.count("candidate_excluded")) config.candidateExcluded = update.candidateExcluded;
    if (update.keys.count("inherit_to_child")) config.inheritToChild = update.inheritToChild;
    if (update.keys.count("progression_mode")) config.progressionMode = update.progressionMode;
    if (update.keys.count("target_increment")) config.targetIncrement = update.targetIncrement;
    if (update.keys.count("max_target_epochs")) config.maxTargetEpochs = update.maxTargetEpochs;
    if (update.keys.count("target_sequence")) config.targetSequence = update.targetSequence;
}

std::optional<std::string> ContinuationPolicyConfigurationError(
    const ContinuationPolicyConfig& config,
    bool requireSelectionConfig)
{
    if (config.targetEpochs.has_value() && *config.targetEpochs <= 0)
        return "target_epochs_must_be_positive";
    if (config.minEvals <= 0) return "min_evals_must_be_positive";
    if (config.patience <= 0) return "patience_must_be_positive";
    if (config.minLeaderScore.has_value() && !std::isfinite(*config.minLeaderScore))
        return "min_leader_score_must_be_finite";
    if (config.minInferAccuracy.has_value() && !std::isfinite(*config.minInferAccuracy))
        return "min_infer_accuracy_must_be_finite";
    if (config.minImprovement.has_value() &&
        (!std::isfinite(*config.minImprovement) || *config.minImprovement < 0.0))
        return "min_improvement_must_be_non_negative_and_finite";
    if (config.maxDegradation.has_value() &&
        (!std::isfinite(*config.maxDegradation) || *config.maxDegradation < 0.0))
        return "max_degradation_must_be_non_negative_and_finite";
    if (config.topN.has_value() && *config.topN <= 0) return "top_n_must_be_positive";
    if (!ValidContinuationScope(config.scope)) return "invalid_scope";
    if (!ValidContinuationTrendMode(config.trendMode)) return "invalid_trend_mode";
    if (!ValidContinuationSourceMode(config.sourceMode)) return "invalid_source_mode";
    if (config.progressionMode.has_value() &&
        !ValidContinuationProgressionMode(*config.progressionMode))
        return "invalid_progression_mode";

    if (config.targetSequence.has_value())
    {
        if (config.targetSequence->empty())
            return "target_sequence_required";
        std::set<int> seen;
        std::optional<int> previous;
        for (const int target : *config.targetSequence)
        {
            if (target <= 0)
                return "target_sequence_contains_nonpositive_value";
            if (!seen.insert(target).second)
                return "target_sequence_contains_duplicate";
            if (previous.has_value() && target < *previous)
                return "target_sequence_must_be_strictly_increasing";
            previous = target;
        }
    }

    const std::optional<std::string> progressionMode =
        EffectiveContinuationProgressionMode(config);
    if (progressionMode == "target_sequence")
    {
        if (!config.targetSequence.has_value() || config.targetSequence->empty())
            return "target_sequence_required";
        if (config.targetIncrement.has_value())
            return "target_increment_disallowed_for_target_sequence";
        if (config.maxTargetEpochs.has_value() &&
            *config.maxTargetEpochs != config.targetSequence->back())
            return "max_target_conflicts_with_sequence";

        const bool terminal = config.inheritanceStatus == "max_target_reached";
        if (terminal)
        {
            if (config.source.targetEpochs != config.targetSequence->back())
                return "sequence_has_no_target_after_current_epoch";
            if (config.targetEpochs.has_value())
                return "policy_target_is_not_next_sequence_target";
        }
        else
        {
            const SequenceContinuationTargetDerivation sequenceTarget =
                DeriveSequenceContinuationChildPolicy(
                    config.source.targetEpochs,
                    config.targetEpochs,
                    config.targetSequence);
            if (!sequenceTarget.error.empty())
                return sequenceTarget.error;
        }
    }
    else
    {
        if (config.targetSequence.has_value())
            return "fixed_increment_disallows_target_sequence";
        if (config.progressionMode == "fixed_increment" &&
            !config.targetIncrement.has_value())
            return "fixed_increment_requires_target_increment";
    }

    if (config.targetIncrement.has_value() && *config.targetIncrement <= 0)
        return "target_increment_must_be_positive";
    if (progressionMode != "target_sequence" && config.maxTargetEpochs.has_value())
    {
        if (*config.maxTargetEpochs <= 0) return "max_target_epochs_must_be_positive";
        const bool terminal = config.inheritanceStatus == "max_target_reached";
        if ((!terminal && *config.maxTargetEpochs <= config.source.targetEpochs) ||
            (terminal && *config.maxTargetEpochs != config.source.targetEpochs))
            return "max_target_must_exceed_current_target";
        if (config.targetEpochs.has_value() && *config.targetEpochs > *config.maxTargetEpochs)
            return "policy_target_exceeds_max_target";
    }
    if (config.inheritToChild && progressionMode != "target_sequence")
    {
        const ContinuationTargetDerivation target =
            DeriveContinuationChildPolicyTarget(config.targetEpochs, config.targetIncrement);
        if (!target.error.empty()) return target.error;
        if (config.maxTargetEpochs.has_value() &&
            *config.targetEpochs < *config.maxTargetEpochs &&
            *target.targetEpochs > *config.maxTargetEpochs)
            return "target_increment_would_skip_past_max_target";
    }
    if (config.trendMode == "none" &&
        (config.minImprovement.has_value() || config.maxDegradation.has_value()))
        return "trend_none_requires_null_trend_thresholds";
    if (config.trendMode == "non_degrading" &&
        (!config.maxDegradation.has_value() || config.minImprovement.has_value()))
        return "non_degrading_requires_only_max_degradation";
    if (config.trendMode == "improving" &&
        (!config.minImprovement.has_value() || config.maxDegradation.has_value()))
        return "improving_requires_only_min_improvement";
    if (requireSelectionConfig && !config.targetEpochs.has_value())
        return "target_epochs_required";
    if (requireSelectionConfig && config.candidateExcluded && !config.includeExcluded)
        return "excluded_candidate_requires_include_excluded";
    if (requireSelectionConfig &&
        !config.minLeaderScore.has_value() &&
        !config.minInferAccuracy.has_value() &&
        !config.topN.has_value() &&
        config.trendMode == "none")
        return "at_least_one_threshold_ranking_or_trend_criterion_required";
    return std::nullopt;
}

std::optional<std::string> ContinuationPolicyEnablementError(
    const ContinuationPolicyConfig& config)
{
    const std::optional<std::string> configurationError =
        ContinuationPolicyConfigurationError(config, true);
    if (configurationError.has_value())
        return configurationError;

    if (config.status == "pending" || config.status == "paused" ||
        config.status == "running" || config.status == "completed")
    {
        return std::nullopt;
    }
    if (config.status == "cancelled")
        return "policy_enablement_disallowed_for_cancelled_source";
    if (config.status == "failed")
        return "policy_enablement_disallowed_for_failed_source";
    return "policy_enablement_disallowed_for_source_status";
}

bool ContinuationPolicySourceCompletionReady(
    const ContinuationPolicyConfig& config)
{
    return config.status == "completed" && config.phase == "done";
}

std::string ContinuationPolicySemanticCanonicalText(const ContinuationPolicyConfig& config)
{
    std::ostringstream out;
    out << "target_epochs=" << ContinuationOptionalIntText(config.targetEpochs)
        << "|min_evals=" << config.minEvals
        << "|patience=" << config.patience
        << "|min_leader_score=" << ContinuationOptionalDoubleText(config.minLeaderScore)
        << "|min_infer_accuracy=" << ContinuationOptionalDoubleText(config.minInferAccuracy)
        << "|min_improvement=" << ContinuationOptionalDoubleText(config.minImprovement)
        << "|max_degradation=" << ContinuationOptionalDoubleText(config.maxDegradation)
        << "|top_n=" << ContinuationOptionalIntText(config.topN)
        << "|scope=" << config.scope
        << "|trend_mode=" << config.trendMode
        << "|source_mode=" << config.sourceMode
        << "|include_excluded=" << (config.includeExcluded ? "true" : "false")
        << "|candidate_excluded=" << (config.candidateExcluded ? "true" : "false")
        << "|inherit_to_child=" << (config.inheritToChild ? "true" : "false");
    if (EffectiveContinuationProgressionMode(config) == "target_sequence")
    {
        out << "|progression_mode=target_sequence|target_sequence=[";
        if (config.targetSequence.has_value())
        {
            for (size_t index = 0; index < config.targetSequence->size(); ++index)
            {
                if (index != 0)
                    out << ',';
                out << (*config.targetSequence)[index];
            }
        }
        out << ']';
    }
    out << "|target_increment=" << ContinuationOptionalIntText(config.targetIncrement);
    if (config.maxTargetEpochs.has_value())
        out << "|max_target_epochs=" << *config.maxTargetEpochs;
    return out.str();
}

std::string StableContinuationPolicyHash(const std::string& canonicalPolicy)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char ch : canonicalPolicy)
    {
        hash ^= static_cast<uint64_t>(ch);
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::string SemanticContinuationPolicyHash(
    const ContinuationPolicyIdentityMaterial& material)
{
    return StableContinuationPolicyHash(material.semanticConfiguration);
}

std::string ContinuationPolicySemanticHash(const ContinuationPolicyConfig& config)
{
    return SemanticContinuationPolicyHash(
        {ContinuationPolicySemanticCanonicalText(config),
         config.policyInherited,
         config.inheritedFromExperimentId,
         config.inheritanceStatus});
}

std::string ContinuationOptionalDoubleText(const std::optional<double>& value)
{
    if (!value.has_value())
        return "NULL";
    std::ostringstream out;
    out << std::setprecision(17) << *value;
    return out.str();
}

std::string ContinuationOptionalIntText(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "NULL";
}

std::string ContinuationPolicyDisplayText(const ContinuationPolicyConfig& config)
{
    return ContinuationPolicySemanticCanonicalText(config) +
           "|policy_revision=" + std::to_string(config.policyRevision) +
           "|policy_inherited=" + (config.policyInherited ? "true" : "false") +
           "|inherited_from_experiment_id=" +
               (config.inheritedFromExperimentId.has_value()
                    ? std::to_string(*config.inheritedFromExperimentId) : "NULL") +
           "|inherited_from_revision=" +
               (config.inheritedFromRevision.has_value()
                    ? std::to_string(*config.inheritedFromRevision) : "NULL") +
           "|inherited_from_hash=" + config.inheritedFromHash.value_or("NULL") +
           "|inheritance_status=" + config.inheritanceStatus;
}

} // namespace EA::ExperimentScheduler
