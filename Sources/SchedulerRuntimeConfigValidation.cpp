#include "SchedulerRuntimeConfigValidation.hpp"

#include <stdexcept>
#include "Donchian20Mode.hpp"
#include "DonchianLookback.hpp"
#include "FeatureWarmupScope.hpp"

namespace EA::SchedulerRuntimeConfigValidation
{
Donchian20Mode LoadExperimentDonchian20Mode(pqxx::work& w,
                                            long long experimentId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT donchian20_mode FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        throw std::runtime_error("experiment_not_found_for_donchian20_mode");
    return ParseDonchian20Mode(rows[0][0].as<std::string>());
}

EA::FeatureWarmupScope LoadExperimentFeatureWarmupScope(
    pqxx::work& w,
    long long experimentId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT feature_warmup_scope FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        throw std::runtime_error("experiment_not_found_for_feature_warmup_scope");
    return EA::ParseFeatureWarmupScope(rows[0][0].as<std::string>());
}

std::size_t LoadExperimentDonchianLookback(pqxx::work& w,
                                           long long experimentId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT donchian_lookback FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        throw std::runtime_error("experiment_not_found_for_donchian_lookback");
    return ParseDonchianLookback(rows[0][0].as<std::string>());
}

void ValidateSchedulerDonchian20Mode(pqxx::work& w,
                                     const EA::LaunchArgs& launchArgs,
                                     Donchian20Mode runtimeMode)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            throw std::runtime_error("checkpoint_eval_not_found_for_donchian20_mode");
        experimentId = rows[0][0].as<long long>();
    }
    if (!experimentId.has_value())
        return;

    const Donchian20Mode persistedMode =
        LoadExperimentDonchian20Mode(w, *experimentId);
    if (persistedMode != runtimeMode)
        throw std::runtime_error(
            std::string{"scheduler experiment Donchian-20 mode mismatch: persisted="} +
            Donchian20ModeText(persistedMode) + ", runtime=" +
            Donchian20ModeText(runtimeMode));
}

void ValidateSchedulerFeatureWarmupScope(
    pqxx::work& w,
    const EA::LaunchArgs& launchArgs,
    EA::FeatureWarmupScope runtimeScope)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            throw std::runtime_error("checkpoint_eval_not_found_for_feature_warmup_scope");
        experimentId = rows[0][0].as<long long>();
    }
    if (experimentId.has_value() &&
        LoadExperimentFeatureWarmupScope(w, *experimentId) != runtimeScope)
        throw std::runtime_error("scheduler experiment feature warmup scope mismatch");
}

void ValidateSchedulerDonchianLookback(pqxx::work& w,
                                       const EA::LaunchArgs& launchArgs,
                                       std::size_t runtimeLookback)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            throw std::runtime_error("checkpoint_eval_not_found_for_donchian_lookback");
        experimentId = rows[0][0].as<long long>();
    }
    if (experimentId.has_value() &&
        LoadExperimentDonchianLookback(w, *experimentId) != runtimeLookback)
        throw std::runtime_error("scheduler experiment Donchian lookback mismatch");
}




} // namespace EA::SchedulerRuntimeConfigValidation
