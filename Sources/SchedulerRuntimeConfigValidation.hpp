#pragma once
#include <cstddef>
#include <pqxx/pqxx>
#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"
#include "LaunchArguments.hpp"
namespace EA::SchedulerRuntimeConfigValidation
{
Donchian20Mode LoadExperimentDonchian20Mode(pqxx::work&, long long);
FeatureWarmupScope LoadExperimentFeatureWarmupScope(pqxx::work&, long long);
std::size_t LoadExperimentDonchianLookback(pqxx::work&, long long);
void ValidateSchedulerDonchian20Mode(pqxx::work&, const LaunchArgs&, Donchian20Mode);
void ValidateSchedulerFeatureWarmupScope(pqxx::work&, const LaunchArgs&, FeatureWarmupScope);
void ValidateSchedulerDonchianLookback(pqxx::work&, const LaunchArgs&, std::size_t);
}
