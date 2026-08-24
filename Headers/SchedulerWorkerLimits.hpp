#pragma once

#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

namespace EA::ExperimentScheduler
{

inline constexpr int kDefaultMaxTrainProcs = 1;
inline constexpr int kDefaultMaxInferProcs = 1;
inline constexpr int kDefaultMaxAnalyzeProcs = 1;

inline int ParseNonNegativeSchedulerWorkerLimit(
    const std::string& optionName,
    const std::string& value)
{
    std::size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(
            "invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed < 0 ||
        parsed > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(
            "invalid " + optionName + " value '" + value + "'");
    }
    return static_cast<int>(parsed);
}

inline bool TryParseSchedulerWorkerLimitArgument(
    int argc,
    const char* argv[],
    int& index,
    int& maxTrainProcs,
    int& maxInferProcs,
    int& maxAnalyzeProcs)
{
    const std::string argument{argv[index]};
    const auto tryParse = [&](const std::string& optionName, int& target) {
        if (argument == optionName)
        {
            if (index + 1 >= argc)
                throw std::invalid_argument(optionName + " requires a value");
            target = ParseNonNegativeSchedulerWorkerLimit(
                optionName, argv[++index]);
            return true;
        }

        const std::string prefix = optionName + "=";
        if (argument.rfind(prefix, 0) != 0)
            return false;
        target = ParseNonNegativeSchedulerWorkerLimit(
            optionName, argument.substr(prefix.size()));
        return true;
    };

    return tryParse("--max-train-procs", maxTrainProcs) ||
           tryParse("--max-infer-procs", maxInferProcs) ||
           tryParse("--max-analyze-procs", maxAnalyzeProcs);
}

inline std::optional<int> ExtractSchedulerWorkerLimitFromCommand(
    const std::string& command,
    const std::string& optionName)
{
    const std::regex optionPattern{
        optionName + R"((?:=|\s+)([0-9]+))"};
    std::smatch match;
    if (!std::regex_search(command, match, optionPattern))
        return std::nullopt;
    return ParseNonNegativeSchedulerWorkerLimit(
        optionName, match[1].str());
}

constexpr bool SchedulerWorkerCapacityHasSlot(
    int maximum,
    int consuming) noexcept
{
    return consuming < maximum;
}

constexpr bool SchedulerWorkerCapacityExceeded(
    int maximum,
    int consuming) noexcept
{
    return consuming > maximum;
}

constexpr int AvailableWorkerProcessSlots(
    int maximum,
    int consuming) noexcept
{
    return SchedulerWorkerCapacityHasSlot(maximum, consuming)
        ? maximum - consuming
        : 0;
}

constexpr int AvailableInferProcessSlots(
    int maximum,
    int runningExperimentInference,
    int runningCheckpointInference) noexcept
{
    return AvailableWorkerProcessSlots(
        AvailableWorkerProcessSlots(
            maximum, runningExperimentInference),
        runningCheckpointInference);
}

} // namespace EA::ExperimentScheduler
