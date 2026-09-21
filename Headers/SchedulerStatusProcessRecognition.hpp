#pragma once

#include <cctype>
#include <string_view>

namespace EA::ExperimentScheduler
{

inline bool SchedulerStatusCommandHasArgument(
    std::string_view command,
    std::string_view argument)
{
    std::size_t position = command.find(argument);
    while (position != std::string_view::npos)
    {
        const std::size_t after = position + argument.size();
        const bool hasLeadingBoundary =
            position == 0 || std::isspace(
                static_cast<unsigned char>(command[position - 1]));
        const bool hasTrailingBoundary =
            after == command.size() || std::isspace(
                static_cast<unsigned char>(command[after]));
        if (hasLeadingBoundary && hasTrailingBoundary)
            return true;
        position = command.find(argument, position + 1);
    }
    return false;
}

inline bool SchedulerStatusCommandHasExecutableBasename(
    std::string_view command,
    std::string_view basename)
{
    const std::size_t first = command.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return false;

    std::size_t position = command.find(basename, first);
    while (position != std::string_view::npos)
    {
        const std::size_t after = position + basename.size();
        const bool hasTrailingBoundary =
            after == command.size() || std::isspace(
                static_cast<unsigned char>(command[after]));
        if (!hasTrailingBoundary)
        {
            position = command.find(basename, position + 1);
            continue;
        }

        if (position == first)
            return true;

        // macOS ps does not preserve quoting around an executable path that
        // contains spaces.  Treat a known basename as executable evidence only
        // when it completes the leading absolute path, before any option.
        const std::string_view prefix = command.substr(first, position - first);
        if (command[first] == '/' &&
            prefix.find(" --") == std::string_view::npos &&
            prefix.find('=') == std::string_view::npos &&
            prefix.back() == '/')
        {
            return true;
        }
        position = command.find(basename, position + 1);
    }
    return false;
}

// Restrict status process evidence to executable identities that have hosted
// the scheduler and require its exact mode argument.  This excludes workers
// and unrelated commands that merely contain legacy LSTM text.
inline bool IsSchedulerStatusSchedulerProcessCommand(
    std::string_view command)
{
    if (!SchedulerStatusCommandHasArgument(
            command, "--schedule-experiments"))
    {
        return false;
    }

    return SchedulerStatusCommandHasExecutableBasename(
               command, "lstm-scheduler") ||
           SchedulerStatusCommandHasExecutableBasename(
               command, "LSTM_Release") ||
           SchedulerStatusCommandHasExecutableBasename(command, "LSTM");
}

} // namespace EA::ExperimentScheduler
