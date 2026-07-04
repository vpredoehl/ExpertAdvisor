#pragma once

#include <array>
#include <cstdio>
#include <optional>
#include <sstream>
#include <string>

namespace EA::RunMetadata
{

constexpr const char* kSchedulerVersion = "lstm-experiment-framework-1";

struct Snapshot
{
    std::string gitCommit = "unknown";
    std::string gitBranch = "unknown";
    std::optional<bool> gitDirty;
    std::string buildConfig;
    std::string compilerVersion;
    std::string schedulerVersion = kSchedulerVersion;
    std::string binaryName;
    std::string invocationMode;
};

inline std::string Trim(std::string value)
{
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                              value.back() == ' ' || value.back() == '\t'))
        value.pop_back();
    size_t first = 0;
    while (first < value.size() &&
           (value[first] == ' ' || value[first] == '\t' ||
            value[first] == '\n' || value[first] == '\r'))
        ++first;
    return value.substr(first);
}

inline std::optional<std::string> CaptureCommandOutput(const char* command)
{
    FILE* pipe = popen(command, "r");
    if (!pipe)
        return std::nullopt;

    std::array<char, 256> buffer{};
    std::ostringstream out;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
        out << buffer.data();
    const int rc = pclose(pipe);
    if (rc != 0)
        return std::nullopt;
    return Trim(out.str());
}

inline Snapshot Capture(const std::string& binaryName,
                        const std::string& invocationMode)
{
    Snapshot snapshot;
    snapshot.binaryName = binaryName;
    snapshot.invocationMode = invocationMode;

#if defined(NDEBUG)
    snapshot.buildConfig = "Release";
#else
    snapshot.buildConfig = "Debug";
#endif

#if defined(__clang_version__)
    snapshot.compilerVersion = __clang_version__;
#elif defined(__VERSION__)
    snapshot.compilerVersion = __VERSION__;
#else
    snapshot.compilerVersion = "unknown";
#endif

    if (const auto commit = CaptureCommandOutput("git rev-parse HEAD 2>/dev/null");
        commit.has_value() && !commit->empty())
    {
        snapshot.gitCommit = *commit;
    }
    if (const auto branch = CaptureCommandOutput("git branch --show-current 2>/dev/null");
        branch.has_value() && !branch->empty())
    {
        snapshot.gitBranch = *branch;
    }
    if (const auto status = CaptureCommandOutput("git status --porcelain 2>/dev/null");
        status.has_value())
    {
        snapshot.gitDirty = !status->empty();
    }

    return snapshot;
}

} // namespace EA::RunMetadata
