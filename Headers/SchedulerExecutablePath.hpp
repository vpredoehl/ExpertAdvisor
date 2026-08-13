#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <mach-o/dyld.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace EA::ExperimentScheduler
{

inline std::string ResolveCanonicalExecutablePath()
{
    uint32_t size = 0;
    if (_NSGetExecutablePath(nullptr, &size) == 0 || size == 0)
        throw std::runtime_error(
            "canonical_executable_path_size_unavailable");

    std::vector<char> buffer(size + 1, '\0');
    uint32_t supplied = static_cast<uint32_t>(buffer.size());
    if (_NSGetExecutablePath(buffer.data(), &supplied) != 0)
    {
        buffer.assign(static_cast<size_t>(supplied) + 1, '\0');
        supplied = static_cast<uint32_t>(buffer.size());
        if (_NSGetExecutablePath(buffer.data(), &supplied) != 0)
            throw std::runtime_error(
                "canonical_executable_path_resolution_failed");
    }

    errno = 0;
    char* resolved = ::realpath(buffer.data(), nullptr);
    if (resolved == nullptr)
    {
        const int errorNumber = errno;
        throw std::runtime_error(
            "canonical_executable_realpath_failed_errno_" +
            std::to_string(errorNumber));
    }
    std::string path{resolved};
    std::free(resolved);

    if (path.empty() || path.front() != '/' ||
        !std::filesystem::path(path).is_absolute())
    {
        throw std::runtime_error(
            "canonical_executable_path_not_absolute");
    }
    if (::access(path.c_str(), X_OK) != 0)
    {
        const int errorNumber = errno;
        throw std::runtime_error(
            "canonical_executable_not_executable_errno_" +
            std::to_string(errorNumber));
    }
    return path;
}

} // namespace EA::ExperimentScheduler
