#pragma once

#include <string>
#include <vector>

namespace EA::Scheduler
{

inline std::vector<std::string> BeginTrainingWorkerCommand(
    const std::string& selectedWorkerExecutable,
    const std::string& canonicalFeatureAblationMask)
{
    std::vector<std::string> argv{
        selectedWorkerExecutable,
        "--train"};
    if (!canonicalFeatureAblationMask.empty())
    {
        argv.push_back(
            "--ablate-features=" + canonicalFeatureAblationMask);
    }
    return argv;
}

} // namespace EA::Scheduler
