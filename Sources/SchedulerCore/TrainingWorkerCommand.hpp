#pragma once

#include <string>
#include <vector>

namespace EA::Scheduler
{

inline std::vector<std::string> BeginTrainingWorkerCommand(
    const std::string& selectedWorkerExecutable)
{
    return {
        selectedWorkerExecutable,
        "--train"};
}

} // namespace EA::Scheduler
