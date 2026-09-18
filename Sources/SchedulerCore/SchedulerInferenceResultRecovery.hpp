#pragma once

namespace EA::SchedulerCore
{

struct AuthoritativeFinalInferenceResult
{
    long long resultId = 0;
    long long modelId = 0;
    bool forcedFinalInferenceRerun = false;
};

} // namespace EA::SchedulerCore
