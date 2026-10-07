#pragma once

namespace EA { struct LaunchArgs; }

namespace EA::LaunchRuntimeConfig
{
void Apply(const LaunchArgs& launchArgs, bool& runtimeInferenceMode);
}
