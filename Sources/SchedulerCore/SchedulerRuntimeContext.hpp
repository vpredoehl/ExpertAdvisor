#pragma once

#include "SchedulerChildCompletionService.hpp"

namespace EA::SchedulerCore
{

// Lifetime-owned state for one scheduler daemon invocation. Launch,
// completion, reconciliation, preemption, and shutdown share this exact
// collection; it is deliberately not process-global scheduler state.
class SchedulerRuntimeContext final
{
public:
    SchedulerOwnedChildren ownedChildren;
};

} // namespace EA::SchedulerCore
