#pragma once
#include "SchedulerPhasePriority.hpp"

namespace EA::SchedulerCore
{
class SchedulerPhasePriorityRepository
{
public:
    virtual ~SchedulerPhasePriorityRepository() = default;
    virtual std::string load() = 0;
    virtual void save(const std::string& canonical) = 0;
};

class SchedulerPhasePriorityService final
{
public:
    explicit SchedulerPhasePriorityService(SchedulerPhasePriorityRepository& repository)
        : repository_{repository} {}
    SchedulerPhasePriority load() { return SchedulerPhasePriority::Parse(repository_.load()); }
    void set(const SchedulerPhasePriority& policy) { repository_.save(policy.canonical()); }
private:
    SchedulerPhasePriorityRepository& repository_;
};
} // namespace EA::SchedulerCore
