#pragma once

#include "SchedulerRepository.hpp"

#include <optional>
#include <string_view>
#include <vector>

namespace EA::SchedulerCore
{

struct PhaseAdmissionSnapshot
{
    std::vector<PendingSchedulerExperimentRecord> candidates;
    int capacityLimit = 0;
    int capacityUsed = 0;

    [[nodiscard]] bool hasCapacity() const noexcept;
    [[nodiscard]] int availableSlots() const noexcept;
};

class SchedulerAdmissionService final
{
public:
    explicit SchedulerAdmissionService(SchedulerRepository& repository);

    PhaseAdmissionSnapshot loadPhase(
        std::string_view phase,
        int capacityLimit,
        bool cancellationOnly = false);
    std::vector<PendingSchedulerExperimentRecord> loadCandidates(
        std::string_view phase,
        bool cancellationOnly = false);
    SchedulerQueueSnapshot loadQueueSnapshot();
    int capacityUsed(std::string_view capacityClass);
    bool hasCapacity(std::string_view capacityClass, int capacityLimit);
    std::optional<PreemptionVictimRecord> selectPreemptionVictim(
        std::string_view phase,
        std::string_view candidatePriority);

private:
    SchedulerRepository& repository_;
};

} // namespace EA::SchedulerCore
