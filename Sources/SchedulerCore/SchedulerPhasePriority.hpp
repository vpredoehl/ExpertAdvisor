#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::SchedulerCore
{
enum class SchedulerPhase { Train, Infer, Analyze };

inline std::string_view SchedulerPhaseName(SchedulerPhase phase)
{
    switch (phase)
    {
    case SchedulerPhase::Train: return "train";
    case SchedulerPhase::Infer: return "infer";
    case SchedulerPhase::Analyze: return "analyze";
    }
    throw std::logic_error("unknown scheduler phase");
}

struct SchedulerPhasePriority
{
    bool concurrent = true;
    std::array<SchedulerPhase, 3> order{
        SchedulerPhase::Train, SchedulerPhase::Infer, SchedulerPhase::Analyze};

    static SchedulerPhasePriority Parse(std::string_view value)
    {
        if (value == "concurrent") return {};
        SchedulerPhasePriority policy;
        policy.concurrent = false;
        std::array<bool, 3> seen{};
        for (std::size_t index = 0; index < 3; ++index)
        {
            const auto separator = value.find(':');
            const auto token = value.substr(0, separator);
            SchedulerPhase phase;
            if (token == "train") phase = SchedulerPhase::Train;
            else if (token == "infer") phase = SchedulerPhase::Infer;
            else if (token == "analyze") phase = SchedulerPhase::Analyze;
            else throw std::invalid_argument("phase priority must be concurrent or a permutation of train:infer:analyze");
            const auto slot = static_cast<std::size_t>(phase);
            if (seen[slot]) throw std::invalid_argument("duplicate phase in phase priority");
            seen[slot] = true;
            policy.order[index] = phase;
            if ((index < 2 && separator == std::string_view::npos) ||
                (index == 2 && separator != std::string_view::npos))
                throw std::invalid_argument("phase priority requires exactly three phases");
            if (index < 2) value.remove_prefix(separator + 1);
        }
        return policy;
    }

    std::string canonical() const
    {
        if (concurrent) return "concurrent";
        return std::string{SchedulerPhaseName(order[0])} + ":" +
            std::string{SchedulerPhaseName(order[1])} + ":" +
            std::string{SchedulerPhaseName(order[2])};
    }
};

struct SchedulerPhaseDemand
{
    bool enabled = false;
    bool eligiblePending = false;
    int activeWorkers = 0;
    // Lowest numeric scheduler-priority rank among eligible pending work.
    std::optional<int> highestEligiblePriorityRank;
};

struct SchedulerPhaseAdmissionPlan
{
    std::array<bool, 3> allowed{true, true, true};
    std::optional<SchedulerPhase> selected;
    std::optional<int> selectedPriorityRank;
    bool draining = false;
    bool ordered = false;
};

// Scheduler priority controls admission across all phases. Persisted phase
// precedence is the tie-breaker among work at that same priority. Existing
// same-phase preemption remains in FinalExperimentDispatchService.
inline SchedulerPhaseAdmissionPlan PlanSchedulerPhases(
    const SchedulerPhasePriority& policy,
    const std::array<SchedulerPhaseDemand, 3>& demand)
{
    SchedulerPhaseAdmissionPlan plan;
    for (const auto& state : demand)
    {
        if (!state.enabled || !state.eligiblePending ||
            !state.highestEligiblePriorityRank)
            continue;
        if (!plan.selectedPriorityRank ||
            *state.highestEligiblePriorityRank < *plan.selectedPriorityRank)
            plan.selectedPriorityRank = state.highestEligiblePriorityRank;
    }

    if (policy.concurrent)
    {
        if (!plan.selectedPriorityRank)
            return plan;
        plan.allowed.fill(false);
        for (std::size_t index = 0; index < demand.size(); ++index)
        {
            const auto& state = demand[index];
            plan.allowed[index] = state.enabled && state.eligiblePending &&
                state.highestEligiblePriorityRank == plan.selectedPriorityRank;
        }
        return plan;
    }
    plan.ordered = true;
    plan.allowed.fill(false);
    for (const auto phase : policy.order)
    {
        const auto& state = demand[static_cast<std::size_t>(phase)];
        const bool selectedPriorityPending = state.enabled &&
            state.eligiblePending &&
            state.highestEligiblePriorityRank == plan.selectedPriorityRank;
        if (selectedPriorityPending ||
            (!plan.selectedPriorityRank && state.enabled &&
             state.activeWorkers > 0))
        {
            plan.selected = phase;
            break;
        }
    }
    if (!plan.selected) return plan;
    for (std::size_t index = 0; index < demand.size(); ++index)
    {
        if (index != static_cast<std::size_t>(*plan.selected) &&
            demand[index].activeWorkers > 0)
        {
            plan.draining = true;
            return plan;
        }
    }
    plan.allowed[static_cast<std::size_t>(*plan.selected)] = true;
    return plan;
}
} // namespace EA::SchedulerCore
