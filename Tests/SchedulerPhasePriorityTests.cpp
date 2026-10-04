#include "SchedulerCore/SchedulerPhasePriorityService.hpp"
#include "SchedulerCore/SchedulerPolicy.hpp"
#include <algorithm>
#include <cassert>

using namespace EA::SchedulerCore;
class MemoryPolicy final : public SchedulerPhasePriorityRepository
{
public:
    std::string value = "concurrent";
    std::string load() override { return value; }
    void save(const std::string& canonical) override { value = canonical; }
};
int main()
{
    const auto policy = SchedulerPhasePriority::Parse("train:infer:analyze");
    std::array<SchedulerPhaseDemand, 3> demand{{
        {true,true,0,std::nullopt}, {true,true,0,std::nullopt},
        {true,true,0,std::nullopt}}};
    auto plan = PlanSchedulerPhases(policy, demand);
    assert((plan.allowed == std::array<bool,3>{true,false,false}));
    demand[0] = {true,false,1,std::nullopt}; // Last train still active: infer must wait.
    assert(PlanSchedulerPhases(policy,demand).selected == SchedulerPhase::Train);
    demand[0].activeWorkers = 0;
    assert(PlanSchedulerPhases(policy,demand).selected == SchedulerPhase::Infer);
    demand[1].eligiblePending = false;
    assert(PlanSchedulerPhases(policy,demand).selected == SchedulerPhase::Analyze);
    demand[2].eligiblePending = false;
    assert(!PlanSchedulerPhases(policy,demand).selected);
    // Live switch and new earlier-phase demand drain competing workers.
    demand = {{{true,true,2,std::nullopt}, {true,true,0,std::nullopt},
               {true,true,0,std::nullopt}}};
    plan = PlanSchedulerPhases(SchedulerPhasePriority::Parse("infer:analyze:train"),demand);
    assert(plan.selected == SchedulerPhase::Infer && plan.draining);
    assert((plan.allowed == std::array<bool,3>{false,false,false}));
    demand[0].activeWorkers = 0;
    assert(PlanSchedulerPhases(SchedulerPhasePriority::Parse("infer:analyze:train"),demand).allowed[1]);
    demand[0].enabled = false; // Disabled phase skips pending work.
    assert(PlanSchedulerPhases(policy,demand).selected == SchedulerPhase::Infer);
    demand[0].activeWorkers = 1; // Disabled but still running must drain.
    assert(PlanSchedulerPhases(policy,demand).draining);
    assert((PlanSchedulerPhases({},demand).allowed == std::array<bool,3>{true,true,true}));

    // Priority is global: a high-priority analyze job wins over normal infer
    // and low train work even when infer is first in the persisted phase order.
    demand = {{{true,true,0,1},{true,true,0,1},{true,true,0,0}}};
    plan = PlanSchedulerPhases(
        SchedulerPhasePriority::Parse("infer:analyze:train"), demand);
    assert(plan.selected == SchedulerPhase::Analyze);
    assert(plan.selectedPriorityRank == 0);
    assert((plan.allowed == std::array<bool,3>{false,false,true}));

    // Phase precedence remains the tie-breaker for equal priorities.
    demand = {{{true,true,0,1},{true,true,0,1},{true,true,0,1}}};
    plan = PlanSchedulerPhases(
        SchedulerPhasePriority::Parse("infer:analyze:train"), demand);
    assert(plan.selected == SchedulerPhase::Infer);
    assert(plan.selectedPriorityRank == 1);

    // A higher-priority phase waiting for its own capacity/drain keeps lower
    // priority work closed; once it drains, normal work is admitted.
    demand = {{{true,true,1,2},{true,true,0,0},{true,true,0,1}}};
    plan = PlanSchedulerPhases(
        SchedulerPhasePriority::Parse("infer:analyze:train"), demand);
    assert(plan.selected == SchedulerPhase::Infer);
    assert(plan.draining);
    assert((plan.allowed == std::array<bool,3>{false,false,false}));
    demand[1] = {true,false,0,std::nullopt};
    plan = PlanSchedulerPhases(
        SchedulerPhasePriority::Parse("infer:analyze:train"), demand);
    assert(plan.selected == SchedulerPhase::Analyze);
    assert(plan.selectedPriorityRank == 1);

    // Concurrent phase policy still blocks lower priorities globally, while
    // allowing every phase that has the highest queued priority.
    demand = {{{true,true,0,0},{true,true,0,1},{true,true,0,0}}};
    plan = PlanSchedulerPhases({}, demand);
    assert((plan.allowed == std::array<bool,3>{true,false,true}));
    assert(plan.selectedPriorityRank == 0);

    // One globally winning class owns execution.  A normal infer candidate
    // must wait for both low train workers (not merely one infer "slot") to
    // relinquish execution, then the low train class becomes resumable once
    // the normal work drains.
    demand = {{{true,true,2,2},{true,true,0,1},{true,false,0,std::nullopt}}};
    plan = PlanSchedulerPhases(policy, demand);
    assert(plan.selected == SchedulerPhase::Infer);
    assert(plan.selectedPriorityRank == 1);
    assert(plan.draining);
    assert((plan.allowed == std::array<bool,3>{false,false,false}));
    demand[0].activeWorkers = 0;
    plan = PlanSchedulerPhases(policy, demand);
    assert((plan.allowed == std::array<bool,3>{false,true,false}));
    demand[1].eligiblePending = false;
    plan = PlanSchedulerPhases(policy, demand);
    assert(plan.selected == SchedulerPhase::Train);
    assert((plan.allowed == std::array<bool,3>{true,false,false}));

    // Phase precedence is exclusive within the winning priority tier.
    demand = {{{true,true,0,1},{true,true,1,1},{true,false,0,std::nullopt}}};
    plan = PlanSchedulerPhases(policy, demand);
    assert(plan.selected == SchedulerPhase::Train);
    assert(plan.selectedPriorityRank == 1);
    assert(plan.draining);
    demand[1].activeWorkers = 0;
    assert((PlanSchedulerPhases(policy, demand).allowed ==
            std::array<bool,3>{true,false,false}));

    // Ineligible high-ranked rows do not contribute a priority rank and
    // therefore cannot starve a runnable lower-ranked class.
    demand = {{{true,true,0,2},{true,false,0,0},{true,false,0,std::nullopt}}};
    plan = PlanSchedulerPhases(policy, demand);
    assert(plan.selected == SchedulerPhase::Train);
    assert(plan.selectedPriorityRank == 2);
    std::array<SchedulerPhase,3> phases{SchedulerPhase::Train,SchedulerPhase::Infer,SchedulerPhase::Analyze};
    int permutations = 0;
    do {
        SchedulerPhasePriority p{false, phases};
        assert(SchedulerPhasePriority::Parse(p.canonical()).order == phases);
        ++permutations;
    } while (std::next_permutation(phases.begin(), phases.end()));
    assert(permutations == 6);
    for (const auto invalid : {"", "train", "train:infer", "train:infer:analyze:train",
            "train:train:analyze", "train:infer:unknown", "train:infer:analyze:", "TRAIN:infer:analyze"})
    {
        bool rejected = false;
        try { (void)SchedulerPhasePriority::Parse(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    MemoryPolicy repository;
    SchedulerPhasePriorityService service{repository};
    assert(service.load().concurrent);
    service.set(policy);
    assert(service.load().canonical() == "train:infer:analyze");
    service.set(SchedulerPhasePriority::Parse("infer:train:analyze"));
    assert(service.load().order[0] == SchedulerPhase::Infer);
    service.set({});
    assert(service.load().concurrent);
    // Existing high/normal/low preemption policy remains independent.
    assert(CanPreempt("high", "low"));
    assert(!CanPreempt("low", "high"));
}
