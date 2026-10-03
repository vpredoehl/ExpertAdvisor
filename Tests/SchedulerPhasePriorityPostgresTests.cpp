#include "SchedulerCore/SchedulerPhasePriorityRepository.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace EA::SchedulerCore;
int main(int argc, const char* argv[])
{
    assert(argc == 3);
    pqxx::connection reader{argv[1]};
    if (std::string{argv[2]} == "unmigrated")
    {
        pqxx::work transaction{reader};
        PostgresSchedulerPhasePriorityRepository repository{transaction};
        SchedulerPhasePriorityService service{repository};
        assert(service.load().concurrent);
        bool rejected = false;
        try { service.set(SchedulerPhasePriority::Parse("train:infer:analyze")); }
        catch (const std::runtime_error& error) { rejected = std::string{error.what()}.find("migration 099") != std::string::npos; }
        assert(rejected);
        return 0;
    }
    pqxx::connection writer{argv[1]};
    const auto load = [&] {
        pqxx::work transaction{reader};
        PostgresSchedulerPhasePriorityRepository repository{transaction};
        return SchedulerPhasePriorityService{repository}.load();
    };
    assert(load().concurrent);
    {
        pqxx::work transaction{writer};
        PostgresSchedulerPhasePriorityRepository repository{transaction};
        SchedulerPhasePriorityService service{repository};
        service.set(SchedulerPhasePriority::Parse("train:infer:analyze"));
        transaction.commit();
    }
    assert(load().canonical() == "train:infer:analyze");
    {
        pqxx::work transaction{writer};
        PostgresSchedulerPhasePriorityRepository repository{transaction};
        SchedulerPhasePriorityService service{repository};
        service.set(SchedulerPhasePriority::Parse("infer:analyze:train"));
        transaction.commit();
    }
    std::array<SchedulerPhaseDemand,3> demand{{{true,true,1},{true,true,0},{true,true,0}}};
    const auto livePlan = PlanSchedulerPhases(load(), demand);
    assert(livePlan.selected == SchedulerPhase::Infer && livePlan.draining);
    {
        pqxx::work transaction{reader};
        assert(transaction.exec("SELECT revision FROM experiment_scheduler_phase_policy").one_row()[0].as<int>() == 2);
    }
    {
        pqxx::work transaction{writer};
        PostgresSchedulerPhasePriorityRepository repository{transaction};
        SchedulerPhasePriorityService service{repository};
        service.set(SchedulerPhasePriority::Parse("infer:analyze:train"));
        transaction.commit();
    }
    {
        pqxx::work transaction{reader};
        assert(transaction.exec("SELECT revision FROM experiment_scheduler_phase_policy").one_row()[0].as<int>() == 2);
    }
    {
        pqxx::work transaction{writer};
        PostgresSchedulerPhasePriorityRepository repository{transaction};
        SchedulerPhasePriorityService{repository}.set({});
        // Abort must leave the durable order intact.
    }
    assert(load().canonical() == "infer:analyze:train");
    for (const auto sql : {
            "UPDATE experiment_scheduler_phase_policy SET phase_priority='train:train:infer'",
            "INSERT INTO experiment_scheduler_phase_policy(singleton) VALUES(false)",
            "INSERT INTO experiment_scheduler_phase_policy(singleton) VALUES(true)"})
    {
        bool rejected = false;
        try { pqxx::work transaction{writer}; transaction.exec(sql); transaction.commit(); }
        catch (const pqxx::sql_error&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "SchedulerPhasePriorityPostgresTests passed\n";
}
