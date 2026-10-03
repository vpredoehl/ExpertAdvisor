#pragma once

#include "SchedulerPhasePriorityService.hpp"
#include <pqxx/pqxx>

namespace EA::SchedulerCore
{
class PostgresSchedulerPhasePriorityRepository final : public SchedulerPhasePriorityRepository
{
public:
    explicit PostgresSchedulerPhasePriorityRepository(pqxx::transaction_base& transaction)
        : transaction_{transaction} {}

    std::string load() override
    {
        // Legacy concurrent scheduling remains available before migration.
        if (!exists()) return "concurrent";
        return transaction_.exec(
            "SELECT phase_priority FROM public.experiment_scheduler_phase_policy "
            "WHERE singleton=true").one_row()[0].as<std::string>();
    }

    void save(const std::string& canonical) override
    {
        if (!exists()) throw std::runtime_error(
            "phase priority requires migration 099_scheduler_phase_priority.sql");
        transaction_.exec_params(
            "UPDATE public.experiment_scheduler_phase_policy "
            "SET phase_priority=$1, revision=revision+1, updated_at=clock_timestamp() "
            "WHERE singleton=true AND phase_priority<>$1", canonical);
    }

private:
    bool exists()
    {
        return !transaction_.exec(
            "SELECT to_regclass('public.experiment_scheduler_phase_policy')").one_row()[0].is_null();
    }
    pqxx::transaction_base& transaction_;
};

} // namespace EA::SchedulerCore
