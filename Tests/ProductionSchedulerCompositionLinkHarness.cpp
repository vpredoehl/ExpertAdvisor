#include "SchedulerCore/ProductionSchedulerDaemon.hpp"

int main()
{
    volatile auto productionRunner =
        &EA::SchedulerCore::RunProductionSchedulerDaemon;
    return productionRunner == nullptr;
}
