#include "SchedulerCore/SchedulerObserverCli.hpp"

#include <iostream>

int main(int argc, const char* argv[])
{
    return EA::SchedulerCore::RunSchedulerObserverCli(
        argc, argv, std::cout, std::cerr);
}
