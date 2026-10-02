#pragma once

#include <iosfwd>

namespace EA::SchedulerCore
{

int RunSchedulerObserverCli(int argc, const char* argv[], std::ostream& output,
                            std::ostream& error);

} // namespace EA::SchedulerCore
