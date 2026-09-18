#pragma once

namespace EA::SchedulerCore
{

// LSTM_Release compatibility adapter.  It is intentionally separate from the
// typed engine so a future scheduler entry point need not accept raw argv.
int RunSchedulerDaemonCli(int argc, const char* argv[]);

} // namespace EA::SchedulerCore
