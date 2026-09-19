#pragma once

namespace EA::SchedulerCore
{

// LSTM_Release compatibility adapter.  It is intentionally separate from the
// typed engine so a future scheduler entry point need not accept raw argv.
int RunSchedulerDaemonCli(int argc, const char* argv[]);

// The standalone executable has a single scheduler-daemon mode, unlike the
// multi-command LSTM_Release compatibility executable. This adapter supplies
// that mode selector and delegates all parsing and validation above.
int RunStandaloneSchedulerDaemonCli(int argc, const char* argv[]);

} // namespace EA::SchedulerCore
