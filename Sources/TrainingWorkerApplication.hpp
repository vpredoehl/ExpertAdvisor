#pragma once

namespace EA::Training
{

// Dedicated scheduler-managed TRAIN application. Legacy LSTM_Release keeps
// its independent compatibility implementation in LSTM/main.cpp.
int RunTrainingWorkerApplication(int argc, const char* argv[]);
int RunDedicatedTrainWorkerMain(int argc, const char* argv[]);

} // namespace EA::Training
