#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <cstddef>

namespace EA::MetalForwardAffine
{
using Memory = MetaNN::ContinuousMemory<float, MetaNN::DeviceTags::Metal>;

// Optional per-call evidence; no global counters or timing in the training path.
struct SynchronizationStats
{
    size_t submissions = 0;
    size_t blockingWaits = 0;
    size_t successfulCompletions = 0;
};

// Process-level opt-in timing for the ExpertAdvisor-owned combined command
// buffer. GPU timestamps use Metal's GPU clock and are never mixed with CPU
// steady-clock values. Disabled unless EA_LSTM_COMMAND_BUFFER_TIMING is set.
struct CommandBufferTimingTotals
{
    size_t commandBuffers = 0;
    size_t validGpuTimestamps = 0;
    size_t invalidGpuTimestamps = 0;
    double cpuCreateUs = 0.0;
    double cpuEncodeUs = 0.0;
    double cpuCommitUs = 0.0;
    double cpuWaitUs = 0.0;
    double cpuSubmitToCompletionUs = 0.0;
    double gpuExecutionUs = 0.0;
    double gpuKernelUs = 0.0;
};

void ResetCommandBufferTiming();
CommandBufferTimingTotals GetCommandBufferTiming();

// Read once per process. Unset/"metann" uses the original implementation;
// "combined" opts into the adapter. Invalid values fail before GPU submission.
const char* SelectedPathName();

// Both paths return only after GPU completion. diagnosticLogging prints the
// selected path once, without reading matrix data.
void ForwardMatMulBias(const Memory& a, const Memory& b, const Memory& bias,
                       Memory& c, size_t m, size_t k, size_t n,
                       bool diagnosticLogging);

// Contiguous float32 row-major matrices, with element offsets carried by Memory.
// Inputs and output must not overlap. Owners are retained until completion.
void CombinedMatMulBias(const Memory& a, const Memory& b, const Memory& bias,
                        Memory& c, size_t m, size_t k, size_t n,
                        SynchronizationStats* stats = nullptr);
}
