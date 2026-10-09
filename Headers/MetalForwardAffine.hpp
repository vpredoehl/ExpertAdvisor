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
