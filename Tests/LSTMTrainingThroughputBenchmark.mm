// Database-free Phase 25B-2 fixture. Production training code is linked unchanged.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "LSTM.hpp"
#include "Tensor.hpp"
#include "PricePoint.hpp"
#include "MetalForwardAffine.hpp"
#include "LSTMNumericalEvidence.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <sys/resource.h>
#include <type_traits>
#include <vector>

namespace {
std::ofstream* tensors = nullptr;
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> void Write(std::ofstream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    Require(bool(out), "evidence write failed");
}
void WriteMatrix(std::ofstream& out, const EA::LSTM::EAMatrix& m) {
    Write(out, uint64_t(m.Shape()[0])); Write(out, uint64_t(m.Shape()[1]));
    const auto low = MetaNN::LowerAccess(m);
    for (size_t i = 0; i < m.Shape()[0] * m.Shape()[1]; ++i) {
        Require(std::isfinite(low.RawMemory()[i]), "nonfinite tensor");
        Write(out, low.RawMemory()[i]);
    }
}
void WriteState(std::ofstream& out, const EA::LSTM& m) {
    for (const auto* matrix : {&m.param, &m.bias, &m.returnHeadWeight,
         &m.returnHeadBias, &m.returnHeadDirWeight, &m.returnHeadDirBias,
         &m.prevHiddenState, &m.prevCellState}) WriteMatrix(out, *matrix);
    Write(out, m.learning_rate); Write(out, m.targetScale);
    Write(out, uint64_t(m.optimizerUpdateCount)); Write(out, uint64_t(m.completedEpochs));
}
void Zero(EA::LSTM::EAMatrix& m) {
    auto low = MetaNN::LowerAccess(m);
    std::fill(low.MutableRawMemory(), low.MutableRawMemory() + m.Shape()[0]*m.Shape()[1], 0.0f);
}
// Same cout silencing behavior as the application's ScopedDiagnosticCoutSilencer.
// Runtime-gated diagnostics are disabled by the bridge below. Remaining legacy
// diagnostic computation is retained, exactly as in the normal training path.
struct Silence {
    struct NullBuffer : std::streambuf { int overflow(int c) override { return c; } } buffer;
    std::streambuf* previous = std::cout.rdbuf(&buffer);
    ~Silence() { std::cout.rdbuf(previous); }
};
}
extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }
void EA::Testing::RecordLSTMNumericalMatrices(const char* stage,
    std::initializer_list<const EA::LSTM::EAMatrix*> matrices) {
    Require(tensors != nullptr, "missing observer evidence stream");
    Write(*tensors, uint64_t(std::strlen(stage)));
    tensors->write(stage, std::strlen(stage));
    Write(*tensors, uint64_t(matrices.size()));
    for (const auto* m : matrices) WriteMatrix(*tensors, *m);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Require(argc == 5 || argc == 6, "usage: benchmark legacy|auxiliary|log|percent warmup measured output-prefix [windows]");
            const std::string mode = argv[1], prefix = argv[4];
            Require(mode == "legacy" || mode == "auxiliary" || mode == "log" || mode == "percent", "unknown target");
            const size_t warmup = std::stoul(argv[2]), measured = std::stoul(argv[3]);
            Require(measured > 0 && measured + warmup <= 256, "invalid update count");
            hidden_size = 64; n_out = 64; window_size = 64; prediction_horizon = 2;
            const size_t windows = argc == 6 ? std::stoul(argv[5]) : 128;
            Require(windows > 0 && windows <= 128, "invalid window count");
            const size_t batchRows = window_size + prediction_horizon - 1 + windows;
            const auto type = mode == "log" ? EA::LSTM::TargetType::LogReturn :
                mode == "percent" ? EA::LSTM::TargetType::PercentReturn : EA::LSTM::TargetType::UpNeutralDownReturn;
            const auto objective = mode == "auxiliary" ? EA::TrainingObjective::ProfitabilityAuxiliary() : EA::TrainingObjective::Legacy();
            std::ofstream state(prefix + ".state", std::ios::binary);
            std::ofstream tensorFile;
#ifdef LSTM_NUMERICAL_TEST_OBSERVERS
            tensorFile.open(prefix + ".tensors", std::ios::binary);
            Require(bool(tensorFile), "cannot open tensor evidence");
            tensors = &tensorFile;
#endif
            Require(bool(state), "cannot open state evidence");
            std::vector<double> latencies, losses;
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            Require(device != nil, "Metal device unavailable");
            NSUInteger allocatedBefore = 0, allocatedAfter = 0, allocatedMax = 0;
            size_t updates = 0;
            float lr = 0;
            {
                Silence quiet;
                Tensor tensor{"usdcadrmp"};
                for (size_t row = 0; row < 520; ++row) {
                    const float close = 1.25f + 0.003f * std::sin(float(row) * 0.7f);
                    Feature bar{close - 0.0001f, close, close + 0.0003f, close - 0.0003f,
                        PriceTP{std::chrono::seconds{900 * long(row)}}};
                    bar.tickVolume = 100.0f + float(row % 7);
                    tensor.Add(bar);
                }
                EA::LSTM model{tensor, 64, 1.0f, 0.0f, type, std::nullopt, {}, 42U};
                if (type == EA::LSTM::TargetType::UpNeutralDownReturn) {
                    Zero(model.returnHeadWeight); Zero(model.returnHeadBias);
                } else { Zero(model.returnHeadDirWeight); Zero(model.returnHeadDirBias); }
                model.SetTrainingObjective(objective);
                if (type != EA::LSTM::TargetType::UpNeutralDownReturn) model.targetScale = 100000.0f;
                Require(model.InputFeatureCount() == 171 && model.param.Shape()[0] == 235 &&
                        model.param.Shape()[1] == 256, "fixture model dimensions changed");
                const Window batch{tensor.end() - std::ptrdiff_t(batchRows), tensor.end()};
                WriteState(state, model);
                for (size_t call = 0; call < warmup + measured; ++call) {
                    if (call == warmup) allocatedBefore = device.currentAllocatedSize;
                    const auto start = std::chrono::steady_clock::now();
                    const auto result = model.CalculateBatch(batch, 2);
                    const auto stop = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
                    Require(std::isfinite(std::get<0>(result)) && std::get<1>(result) == windows &&
                            std::get<2>(result) == 0 && model.optimizerUpdateCount == call + 1,
                            "training failed loss/window/update assertions");
                    Write(state, std::get<0>(result)); Write(state, uint64_t(std::get<1>(result)));
                    Write(state, uint64_t(std::get<2>(result))); WriteState(state, model);
                    allocatedAfter = device.currentAllocatedSize;
                    allocatedMax = std::max(allocatedMax, allocatedAfter);
                    if (call >= warmup) { latencies.push_back(ms); losses.push_back(std::get<0>(result)); }
                }
                updates = model.optimizerUpdateCount; lr = model.learning_rate;
            }
            tensors = nullptr;
            struct rusage usage{};
            Require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
            std::cout << std::setprecision(15);
            double total = 0;
            for (size_t i = 0; i < latencies.size(); ++i) {
                total += latencies[i];
                std::cout << "UPDATE,index=" << i << ",ms=" << latencies[i] << ",loss=" << losses[i] << '\n';
            }
            std::cout << "SUMMARY,path=" << EA::MetalForwardAffine::SelectedPathName()
                << ",mode=" << mode << ",warmup=" << warmup << ",measured=" << measured
                << ",updates=" << updates << ",windows=" << windows << ",input=171,hidden=64,sequence=64"
                << ",learning_rate=" << lr << ",mean_ms=" << total/measured
                << ",updates_per_second=" << measured*1000.0/total
                << ",peak_rss_bytes=" << usage.ru_maxrss
                << ",metal_before_bytes=" << allocatedBefore << ",metal_after_bytes=" << allocatedAfter
                << ",metal_max_boundary_bytes=" << allocatedMax
                << ",metal_after_model_release_bytes=" << device.currentAllocatedSize << '\n';
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
