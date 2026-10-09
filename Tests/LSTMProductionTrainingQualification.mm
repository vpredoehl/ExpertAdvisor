// Qualification only: canonical read-only preparation followed by database-free
// CalculateBatch calls. No production worker or persistence workflow is launched.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "LSTM.hpp"
#include "PgModelIO.hpp"
#include "ModelInputPreparation.hpp"
#include "MetalForwardAffine.hpp"
#include "LSTMNumericalEvidence.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mach/mach.h>
#include <streambuf>
#include <sys/resource.h>

namespace {
std::ofstream* observed = nullptr;
void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> void Write(std::ofstream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    Require(bool(out), "evidence_write_failed");
}
void Matrix(std::ofstream& out, const EA::LSTM::EAMatrix& matrix) {
    Write(out, uint64_t(matrix.Shape()[0])); Write(out, uint64_t(matrix.Shape()[1]));
    const auto access = MetaNN::LowerAccess(matrix);
    for (size_t i = 0; i < matrix.Shape()[0] * matrix.Shape()[1]; ++i) {
        Require(std::isfinite(access.RawMemory()[i]), "nonfinite_matrix");
        Write(out, access.RawMemory()[i]);
    }
}
void State(std::ofstream& out, const EA::LSTM& model) {
    for (const auto* matrix : {&model.param, &model.bias, &model.returnHeadWeight,
         &model.returnHeadBias, &model.returnHeadDirWeight, &model.returnHeadDirBias,
         &model.prevHiddenState, &model.prevCellState}) Matrix(out, *matrix);
    Write(out, model.learning_rate); Write(out, model.targetScale);
    Write(out, model.targetBias); Write(out, model.targetUseZScore);
    Write(out, model.targetMean); Write(out, model.targetStd);
    Write(out, uint64_t(model.optimizerUpdateCount));
    Write(out, uint64_t(model.completedEpochs));
}
struct Silence {
    struct Null : std::streambuf { int overflow(int c) override { return c; } } buffer;
    std::streambuf* previous = std::cout.rdbuf(&buffer);
    ~Silence() { std::cout.rdbuf(previous); }
};
uint64_t ResidentBytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    Require(task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS, "task_info_failed");
    return info.resident_size;
}
double CpuSeconds() {
    rusage usage{}; Require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage_failed");
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}
}
extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }
namespace EA { bool RuntimeDiagnosticLoggingEnabled() noexcept { return false; } }
void EA::Testing::RecordLSTMNumericalMatrices(const char* stage,
    std::initializer_list<const EA::LSTM::EAMatrix*> matrices) {
    Require(observed != nullptr, "observer_stream_missing");
    Write(*observed, uint64_t(std::strlen(stage)));
    observed->write(stage, std::strlen(stage));
    Write(*observed, uint64_t(matrices.size()));
    for (const auto* matrix : matrices) Matrix(*observed, *matrix);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Require(argc == 4, "usage: fixture warmup measured output-prefix");
            const size_t warmup = std::stoul(argv[1]), measured = std::stoul(argv[2]);
            const std::string prefix = argv[3];
            Require(measured > 0 && warmup + measured <= 1024, "invalid_update_count");
            // Explicit read-only sessions; connection strings contain no credentials.
            const std::string options = " options='-c default_transaction_read_only=on -c statement_timeout=120000' application_name=phase25b3_read_only";
            const std::string lstmConnection = "dbname=LSTM" + options;
            DBIO::PgModelIO::PersistedModelMaterialization persisted;
            unsigned int seed = 0;
            std::string from, to;
            {
                pqxx::connection connection{lstmConnection};
                pqxx::work read{connection}; read.exec("SET TRANSACTION READ ONLY;");
                Require(read.exec("SHOW transaction_read_only").one_row()[0].as<std::string>() == "on", "read_only_required");
                const auto row = read.exec("SELECT status,symbol,fresh_initialization_seed,model_input_width,model_input_semantic_layout_version,train_start::date::text,train_end::date::text,last_model_id FROM experiment WHERE experiment_id=746").one_row();
                Require(row[0].as<std::string>() == "completed" && row[1].as<std::string>() == "cadchfrmp" && row[3].as<int>() == 171 && row[4].as<int>() == 13 && row[7].as<long long>() == 2090, "baseline_experiment_changed");
                seed = row[2].as<unsigned int>(); from = row[5].as<std::string>(); to = row[6].as<std::string>();
                persisted = DBIO::PgModelIO::ReadPersistedModelMaterialization(read, 2090);
                read.commit();
            }
            Require(persisted.trainConfigMeta.has_value() && persisted.identity.economicCalendarSnapshotId.has_value() && persisted.identity.economicCalendarSnapshotHash.has_value(), "persisted_contract_incomplete");
            const auto& config = persisted.trainConfigMeta->values;
            Require(config.size() == 14 && config[1] == 4 && config[3] == 64 && persisted.modelMeta.hiddenSize == 64 && persisted.modelMeta.inputWidth == 171, "persisted_dimensions_changed");
            prediction_horizon = size_t(config[1]); c_next_threshold = float(config[2]);
            window_size = size_t(config[3]); hidden_size = persisted.modelMeta.hiddenSize;
            n_out = hidden_size; num_layers = size_t(config[8]); normalization_version = int(config[9]);
            epoch_count = int(config[10]); core_lr_mult = float(config[11]);
            head_weight_lr_mult = float(config[12]); head_bias_lr_mult = float(config[13]);
            std::ofstream state(prefix + ".state", std::ios::binary);
            std::ofstream inputs(prefix + ".inputs", std::ios::binary);
            std::ofstream tensors;
#ifdef LSTM_NUMERICAL_TEST_OBSERVERS
            tensors.open(prefix + ".tensors", std::ios::binary); observed = &tensors;
#endif
            std::vector<double> latency, cpu;
            std::vector<uint64_t> rss, metal;
            std::vector<float> losses;
            std::vector<size_t> accepted;
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            Require(device != nil, "metal_unavailable");
            const auto preparationStart = std::chrono::steady_clock::now();
            auto prepared = EA::ModelInputPreparation::Prepare(
                {"cadchfrmp", from, to, persisted.featureWarmupScope,
                 persisted.donchian20Mode, persisted.donchianLookback,
                 EA::EconomicCalendar::EconomicCalendarSnapshotIdentity{
                     *persisted.identity.economicCalendarSnapshotId,
                     *persisted.identity.economicCalendarSnapshotHash}},
                {"dbname=forex" + options, lstmConnection});
            const auto preparationStop = std::chrono::steady_clock::now();
            Tensor& tensor = prepared.tensor;
            Require(prepared.logicalOutputStartIndex + (warmup + measured) * batch_size <= tensor.RowCount(), "insufficient_full_batches");
            // Freeze the actual canonical physical rows and raw target inputs.
            // Appended return features are computed by unchanged CalculateBatch.
            Write(inputs, uint64_t(tensor.RowCount()));
            Write(inputs, uint64_t(prepared.logicalOutputStartIndex));
            for (auto it = tensor.begin(); it != tensor.end(); ++it) {
                Matrix(inputs, *it);
                Write(inputs, tensor.RawOpenAtIterator(it)); Write(inputs, tensor.RawCloseAtIterator(it));
                Write(inputs, tensor.RawHighAtIterator(it)); Write(inputs, tensor.RawLowAtIterator(it));
                Write(inputs, int64_t(tensor.RawTimeAtIterator(it).time_since_epoch().count()));
            }
            inputs.close();
            std::cout << "DATASET,rows=" << tensor.RowCount() << ",warmup_rows=" << prepared.logicalOutputStartIndex
                << ",first_seconds=" << tensor.RawTimeAtIterator(tensor.begin()).time_since_epoch().count()
                << ",last_seconds=" << tensor.RawTimeAtIterator(tensor.end()-1).time_since_epoch().count()
                << ",preparation_seconds=" << std::chrono::duration<double>(preparationStop-preparationStart).count() << std::endl;
            uint64_t allocatedBefore = 0, allocatedAfter = 0;
            double trainingSeconds = 0;
            {
                Silence silence;
                EA::LSTM model{tensor, hidden_size, 1, 0, EA::LSTM::TargetType::UpNeutralDownReturn,
                    size_t{171}, persisted.identity.featureAblationMask, seed};
                model.SetTrainingObjective(persisted.trainingObjective);
                State(state, model);
                rss.push_back(ResidentBytes()); metal.push_back(device.currentAllocatedSize);
                for (size_t call = 0; call < warmup + measured; ++call) {
                    const size_t first = prepared.logicalOutputStartIndex + call * batch_size;
                    const Window batch{tensor.begin() + std::ptrdiff_t(first), tensor.begin() + std::ptrdiff_t(first + batch_size)};
                    if (call == warmup) allocatedBefore = device.currentAllocatedSize;
                    const double cpuStart = CpuSeconds();
                    const auto start = std::chrono::steady_clock::now();
                    const auto result = model.CalculateBatch(batch, 0);
                    const auto stop = std::chrono::steady_clock::now();
                    const double cpuEnd = CpuSeconds();
                    const double seconds = std::chrono::duration<double>(stop-start).count();
                    Require(std::isfinite(std::get<0>(result)) && std::get<1>(result) + std::get<2>(result) == 189 && model.optimizerUpdateCount == call + 1, "training_assertion_failed");
                    trainingSeconds += seconds;
                    Write(state, std::get<0>(result)); Write(state, uint64_t(std::get<1>(result)));
                    Write(state, uint64_t(std::get<2>(result))); State(state, model);
                    latency.push_back(seconds * 1000); cpu.push_back(cpuEnd-cpuStart);
                    rss.push_back(ResidentBytes()); metal.push_back(device.currentAllocatedSize);
                    losses.push_back(std::get<0>(result)); accepted.push_back(std::get<1>(result));
                }
                allocatedAfter = device.currentAllocatedSize;
            }
            observed = nullptr;
            rusage usage{}; Require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage_failed");
            std::cout << std::setprecision(15);
            for (size_t i = 0; i < latency.size(); ++i)
                std::cout << "UPDATE,index=" << i << ",measured=" << (i >= warmup) << ",ms=" << latency[i]
                    << ",cpu_seconds=" << cpu[i] << ",loss=" << losses[i] << ",windows=" << accepted[i]
                    << ",rss_bytes=" << rss[i+1] << ",metal_bytes=" << metal[i+1] << '\n';
            std::cout << "SUMMARY,path=" << EA::MetalForwardAffine::SelectedPathName()
                << ",warmup=" << warmup << ",measured=" << measured << ",updates=" << latency.size()
                << ",training_seconds=" << trainingSeconds << ",peak_rss_bytes=" << usage.ru_maxrss
                << ",initial_rss_bytes=" << rss.front() << ",initial_metal_bytes=" << metal.front()
                << ",metal_before_bytes=" << allocatedBefore << ",metal_after_bytes=" << allocatedAfter
                << ",metal_after_model_release_bytes=" << device.currentAllocatedSize << '\n';
        } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    }
}
