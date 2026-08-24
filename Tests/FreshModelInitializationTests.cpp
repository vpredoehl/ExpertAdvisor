#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "LSTM.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"
#include "TrainingObjective.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{

template <typename Value>
void WriteValue(std::ofstream& output, const Value& value)
{
    static_assert(std::is_trivially_copyable_v<Value>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output)
        throw std::runtime_error("initialization_snapshot_write_failed");
}

template <typename Matrix>
void WriteMatrix(std::ofstream& output, const Matrix& matrix)
{
    const std::uint64_t rows = matrix.Shape()[0];
    const std::uint64_t columns = matrix.Shape()[1];
    WriteValue(output, rows);
    WriteValue(output, columns);

    auto access = MetaNN::LowerAccess(matrix);
    const float* values = access.RawMemory();
    const std::size_t bytes =
        static_cast<std::size_t>(rows * columns) * sizeof(float);
    output.write(reinterpret_cast<const char*>(values),
                 static_cast<std::streamsize>(bytes));
    if (!output)
        throw std::runtime_error("initialization_snapshot_write_failed");
}

template <typename Matrix>
void RequireAllEqual(const Matrix& matrix, float expected,
                     const char* parameterName)
{
    auto access = MetaNN::LowerAccess(matrix);
    const float* values = access.RawMemory();
    const std::size_t count = matrix.Shape()[0] * matrix.Shape()[1];
    for (std::size_t index = 0; index < count; ++index)
    {
        if (values[index] != expected)
            throw std::runtime_error(
                std::string{"unexpected_initial_value:"} + parameterName);
    }
}

void WriteSharedInitialization(const EA::LSTM& model,
                               const std::string& path)
{
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    if (!output)
        throw std::runtime_error("initialization_snapshot_open_failed");

    // These are every trainable family shared by both Phase 4C arms.
    WriteMatrix(output, model.param);
    WriteMatrix(output, model.bias);
    WriteMatrix(output, model.returnHeadDirWeight);
    WriteMatrix(output, model.returnHeadDirBias);

    // These are not trainable parameters, but freezing them in the snapshot
    // proves both fresh workers also start with identical recurrent/optimizer
    // state before the first CalculateBatch update.
    WriteMatrix(output, model.prevHiddenState);
    WriteMatrix(output, model.prevCellState);
    WriteValue(output, model.optimizerUpdateCount);
    WriteValue(output, model.completedEpochs);
}

void WriteAuxiliaryInitialization(const EA::LSTM& model,
                                  const std::string& path)
{
    RequireAllEqual(model.returnHeadWeight, 0.01f,
                    "returnHeadWeight");
    RequireAllEqual(model.returnHeadBias, 0.0f,
                    "returnHeadBias");

    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    if (!output)
        throw std::runtime_error("auxiliary_snapshot_open_failed");
    WriteMatrix(output, model.returnHeadWeight);
    WriteMatrix(output, model.returnHeadBias);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: FreshModelInitializationTests "
                     "legacy|auxiliary output-prefix\n";
        return 2;
    }

    try
    {
        const std::string selection = argv[1];
        const std::string outputPrefix = argv[2];
        const EA::TrainingObjective::Configuration objective =
            selection == "legacy"
                ? EA::TrainingObjective::Legacy()
                : (selection == "auxiliary"
                       ? EA::TrainingObjective::ProfitabilityAuxiliary()
                       : throw std::invalid_argument(
                             "unknown_test_objective"));

        // Match the scheduler fresh-training architecture defaults. One row
        // is enough to establish the current physical Tensor feature width;
        // no forward pass or optimizer update is performed.
        hidden_size = default_hidden_size;
        n_out = default_hidden_size;
        Tensor tensor{"usdcadrmp"};
        Feature bar{1.25f, 1.2501f, 1.2502f, 1.2499f,
                    PriceTP{std::chrono::seconds{900}}};
        bar.tickVolume = 100.0f;
        tensor.Add(bar);

        // This is the production fresh-training order in main.cpp:
        // construct EA::LSTM, then apply the resolved objective.
        EA::LSTM model{tensor, 1.0f, 0.0f,
                       EA::LSTM::TargetType::UpNeutralDownReturn};
        model.SetTrainingObjective(objective);

        WriteSharedInitialization(model, outputPrefix + ".shared.bin");
        if (EA::TrainingObjective::AuxiliaryEnabled(objective))
            WriteAuxiliaryInitialization(model,
                                         outputPrefix + ".auxiliary.bin");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
