// Database-free, deterministic baseline/current training comparison fixture.
// The binary output is test evidence, not a production checkpoint format.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "LSTM.hpp"
#include "MatrixUtils.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

namespace
{
bool diagnostics = false;

void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Value>
void Write(std::ofstream& stream, const Value& value)
{
    static_assert(std::is_trivially_copyable_v<Value>);
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    Require(static_cast<bool>(stream), "numerical evidence write failed");
}

void WriteMatrix(std::ofstream& stream, const EA::LSTM::EAMatrix& matrix)
{
    Write(stream, static_cast<uint64_t>(matrix.Shape()[0]));
    Write(stream, static_cast<uint64_t>(matrix.Shape()[1]));
    const auto low = MetaNN::LowerAccess(matrix);
    const auto* data = low.RawMemory();
    for (size_t index = 0; index < matrix.Shape()[0] * matrix.Shape()[1]; ++index)
    {
        Require(std::isfinite(data[index]), "nonfinite model state");
        Write(stream, data[index]);
    }
}

void WriteState(std::ofstream& stream, const EA::LSTM& model)
{
    for (const auto* matrix : {&model.param, &model.bias, &model.returnHeadWeight,
                               &model.returnHeadBias, &model.returnHeadDirWeight,
                               &model.returnHeadDirBias, &model.prevHiddenState,
                               &model.prevCellState})
        WriteMatrix(stream, *matrix);
    Write(stream, model.learning_rate);
    Write(stream, model.targetScale);
    Write(stream, static_cast<uint64_t>(model.optimizerUpdateCount));
    Write(stream, static_cast<uint64_t>(model.completedEpochs));
}

void Zero(EA::LSTM::EAMatrix& matrix)
{
    auto low = MetaNN::LowerAccess(matrix);
    std::fill(low.MutableRawMemory(), low.MutableRawMemory() +
              matrix.Shape()[0] * matrix.Shape()[1], 0.0f);
}

void TestMetalForgetSlice()
{
    EA::LSTM::EAMatrix source(3, 32);
    for (size_t row = 0; row < 3; ++row)
        for (size_t column = 0; column < 32; ++column)
            source.SetValue(row, column, static_cast<float>(32 * row + column) - 35.5f);
    auto logits = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(source, 8, 8);
    auto oldCache = NNUtils::DeepCopyMatrix(logits);
    const auto* storage = MetaNN::LowerAccess(logits).RawMemory();
    auto cache = std::move(logits);
    Require(MetaNN::LowerAccess(cache).RawMemory() == storage &&
            storage != MetaNN::LowerAccess(source).RawMemory(), "Metal slice ownership");
    for (size_t row = 0; row < 3; ++row)
        for (size_t column = 0; column < 8; ++column)
            Require(cache(row, column) == oldCache(row, column) &&
                    cache(row, column) == source(row, 8 + column), "Metal forget logits");
    source = EA::LSTM::EAMatrix(1, 1); // release the original gate storage
    Require(cache(2, 7) == oldCache(2, 7), "Metal slice lifetime");
}

void RestoreMatrices(const EA::LSTM& saved, EA::LSTM& restored)
{
    // Exercise reconstruction from the matrix families persisted by saveAll;
    // database serialization and scheduler resume are intentionally not invoked.
    EA::TrainingObjective::RequireResumeCompatible(saved.trainingObjective,
                                                   restored.trainingObjective);
    restored.param = NNUtils::DeepCopyMatrix(saved.param);
    restored.bias = NNUtils::DeepCopyMatrix(saved.bias);
    restored.returnHeadWeight = NNUtils::DeepCopyMatrix(saved.returnHeadWeight);
    restored.returnHeadBias = NNUtils::DeepCopyMatrix(saved.returnHeadBias);
    restored.returnHeadDirWeight = NNUtils::DeepCopyMatrix(saved.returnHeadDirWeight);
    restored.returnHeadDirBias = NNUtils::DeepCopyMatrix(saved.returnHeadDirBias);
    restored.learning_rate = saved.learning_rate;
    restored.targetScale = saved.targetScale;
    restored.optimizerUpdateCount = saved.optimizerUpdateCount;
    restored.completedEpochs = saved.completedEpochs;
}
} // namespace

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return diagnostics; }

int main(int argc, char** argv)
{
    try
    {
        Require(argc == 4, "usage: fixture legacy|auxiliary|log|percent on|off output");
        diagnostics = std::string{argv[2]} == "on";
        const std::string mode = argv[1];
        const auto type = mode == "log" ? EA::LSTM::TargetType::LogReturn :
            mode == "percent" ? EA::LSTM::TargetType::PercentReturn :
            EA::LSTM::TargetType::UpNeutralDownReturn;
        const auto objective = mode == "auxiliary" ?
            EA::TrainingObjective::ProfitabilityAuxiliary() : EA::TrainingObjective::Legacy();
        // Fixture architecture and window lengths only; production defaults,
        // input width and semantic layout are unchanged.
        hidden_size = 8;
        n_out = hidden_size;
        window_size = 4;
        prediction_horizon = 2;
        TestMetalForgetSlice();
        Tensor tensor{"usdcadrmp"};
        for (size_t row = 0; row < 520; ++row)
        {
            const float close = 1.25f + 0.003f * std::sin(static_cast<float>(row) * 0.7f);
            Feature bar{close - 0.0001f, close, close + 0.0003f, close - 0.0003f,
                        PriceTP{std::chrono::seconds{900 * static_cast<long>(row)}}};
            bar.tickVolume = 100.0f + static_cast<float>(row % 7);
            tensor.Add(bar);
        }
        EA::LSTM model{tensor, hidden_size, 1.0f, 0.0f, type,
                       std::nullopt, {}, 42U};
        // Inactive heads otherwise have unspecified initialization for some
        // targets. Give only those inactive families reproducible fixture values.
        if (type == EA::LSTM::TargetType::UpNeutralDownReturn)
        {
            Zero(model.returnHeadWeight);
            Zero(model.returnHeadBias);
        }
        else
        {
            Zero(model.returnHeadDirWeight);
            Zero(model.returnHeadDirBias);
        }
        model.SetTrainingObjective(objective);
        if (type != EA::LSTM::TargetType::UpNeutralDownReturn)
            model.targetScale = 100000.0f; // regression fixture exercises real clipping
        std::ofstream output{argv[3], std::ios::binary | std::ios::trunc};
        Require(static_cast<bool>(output), "cannot open evidence file");
        std::cout << std::setprecision(15);
        const Window batch{tensor.end() - 8, tensor.end()};
        WriteState(output, model);
        for (unsigned short call = 0; call < 66; ++call)
        {
            const auto result = model.CalculateBatch(batch, 2);
            Require(std::isfinite(std::get<0>(result)) && model.optimizerUpdateCount == call + 1U,
                    "training failed to update");
            Write(output, std::get<0>(result));
            Write(output, static_cast<uint64_t>(std::get<1>(result)));
            Write(output, static_cast<uint64_t>(std::get<2>(result)));
            model.completedEpochs = call + 1U;
            WriteState(output, model);
        }
        EA::LSTM restored{tensor, hidden_size, 1.0f, 0.0f, type,
                          std::nullopt, {}, 43U};
        restored.SetTrainingObjective(objective);
        RestoreMatrices(model, restored);
        const auto continuing = model.CalculateBatch(batch, 2);
        const auto resuming = restored.CalculateBatch(batch, 2);
        Require(continuing == resuming, "restored matrix continuation loss changed");
        for (const auto pair : {std::pair{&model.param, &restored.param},
                                std::pair{&model.bias, &restored.bias},
                                std::pair{&model.returnHeadWeight, &restored.returnHeadWeight},
                                std::pair{&model.returnHeadBias, &restored.returnHeadBias},
                                std::pair{&model.returnHeadDirWeight, &restored.returnHeadDirWeight},
                                std::pair{&model.returnHeadDirBias, &restored.returnHeadDirBias}})
            Require(pair.first->Shape() == pair.second->Shape() &&
                    std::memcmp(MetaNN::LowerAccess(*pair.first).RawMemory(),
                                MetaNN::LowerAccess(*pair.second).RawMemory(),
                                pair.first->Shape()[0] * pair.first->Shape()[1] * sizeof(float)) == 0,
                    "restored matrix continuation parameters changed");
        WriteState(output, restored);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
