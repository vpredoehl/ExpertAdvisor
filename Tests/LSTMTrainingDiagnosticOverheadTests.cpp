#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "LstmRuntimeLogging.hpp"
#include "MatrixUtils.hpp"

namespace
{
using Matrix = MetaNN::Matrix<float, MetaNN::DeviceTags::CPU>;
using TargetType = EA::LSTM::TargetType;
std::vector<const float*> inspected;
size_t copies = 0;

void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

Matrix Filled(float value)
{
    Matrix matrix(1, 2);
    matrix.SetValue(0, 0, value);
    matrix.SetValue(0, 1, value + 1.0f);
    return matrix;
}

const float* Data(const Matrix& matrix)
{
    return MetaNN::LowerAccess(matrix).RawMemory();
}

Matrix CountEvaluate(const Matrix& matrix)
{
    inspected.push_back(Data(matrix));
    return MetaNN::Evaluate(matrix);
}

Matrix CountDeepCopy(const Matrix& matrix)
{
    ++copies;
    return NNUtils::DeepCopyMatrix(matrix);
}

bool LogDiagnostic() { return EA::RuntimeDiagnosticLoggingEnabled(); }
bool RuntimeDiagnosticLoggingEnabled() { return LogDiagnostic(); }
std::ostream& DiagnosticOut() { return std::cout; }

struct ScopedDiagnosticCoutSilencer { ~ScopedDiagnosticCoutSilencer() {} };

struct Model
{
    TargetType targetType;
    Matrix param = Filled(1.0f), bias = Filled(2.0f);
    Matrix returnHeadWeight = Filled(3.0f), returnHeadBias = Filled(4.0f);
    Matrix returnHeadDirWeight = Filled(5.0f), returnHeadDirBias = Filled(6.0f);
    size_t updates = 0;

    auto CalculateBatch(int, int)
    {
        Require(inspected.size() == (LogDiagnostic() ? 4 : 0),
                "pre-update norm count/timing");
        for (auto* matrix : {&param, &bias, &returnHeadWeight,
                             &returnHeadBias, &returnHeadDirWeight,
                             &returnHeadDirBias})
        {
            for (size_t column = 0; column < 2; ++column)
                matrix->SetValue(0, column, (*matrix)(0, column) + 0.25f);
        }
        ++updates;
        return std::tuple{0.125f, size_t{1}, size_t{0}};
    }
};

double Norm(const Matrix& matrix)
{
    double sum = 0.0;
    for (size_t column = 0; column < 2; ++column)
    {
        const double value = matrix(0, column);
        sum += value * value;
    }
    return std::sqrt(sum);
}

void TestWorker(EA::RuntimeLogLevel level, TargetType type)
{
    EA::SetRuntimeLogLevel(level);
    inspected.clear();
    Model l{type};
    const bool direction = type == TargetType::UpNeutralDownReturn;
    const double p0 = Norm(l.param), b0 = Norm(l.bias);
    const double w0 = Norm(direction ? l.returnHeadDirWeight : l.returnHeadWeight);
    const double hb0 = Norm(direction ? l.returnHeadDirBias : l.returnHeadBias);
    const int b = 0, e = 2;
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    {
#include "WorkerDiagnosticBlock.inc"
    }
    std::cout.rdbuf(previous);
    Require(l.updates == 1 && l.param(0, 0) == 1.25f, "worker update changed");
    if (!LogDiagnostic())
    {
        Require(inspected.empty() && output.str().empty(), "disabled norm overhead");
        return;
    }
    Require(inspected.size() == 8, "enabled norm count");
    const float* inactiveW = Data(direction ? l.returnHeadWeight : l.returnHeadDirWeight);
    const float* inactiveB = Data(direction ? l.returnHeadBias : l.returnHeadDirBias);
    for (auto* pointer : inspected)
        Require(pointer != inactiveW && pointer != inactiveB, "inactive head inspected");
    std::ostringstream expected;
    expected << "epoch 3 loss=0.125 ||param|| " << p0 << " -> " << Norm(l.param)
             << " ||bias|| " << b0 << " -> " << Norm(l.bias)
             << (direction ? " ||dirHeadW|| " : " ||headW|| ") << w0 << " -> "
             << Norm(direction ? l.returnHeadDirWeight : l.returnHeadWeight)
             << (direction ? " ||dirHeadB|| " : " ||headB|| ") << hb0 << " -> "
             << Norm(direction ? l.returnHeadDirBias : l.returnHeadBias) << '\n';
    Require(output.str() == expected.str(), "enabled worker output changed");
}

void TestSnapshot(size_t call, bool enabled, TargetType targetType)
{
    EA::SetRuntimeLogLevel(enabled ? EA::RuntimeLogLevel::Diagnostic : EA::RuntimeLogLevel::Quiet);
    using EAMatrix = Matrix;
    constexpr size_t LSTM_PHASE3_HEAD_DIAG_LIMIT = 64;
    auto d_param_f = Filled(20.0f), d_bias_f = Filled(30.0f);
    auto d_headW_f = Filled(40.0f), d_headB_f = Filled(50.0f);
    auto d_headDirW_f = Filled(60.0f), d_headDirB_f = Filled(70.0f);
    copies = 0;
#include "GradientSnapshotBlock.inc"
    Require(phase3ClipFullDiagIdx == call, "diagnostic counter timing changed");
    const bool required = enabled && call < 64;
    Require(phase3ClipFullDiagEnabled == required, "snapshot consumer condition");
    Require(copies == (required ? 4 : 0), "snapshot copy count");
    Require(d_param_preclip.has_value() == required && d_bias_preclip.has_value() == required,
            "core snapshot presence");
    const bool direction = targetType == TargetType::UpNeutralDownReturn;
    Require(d_headDirW_preclip.has_value() == (required && direction) &&
            d_headDirB_preclip.has_value() == (required && direction) &&
            d_headW_preclip.has_value() == (required && !direction) &&
            d_headB_preclip.has_value() == (required && !direction), "inactive snapshot");
    if (required)
    {
        d_param_f.SetValue(0, 0, 10.0f); // represent clipping after the snapshot
        Require((*d_param_preclip)(0, 0) == 20.0f && Data(*d_param_preclip) != Data(d_param_f),
                "snapshot is not independent of clipped gradient");
    }
}

void TestForgetGateSlice()
{
    // Multiple rows detect accidental striding; all-column slicing tests its fast path.
    for (const size_t rows : {size_t{1}, size_t{3}})
    {
        Matrix gates(rows, 12);
        for (size_t row = 0; row < rows; ++row)
            for (size_t column = 0; column < 12; ++column)
                gates.SetValue(row, column, static_cast<float>(row * 12 + column) - 10.5f);
        for (const bool all : {false, true})
        {
            const size_t offset = all ? 0 : 3, width = all ? 12 : 3;
            auto slice = NNUtils::ViewCols<float, MetaNN::DeviceTags::CPU>(gates, offset, width);
            auto previousCache = NNUtils::DeepCopyMatrix(slice);
            const float* storage = Data(slice);
            auto cache = std::move(slice);
            Require(Data(cache) == storage && storage != Data(gates), "slice ownership changed");
            for (size_t row = 0; row < rows; ++row)
                for (size_t column = 0; column < width; ++column)
                {
                    const float value = gates(row, offset + column);
                    Require(Data(cache)[row * width + column] == value &&
                            cache(row, column) == previousCache(row, column), "forget logits changed");
                    Require(1.0f / (1.0f + std::exp(-cache(row, column))) ==
                            1.0f / (1.0f + std::exp(-previousCache(row, column))), "forget gate changed");
                }
            gates = Matrix(0, 0); // cached logits must survive source destruction
            Require(cache(0, 0) == previousCache(0, 0), "slice lifetime changed");
            gates = Matrix(rows, 12);
            for (size_t row = 0; row < rows; ++row)
                for (size_t column = 0; column < 12; ++column)
                    gates.SetValue(row, column, static_cast<float>(row * 12 + column) - 10.5f);
        }
    }
}
} // namespace

int main()
{
    for (auto level : {EA::RuntimeLogLevel::Quiet, EA::RuntimeLogLevel::Summary,
                       EA::RuntimeLogLevel::Diagnostic})
        for (auto type : {TargetType::LogReturn, TargetType::PercentReturn,
                          TargetType::UpNeutralDownReturn})
            TestWorker(level, type);
    // Disabled calls still consume the original diagnostic budget; re-enabling
    // neither resets the counter nor enables copies after the 64-call boundary.
    for (size_t call = 0; call < 68; ++call)
        TestSnapshot(call, call >= 4 && call != 62,
                     call % 2 ? TargetType::UpNeutralDownReturn : TargetType::LogReturn);
    TestForgetGateSlice();
    std::cout << "LSTMTrainingDiagnosticOverheadTests passed (CPU; no database/Metal work)\n";
}
