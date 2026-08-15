#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <MetaNN/data/facilities/lower_access.h>

#include "PricePoint.hpp"
#include "ReturnFeatureHistory.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{

constexpr std::size_t kBaseFeatureCount = 32;
constexpr std::size_t kModelFeatureCount =
    kBaseFeatureCount + EA::kMultiHorizonReturnLookbacks.size();

std::vector<Feature> MakeSourceRows(std::size_t count)
{
    std::vector<Feature> rows;
    rows.reserve(count);
    const PriceTP first {};
    for (std::size_t i = 0; i < count; ++i)
    {
        const float trend = 1.20f + static_cast<float>(i) * 0.00007f;
        const float wave = std::sin(static_cast<float>(i) * 0.17f) * 0.0003f;
        const float open = trend + wave;
        const float close = trend + std::sin(static_cast<float>(i) * 0.23f) * 0.00035f;
        rows.push_back(Feature{
            open,
            close,
            std::max(open, close) + 0.00021f + static_cast<float>(i % 5) * 0.00001f,
            std::min(open, close) - 0.00019f - static_cast<float>(i % 3) * 0.00001f,
            first + std::chrono::seconds(static_cast<long long>(i * 15 * 60))
        });
    }
    return rows;
}

Tensor BuildTensor(const std::vector<Feature>& rows,
                   std::size_t sourceBegin = 0)
{
    Tensor tensor{"warmup-test"};
    for (std::size_t i = sourceBegin; i < rows.size(); ++i)
        tensor.Add(rows.at(i));
    return tensor;
}

std::array<float, kBaseFeatureCount> BaseFeaturesAt(const Tensor& tensor,
                                                     std::size_t globalIndex)
{
    const auto row = *(tensor.begin() + static_cast<std::ptrdiff_t>(globalIndex));
    auto access = MetaNN::LowerAccess(row);
    std::array<float, kBaseFeatureCount> values {};
    std::memcpy(values.data(), access.RawMemory(), values.size() * sizeof(float));
    return values;
}

std::array<float, kModelFeatureCount> AssembleModelInputAt(
    const Tensor& tensor,
    std::size_t globalIndex)
{
    const auto base = BaseFeaturesAt(tensor, globalIndex);
    std::array<float, kModelFeatureCount> row {};
    const std::size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
        row.data(), base.data(), base.size(), globalIndex, 1000.0f,
        [&tensor](std::size_t position)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    assert(appended == EA::kMultiHorizonReturnLookbacks.size());
    return row;
}

void AssertByteIdentical(const void* lhs, const void* rhs, std::size_t size)
{
    assert(std::memcmp(lhs, rhs, size) == 0);
}

} // namespace

int main()
{
    static_assert(feature_size == kBaseFeatureCount);
    static_assert(EA::kMultiHorizonReturnLookbacks.size() == 4);
    static_assert(kModelFeatureCount == 36);

    const auto source = MakeSourceRows(512);
    constexpr std::size_t requestedStart = 160;

    // The production warmup path builds state from all predecessor source rows.
    const Tensor widerQuery = BuildTensor(source);
    const auto wideAtBoundary = BaseFeaturesAt(widerQuery, requestedStart);

    // A narrower logical interval uses the same warmed Tensor but emits only
    // source rows at or after requestedStart.
    const auto narrowTrainAtBoundary = AssembleModelInputAt(widerQuery, requestedStart);
    const auto narrowInferAtBoundary = AssembleModelInputAt(widerQuery, requestedStart);
    AssertByteIdentical(narrowTrainAtBoundary.data(), narrowInferAtBoundary.data(),
                        narrowTrainAtBoundary.size() * sizeof(float));
    AssertByteIdentical(wideAtBoundary.data(), narrowTrainAtBoundary.data(),
                        wideAtBoundary.size() * sizeof(float));

    // Every active stateful base channel is genuinely path-dependent at the
    // requested boundary. Starting Tensor::Add there would reset the first
    // row to zero instead of using predecessor source history.
    const Tensor coldBoundaryQuery = BuildTensor(source, requestedStart);
    const auto coldAtBoundary = BaseFeaturesAt(coldBoundaryQuery, 0);
    for (const std::size_t column : {6u, 7u, 14u, 15u, 16u, 17u, 18u,
                                     19u, 20u, 21u, 22u, 23u, 24u, 25u,
                                     26u, 27u, 28u, 29u, 31u})
    {
        assert(wideAtBoundary.at(column) != coldAtBoundary.at(column));
    }

    // Warmup rows never become logical output batches.
    std::vector<std::size_t> emittedBatchStarts;
    widerQuery.ForEachBatchFrom(requestedStart, [&](const auto& batch)
    {
        emittedBatchStarts.push_back(
            static_cast<std::size_t>(batch.begin() - widerQuery.begin()));
        assert(batch.begin() >= widerQuery.begin() +
               static_cast<std::ptrdiff_t>(requestedStart));
    });
    assert(!emittedBatchStarts.empty());
    assert(emittedBatchStarts.front() == requestedStart);

    // Tensor state is object-local: building an independent interval cannot
    // leak EMA, ATR, or rolling state into a later reconstruction.
    const Tensor unrelatedInterval = BuildTensor(source, 64);
    (void)unrelatedInterval;
    const Tensor rebuiltWiderQuery = BuildTensor(source);
    const auto rebuiltAtBoundary = AssembleModelInputAt(rebuiltWiderQuery, requestedStart);
    AssertByteIdentical(narrowTrainAtBoundary.data(), rebuiltAtBoundary.data(),
                        rebuiltAtBoundary.size() * sizeof(float));

    return 0;
}
