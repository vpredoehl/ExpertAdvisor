#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include "Tensor.hpp"

namespace NNUtils {

// Concatenate matrices along columns.
// All input parts must have the same number of rows. Typical use: concatenate 1xK_i row vectors.
// Works for general R x K_i by copying per row.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> ConcatCols(std::initializer_list<MetaNN::Matrix<T, DevTag>> parts)
{
    // Evaluate and collect parts to ensure concrete matrices
    std::vector<MetaNN::Matrix<T, DevTag>> mats;
    mats.reserve(parts.size());

    size_t rows = 0;
    size_t totalCols = 0;
    bool first = true;
    for (const auto& p : parts)
    {
        auto m = MetaNN::Evaluate(p);
        if (first)
        {
            rows = m.Shape()[0];
            first = false;
        }
        totalCols += m.Shape()[1];
        mats.push_back(std::move(m));
    }

    MetaNN::Matrix<T, DevTag> out(rows, totalCols);
    auto lowOut = MetaNN::LowerAccess(out);
    T* outRaw = lowOut.MutableRawMemory();

    // Copy per row to correctly handle R > 1
    for (size_t r = 0; r < rows; ++r)
    {
        size_t colOffset = 0;
        for (const auto& m : mats)
        {
            const size_t cols = m.Shape()[1];
            auto lowM = MetaNN::LowerAccess(m);
            const T* src = lowM.RawMemory();
            // Row-major layout: row r starts at r * cols
            std::copy(src + r * cols, src + r * cols + cols,
                      outRaw + r * totalCols + colOffset);
            colOffset += cols;
        }
    }

    return out;
}

} // namespace NNUtils
