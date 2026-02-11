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

// Slice a contiguous range of rows [rowOffset, rowOffset + rowCount) from a matrix.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> SliceRows(const MetaNN::Matrix<T, DevTag>& src,
                                    size_t rowOffset,
                                    size_t rowCount)
{
    const size_t cols = src.Shape()[1];
    MetaNN::Matrix<T, DevTag> out(rowCount, cols);
    auto lowSrc = MetaNN::LowerAccess(src);
    auto lowOut = MetaNN::LowerAccess(out);
    const T* s = lowSrc.RawMemory();
    T* d = lowOut.MutableRawMemory();
    for (size_t r = 0; r < rowCount; ++r)
    {
        const T* srcRowPtr = s + (rowOffset + r) * cols;
        T* dstRowPtr = d + r * cols;
        std::copy(srcRowPtr, srcRowPtr + cols, dstRowPtr);
    }
    return out;
}

// Convenience: take the bottom N rows of a matrix.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> TakeBottomRows(const MetaNN::Matrix<T, DevTag>& src,
                                         size_t rowCount)
{
    const size_t rows = src.Shape()[0];
    return SliceRows<T, DevTag>(src, rows - rowCount, rowCount);
}

// Slice a contiguous range of columns [colOffset, colOffset + colCount) from a matrix.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> SliceCols(const MetaNN::Matrix<T, DevTag>& src,
                                    size_t colOffset,
                                    size_t colCount)
{
    const size_t rows = src.Shape()[0];
    const size_t srcCols = src.Shape()[1];
    MetaNN::Matrix<T, DevTag> out(rows, colCount);
    auto lowSrc = MetaNN::LowerAccess(src);
    auto lowOut = MetaNN::LowerAccess(out);
    const T* s = lowSrc.RawMemory();
    T* d = lowOut.MutableRawMemory();
    for (size_t r = 0; r < rows; ++r)
    {
        const T* srcPtr = s + r * srcCols + colOffset;
        T* dstPtr = d + r * colCount;
        std::copy(srcPtr, srcPtr + colCount, dstPtr);
    }
    return out;
}

// Convenience: take the rightmost N columns of a matrix.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> TakeRightCols(const MetaNN::Matrix<T, DevTag>& src,
                                        size_t colCount)
{
    const size_t srcCols = src.Shape()[1];
    return SliceCols<T, DevTag>(src, srcCols - colCount, colCount);
}

// Convenience: take the top N rows of a matrix.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> TakeTopRows(const MetaNN::Matrix<T, DevTag>& src,
                                      size_t rowCount)
{
    return SliceRows<T, DevTag>(src, /*rowOffset*/ 0, rowCount);
}

// Convenience: take the leftmost N columns of a matrix.
template <typename T, typename DevTag>
MetaNN::Matrix<T, DevTag> TakeLeftCols(const MetaNN::Matrix<T, DevTag>& src,
                                       size_t colCount)
{
    return SliceCols<T, DevTag>(src, /*colOffset*/ 0, colCount);
}

} // namespace NNUtils

