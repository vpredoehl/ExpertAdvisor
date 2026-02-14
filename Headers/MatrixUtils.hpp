#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include "Tensor.hpp"

namespace NNUtils {

// Works for general R x K_i by copying per row.
template <typename T, typename DevT, typename... Parts>
MetaNN::Matrix<T, DevT> ConcatCols(Parts&&... parts)
{
    // Evaluate and collect parts to ensure concrete matrices, even for heterogeneous expressions
    std::vector<MetaNN::Matrix<T, DevT>> mats;
    mats.reserve(sizeof...(Parts));
    auto collect = [&](auto&& p)
    {
        mats.push_back(MetaNN::Evaluate(std::forward<decltype(p)>(p)));
    };
    (collect(std::forward<Parts>(parts)), ...);

    size_t rows = 0;
    size_t totalCols = 0;
    bool first = true;
    for (const auto& m : mats)
    {
        if (first)
        {
            rows = m.Shape()[0];
            first = false;
        }
        totalCols += m.Shape()[1];
    }

    MetaNN::Matrix<T, DevT> out(rows, totalCols);
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

// Concatenate matrices along columns.
// All input parts must have the same number of rows. Typical use: concatenate 1xK_i row vectors.
// Works for general R x K_i by copying per row.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> ConcatCols(std::initializer_list<MetaNN::Matrix<T, DevT>> parts)
{
    // Evaluate and collect parts to ensure concrete matrices
    std::vector<MetaNN::Matrix<T, DevT>> mats;
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

    MetaNN::Matrix<T, DevT> out(rows, totalCols);
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
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> SliceRows(const MetaNN::Matrix<T, DevT>& src,
                                    size_t rowOffset,
                                    size_t rowCount)
{
    const size_t cols = src.Shape()[1];
    MetaNN::Matrix<T, DevT> out(rowCount, cols);
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
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> TakeBottomRows(const MetaNN::Matrix<T, DevT>& src,
                                         size_t rowCount)
{
    const size_t rows = src.Shape()[0];
    return SliceRows<T, DevT>(src, rows - rowCount, rowCount);
}

// Slice a contiguous range of columns [colOffset, colOffset + colCount) from a matrix.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> SliceCols(const MetaNN::Matrix<T, DevT>& src,
                                    size_t colOffset,
                                    size_t colCount)
{
    const size_t rows = src.Shape()[0];
    const size_t srcCols = src.Shape()[1];
    // Fast path: if taking all columns starting at 0, return a direct copy
    if (colOffset == 0 && colCount == srcCols)
    {
        return src;
    }
    MetaNN::Matrix<T, DevT> out(rows, colCount);
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
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> TakeRightCols(const MetaNN::Matrix<T, DevT>& src,
                                        size_t colCount)
{
    const size_t srcCols = src.Shape()[1];
    return SliceCols<T, DevT>(src, srcCols - colCount, colCount);
}

// Convenience: take the top N rows of a matrix.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> TakeTopRows(const MetaNN::Matrix<T, DevT>& src,
                                      size_t rowCount)
{
    return SliceRows<T, DevT>(src, /*rowOffset*/ 0, rowCount);
}

// Convenience: take the leftmost N columns of a matrix.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> TakeLeftCols(const MetaNN::Matrix<T, DevT>& src,
                                       size_t colCount)
{
    return SliceCols<T, DevT>(src, /*colOffset*/ 0, colCount);
}

// View (no-copy): create a matrix referencing a contiguous range of rows.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> ViewRows(const MetaNN::Matrix<T, DevT>& src,
                                   size_t rowOffset,
                                   size_t rowCount)
{
    const size_t cols = src.Shape()[1];
    auto lowSrc = MetaNN::LowerAccess(src);
    auto mem = lowSrc.SharedMemory().Shift(rowOffset * cols);
    return MetaNN::Matrix<T, DevT>(mem, MetaNN::Shape(rowCount, cols));
}

// Convenience: view the bottom N rows (no copy).
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> ViewBottomRows(const MetaNN::Matrix<T, DevT>& src,
                                         size_t rowCount)
{
    const size_t rows = src.Shape()[0];
    return ViewRows<T, DevT>(src, rows - rowCount, rowCount);
}

// Convenience: view the top N rows (no copy).
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> ViewTopRows(const MetaNN::Matrix<T, DevT>& src, size_t rowCount)
{
    return ViewRows<T, DevT>(src, /*rowOffset*/ 0, rowCount);
}

// View a contiguous block of columns. If MetaNN does not support strided column views,
// fall back to a slicing copy to produce a concrete matrix with the requested columns.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> ViewCols(const MetaNN::Matrix<T, DevT>& src, size_t colOffset, size_t colCount)
{
    // Use SliceCols (copy) to obtain the requested column block.
    return SliceCols<T, DevT>(src, colOffset, colCount);
}

// Generic matrix type cast helper: casts element type and device tag
template <typename T, typename DevT, typename SrcMat>
MetaNN::Matrix<T, DevT> CastMatrix(const SrcMat& src)
{
    // Ensure we operate on a concrete matrix to safely access raw memory
    auto eval = MetaNN::Evaluate(src);
    using EvalMat = decltype(eval);
    using SrcElem = typename EvalMat::ElementType;
    using SrcDev  = typename EvalMat::DeviceType;

    // Fast path: if element and device types already match, return as-is
    if constexpr (std::is_same_v<SrcElem, T> && std::is_same_v<SrcDev, DevT>)
    {
        return eval;
    }

    // Otherwise, cast element type and/or device by copying
    MetaNN::Matrix<T, DevT> dst(eval.Shape()[0], eval.Shape()[1]);
    auto lowSrc = MetaNN::LowerAccess(eval);
    auto lowDst = MetaNN::LowerAccess(dst);
    const auto* s = lowSrc.RawMemory();
    T* d = lowDst.MutableRawMemory();
    const size_t len = eval.Shape()[0] * eval.Shape()[1];
    for (size_t i = 0; i < len; ++i) d[i] = static_cast<T>(s[i]);
    return dst;
}

} // namespace NNUtils

