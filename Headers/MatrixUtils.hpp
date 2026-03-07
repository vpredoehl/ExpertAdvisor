#pragma once

#include <vector>
#include <algorithm>
#include <tuple>
#include <cstddef>
#include <type_traits>
#include <cassert>
#include <cstring>
#include "Tensor.hpp"
#include <MetaNN/operation/tensor/slice.h>
#include "LSTM.hpp"

namespace NNUtils {

// Internal helper: fill an already-allocated matrix with the horizontal concatenation of parts
template <typename T, typename DevT>
inline void FillConcatCols(MetaNN::Matrix<T, DevT>& out,
                           const std::vector<MetaNN::Matrix<T, DevT>>& mats)
{
    const size_t rows = out.Shape()[0];
    const size_t totalCols = out.Shape()[1];

    // Sanity checks: all parts have the same number of rows and total columns match
#ifndef NDEBUG
    size_t checkTotalCols = 0;
    for (const auto& m : mats)
    {
        assert(m.Shape()[0] == rows && "ConcatCols: all parts must have the same number of rows");
        checkTotalCols += m.Shape()[1];
    }
    assert(checkTotalCols == totalCols && "ConcatCols: output has incorrect column count");
#endif

    auto lowOut = MetaNN::LowerAccess(out);
    T* outRaw = lowOut.MutableRawMemory();

    // Precompute raw pointers and column counts once
    struct SrcInfo { const T* ptr; size_t cols; };
    std::vector<SrcInfo> srcs;
    srcs.reserve(mats.size());
    for (const auto& m : mats)
    {
        auto lowM = MetaNN::LowerAccess(m);
        srcs.push_back(SrcInfo{ lowM.RawMemory(), m.Shape()[1] });
    }

    // Row-wise copy
    for (size_t r = 0; r < rows; ++r)
    {
        size_t colOffset = 0;
        for (const auto& s : srcs)
        {
            const T* srcRow = s.ptr + r * s.cols;
            T* dstRow = outRaw + r * totalCols + colOffset;
            if constexpr (std::is_trivially_copyable_v<T>)
            {
                std::memcpy(dstRow, srcRow, s.cols * sizeof(T));
            }
            else
            {
                std::copy(srcRow, srcRow + s.cols, dstRow);
            }
            colOffset += s.cols;
        }
    }
}

// Overloads that write into a preallocated output buffer to avoid per-call allocation
template <typename T, typename DevT, typename... Parts>
inline void ConcatColsInto(MetaNN::Matrix<T, DevT>& out, Parts&&... parts)
{
    std::vector<MetaNN::Matrix<T, DevT>> mats;
    mats.reserve(sizeof...(Parts));
    auto collect = [&](auto&& p)
    {
        mats.push_back(MetaNN::Evaluate(std::forward<decltype(p)>(p)));
    };
    (collect(std::forward<Parts>(parts)), ...);

#ifndef NDEBUG
    // Validate output shape against inputs
    size_t rowsCheck = 0;
    size_t colsCheck = 0;
    bool first = true;
    for (const auto& m : mats)
    {
        if (first) { rowsCheck = m.Shape()[0]; first = false; }
        else { assert(m.Shape()[0] == rowsCheck && "ConcatColsInto: all parts must have the same number of rows"); }
        colsCheck += m.Shape()[1];
    }
    assert(out.Shape()[0] == rowsCheck && "ConcatColsInto: output has incorrect row count");
    assert(out.Shape()[1] == colsCheck && "ConcatColsInto: output has incorrect column count");
#endif

#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(!mats.empty(), "ConcatColsInto: no parts provided");
    size_t rowsCheck2 = mats[0].Shape()[0];
    size_t colsSum2 = 0;
    for (const auto& m : mats) { colsSum2 += m.Shape()[1]; }
    LSTM_ASSERT(out.Shape()[0] == rowsCheck2, "ConcatColsInto: output rows mismatch");
    LSTM_ASSERT(out.Shape()[1] == colsSum2, "ConcatColsInto: output cols mismatch");
#endif

    FillConcatCols(out, mats);
}

template <typename T, typename DevT>
inline void ConcatColsInto(MetaNN::Matrix<T, DevT>& out,
                           std::initializer_list<MetaNN::Matrix<T, DevT>> parts)
{
    std::vector<MetaNN::Matrix<T, DevT>> mats;
    mats.reserve(parts.size());
    for (const auto& p : parts)
    {
        mats.push_back(MetaNN::Evaluate(p));
    }

#ifndef NDEBUG
    // Validate output shape against inputs
    size_t rowsCheck = 0;
    size_t colsCheck = 0;
    bool first = true;
    for (const auto& m : mats)
    {
        if (first) { rowsCheck = m.Shape()[0]; first = false; }
        else { assert(m.Shape()[0] == rowsCheck && "ConcatColsInto: all parts must have the same number of rows"); }
        colsCheck += m.Shape()[1];
    }
    assert(out.Shape()[0] == rowsCheck && "ConcatColsInto: output has incorrect row count");
    assert(out.Shape()[1] == colsCheck && "ConcatColsInto: output has incorrect column count");
#endif

#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(!mats.empty(), "ConcatColsInto(init_list): no parts provided");
    size_t rowsCheck2 = mats[0].Shape()[0];
    size_t colsSum2 = 0;
    for (const auto& m : mats) { colsSum2 += m.Shape()[1]; }
    LSTM_ASSERT(out.Shape()[0] == rowsCheck2, "ConcatColsInto(init_list): output rows mismatch");
    LSTM_ASSERT(out.Shape()[1] == colsSum2, "ConcatColsInto(init_list): output cols mismatch");
#endif

    FillConcatCols(out, mats);
}

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
#ifndef NDEBUG
        else
        {
            assert(m.Shape()[0] == rows && "ConcatCols: all parts must have the same number of rows");
        }
#endif
        totalCols += m.Shape()[1];
    }

    MetaNN::Matrix<T, DevT> out(rows, totalCols);
    FillConcatCols(out, mats);
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
        else
        {
#ifndef NDEBUG
            assert(m.Shape()[0] == rows && "ConcatCols: all parts must have the same number of rows");
#endif
        }
        totalCols += m.Shape()[1];
        mats.push_back(std::move(m));
    }

    MetaNN::Matrix<T, DevT> out(rows, totalCols);
    FillConcatCols(out, mats);
    return out;
}

// Slice a contiguous range of rows [rowOffset, rowOffset + rowCount) from a matrix.
template <typename T, typename DevT>
MetaNN::Matrix<T, DevT> SliceRows(const MetaNN::Matrix<T, DevT>& src,
                                    size_t rowOffset,
                                    size_t rowCount)
{
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(rowOffset + rowCount <= src.Shape()[0], "SliceRows: row range out of bounds");
#endif
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
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(colOffset + colCount <= src.Shape()[1], "SliceCols: col range out of bounds");
#endif
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
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(rowOffset + rowCount <= src.Shape()[0], "ViewRows: row range out of bounds");
#endif
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
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(colOffset + colCount <= src.Shape()[1], "ViewCols: col range out of bounds");
#endif
    // Use SliceCols (copy) to obtain the requested column block.
    return SliceCols<T, DevT>(src, colOffset, colCount);
}

// View a contiguous block of columns, falling back to a concrete sliced copy when
// a lazy MetaNN::Slice expression is unavailable in this build.
template <typename Mat>
auto ViewColsExpr(const Mat& src, size_t colOffset, size_t colCount)
{
    // Fallback: evaluate and return a concrete matrix with the requested columns.
    using Decayed = std::decay_t<Mat>;
    using T = typename Decayed::ElementType;
    using Dev = typename Decayed::DeviceType;
    auto eval = MetaNN::Evaluate(src);
    return SliceCols<T, Dev>(eval, colOffset, colCount);
}

// Zero-copy helpers: split a (1 x 4H) or (B x 4H) gate matrix into four lazy views without copying
// For a single row (batch=1): returns four (1 x H) views (i, f, g, o)
template <typename Mat>
auto SplitGatesRowExpr(const Mat& y)
{
    // Expect y shape: (1, 4H)
    #ifndef NDEBUG
    assert(y.Shape()[0] == 1 && "SplitGatesRowExpr expects a (1 x 4H) matrix");
    assert(y.Shape()[1] % 4 == 0 && "SplitGatesRowExpr: columns must be divisible by 4");
    #endif
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(y.Shape()[0] == 1, "SplitGatesRowExpr expects (1 x 4H)");
    LSTM_ASSERT(y.Shape()[1] % 4 == 0, "SplitGatesRowExpr: columns must be divisible by 4");
#endif
    const size_t H = y.Shape()[1] / 4;
    auto gates2D = MetaNN::Reshape(y, MetaNN::Shape(4, H));
    // Each gates2D[k] is a 1D view of length H; reshape to (1 x H)
    auto i = MetaNN::Reshape(gates2D[0], MetaNN::Shape(1, H));
    auto f = MetaNN::Reshape(gates2D[1], MetaNN::Shape(1, H));
    auto g = MetaNN::Reshape(gates2D[2], MetaNN::Shape(1, H));
    auto o = MetaNN::Reshape(gates2D[3], MetaNN::Shape(1, H));
    return std::make_tuple(i, f, g, o);
}

// For batched outputs: Y shape (B x 4H). Returns four (B x H) views (i, f, g, o)
template <typename Mat>
auto SplitGatesBatchExpr(const Mat& Y)
{
    const size_t B = Y.Shape()[0];
    const size_t W = Y.Shape()[1];
    #ifndef NDEBUG
    assert(W % 4 == 0 && "SplitGatesBatchExpr: columns must be divisible by 4");
    #endif
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(W % 4 == 0, "SplitGatesBatchExpr: columns must be divisible by 4");
#endif
    const size_t H = W / 4;
    // Reshape to (4, B, H) so indexing the first dimension yields (B x H) gate views
    auto Y3 = MetaNN::Reshape(Y, MetaNN::Shape(4, B, H));
    auto i = Y3[0];
    auto f = Y3[1];
    auto g = Y3[2];
    auto o = Y3[3];
    return std::make_tuple(i, f, g, o);
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
    if constexpr (std::is_same_v<SrcElem, T> && std::is_same_v<SrcDev, DevT>)   return eval;

    // Otherwise, cast element type and/or device by copying
    MetaNN::Matrix<T, DevT> dst(eval.Shape()[0], eval.Shape()[1]);
    auto lowSrc = MetaNN::LowerAccess(eval);
    auto lowDst = MetaNN::LowerAccess(dst);
    const auto* s = lowSrc.RawMemory();
    T* d = lowDst.MutableRawMemory();
    const size_t len = eval.Shape()[0] * eval.Shape()[1];
#pragma omp parallel for
    for (size_t i = 0; i < len; ++i) d[i] = static_cast<T>(s[i]);
    return dst;
}

} // namespace NNUtils



