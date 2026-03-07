//
//  Tensor.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/3/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <array>
#include <vector>
#include <list>
#include <ranges>
#include <iostream>

#include "Params.hpp"
#include "LSTM.hpp"   // for LSTM_TRAINING_ASSERTS

using std::string;
using std::list;

using PriceTP = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;
struct Feature;

class Tensor
{
    bool has_prev_close = false;
    float prev_close = 0.0f;
    string table;
    DataSet ds;
    std::vector<float> raw_close;
    
public:
    // Number of full batches (size divisible by batch_size)
    // Number of batches including a trailing partial batch
    size_t NumberOfBatchesIncludingRemainder() const {  return (ds.size() + batch_size - 1) / batch_size;   }
    
    // Safe batch accessor that clamps the end iterator (and handles out-of-range idx)
    auto GetBatchClamped(long idx) const
    {
        size_t startIndex = 0;
        if (idx > 0) startIndex = static_cast<size_t>(idx) * batch_size;
        if (startIndex > ds.size()) startIndex = ds.size();
        size_t endIndex = startIndex + batch_size;
        if (endIndex > ds.size()) endIndex = ds.size();
#if LSTM_TRAINING_ASSERTS
        LSTM_ASSERT(startIndex <= endIndex && endIndex <= ds.size(), "GetBatchClamped: invalid range");
#endif
        auto startIt = ds.cbegin() + startIndex;
        auto endIt = ds.cbegin() + endIndex;
        return std::ranges::subrange(startIt, endIt);
    }

    // Convenience: iterate all batches (including remainder) and invoke a callable
    template <class Func>
    void ForEachBatch(Func&& f) const
    {
        const size_t n = NumberOfBatchesIncludingRemainder();
#if LSTM_TRAINING_ASSERTS
        LSTM_ASSERT(n == NumberOfBatchesIncludingRemainder(), "ForEachBatch: n mismatch");
#endif
        for (size_t i = 0; i < n; ++i)  f(GetBatchClamped(i));
    }

    
    Tensor(string name) : table { name }    {}
    
    void Add(Feature f);
    
    DataSet::const_iterator begin() const  { return ds.cbegin(); }
    DataSet::const_iterator end() const { return ds.cend(); }
    
    auto GetWindow(DataSet::const_iterator iter) const -> Window
    {
#if LSTM_TRAINING_ASSERTS
        LSTM_ASSERT(iter >= ds.cbegin() && iter <= ds.cend(), "GetWindow: iterator out of bounds");
        LSTM_ASSERT(static_cast<size_t>(ds.cend() - iter) >= window_size, "GetWindow: not enough elements for window");
#endif
        return { iter, iter + window_size };
    }
    size_t NumberOfBatches() const    {   return ds.size() / batch_size;   }
    auto GetBatch(long idx) const
    {
#if LSTM_TRAINING_ASSERTS
        LSTM_ASSERT(idx >= 0, "GetBatch: negative index");
        size_t start = static_cast<size_t>(idx) * batch_size;
        LSTM_ASSERT(start + batch_size <= ds.size(), "GetBatch: out of range");
#endif
        auto batchIdx = ds.cbegin() + idx * batch_size;
        return std::ranges::subrange(batchIdx, batchIdx + batch_size);
    }

    float RawCloseAtIterator(DataSet::const_iterator it) const;

};

std::ostream& operator<<(std::ostream&, Window);

template <class Mat>
void printMatrix(const char* name, const Mat& mat)
{
    // Materialize MetaNN expressions (e.g., Reshape) before element access
    auto cm = MetaNN::Evaluate(mat);
    std::cout << name << " (" << cm.Shape()[0] << " x " << cm.Shape()[1] << ")\n";
    for (size_t i = 0; i < cm.Shape()[0]; ++i)
        for (size_t j = 0; j < cm.Shape()[1]; ++j)
            std::cout << cm(i, j) << (j + 1 == cm.Shape()[1] ? '\n' : ' ');
    std::cout << std::endl;
}

#endif /* Tensor_hpp */

