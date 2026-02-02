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

#include "PricePoint.hpp"

#include "MetaNN/meta_nn.h"

using FeatureMatrix = MetaNN::Matrix<float, MetaNN::DeviceTags::CPU>;
using DataSet = std::vector<FeatureMatrix>;
using Window = std::ranges::subrange<DataSet::const_iterator, DataSet::const_iterator>;
using Batch = Window;

using std::string;
using std::list;


// sequence of features
constexpr auto window_size = 5;
constexpr auto batch_size = 15;

class Tensor
{
    string table;
    DataSet ds;
    
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
        auto startIt = ds.cbegin() + startIndex;
        auto endIt = ds.cbegin() + endIndex;
        return std::ranges::subrange(startIt, endIt);
    }

    // Convenience: iterate all batches (including remainder) and invoke a callable
    template <class Func>
    void ForEachBatch(Func&& f) const
    {
        const size_t n = NumberOfBatchesIncludingRemainder();
        for (size_t i = 0; i < n; ++i)  f(i);
    }

    
    Tensor(string name) : table { name }    {}
    
    void Add(Feature f);
    
    DataSet::const_iterator begin() const  { return ds.cbegin(); }
    DataSet::const_iterator end() const { return ds.cend(); }
    
    auto GetWindow(DataSet::const_iterator iter) const -> Window    {   return { iter, iter + window_size };  }
    size_t NumberOfBatches() const    {   return ds.size() / batch_size;   }
    auto GetBatch(long idx) const
    {
        auto batchIdx = ds.cbegin() + idx * batch_size;
        return std::ranges::subrange(batchIdx, batchIdx + batch_size);
    }
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


