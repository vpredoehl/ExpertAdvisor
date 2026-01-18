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


#include "PricePoint.hpp"

using DataSet = std::vector<Feature>;
using Window = std::ranges::subrange<DataSet::const_iterator, DataSet::const_iterator>;

using std::string;
using std::list;


// sequence of features
constexpr auto window_size = 5;
constexpr auto batch_size = 64;

class Tensor
{
    string table;
    DataSet b;
    short window_size;
    
public:
    Tensor(string name, short ws) : table { name }, window_size { ws }    {}
    
    void Add(Feature f)   {   b.push_back(f);   }
    
    DataSet::const_iterator begin() const  { return b.cbegin(); }
    DataSet::const_iterator end() const { return b.cend(); }
    
    auto GetWindow(long idx) const  {   return std::ranges::subrange(b.cbegin() + idx, b.cbegin() + idx + window_size); }
    auto GetWindow(DataSet::const_iterator iter) -> Window    {   return { iter, iter + window_size };  }
    auto GetBatch(long idx ) const
    {
        auto batchIdx = b.cbegin() + idx * batch_size;
        return std::ranges::subrange(batchIdx, batchIdx + batch_size);
    }
};

std::ostream& operator<<(std::ostream&, Window);


#endif /* Tensor_hpp */
