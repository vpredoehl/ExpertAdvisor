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


#include "PricePoint.hpp"

constexpr auto window_size = 5;

using Window = std::array<Feature,window_size>;
using Batch = std::vector<Window>;

using std::string;
using std::list;

class Tensor
{
    string table;
    Batch b;
    list<Window> window_seq { window_size };
    short seq_size = 1;
    
public:
    Tensor(string name) : table { name }    {}
    
    void Add(Feature);  // adds feature to rolling sequence of feature windows
};


#endif /* Tensor_hpp */
