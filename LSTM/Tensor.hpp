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

using Batch = std::vector<Feature>;
using Window = std::ranges::subrange<Batch::const_iterator, Batch::const_iterator>;

using std::string;
using std::list;


// sequence of features
constexpr auto window_size = 5;

class Tensor
{
    string table;
    Batch b;
    short window_size;
    
public:
    Tensor(string name, short ws) : table { name }, window_size { ws }    {}
    
    void Add(Feature f)   {   b.push_back(f);   }
    
    Batch::const_iterator begin() const  { return b.cbegin(); }
    Batch::const_iterator end() const { return b.cend(); }
    
    auto GetWindow(long idx) const  {   return std::ranges::subrange(b.cbegin() + idx, b.cbegin() + idx + window_size); }
};

std::ostream& operator<<(std::ostream&, Window);


#endif /* Tensor_hpp */
