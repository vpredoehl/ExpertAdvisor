//
//  Tensor.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/3/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include <iomanip>

#include "Tensor.hpp"

using std::setw;

std::ostream& operator<<(std::ostream& o, Window w)
{
    for(auto f : w)     o <<  setw(10) << f.open << setw(10) << f.close << setw(10) << f.high <<  setw(10) << f.low << std::endl;
    return o;
}
