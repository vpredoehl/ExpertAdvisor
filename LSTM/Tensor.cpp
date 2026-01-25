//
//  Tensor.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/3/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include <iomanip>

#include "Tensor.hpp"
#include "LSTM.hpp"

using std::setw;

std::ostream& operator<<(std::ostream& o, Window w)
{
    for(auto f : w)
    {
        float open = f.Shape()[0];
        float close = f.Shape()[0];
        float high = f.Shape()[0];
        float low = f.Shape()[0];
        o <<  setw(10) << open << setw(10) << close << setw(10) << high <<  setw(10) << low << std::endl;
    }
    return o;
}

void Tensor::Add(Feature f)
{
    FeatureMatrix fm(1,feature_size);
    fm.SetValue(0, 0, f.open);
    fm.SetValue(0, 1, f.close);
    fm.SetValue(0, 2, f.high);
    fm.SetValue(0, 3, f.low);
    b.push_back(fm);

}
