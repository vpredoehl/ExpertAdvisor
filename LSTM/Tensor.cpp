//
//  Tensor.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/3/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include <iomanip>
#include <iostream>

#include "Tensor.hpp"
#include "LSTM.hpp"
#include "PricePoint.hpp"

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

//void Tensor::Add(Feature f)
//{
//    FeatureMatrix fm(1,feature_size);
//    fm.SetValue(0, 0, f.open);
//    fm.SetValue(0, 1, f.close);
//    fm.SetValue(0, 2, f.high);
//    fm.SetValue(0, 3, f.low);
//    ds.push_back(fm);
#include <cmath>

void Tensor::Add(Feature f)
{
    FeatureMatrix fm(1, feature_size);

    if (!has_prev_close)
    {
        // first sample: no previous reference, just emit zeros
        fm.SetValue(0, 0, 0.0f);
        fm.SetValue(0, 1, 0.0f);
        fm.SetValue(0, 2, 0.0f);
        fm.SetValue(0, 3, 0.0f);
        has_prev_close = true;
        prev_close = f.close;
        ds.push_back(fm);
        return;
    }
    

    const float ref = prev_close;
    const float o = std::log(f.open  / ref);
    const float c = std::log(f.close / ref);
    const float h = std::log(f.high  / ref);
    const float l = std::log(f.low   / ref);

    fm.SetValue(0, 0, o);
    fm.SetValue(0, 1, c);
    fm.SetValue(0, 2, h);
    fm.SetValue(0, 3, l);

    prev_close = f.close;
    ds.push_back(fm);
}

