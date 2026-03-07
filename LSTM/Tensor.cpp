//
//  Tensor.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/3/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include <iomanip>
#include <iostream>
#include <cmath>
#include <numbers>

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

void Tensor::Add(Feature f)
{
    FeatureMatrix fm(1, feature_size);

    if (!has_prev_close) {
        fm.SetValue(0, 0, 0.0f);
        fm.SetValue(0, 1, 0.0f);
        fm.SetValue(0, 2, 0.0f);
        fm.SetValue(0, 3, 0.0f);
        fm.SetValue(0, 4, 0.0f);
        fm.SetValue(0, 5, 0.0f);
        fm.SetValue(0, 6, 0.0f);
        fm.SetValue(0, 7, 0.0f);
        fm.SetValue(0, 8, 0.0f);
        fm.SetValue(0, 9, 0.0f);
        has_prev_close = true;
        prev_close = f.close;
        ds.push_back(fm);
        raw_close.push_back(f.close);
        return;
    }

    const float ref = prev_close;
    const float o = std::log(f.open  / ref) * kFeatureScale;
    const float c = std::log(f.close / ref) * kFeatureScale;
    const float h = std::log(f.high  / ref) * kFeatureScale;
    const float l = std::log(f.low   / ref) * kFeatureScale;

    fm.SetValue(0, 0, o);
    fm.SetValue(0, 1, c);
    fm.SetValue(0, 2, h);
    fm.SetValue(0, 3, l);

    const float body = c - o;
    fm.SetValue(0, 4, body);

    const float range = h - l;
    fm.SetValue(0, 5, range);

    // Rolling volatility of log returns over lookback
    double sum = 0.0, sumsq = 0.0;
    size_t count = 0;
    const size_t n = raw_close.size(); // number of prior closes
    const size_t maxPrev = (rolling_vol_lookback > 0 ? rolling_vol_lookback - 1 : 0);
    size_t startIdx = 1;
    if (n > maxPrev) startIdx = n - maxPrev;
    // accumulate previous returns: log(raw_close[j] / raw_close[j-1]) for j = startIdx..(n-1)
    for (size_t j = startIdx; j < n; ++j)
    {
        double r = std::log(static_cast<double>(raw_close[j]) / static_cast<double>(raw_close[j-1]));
        sum += r; sumsq += r * r; ++count;
    }
    // include current return
    double r_cur = std::log(static_cast<double>(f.close) / static_cast<double>(ref));
    sum += r_cur; sumsq += r_cur * r_cur; ++count;

    float vol_scaled = 0.0f;
    if (count > 1)
    {
        double mean = sum / static_cast<double>(count);
        double var = sumsq / static_cast<double>(count) - mean * mean;
        if (var < 0.0) var = 0.0;
        vol_scaled = static_cast<float>(std::sqrt(var) * kFeatureScale);
    }
    else vol_scaled = 0.0f;
    fm.SetValue(0, 6, vol_scaled);

    // Rolling cumulative log return over lookback (including current)
    double sumRet = 0.0;
    size_t countRet = 0;
    const size_t n2 = raw_close.size();
    const size_t maxPrevRet = (rolling_ret_lookback > 0 ? rolling_ret_lookback - 1 : 0);
    size_t startIdxRet = 1;
    if (n2 > maxPrevRet) startIdxRet = n2 - maxPrevRet;
    for (size_t j = startIdxRet; j < n2; ++j)
    {
        double r = std::log(static_cast<double>(raw_close[j]) / static_cast<double>(raw_close[j-1]));
        sumRet += r; ++countRet;
    }
    double r_cur2 = std::log(static_cast<double>(f.close) / static_cast<double>(ref));
    sumRet += r_cur2; ++countRet;
    float roll_ret_scaled = static_cast<float>(sumRet * kFeatureScale);
    fm.SetValue(0, 7, roll_ret_scaled);

    // Time-of-day cyclical features (sin/cos)
    constexpr double twoPi = 2 * std::numbers::pi;  // 6.28318530717958647692;
    const int cycSec = time_cycle_seconds(candle_duration);
    long long epochSec = std::chrono::duration_cast<std::chrono::seconds>(f.time.time_since_epoch()).count();
    int secInCycle = (cycSec > 0) ? static_cast<int>(epochSec % cycSec) : 0;
    double phase = (cycSec > 0) ? (twoPi * (static_cast<double>(secInCycle) / static_cast<double>(cycSec))) : 0.0;
    float sin_t = static_cast<float>(std::sin(phase));
    float cos_t = static_cast<float>(std::cos(phase));
    fm.SetValue(0, 8, sin_t);
    fm.SetValue(0, 9, cos_t);
    
    printMatrix("fm: ", fm);

    prev_close = f.close;
    ds.push_back(fm);
    raw_close.push_back(f.close);
}
float Tensor::RawCloseAtIterator(DataSet::const_iterator it) const
{
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(it >= ds.cbegin() && it < ds.cend(), "RawCloseAtIterator: iterator out of bounds");
#endif
    size_t idx = static_cast<size_t>(it - ds.cbegin());
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(idx < raw_close.size(), "RawCloseAtIterator: index out of raw_close bounds");
#endif
    return raw_close[idx];
}

