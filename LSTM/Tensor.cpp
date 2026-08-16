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
#include <ctime>
#include <numbers>

#include "Tensor.hpp"
#include "LSTM.hpp"
#include "PricePoint.hpp"
#include "DonchianFeatures.hpp"
#include "SessionPhaseFeatures.hpp"
#include "CausalReturnSurpriseFeatures.hpp"
#include "CausalVolatilityRegimeFeatures.hpp"

using std::setw;

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() __attribute__((weak_import));

namespace
{
bool RuntimeDiagnosticLoggingEnabled()
{
    return (LstmRuntimeDiagnosticLoggingEnabled == nullptr) ||
           LstmRuntimeDiagnosticLoggingEnabled();
}
}

std::ostream& operator<<(std::ostream& o, Window w)
{
    for (const auto& f : w)
    {
        float open = f.Shape()[0];
        float close = f.Shape()[0];
        float high = f.Shape()[0];
        float low = f.Shape()[0];
        o <<  setw(10) << open << setw(10) << close << setw(10) << high <<  setw(10) << low << std::endl;
    }
    return o;
}

//std::vector<float> Tensor::rolling_mean(const std::vector<float>& data, size_t window)
//{
//    float sum = 0.0f;
//    std::vector<float> result(data.size(), 0.0f);
//
//    if (window == 0 || data.empty())    return result;
//
//
//    for (size_t i = 0; i < data.size(); ++i)
//    {
//        sum += data[i];
//
//        if (i >= window)    sum -= data[i - window];
//
//        if (i >= window - 1)    result[i] = sum / window;
//        else                    result[i] = sum / (i + 1);  // partial window at start
//    }
//
//    return result;
//}
//
//float Tensor::rolling_mean_at(const std::vector<float>& data, size_t idx, size_t window)
//{
//    size_t start = (idx >= window - 1) ? idx - window + 1 : 0;
//    float sum = 0.0f;
//    size_t count = 0;
//
//#pragma openmp parallel for
//    for (size_t i = start; i <= idx; ++i)   {   sum += data[i]; count++;    }
//    return sum / count;
//}

void Tensor::Add(Feature f)
{
    static size_t s_featureRangeGuardDiagCount = 0;
    constexpr size_t kFeatureRangeGuardDiagLimit = 50;
    constexpr float kMinRealisticFxRangeRaw = 1.0e-4f; // 1 pip floor for 15m FX bars

    // Reduce reallocations by reserving capacity in chunks
    if (ds.size() == ds.capacity()) ds.reserve(ds.size() + 4096);
    if (raw_open.size() == raw_open.capacity())     raw_open.reserve(raw_open.size() + 4096);
    if (raw_close.size() == raw_close.capacity())   raw_close.reserve(raw_close.size() + 4096);
    if (raw_high.size() == raw_high.capacity())     raw_high.reserve(raw_high.size() + 4096);
    if (raw_low.size() == raw_low.capacity())       raw_low.reserve(raw_low.size() + 4096);
    if (raw_time.size() == raw_time.capacity())     raw_time.reserve(raw_time.size() + 4096);

    FeatureMatrix fm(1, feature_size);
    // Calculate before retaining this completed bar so the appended causal
    // features reference only their up-to-32 predecessors.
    const float relativeVolume = relativeTickVolume.AddCompletedBar(f.tickVolume);
    const float returnSurprise = causalReturnSurprise.AddCompletedClose(f.close);
    const float volatilityRegime = causalVolatilityRegime.AddCompletedClose(f.close);

    if (!has_prev_close) {
        auto low = MetaNN::LowerAccess(fm);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + feature_size, 0.0f);
        const auto [sessionPhaseSin, sessionPhaseCos] =
            ComputeUtcSessionPhase(f.time);
        low.MutableRawMemory()[sessionPhaseSinCol] = sessionPhaseSin;
        low.MutableRawMemory()[sessionPhaseCosCol] = sessionPhaseCos;
        low.MutableRawMemory()[relativeTickVolumeCol] = relativeVolume;
        low.MutableRawMemory()[causalReturnSurpriseCol] = returnSurprise;
        low.MutableRawMemory()[causalVolatilityRegimeCol] = volatilityRegime;
        has_prev_close = true;
        prev_close = f.close;
        ds.push_back(std::move(fm));
        raw_open.push_back(f.open);
        raw_close.push_back(f.close);
        raw_high.push_back(f.high);
        raw_low.push_back(f.low);
        raw_time.push_back(f.time);

        // Initialize EMA baselines on first sample
        has_ema = true;
        ema8 = f.close;
        ema21 = f.close;
        ema50 = f.close;

        return;
    }

    const float ref = prev_close;
    const float o = std::log(f.open  / ref) * kFeatureScale;
    const float c = std::log(f.close / ref) * kFeatureScale;
    const float h = std::log(f.high  / ref) * kFeatureScale;
    const float l = std::log(f.low   / ref) * kFeatureScale;

    auto low = MetaNN::LowerAccess(fm);
    float* p = low.MutableRawMemory();
    p[0] = o;
    p[1] = c;
    p[2] = h;
    p[3] = l;

    const float body = c - o;
    p[4] = body;

    const float range =  h - l;
    p[5] = range;

    // Causal Donchian-20 distances. raw_high/raw_low contain prior rows at
    // this point, so the current bar cannot affect either extrema.
    const auto [donchianUp, donchianDown] = ComputeCausalDonchian(
        raw_high, raw_low, f.close, kFeatureScale, donchianLookback);
    if (donchian20Mode == Donchian20Mode::Enabled)
    {
        p[donchianUpCol] = donchianUp;
        p[donchianDownCol] = donchianDown;
    }
    else
    {
        p[donchianUpCol] = 0.0f;
        p[donchianDownCol] = 0.0f;
    }

    // Range expansion: (high - low) / avg_range, use rolling mean of raw ranges
    const float raw_range = f.high - f.low;
    const float avg_range = rangeMean.update(raw_range);
    const float range_expansion = (avg_range > 1e-12f ? raw_range / avg_range : 0.0f);
    p[31] = range_expansion;

    // Candle body strength: (close - open) / (high - low) in scaled log space
    const float body_strength = (range != 0.0f ? (c - o) / range : 0.0f);
    p[30] = body_strength;

    const float denom = std::max(range, 1e-6f);

    // Update EMAs on raw close
    if (!has_ema) {
        has_ema = true;
        ema8 = prev_close;
        ema21 = prev_close;
        ema50 = prev_close;
    }
    const float ema8_prev  = ema8;
    const float ema21_prev = ema21;
    const float ema50_prev = ema50;

    const float alpha8 = 2.0f / (8.0f + 1.0f);
    const float alpha21 = 2.0f / (21.0f + 1.0f);
    const float alpha50 = 2.0f / (50.0f + 1.0f);
    ema8  = alpha8  * f.close + (1.0f - alpha8)  * ema8;
    ema21 = alpha21 * f.close + (1.0f - alpha21) * ema21;
    ema50 = alpha50 * f.close + (1.0f - alpha50) * ema50;

    // Update ATR(14) on raw prices
    const float tr = std::max({ f.high - f.low, std::fabs(f.high - prev_close), std::fabs(f.low - prev_close) });
    const float alphaATR = 1.0f / 14.0f;
    if (!has_atr) { has_atr = true; atr14 = tr; }
    else { atr14 = alphaATR * tr + (1.0f - alphaATR) * atr14; }

    const float upper_wick = (h - std::max(o, c)) / denom;
    p[12] = upper_wick;

    const float lower_wick = (std::min(o, c) - l) / denom;
    p[13] = lower_wick;

    // EMA-derived features: normalized distance (scaled log space) by current candle range
    const float ema8_s  = std::log(ema8  / ref) * kFeatureScale;
    const float ema21_s = std::log(ema21 / ref) * kFeatureScale;
    const float ema50_s = std::log(ema50 / ref) * kFeatureScale;
    const float denom_range_old = std::max(range, 1e-6f);
    const float typical_raw_range = std::max(std::max(avg_range, atr14), 0.0f);
    const float fallback_raw_range = std::max(kMinRealisticFxRangeRaw, 0.25f * typical_raw_range);
    const float denom_range =
        (std::isfinite(ref) && ref > 0.0f)
            ? std::max(denom_range_old,
                       std::fabs(std::log((ref + fallback_raw_range) / ref) * kFeatureScale))
            : denom_range_old;
    const float col14_before = (c - ema8_s)  / denom_range_old;
    const float col15_before = (c - ema21_s) / denom_range_old;
    const float col16_before = (c - ema50_s) / denom_range_old;
    const float col17_before = (ema8_s  - ema21_s) / denom_range_old;
    const float col18_before = (ema21_s - ema50_s) / denom_range_old;
    const float col14_after = (c - ema8_s)  / denom_range;
    const float col15_after = (c - ema21_s) / denom_range;
    const float col16_after = (c - ema50_s) / denom_range;
    const float col17_after = (ema8_s  - ema21_s) / denom_range;
    const float col18_after = (ema21_s - ema50_s) / denom_range;
    p[14] = col14_after;
    p[15] = col15_after;
    p[16] = col16_after;

    // EMA slope (log space) normalized by current candle range
    const float slopeLog8  = std::log(std::max(ema8,  1e-12f) / std::max(ema8_prev,  1e-12f)) * kFeatureScale;
    const float slopeLog21 = std::log(std::max(ema21, 1e-12f) / std::max(ema21_prev, 1e-12f)) * kFeatureScale;
    const float slopeLog50 = std::log(std::max(ema50, 1e-12f) / std::max(ema50_prev, 1e-12f)) * kFeatureScale;
    const float col24_before = slopeLog8  / denom_range_old;
    const float col25_before = slopeLog21 / denom_range_old;
    const float col26_before = slopeLog50 / denom_range_old;
    const float col24_after = slopeLog8  / denom_range;
    const float col25_after = slopeLog21 / denom_range;
    const float col26_after = slopeLog50 / denom_range;
    p[24] = col24_after;
    p[25] = col25_after;
    p[26] = col26_after;

    // EMA spread features normalized by current candle range (scaled log space)
    p[17] = col17_after;
    p[18] = col18_after;

    const bool featureRangeGuardTriggered = denom_range > denom_range_old;
    if (featureRangeGuardTriggered &&
        RuntimeDiagnosticLoggingEnabled() &&
        s_featureRangeGuardDiagCount < kFeatureRangeGuardDiagLimit)
    {
        std::cout << "DIAG_FEATURE_RANGE_GUARD"
                  << ",row=" << ds.size()
                  << ",dt=" << f.time
                  << ",range=" << range
                  << ",denom_range_old=" << denom_range_old
                  << ",denom_range_new=" << denom_range
                  << ",avg_range_raw=" << avg_range
                  << ",atr14_raw=" << atr14
                  << ",fallback_raw_range=" << fallback_raw_range
                  << ",cols=14|15|16|17|18|24|25|26"
                  << ",before=14:" << col14_before
                  << "|15:" << col15_before
                  << "|16:" << col16_before
                  << "|17:" << col17_before
                  << "|18:" << col18_before
                  << "|24:" << col24_before
                  << "|25:" << col25_before
                  << "|26:" << col26_before
                  << ",after=14:" << col14_after
                  << "|15:" << col15_after
                  << "|16:" << col16_after
                  << "|17:" << col17_after
                  << "|18:" << col18_after
                  << "|24:" << col24_after
                  << "|25:" << col25_after
                  << "|26:" << col26_after
                  << std::endl;
        ++s_featureRangeGuardDiagCount;
    }

    // ATR-normalized EMA distance features in raw price space
    const float denom_atr = std::max(atr14, 1e-12f);
    p[19] = (f.close - ema8)  / denom_atr;
    p[20] = (f.close - ema21) / denom_atr;
    p[21] = (f.close - ema50) / denom_atr;

    // ATR-normalized EMA spread features in raw price space
    p[22] = (ema8  - ema21) / denom_atr;
    p[23] = (ema21 - ema50) / denom_atr;

    // EMA slope in raw price space normalized by ATR
    p[27] = (ema8  - ema8_prev)  / denom_atr;
    p[28] = (ema21 - ema21_prev) / denom_atr;
    p[29] = (ema50 - ema50_prev) / denom_atr;

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
    p[6] = vol_scaled;

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
    p[7] = roll_ret_scaled;

    // Time-of-day cyclical features (sin/cos)
    constexpr double twoPi = 2 * std::numbers::pi;  // 6.28318530717958647692;
    const int cycSec = time_cycle_seconds(candle_duration);
    long long epochSec = std::chrono::duration_cast<std::chrono::seconds>(f.time.time_since_epoch()).count();
    int secInCycle = (cycSec > 0) ? static_cast<int>(epochSec % cycSec) : 0;
    double phase = (cycSec > 0) ? (twoPi * (static_cast<double>(secInCycle) / static_cast<double>(cycSec))) : 0.0;
    float sin_t = static_cast<float>(std::sin(phase));
    float cos_t = static_cast<float>(std::cos(phase));
    p[8] = sin_t;
    p[9] = cos_t;

    // True intraday phase of the canonical UTC candle timestamp. Keep the
    // historical 15-minute-cycle channels above unchanged for old models.
    const auto [sessionPhaseSin, sessionPhaseCos] = ComputeUtcSessionPhase(f.time);
    p[sessionPhaseSinCol] = sessionPhaseSin;
    p[sessionPhaseCosCol] = sessionPhaseCos;
    p[relativeTickVolumeCol] = relativeVolume;
    p[causalReturnSurpriseCol] = returnSurprise;
    p[causalVolatilityRegimeCol] = volatilityRegime;

    // Day-of-week cyclical features (sin/cos)
    const int weekSec = 7 * 24 * 60 * 60;
    auto tp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(f.time);
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    gmtime_r(&t, &tm);
    int secOfWeek = tm.tm_wday * 24 * 60 * 60 + tm.tm_hour * 60 * 60 + tm.tm_min * 60 + tm.tm_sec;
    double phase_w = twoPi * (static_cast<double>(secOfWeek) / static_cast<double>(weekSec));
    float sin_w = static_cast<float>(std::sin(phase_w));
    float cos_w = static_cast<float>(std::cos(phase_w));
    p[10] = sin_w;
    p[11] = cos_w;
    
    prev_close = f.close;
    ds.push_back(std::move(fm));
    raw_open.push_back(f.open);
    raw_close.push_back(f.close);
    raw_high.push_back(f.high);
    raw_low.push_back(f.low);
    raw_time.push_back(f.time);
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

float Tensor::RawHighAtIterator(DataSet::const_iterator it) const
{
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(it >= ds.cbegin() && it < ds.cend(), "RawHighAtIterator: iterator out of bounds");
#endif
    size_t idx = static_cast<size_t>(it - ds.cbegin());
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(idx < raw_high.size(), "RawHighAtIterator: index out of raw_high bounds");
#endif
    return raw_high[idx];
}

float Tensor::RawLowAtIterator(DataSet::const_iterator it) const
{
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(it >= ds.cbegin() && it < ds.cend(), "RawLowAtIterator: iterator out of bounds");
#endif
    size_t idx = static_cast<size_t>(it - ds.cbegin());
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(idx < raw_low.size(), "RawLowAtIterator: index out of raw_low bounds");
#endif
    return raw_low[idx];
}

float Tensor::RawOpenAtIterator(DataSet::const_iterator it) const
{
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(it >= ds.cbegin() && it < ds.cend(), "RawOpenAtIterator: iterator out of bounds");
#endif
    size_t idx = static_cast<size_t>(it - ds.cbegin());
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(idx < raw_open.size(), "RawOpenAtIterator: index out of raw_open bounds");
#endif
    return raw_open[idx];
}

PriceTP Tensor::RawTimeAtIterator(DataSet::const_iterator it) const
{
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(it >= ds.cbegin() && it < ds.cend(), "RawTimeAtIterator: iterator out of bounds");
#endif
    size_t idx = static_cast<size_t>(it - ds.cbegin());
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(idx < raw_time.size(), "RawTimeAtIterator: index out of raw_time bounds");
#endif
    return raw_time[idx];
}
