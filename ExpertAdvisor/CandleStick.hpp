//
//  CandleStick.hpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/4/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#ifndef CandleStick_hpp
#define CandleStick_hpp

#include "PricePoint.hpp"
#include <iostream>
#include <chrono>
#include <vector>

class CandleStick;
using days = std::chrono::duration<long, std::ratio<24 * 3600>>;
using weeks = std::chrono::duration<long, std::ratio<7 * 24 * 3600>>;
using ChartCandle = std::vector<CandleStick>;


struct CandlePrice
{
    float high, low, open, close;
    
    CandlePrice(PricePoint p)    {   operator=(static_cast<float>(p));  }
    CandlePrice(RawMarketPrice::const_iterator, RawMarketPrice::const_iterator);
    CandlePrice(ChartCandle::const_iterator, ChartCandle::const_iterator);
    
    CandlePrice(float f) {   operator=(f);   }
    
    auto operator=(float candleValue) -> float    {  return high = low = open = close = candleValue;   }
    operator float() const  {   return close;   }
    auto operator!=(CandlePrice c) const -> bool    {   return high != c.high || low != c.low || open != c.open || close != c.close; }
};

class CandleStick
{
    CandlePrice priceInfo;
    PriceTP time;
    bool isFiller = false;  // no activity during candle time period

    friend class Chart;
    friend class CandlePrice;
    friend auto FindOpenThatIsNotFiller(ChartCandle::const_iterator s, ChartCandle::const_iterator e);
    friend auto FindCloseThatIsNotFiller(ChartCandle::const_iterator s, ChartCandle::const_iterator e);
    friend std::ostream& operator<<(std::ostream &o, CandleStick c);
public:
    CandleStick(PriceTP candleTime, RawMarketPrice::const_iterator start, RawMarketPrice::const_iterator end)
        :   priceInfo { start, end }   {   time = candleTime;   }
    CandleStick(PriceTP candleTime, ChartCandle::const_iterator start, ChartCandle::const_iterator end);

    auto closePrice() -> float   {   return priceInfo.close;   }
    auto operator=(float candleValue) -> float  {   isFiller = true;   return priceInfo = candleValue;  }
    
    auto operator!=(const CandleStick& c) const -> bool {   return priceInfo != c.priceInfo;   }
};

std::ostream& operator<<(std::ostream &o, CandleStick cs);

#endif /* CandleStick_hpp */
