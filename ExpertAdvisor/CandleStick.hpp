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
#include <vector>

enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

using MarketData = std::vector<PricePoint>;

using days = std::chrono::duration<long, std::ratio<24 * 3600>>;
using weeks = std::chrono::duration<long, std::ratio<7 * 24 * 3600>>;
using ChartType = std::pair<std::string, TimeFrame>;

class CandleStick
{
    float high, low, open, close;
    MarketData::const_iterator seqIter;
    PriceTP when;
    
    friend std::ostream& operator<<(std::ostream &o, CandleStick c);
public:
    CandleStick(PriceTP candleTime, MarketData::const_iterator start, MarketData::const_iterator end);
    
    auto closePrice() -> float   {   return close;   }
    auto operator=(float candleValue) -> float  {   return high = low = open = close = candleValue;  }
};

std::ostream& operator<<(std::ostream &o, CandleStick cs);

#endif /* CandleStick_hpp */
