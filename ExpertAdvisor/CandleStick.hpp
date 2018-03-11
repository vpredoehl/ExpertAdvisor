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

using days = std::chrono::duration<long, std::ratio<24 * 3600>>;
using weeks = std::chrono::duration<long, std::ratio<7 * 24 * 3600>>;

class MarketData_iterator;
class CandleStick
{
    float high, low, open, close;
    MarketData::const_iterator seqIter, endIter;
    PriceTP when;
    
    friend std::ostream& operator<<(std::ostream &o, CandleStick c);
public:
    CandleStick(PriceTP candleTime, MarketData_iterator start, MarketData_iterator end);
    
    auto closePrice() -> float   {   return close;   }
    auto operator=(float candleValue) -> float  {   return high = low = open = close = candleValue;  }
    
    auto operator!=(const CandleStick& c) const -> bool {   return high != c.high || low != c.low || open != c.open || close != c.close;   }
    operator MarketData::const_iterator() const {   return seqIter; }
};
using ChartData = std::vector<CandleStick>;

std::ostream& operator<<(std::ostream &o, CandleStick cs);

#endif /* CandleStick_hpp */
