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


using days = std::chrono::duration<long, std::ratio<24 * 3600>>;
using weeks = std::chrono::duration<long, std::ratio<7 * 24 * 3600>>;

class CandleStick
{
    CandlePrice priceInfo;
    PriceTP time;

    friend class Chart;
    friend std::ostream& operator<<(std::ostream &o, CandleStick c);
public:
    CandleStick(PriceTP candleTime, MarketData::const_iterator start, MarketData::const_iterator end);
    
    auto closePrice() -> float   {   return priceInfo.close;   }
    auto operator=(float candleValue) -> float  {   return priceInfo = candleValue;  }
    
    auto operator!=(const CandleStick& c) const -> bool {   return priceInfo == c.priceInfo;   }
};
using ChartData = std::vector<CandleStick>;

std::ostream& operator<<(std::ostream &o, CandleStick cs);

#endif /* CandleStick_hpp */
