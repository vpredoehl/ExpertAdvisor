//
//  Chart.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/5/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "Chart.hpp"
#include <iostream>

using namespace std::chrono;

Chart::Chart(const MarketData &priceD, minutes candleDuration)
{
    auto candleStartIter = priceD.cbegin(), candleEndIter = candleStartIter;
    auto candleStartTime = candleStartIter->time;
    auto candleEndTime = candleStartIter->time + candleDuration;
    auto TimeNotInCandle = [&candleEndTime](const PricePoint &pp) -> bool    {   return pp.time >= candleEndTime;   };
    
    float lastCandlePrice = 0;
    do
    {
        CandleStick cs(candleStartTime, candleStartIter, candleEndIter = std::find_if(candleStartIter, priceD.cend(), TimeNotInCandle));
        
        if(candleStartIter == candleEndIter)   cs = lastCandlePrice;
        candleStartIter = candleEndIter;
        candleStartTime = candleEndTime;
        candleEndTime += candleDuration;
        lastCandlePrice = cs.closePrice();
        candles.push_back(cs);
    } while (priceD.end() != candleEndIter);
}

using std::ostream;

ostream& operator<<(ostream &o, const Chart &c)
{
    using std::endl;
    o << "Chart: " << c.sym << "  TimeFrame: ";
    switch (c.chartTF) {
        case Chart::minutely:
            o << "minute" << endl;
            break;
        case Chart::hourly:
            o << "hourly" << endl;
            break;
        case Chart::daily:
            o << "daily" << endl;
            break;
        case Chart::weekly:
            o << "weekly" << endl;
            break;
        case Chart::monthly:
            o << "monthly" << endl;
    }
    std::for_each(c.cbegin(), c.cend(), [&o](CandleStick c)
                  {
                      o << c << std::endl;
                  });
    return o;
}
