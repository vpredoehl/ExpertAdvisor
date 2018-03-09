//
//  Chart.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/5/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "Chart.hpp"
#include "MarketData_iterator.hpp"
#include <iostream>

using namespace std::chrono;

Chart::Chart(MarketData_iterator startIter, MarketData_iterator endIter, minutes dur)
: candleStartIter(startIter), candleEndIter(endIter)
{
    auto startTime = startIter->time;
    auto endTime = startIter->time + dur;
    auto TimeNotInCandle = [&endTime](const PricePoint &pp) -> bool
    {
        std::cout << pp << std::endl;
        return pp.time >= endTime;
    };
    float lastPrice = 0;
    
    endIter = startIter;
    do
    {
        CandleStick cs(startTime, startIter, endIter = std::find_if(startIter, MarketData_iterator(candleEndIter), TimeNotInCandle));
        
        if(startIter == endIter)   cs = lastPrice;
        std::cout << cs << std::endl;
        seqOffset.push_back(endIter - startIter);
        startIter = endIter;
        startTime = endTime;
        endTime += dur;
        lastPrice = cs.closePrice();
        candles.push_back(cs);
    } while (endIter != candleEndIter);
}

using std::ostream;

ostream& operator<<(ostream &o, const Chart &ch)
{
    using std::endl;
    o << "Chart: " << ch.sym << "  TimeFrame: ";
    switch (ch.chartTF) {
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
    std::for_each(ch.candles.cbegin(), ch.candles.cend(), [&o](CandleStick c)
                  {
                      o << c << std::endl;
                  });
    return o;
}
