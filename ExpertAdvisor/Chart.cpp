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

Chart::Chart(MarketData::const_iterator startIter, MarketData::const_iterator endIter, minutes dur)
: candleStartIter(startIter), candleEndIter(endIter)
{
    auto startTime = startIter->time;
    auto endTime = startIter->time + dur;
    auto TimeNotInCandle = [&endTime](const PricePoint &pp) -> bool {   return pp.time >= endTime;  };
    float lastPrice = 0;
    
    endIter = startIter;
    do
    {
        CandleStick cs(startTime, startIter, endIter = std::find_if(startIter, candleEndIter, TimeNotInCandle));
        
        if(startIter == endIter)   cs = lastPrice;
        std::cout << cs << std::endl;
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

auto Chart::operator==(const Chart& ch) const -> bool
{
//    using ForwardIter = ChartData::const_iterator;
//    ForwardIter c2;
//
//    for(ForwardIter c1 = candles.cbegin(), c2 = ch.candles.cbegin();
//            c1 != candles.cend() && c2 != ch.candles.cend();    ++c1, ++c2)
//    {
//        std::cout << "Comparing: " << *c1 << std::endl << *c2 << std::endl;
//        if(*c1 != *c2)    return false;
//    }
    return true;
}
