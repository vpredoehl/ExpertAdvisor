//
//  Chart.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/5/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "PricePoint.hpp"
#include "Chart.hpp"
#include <iostream>

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
    auto closeSum = 0.0;
    std::for_each(ch.candles.cbegin(), ch.candles.cend(), [&o, &closeSum](CandleStick c)
                  {
                      closeSum += c.closePrice();
                      o << c << std::endl;
                  });
    o << "close sum: " << closeSum << std::endl;
    return o;
}

auto Chart::operator==(const Chart& ch) const -> bool
{
    using ForwardIter = ChartCandle::const_iterator;

    for(ForwardIter c1 = candles.cbegin(), c2 = ch.candles.cbegin();
        c1 != candles.cend() && c2 != ch.candles.cend();    ++c1, ++c2)
            if(*c1 != *c2)  return false;
    return true;
}
