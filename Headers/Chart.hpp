//
//  Chart.hpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/5/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#ifndef Chart_hpp
#define Chart_hpp

#include "CandleStick.hpp"
#include <chrono>
#include <vector>


class Chart
{
    enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

    std::string sym;
    TimeFrame chartTF = minutely;
    ChartCandle candles;
    
    friend std::ostream& operator<<(std::ostream &o, const Chart&);
public:
    template<class ForwardIter>
    Chart(ForwardIter s, ForwardIter e, std::chrono::minutes);
    
    auto cbegin() const -> ChartCandle::const_iterator    {   return candles.cbegin();    }
    auto cend() const -> ChartCandle::const_iterator  {   return candles.cend();  }
    void clear() noexcept   {   candles.clear();    }

    auto operator==(const Chart& ch) const -> bool;
};
inline auto operator!=(const Chart& c1, const Chart& c2) -> bool {   return !(c1 == c2);  }

std::ostream& operator<<(std::ostream &o, const Chart&);

template<class ForwardIter>
Chart::Chart(ForwardIter startIter, ForwardIter endIter, std::chrono::minutes dur)
{
    auto candleEndIter = endIter;
    auto startTime = startIter->time;
    auto endTime = startIter->time + dur;
    auto TimeNotInCandle = [&endTime](auto &pp) -> bool {   return pp.time >= endTime;  };
    float lastPrice = 0;
    
    endIter = startIter;
    do
    {
        CandleStick cs(startTime, startIter, endIter = std::find_if(startIter, candleEndIter, TimeNotInCandle));
        
        if(startIter == endIter)   cs = lastPrice;
        startIter = endIter;
        startTime = endTime;
        endTime += dur;
        lastPrice = cs.closePrice();
        candles.push_back(cs);
    } while (endIter != candleEndIter);
}

#endif /* Chart_hpp */
