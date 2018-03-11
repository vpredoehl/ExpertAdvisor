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
#include "MarketData_iterator.hpp"
#include <chrono>
#include <vector>

using ChartData = std::vector<CandleStick>;

class MarketData_iterator;
class Chart
{
    enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

    std::string sym;
    TimeFrame chartTF = minutely;
    MarketData::const_iterator candleStartIter, candleEndIter;
    NextCandleOffset seqOffset;
    ChartData candles;
    
    friend std::ostream& operator<<(std::ostream &o, const Chart&);
public:
    Chart(MarketData_iterator s, MarketData_iterator e, std::chrono::minutes = std::chrono::minutes { 1 });

    auto cbegin() const -> MarketData_iterator  {   return MarketData_iterator(candleStartIter, MarketData_iterator::OffsetPair(&seqOffset, seqOffset.cbegin())); }
    auto cend() const -> MarketData_iterator    {   return candleEndIter;   }

    auto operator==(const Chart& ch) const -> bool;
};
inline auto operator!=(const Chart& c1, const Chart& c2) -> bool {   return !(c1 == c2);  }

std::ostream& operator<<(std::ostream &o, const Chart&);

#endif /* Chart_hpp */
