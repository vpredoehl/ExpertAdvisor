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

using ChartData = std::vector<CandleStick>;

class MarketData_iterator;
class Chart
{
    enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

    std::string sym;
    TimeFrame chartTF = minutely;
    MarketData::const_iterator candleStartIter, candleEndIter;
    std::vector<MarketData::const_iterator::difference_type> seqOffset;
    ChartData candles;
    
    friend std::ostream& operator<<(std::ostream &o, const Chart&);
public:
    Chart(MarketData_iterator s, MarketData_iterator e, std::chrono::minutes = std::chrono::minutes { 1 });

    MarketData::const_iterator cbegin() const    {   return candleStartIter;    }
    MarketData::const_iterator cend() const  {   return candleEndIter;  }
};

std::ostream& operator<<(std::ostream &o, const Chart&);

#endif /* Chart_hpp */
