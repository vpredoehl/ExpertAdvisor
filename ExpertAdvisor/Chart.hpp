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

class Chart
{
    enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

    std::string sym;
    TimeFrame chartTF = minutely;
    ChartData candles;
    
    friend std::ostream& operator<<(std::ostream &o, const Chart&);
public:
    Chart(const MarketData&, std::chrono::minutes = std::chrono::minutes { 1 });

    ChartData::const_iterator cbegin() const    {   return candles.cbegin();    }
    ChartData::const_iterator cend() const  {   return candles.cend();  }
    
};

std::ostream& operator<<(std::ostream &o, const Chart&);

#endif /* Chart_hpp */
