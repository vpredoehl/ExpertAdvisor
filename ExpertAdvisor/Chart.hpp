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
    MarketData::const_iterator candleStartIter, candleEndIter;
    ChartData candles;
    
    friend std::ostream& operator<<(std::ostream &o, const Chart&);
public:
    Chart(MarketData::const_iterator s, MarketData::const_iterator e, std::chrono::minutes = std::chrono::minutes { 1 });

//    auto cbegin() const -> MarketData_iterator  {   return candles.cbegin(); }
//    auto cend() const -> MarketData_iterator    {   return candles.cend();   }

    auto operator==(const Chart& ch) const -> bool;
};
inline auto operator!=(const Chart& c1, const Chart& c2) -> bool {   return !(c1 == c2);  }

std::ostream& operator<<(std::ostream &o, const Chart&);

#endif /* Chart_hpp */
