//
//  CandleStick.hpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/1/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#ifndef CandleStick_hpp
#define CandleStick_hpp

#include <chrono>
#include <string>

using namespace std::chrono;

enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

using days = duration<long, std::ratio<24 * 3600>>;
using weeks = duration<long, std::ratio<7 * 24 * 3600>>;
using CandleTime = time_point<system_clock, minutes>;


class PricePoint
{
    float bid, ask;
    
public:
    
    static std::string sym;
    
    PricePoint(TimeFrame, CandleTime, float bid, float ask);
    PricePoint() {}
    
private:
    CandleTime time;
    TimeFrame frame;
    
    friend std::ostream& operator<<(std::ostream& o, PricePoint cs);
};
    
std::istream& operator>>(std::istream&, CandleTime&);
std::istream& operator>>(std::istream&, PricePoint&);

std::ostream& operator<<(std::ostream&, CandleTime);
std::ostream& operator<<(std::ostream&, PricePoint);

#endif /* CandleStick_hpp */
