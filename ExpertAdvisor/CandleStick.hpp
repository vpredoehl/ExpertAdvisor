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

class CandleStick
{
    float bid, ask;
    
public:
    using CandleTime = time_point<system_clock, minutes>;
    enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };
    
    CandleStick(TimeFrame, CandleTime, std::string sym, float bid, float ask);
    CandleStick() {}
    
private:
    CandleTime time;
    TimeFrame frame;
    
    friend std::ostream& operator<<(std::ostream& o, CandleStick cs);
};
    
std::istream& operator>>(std::istream&, CandleStick::CandleTime&);
std::istream& operator>>(std::istream&, CandleStick&);

std::ostream& operator<<(std::ostream&, CandleStick::CandleTime);
std::ostream& operator<<(std::ostream&, CandleStick);

#endif /* CandleStick_hpp */
