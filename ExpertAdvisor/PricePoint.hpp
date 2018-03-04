//
//  PricePoint.hpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/1/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#ifndef PricePoint_hpp
#define PricePoint_hpp

#include <chrono>
#include <string>

using namespace std::chrono;

enum TimeFrame : char   {   minutely, hourly, daily, weekly, monthly  };

using days = duration<long, std::ratio<24 * 3600>>;
using weeks = duration<long, std::ratio<7 * 24 * 3600>>;
using PriceTime = time_point<system_clock, minutes>;


class PricePoint
{
    float bid, ask;
    
public:
    
    static std::string sym;
    
    PricePoint(TimeFrame, PriceTime, float bid, float ask);
    PricePoint() {}
    
private:
    PriceTime time;
    TimeFrame frame;
    
    friend std::ostream& operator<<(std::ostream& o, PricePoint cs);
};
    
std::istream& operator>>(std::istream&, PriceTime&);
std::istream& operator>>(std::istream&, PricePoint&);

std::ostream& operator<<(std::ostream&, PriceTime);
std::ostream& operator<<(std::ostream&, PricePoint);

#endif /* PricePoint_hpp */
