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

using PriceTP = std::chrono::time_point<std::chrono::system_clock, std::chrono::minutes>;

class PricePoint
{
    float bid, ask;
    
public:
    static std::string sym;
    PriceTP time;

    PricePoint(PriceTP, float bid, float ask);
    PricePoint() {}
    
    operator float() const  {   return bid; }
private:

    friend std::ostream& operator<<(std::ostream& o, PricePoint cs);
};
using MarketData = std::vector<PricePoint>;

std::istream& operator>>(std::istream&, PriceTP&);
std::istream& operator>>(std::istream&, PricePoint&);

std::ostream& operator<<(std::ostream&, PriceTP);
std::ostream& operator<<(std::ostream&, PricePoint);

#endif /* PricePoint_hpp */
