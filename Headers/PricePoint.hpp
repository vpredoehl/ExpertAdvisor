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

using PriceTP = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;
struct PricePoint
{
    PriceTP time;
    float bid, ask;

    auto operator<(PricePoint pp) const -> bool  {   return ask < pp.ask; }
    auto operator==(PricePoint pp) const -> bool    {   return bid == pp.bid && ask == pp.ask && time == pp.time;   }

    operator float() const  {   return ask; }
private:

    friend std::ostream& operator<<(std::ostream& o, PricePoint cs);
};

struct CandlestickRow
{
    float open, close, high, low;
    PriceTP time;
    
    
};

#include <list>
using RawMarketPrice = std::list<PricePoint>;


std::istream& operator>>(std::istream&, PriceTP&);
std::istream& operator>>(std::istream&, PricePoint&);

std::ostream& operator<<(std::ostream&, PriceTP);
std::ostream& operator<<(std::ostream&, PricePoint);
std::ostream& operator<<(std::ostream&, CandlestickRow);
std::ostream& operator<<(std::ostream&, RawMarketPrice);



#endif /* PricePoint_hpp */
