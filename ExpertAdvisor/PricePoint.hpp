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
    
    auto operator<(PricePoint pp) const -> bool  {   return bid < pp.bid; }
    auto operator==(PricePoint pp) const -> bool    {   return bid == pp.bid && ask == pp.ask;   }

    operator float() const  {   return bid; }
private:

    friend std::ostream& operator<<(std::ostream& o, PricePoint cs);
};

#include <vector>
using MarketData = std::vector<PricePoint>;
using NextCandleOffset = std::vector<MarketData::const_iterator::difference_type>;

struct CandlePrice
{
    float high, low, open, close;

    CandlePrice(PricePoint p)    {   operator=(static_cast<float>(p));  }
    CandlePrice(MarketData::const_iterator, MarketData::const_iterator);
    CandlePrice(float f) {   operator=(f);   }
    
    auto operator=(float candleValue) -> float    {  return high = low = open = close = candleValue;   }
    operator float() const  {   return close;   }
    auto operator==(CandlePrice c) const -> bool    {   return high != c.high || low != c.low || open != c.open || close != c.close; }
};

std::istream& operator>>(std::istream&, PriceTP&);
std::istream& operator>>(std::istream&, PricePoint&);

std::ostream& operator<<(std::ostream&, PriceTP);
std::ostream& operator<<(std::ostream&, PricePoint);

#endif /* PricePoint_hpp */
