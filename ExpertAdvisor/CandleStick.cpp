//
//  CandleStick.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/4/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"
#include <iostream>

CandleStick::CandleStick(PriceTP t, MarketPrice::const_iterator s, MarketPrice::const_iterator e)
:  priceInfo { s, e }
{
    std::for_each(s, e, [](PricePoint pp)
                  {
                      std::cout << pp << std::endl;
                  });
    time = t;
}

CandleStick::CandleStick(PriceTP t, ChartCandle::const_iterator s, ChartCandle::const_iterator e)
: priceInfo { s,e }
{
    std::for_each(s, e, [](CandleStick pp)
                  {
                      std::cout << pp << std::endl;
                  });
    time = t;
}


CandlePrice::CandlePrice(MarketPrice::const_iterator s, MarketPrice::const_iterator e)
: open { *s }, close { *(e-1) }
{
    auto mm = std::minmax_element(s, e);
    
    high = *mm.second;
    low = *mm.first;
}
inline auto FindOpenThatIsNotFiller(ChartCandle::const_iterator s, ChartCandle::const_iterator e)
{
    auto iter = std::find_if(s, e, [](CandleStick cs) -> bool {   std::cout << "open isFiller: " << (cs.isFiller ? "Yes " : "No ") << cs << std::endl;   return !cs.isFiller;  });
    return iter == e ? s : iter;
}
inline auto FindCloseThatIsNotFiller(ChartCandle::const_iterator s, ChartCandle::const_iterator e)
{
    auto beg = std::make_reverse_iterator(e);
    auto end = std::make_reverse_iterator(s);
    auto iter = std::find_if(beg, end, [](CandleStick cs) -> bool {   std::cout << "close isFiller: " << (cs.isFiller ? "Yes " : "No ") << cs << std::endl;   return !cs.isFiller;    });

    return iter == end ? beg : iter;
}
CandlePrice::CandlePrice(ChartCandle::const_iterator s, ChartCandle::const_iterator e)
:   open { FindOpenThatIsNotFiller(s, e)->priceInfo.open },
    close { FindCloseThatIsNotFiller(s, e)->priceInfo.close }
{
    auto max = std::max_element(s, e, [](const CandleStick &c1, const CandleStick &c2) -> bool
                                {
                                    // *largest < *first
                                    if(c1.isFiller) return true;
                                    if(c2.isFiller) return false;
                                    return c1.priceInfo.high < c2.priceInfo.high;
                                });
    auto min = std::min_element(s, e, [](const CandleStick &c1, const CandleStick &c2) -> bool
                                {
                                    // *first < *smallest
                                    if(c2.isFiller) return true;
                                    if(c1.isFiller) return  false;
                                    return c1.priceInfo.low < c2.priceInfo.low;
                                });
    high = max->priceInfo.high;
    low = min->priceInfo.low;
}


using std::ostream;
using std::endl;

#include <locale>
#include <iomanip>
ostream& operator<<(ostream &o, CandleStick c)
{
    o << "CandleStick: " << c.time << endl;
    o << "\tHigh: " << std::setw(10) << c.priceInfo.high << endl;
    o << "\tLow: " << std::setw(10) << c.priceInfo.low << endl;
    o << "\tOpen: " << std::setw(10) << c.priceInfo.open << endl;
    o << "\tClose: " << std::setw(10) << c.priceInfo.close << endl;
    return  o;
}

