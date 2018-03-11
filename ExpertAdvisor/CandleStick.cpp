//
//  CandleStick.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/4/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"
#include "MarketData_iterator.hpp"
#include <iostream>

CandleStick::CandleStick(PriceTP t, MarketData_iterator s, MarketData_iterator e)
:  open(*s), close(*(e-1))
{
    auto mm = std::minmax_element(s, e);
    
    std::for_each(s, e, [](PricePoint pp)
                  {
                      std::cout << pp << std::endl;
                  });
    low = *mm.first;
    high = *mm.second;
    seqIter = s;
    when = t;
}

using std::ostream;
using std::endl;

#include <locale>
#include <iomanip>
ostream& operator<<(ostream &o, CandleStick c)
{
    o << "CandleStick: " << c.when << endl;
    o << "\tHigh: " << std::setw(10) << c.high << endl;
    o << "\tLow: " << std::setw(10) << c.low << endl;
    o << "\tOpen: " << std::setw(10) << c.open << endl;
    o << "\tClose: " << std::setw(10) << c.close << endl;
    return  o;
}

