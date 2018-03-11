//
//  CandleStick.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/4/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"
#include "PricePoint.hpp"
#include <iostream>

CandleStick::CandleStick(PriceTP t, MarketData::const_iterator s, MarketData::const_iterator e)
:  priceInfo { s, e }
{
    std::for_each(s, e, [](PricePoint pp)
                  {
                      std::cout << pp << std::endl;
                  });
    time = t;
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

