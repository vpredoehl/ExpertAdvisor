//
//  CandleStick.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/4/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"
#include <iostream>

CandleStick::CandleStick(PriceTP t, MarketData::const_iterator s, MarketData::const_iterator e)
:  open(*s), close(*(e-1))
{
    std::for_each(s, e, [this,s,e](PricePoint pp)
                  {
                      std::cout << pp << std::endl;
                      high = *std::max_element(s,e);
                      low = *std::min_element(s,e);
                  });
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


    //    switch (cs.frame) {
    //        case minutely:
    //            o << "minute" << endl;
    //            break;
    //        case hourly:
    //            o << "hourly" << endl;
    //            break;
    //        case daily:
    //            o << "daily" << endl;
    //            break;
    //        case weekly:
    //            o << "weekly" << endl;
    //            break;
    //        case monthly:
    //            o << "monthly" << endl;
    //    }
