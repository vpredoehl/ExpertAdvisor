//
//  PricePoint.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/1/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "PricePoint.hpp"
#include <iomanip>

using namespace std::chrono;
using std::string;
using std::istream;
using std::ostream;
using std::endl;
using std::setw;
using std::showbase;
using std::put_money;

istream& operator>>(istream& i, PriceTP &t)
{
    struct tm ct;
    
    i >> std::get_time(&ct, "%F %T:%S");
    t = time_point_cast<seconds>(std::chrono::system_clock::from_time_t(mktime(&ct)));
    return i;
}


ostream& operator<<(ostream& o, PriceTP t)
{
    auto tt = system_clock::to_time_t(t);
    o << std::put_time(localtime(&tt), "%F %R:%S");
    return o;
}
ostream& operator<<(ostream& o, PricePoint pp)
{
    o << "\tTime: " << pp.time << "\tBid: " << setw(10) << pp.bid << "\tAsk: " << setw(10) << pp.ask;
    return o;
}
ostream& operator<<(ostream& o, CandlestickRow row)
{
    o << "\tTime: " << row.time << "\tOpen: " << setw(10) << row.open << "\tClose: " << setw(10) << row.close << "\tHigh: " << setw(10) << row.high << "\tLow: " << setw(10) << row.low;
    return o;
}
