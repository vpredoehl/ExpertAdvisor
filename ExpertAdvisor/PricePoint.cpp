//
//  PricePoint.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/1/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "PricePoint.hpp"

#include <iomanip>


using std::string;

string PricePoint::sym;

PricePoint::PricePoint(TimeFrame tf, PriceTime ct, float bid, float ask)
{
    frame = tf;  time = ct;
    PricePoint::bid = bid; PricePoint::ask = ask;
}

using std::istream;
using std::ostream;
using std::endl;

istream& operator>>(istream& i, PriceTime &t)
{
    struct tm ct;
    
    i >> std::get_time(&ct, "%F %R:%S");
    t = time_point_cast<minutes>(std::chrono::system_clock::from_time_t(mktime(&ct)));
    return i;
}

istream& operator>>(istream& i, PricePoint &t)
{
    using std::getline;
    PriceTime ct;
    float bid,ask;
    
    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    getline(i, PricePoint::sym, ',');
    i >> ct;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i >> bid;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i >> ask;
    t = PricePoint { minutely, ct, bid, ask };
    return i;
}

ostream& operator<<(ostream& o, PriceTime t)
{
    auto tt = system_clock::to_time_t(t);
    o << std::put_time(localtime(&tt), "%F %R:%S");
    return o;
}
ostream& operator<<(ostream& o, PricePoint cs)
{
    o << "Symbol: " << PricePoint::sym << "\tChart: ";
    switch (cs.frame) {
        case minutely:
            o << "minute" << endl;
            break;
        case hourly:
            o << "hourly" << endl;
            break;
        case daily:
            o << "daily" << endl;
            break;
        case weekly:
            o << "weekly" << endl;
            break;
        case monthly:
            o << "monthly" << endl;
    }
    o << "\tTime: " << cs.time << "\tBid: " << cs.bid << "\tAsk: " << cs.ask << endl;
    return o;
}
