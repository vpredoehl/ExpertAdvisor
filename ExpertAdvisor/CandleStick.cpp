//
//  CandleStick.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/1/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"

#include <iomanip>

using ChartTimeFrame = CandleStick::TimeFrame;
using CandleTime = CandleStick::CandleTime;


using std::string;

CandleStick::CandleStick(TimeFrame tf, CandleTime ct, string sym, float bid, float ask)
{
    frame = tf;  time = ct;
    CandleStick::bid = bid; CandleStick::ask = ask;
}

using std::istream;
using std::ostream;
using std::endl;

istream& operator>>(istream& i, CandleTime &t)
{
    struct tm ct;
    
    i >> std::get_time(&ct, "%F %R:%S");
    t = time_point_cast<minutes>(std::chrono::system_clock::from_time_t(mktime(&ct)));
    return i;
}

istream& operator>>(istream& i, CandleStick &t)
{
    using std::getline;
    string sym;
    CandleTime ct;
    float bid,ask;
    
    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    getline(i, sym, ',');
    i >> ct;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i >> bid;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i >> ask;
    t = CandleStick { CandleStick::minutely, ct, sym, bid, ask };
    return i;
}

ostream& operator<<(ostream& o, CandleStick::CandleTime t)
{
    auto tt = system_clock::to_time_t(t);
    o << std::put_time(localtime(&tt), "%F %R:%S");
    return o;
}
ostream& operator<<(ostream& o, CandleStick cs)
{
    o << "Chart: ";
    switch (cs.frame) {
        case CandleStick::minutely:
            o << "minute" << endl;
            break;
        case CandleStick::hourly:
            o << "hourly" << endl;
            break;
        case CandleStick::daily:
            o << "daily" << endl;
            break;
        case CandleStick::weekly:
            o << "weekly" << endl;
            break;
        case CandleStick::monthly:
            o << "monthly" << endl;
    }
    o << "\tTime: " << cs.time << "\tBid: " << cs.bid << "\tAsk: " << cs.ask << endl;
    return o;
}
