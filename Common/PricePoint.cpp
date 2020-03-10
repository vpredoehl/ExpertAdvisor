//
//  PricePoint.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/1/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"
#include <iomanip>

using namespace std::chrono;
using std::string;

PricePoint::PricePoint(PriceTP ct, float bid, float ask)
{
    time = ct;
    PricePoint::bid = bid; PricePoint::ask = ask;
}

using std::istream;
using std::ostream;
using std::endl;
using std::setw;
using std::showbase;
using std::put_money;

istream& operator>>(istream& i, PriceTP &t)
{
    struct tm ct;
    
    i >> std::get_time(&ct, "%F %R:%S");
    t = time_point_cast<minutes>(std::chrono::system_clock::from_time_t(mktime(&ct)));
    return i;
}

//istream& operator>>(istream& i, PricePoint &t)
//{
//    using std::getline;
//    PriceTP ct;
//    float bid,ask;
//
//    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
//    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
//    getline(i, lastParsedSym, ',');
//    i >> ct;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
//    i >> bid;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
//    i >> ask;
//    t = PricePoint { ct, bid, ask };
//    return i;
//}

istream& operator>>(istream& i, PricePoint &t)
{
    using std::getline;
    PriceTP ct;
    float bid,ask;
    char year[5], month[3], day[3], hour[3], min[3], sec[3], nano[4];


    i.get(year, 5); i.get(month, 3);    i.get(day, 3);  i.ignore(std::numeric_limits<std::streamsize>::max(),' ');
    i.get(hour, 3);    i.get(min, 3);    i.get(sec, 3); i.get(nano,4); i.ignore(std::numeric_limits<std::streamsize>::max(),',');

    std::istringstream dateTime { std::string { year } + "-" + std::string { month } + "-" + std::string { day }
        + " " + std::string { hour } + ":" + std::string { min } + ":" + std::string { sec } + "." + std::string { nano }  };

    dateTime >> ct;
    i >> bid;    i.ignore(std::numeric_limits<std::streamsize>::max(),',');
    i >> ask;
    i.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    t = PricePoint { ct, bid, ask };
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
ostream& operator<<(ostream& o, RawMarketPrice rmp)
{
    o << "size: " << rmp.size() << std::endl;
    std::for_each(rmp.begin(), rmp.end(), [&o](PricePoint pp)   {   o << pp << endl;    });
    return o;
}
