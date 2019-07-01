//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "rmp_cursor_iterator.hpp"
#include <iostream>


auto rmp_cursor_iterator::ExractPP() const -> PricePoint
{
    pqxx::result r = cur->retrieve(idx, idx+1);
    auto row = r.cbegin();

    if(row["time"].c_str() == nullptr) return PricePoint();

    auto bid { row["bid"] }, ask { row["ask"] };
    std::istringstream time { row["time"].c_str() };

    time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    return pp;
}

auto rmp_cursor_iterator::operator++() -> rmp_cursor_iterator      {   return { cur, ++idx, idx == cur->size() }; }
