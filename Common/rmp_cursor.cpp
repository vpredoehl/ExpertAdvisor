//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "rmp_result_iterator.hpp"
#include <iostream>

rmp_cursor::rmp_cursor(pqxx::work &w, std::string query, std::string curName,  unsigned long posn)
    : idx { posn }, Cursor { static_cast<pqxx::transaction_base&>(w), query, curName, false }
{
    auto sz = Cursor::size();
    auto lastSliceSize = sz % sliceSize;

    cur_slice = retrieve(0,sliceSize);
    last_slice = retrieve(sz - lastSliceSize, sz);

    std::cout << "cur_slice size: " << cur_slice.size() << std::endl << "last_slice size: " << last_slice.size() << " / " << lastSliceSize << std::endl;
}



auto rmp_result::cend() const -> rmp_result_iterator    {   return { cur->last_slice.cend(), this };    }
auto rmp_result::size() const  {   return cur->size(); }

auto rmp_result_iterator::ExractPP() const -> PricePoint
{
    auto row = const_result_iterator::operator*();

    if(row["time"].c_str() == nullptr) return PricePoint();

    auto bid { row["bid"] }, ask { row["ask"] };
    std::istringstream time { row["time"].c_str() };

    time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    return pp;
}

auto rmp_stream_iterator::ExtractPP(pqxx::result &r) -> PricePoint
{
    auto row = r.cbegin();    // result set has one row
    auto bid { row["bid"] }, ask { row["ask"] };
    std::istringstream time { row["time"].c_str() };

    time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    return pp;
}
rmp_stream_iterator::rmp_stream_iterator(rmp_stream *s)
    : stream { s }
{
    pqxx::result r;
    if((isEnd = (*s >> r))) return;
    ExtractPP(r);
}

