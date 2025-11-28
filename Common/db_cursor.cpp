//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "db_cursor_iterator.hpp"
#include <iostream>


rmp_result_block::rmp_result_block(db_cursor *cur, pqxx::result::difference_type idx)
: pqxx::result { cur->retrieve(idx, idx + block_size) }
{
    fromIdx = idx;
}

template<>
auto db_cursor_iterator<PricePoint>::ExractPP() const -> PricePoint
{
    if(blk == nullptr || !blk->IsCached(idx))
            // requested index is not cached - retrieve new block
        blk = std::make_shared<rmp_result_block>(cur, idx);

    auto row = blk->cbegin() + idx - blk->fromIdx;

    if(row["time"].c_str() == nullptr) return PricePoint();

    auto bid { row["bid"] }, ask { row["ask"] };
    std::istringstream time { row["time"].c_str() };

    time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    return pp;
}

template<>
auto db_cursor_iterator<PricePoint>::operator++() -> db_cursor_iterator      {   return { cur, ++idx, idx == cur->size() }; }

template<>
bool db_forward_iterator<PricePoint>::ReadPP()
{
    pqxx::result r;
    bool lineRead = *cur >> r;

    if(lineRead)
    {
        auto bid { r[0]["bid"] }, ask { r[0]["ask"] };
        std::istringstream time { r[0]["time"].c_str() };

        time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    }
    return lineRead;
}


template<>
bool db_forward_iterator<CandlestickRow>::ReadPP()
{
    pqxx::result r;
    bool lineRead = *cur >> r;

    if(lineRead)
    {
        auto open { r[0]["open"] }, close { r[0]["close"] }, high { r[0]["high"] }, low { r[0]["low"] };
        std::istringstream time { r[0]["dt"].c_str() };

        time >> pp.time;  open >> pp.open; close >> pp.close;   high >> pp.high;    low >> pp.low;
    }
    return lineRead;
}

