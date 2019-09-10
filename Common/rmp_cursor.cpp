//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "rmp_cursor_iterator.hpp"
#include <iostream>

rmp_result_block *rmp_cursor_iterator::blk = nullptr;

rmp_result_block::rmp_result_block(rmp_cursor *cur, pqxx::result::difference_type idx)
: pqxx::result { cur->retrieve(idx, idx + block_size) }
{
    fromIdx = idx;
}

auto rmp_cursor_iterator::ExractPP() const -> PricePoint
{
    if(blk == nullptr || !blk->IsCached(idx))
    {
            // requested index is not cached - retrieve new block
        delete blk;
        blk = new rmp_result_block { cur, idx };
    }
    auto row = blk->cbegin() + idx - blk->fromIdx;

    if(row["time"].c_str() == nullptr) return PricePoint();

    auto bid { row["bid"] }, ask { row["ask"] };
    std::istringstream time { row["time"].c_str() };

    time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    return pp;
}

auto rmp_cursor_iterator::operator++() -> rmp_cursor_iterator      {   return { cur, ++idx, idx == cur->size() }; }
