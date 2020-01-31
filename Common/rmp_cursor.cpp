//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "rmp_cursor_iterator.hpp"
#include <iostream>


rmp_result_block::rmp_result_block(rmp_cursor *cur, pqxx::result::difference_type idx)
: pqxx::result { cur->retrieve(idx, idx + block_size) }
{
    fromIdx = idx;
}

auto rmp_cursor_iterator::ExractPP() const -> PricePoint
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

auto rmp_cursor_iterator::operator++() -> rmp_cursor_iterator      {   return { cur, ++idx, idx == cur->size() }; }

thread_local unsigned long rmp_forward_iterator::magic = 0;
rmp_forward_iterator::rmp_forward_iterator(rmp_cursor_stream *c, bool end)
    : cur { c }
{
    if(!(isSTLEnd = end))
    {
        uniqID = magic++;
        isSTLEnd = !ReadPP();
    }
}

bool rmp_forward_iterator::ReadPP()
{
    pqxx::result r;
    bool lineRead = *cur >> r;

    if(lineRead)
    {
        auto bid { r[0]["bid"] }, ask { r[0]["ask"] };
        std::istringstream time { r[0]["time"].c_str() };

        time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
//        std::cout << pp.time << "\t" << pp.ask << '\t' << pp.bid << std::endl;
    }
    return lineRead;
}

