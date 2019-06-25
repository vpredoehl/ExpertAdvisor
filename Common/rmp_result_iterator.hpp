//
//  ResultIter.hpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#ifndef ResultIter_hpp
#define ResultIter_hpp

#include "PricePoint.hpp"

#include <iostream>
#include <pqxx/pqxx>
#include <pqxx/cursor>


    // pqxx postgresql cursor iterator to be used with Chart template constructor
struct rmp_result_iterator : pqxx::const_result_iterator
{
    using value_type = PricePoint;

    rmp_result_iterator(pqxx::const_result_iterator i) : const_result_iterator { i } {}
    rmp_result_iterator() = delete;

    const PricePoint* operator->() const { ExractPP(); return &pp; }
    bool operator>=(rmp_result_iterator i) { return ExractPP().time >= i.ExractPP().time; }
    const PricePoint operator*() const { return ExractPP(); }

private:
    mutable PricePoint pp;

        auto ExractPP() const -> PricePoint
        {
            auto row = const_result_iterator::operator*();

            if(row["time"].c_str() == nullptr) return PricePoint();

            auto bid { row["bid"] }, ask { row["ask"] };
            std::istringstream time { row["time"].c_str() };

            time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
            return pp;
        }
};

using Cursor = pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned>;
class rmp_cursor : Cursor
{
    const unsigned sliceSize = 1000;
    unsigned long idx = 0;
    pqxx::result cur_slice;
    Cursor *cur;    // kinda stupid, but can't construct base class Cursor from parameters, so will have to pass in as a pointer

public:
//    rmp_cursor(Cursor *c, unsigned long posn = 0)
//    : idx { posn }, cur { c } {}
    rmp_cursor(pqxx::work &w, std::string query, std::string curName,  unsigned long posn = 0)
        : idx { posn }, Cursor { static_cast<pqxx::transaction_base&>(w), query, curName, false }
            {   cur_slice = retrieve(0,sliceSize);  }
//    : idx { posn }, Cursor { w, "select * from " + rawPriceTableName + " where time between '" + fromDate + "' and '" + toDate + "' order by time;", rawPriceTableName + "_cursor", false } {}

    auto next(unsigned long count = 1) -> rmp_result_iterator;

    operator pqxx::result() {   return cur_slice;   }
};

#endif /* ResultIter_hpp */
