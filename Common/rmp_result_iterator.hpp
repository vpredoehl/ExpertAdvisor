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

#include <pqxx/pqxx>
#include <pqxx/cursor>

#include <iostream>

using Cursor = pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned>;


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

#endif /* ResultIter_hpp */
