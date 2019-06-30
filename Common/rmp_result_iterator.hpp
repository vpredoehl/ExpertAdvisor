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


    // pqxx postgresql cursor iterator to be used with Chart template constructor
class rmp_result;
struct rmp_result_iterator : pqxx::const_result_iterator
{
    using value_type = PricePoint;

    const rmp_result *rr;

    rmp_result_iterator(pqxx::const_result_iterator i, const rmp_result *r = nullptr) : const_result_iterator { i } { rr = r; }
    rmp_result_iterator() = delete;

    const PricePoint* operator->() const { ExractPP(); return &pp; }
    bool operator>=(rmp_result_iterator i) { return ExractPP().time >= i.ExractPP().time; }
    const PricePoint operator*() const { return ExractPP(); }

private:
    mutable PricePoint pp;

    auto ExractPP() const -> PricePoint;
};

class rmp_cursor;
class rmp_result : pqxx::result
{
    rmp_cursor *cur;

public:
    rmp_result(const pqxx::result &r, rmp_cursor *c)
        : pqxx::result { r }, cur { c }
        {   }

    auto cbegin() const -> rmp_result_iterator { return { pqxx::result::cbegin(), this };   }
    auto cend() const -> rmp_result_iterator;
    auto size() const;
};

using Cursor = pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned>;
class rmp_cursor : Cursor
{
    const unsigned sliceSize = 10;
    unsigned long idx = 0;
    pqxx::result cur_slice, last_slice;

public:
    rmp_cursor(pqxx::work &w, std::string query, std::string curName,  unsigned long posn = 0);

    operator rmp_result() {   return { cur_slice, this };   }
    auto size() -> Cursor::size_type { return Cursor::size(); }

    friend class rmp_result;
};

class rmp_stream;
class rmp_stream_iterator {
    bool isEnd;
    rmp_stream *stream;
    PricePoint pp;

    auto ExtractPP(pqxx::result&) -> PricePoint;
public:
    rmp_stream_iterator(rmp_stream *s);
    rmp_stream_iterator()   {   isEnd = true; stream = nullptr; }

    auto operator*() const -> PricePoint    {   return pp;  }
    auto operator->() -> PricePoint*  {   return &pp; }

    auto operator++() -> rmp_stream_iterator;
};

class rmp_stream: public pqxx::icursorstream
{
public:
    rmp_stream(pqxx::work &w, std::string query, std::string streamName)
    :   pqxx::icursorstream { w, query, streamName } {}

    auto cbegin()   -> rmp_stream_iterator  {   return { this }; }
    auto cend() const -> rmp_stream_iterator {  return {}; }
};

#endif /* ResultIter_hpp */
