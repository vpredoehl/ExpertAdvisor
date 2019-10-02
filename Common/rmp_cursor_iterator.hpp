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

const auto block_size = 10000;

    // pqxx postgresql cursor iterator to be used with Chart template constructor

class rmp_cursor;

struct rmp_result_block :  pqxx::result
{
    pqxx::result::difference_type fromIdx;

    rmp_result_block(rmp_cursor *cur, pqxx::result::difference_type idx);

    auto IsCached(pqxx::result::difference_type idx) {   return idx >= fromIdx && idx < fromIdx + block_size;   }
    auto retrieve(pqxx::result::difference_type idx) -> PricePoint;
};

struct rmp_cursor_iterator
{
    using value_type = PricePoint;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = signed long;

    mutable std::shared_ptr<rmp_result_block> blk;

    rmp_cursor *cur;
    difference_type idx;
    bool isEnd;

    rmp_cursor_iterator(rmp_cursor *c,difference_type posn, bool end) : cur { c }  {   isEnd = end;    idx = posn;   }

    const PricePoint* operator->() const { ExractPP(); return &pp; }
    const PricePoint operator*() const { return ExractPP(); }

    bool operator>=(rmp_cursor_iterator i) const { return ExractPP().time >= i.ExractPP().time; }
    bool operator!=(rmp_cursor_iterator i) const  {   return i.idx != idx;   }
    bool operator==(rmp_cursor_iterator i) const  {   return i.idx == idx;   }

    auto operator-(rmp_cursor_iterator i) const   {   return idx - i.idx; }
    auto operator-(int n) const {   return rmp_cursor_iterator { cur, idx - n, false }; }

    auto operator--() -> rmp_cursor_iterator    {   return { cur, --idx, false }; }
    auto operator++() -> rmp_cursor_iterator;

private:
    mutable PricePoint pp;

    auto ExractPP() const -> PricePoint;
};

template<> struct std::iterator_traits<rmp_cursor_iterator>
{
    using value_type = rmp_cursor_iterator::value_type;
    using iterator_category = std::random_access_iterator_tag;
};

using Cursor = pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned>;
struct rmp_cursor :  Cursor
{
    rmp_cursor(pqxx::work &w, std::string query, std::string curName,  unsigned long posn = 0)
        : Cursor { static_cast<pqxx::transaction_base&>(w), query, curName, false } {}


    auto size() -> Cursor::size_type { return Cursor::size(); }

    auto cbegin() -> rmp_cursor_iterator {   return { this, 0, false }; }
    auto cend()  -> rmp_cursor_iterator {  return { this, static_cast<difference_type>(Cursor::size()), true };  }
};


#endif /* ResultIter_hpp */
