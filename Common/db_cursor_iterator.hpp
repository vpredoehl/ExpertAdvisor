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

class db_cursor;

struct rmp_result_block :  pqxx::result
{
    pqxx::result::difference_type fromIdx;

    rmp_result_block(db_cursor *cur, pqxx::result::difference_type idx);

    auto IsCached(pqxx::result::difference_type idx) {   return idx >= fromIdx && idx < fromIdx + block_size;   }
    auto retrieve(pqxx::result::difference_type idx) -> PricePoint;
};

template<typename T>
struct db_cursor_iterator
{
    using value_type = T;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = signed long;

    mutable std::shared_ptr<rmp_result_block> blk;

    db_cursor *cur;
    difference_type idx;
    bool isEnd;

    db_cursor_iterator(db_cursor *c,difference_type posn, bool end) : cur { c }  {   isEnd = end;    idx = posn;   }

    const T* operator->() const { ExractPP(); return &pp; }
    const T operator*() const { return ExractPP(); }

    bool operator>=(db_cursor_iterator i) const { return ExractPP().time >= i.ExractPP().time; }
    bool operator!=(db_cursor_iterator i) const  {   return i.idx != idx;   }
    bool operator==(db_cursor_iterator i) const  {   return i.idx == idx;   }

    auto operator++() -> db_cursor_iterator;

private:
    mutable T pp;

    auto ExractPP() const -> T;
};

template<typename T> struct std::iterator_traits<db_cursor_iterator<T>>
{
    using value_type = db_cursor_iterator<T>::value_type;
    using iterator_category = std::random_access_iterator_tag;
};

using Cursor = pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned>;
struct db_cursor :  Cursor
{
    db_cursor(pqxx::work &w, std::string query, std::string curName,  unsigned long posn = 0)
        : Cursor { static_cast<pqxx::transaction_base&>(w), query, curName, false } {}


    auto size() -> Cursor::size_type { return Cursor::size(); }

    auto cbegin() -> db_cursor_iterator<PricePoint> {   return { this, 0, false }; }
    auto cend()  -> db_cursor_iterator<PricePoint> {  return { this, static_cast<difference_type>(Cursor::size()), true };  }
};

template<typename T> struct db_cursor_stream;

template<typename T>
struct db_forward_iterator
{
    using value_type = T;
    using iterator_category = std::forward_iterator_tag;
    using difference_type = signed long;

    db_forward_iterator(db_cursor_stream<T> *c, bool end);

    const T operator*() const    {   return pp;  }
    const T* operator->() const    {   return &pp;  }

    bool operator!=(db_forward_iterator i) const  {   return !operator==(i);   }
    bool operator==(db_forward_iterator i) const
    {
        if(isSTLEnd == i.isSTLEnd)  return true;
        if(!isSTLEnd && !i.isSTLEnd) return uniqID == i.uniqID;
        return false;
    }

    auto operator++(int) -> db_forward_iterator
    {
        db_forward_iterator tmp(*this);
        ++(*this);
        return tmp;
    }
    auto operator++() -> db_forward_iterator
    {
        if(isSTLEnd)    throw std::range_error { "Can't advance rmp_forward_iterator past end" };
        return { cur, isSTLEnd = !ReadPP() };
    }

private:
    thread_local static unsigned long magic;
    unsigned long uniqID;
    db_cursor_stream<T> *cur;
    T pp;
    bool isSTLEnd;

    bool ReadPP();  // returns true if row was read
};
template<typename T> struct std::iterator_traits<db_forward_iterator<T>>
{
    using value_type = typename db_forward_iterator<T>::value_type;
    using iterator_category = std::forward_iterator_tag;
};

template<typename T> thread_local unsigned long db_forward_iterator<T>::magic = 0;
template<> bool db_forward_iterator<PricePoint>::ReadPP();
template<typename T> db_forward_iterator<T>::db_forward_iterator(db_cursor_stream<T> *c, bool end)
    : cur { c }
{
    if(!(isSTLEnd = end))
    {
        uniqID = magic++;
        isSTLEnd = !ReadPP();
    }
}

template<typename T>
struct db_cursor_stream : public pqxx::icursorstream
{
    db_cursor_stream(pqxx::work &w, std::string query, std::string curName)
    : pqxx::icursorstream { static_cast<pqxx::transaction_base&>(w), query, curName } {}

    auto cbegin() -> db_forward_iterator<T> { return { this, false }; }
    auto cend() -> db_forward_iterator<T>   {   return { this, true }; }
};

#endif /* ResultIter_hpp */
