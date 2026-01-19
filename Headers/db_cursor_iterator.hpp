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


template<typename T> struct db_cursor_stream;

template<typename T>
struct db_forward_iterator
{
    using value_type = T;
    using iterator_category = std::forward_iterator_tag;
    using difference_type = signed long;

    db_forward_iterator(db_cursor_stream<T> *c, bool end, bool = true);

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
        return { cur, isSTLEnd = !ReadPP(), false };
    }

private:
    inline static bool isValidPP = false;
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
template<typename T> db_forward_iterator<T>::db_forward_iterator(db_cursor_stream<T> *c, bool end, bool advance)
    : cur { c }
{
    if(!(isSTLEnd = end))
    {
        uniqID = magic++;
        if(advance || !isValidPP) isSTLEnd = !ReadPP();
    }
}

template<typename T>
struct db_cursor_stream : public pqxx::icursorstream
{
    db_cursor_stream(pqxx::work &w, std::string query, std::string curName)
    : pqxx::icursorstream { static_cast<pqxx::transaction_base&>(w), query, curName } {}

    auto cbegin() -> db_forward_iterator<T> { return { this, false, false }; }
    auto cend() -> db_forward_iterator<T>   {   return { this, true, false }; }
};

#endif /* ResultIter_hpp */
