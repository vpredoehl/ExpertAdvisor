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
struct db_input_iterator
{
    using value_type = T;
    using iterator_category = std::input_iterator_tag;
    using difference_type = signed long;

    db_input_iterator(db_cursor_stream<T> *c, bool end, bool = true);

    const T operator*() const    {   return pp;  }
    const T* operator->() const    {   return &pp;  }

    bool operator!=(db_input_iterator i) const  {   return !(*this == i);   }
    bool operator==(const db_input_iterator& i) const
    {
        if (isSTLEnd && i.isSTLEnd) return true;
        if (isSTLEnd != i.isSTLEnd) return false;
        return cur == i.cur && uniqID == i.uniqID;
    }
    auto operator++(int) -> db_input_iterator
    {
        db_input_iterator tmp(*this);
        ++(*this);
        return tmp;
    }
    auto operator++() -> db_input_iterator&
    {
        if(isSTLEnd)    throw std::range_error { "Can't advance rmp_forward_iterator past end" };
        isSTLEnd = !ReadPP();
        return *this;
    }

private:
    thread_local static unsigned long magic;
    unsigned long uniqID = 0;
    db_cursor_stream<T> *cur = nullptr;
    T pp {};
    bool isSTLEnd = true;

    bool ReadPP();  // returns true if row was read
};

template<typename T> thread_local unsigned long db_input_iterator<T>::magic = 0;
template<> bool db_input_iterator<PricePoint>::ReadPP();
template<typename T> db_input_iterator<T>::db_input_iterator(db_cursor_stream<T> *c, bool end, bool advance)
    : cur { c }
{
    if(!(isSTLEnd = end))
    {
        uniqID = magic++;
        if(advance) isSTLEnd = !ReadPP();
    }
}

template<typename T>
struct db_cursor_stream : public pqxx::icursorstream
{
    std::string queryText;
    std::string cursorName;
    size_t nextRowIndex = 0;
    size_t parseFailDiagCount = 0;
    size_t badRowDiagCount = 0;

    db_cursor_stream(pqxx::work &w, std::string query, std::string curName)
    : pqxx::icursorstream { static_cast<pqxx::transaction_base&>(w), query, curName },
      queryText { query },
      cursorName { curName } {}

    auto begin() -> db_input_iterator<T> { return { this, false, true }; }
    auto end() -> db_input_iterator<T>   {   return { this, true, false }; }
};

#endif /* ResultIter_hpp */
