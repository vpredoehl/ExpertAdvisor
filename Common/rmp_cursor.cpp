//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "rmp_result_iterator.hpp"


auto rmp_cursor::next(unsigned long count) -> rmp_result_iterator
{
    cur_slice = cur->retrieve(idx, idx += count);
    return cur_slice.cbegin();
}
