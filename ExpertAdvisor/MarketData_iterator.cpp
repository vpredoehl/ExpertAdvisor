//
//  MarketData_iterator.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/6/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "MarketData_iterator.hpp"

MarketData_iterator::MarketData_iterator(MarketData::const_iterator s)
{
    curMD = s;
}

MarketData_iterator::MarketData_iterator(ChartData::const_iterator iter)
{
    
}

auto MarketData_iterator::operator++(int) -> MarketData_iterator
{
    return curMD++;
}

auto MarketData_iterator::operator++() -> MarketData_iterator
{
    return ++curMD;
}

auto MarketData_iterator::operator=(const MarketData_iterator &mdi) -> MarketData_iterator
{
    curMD = mdi.curMD;
    return mdi;
}

