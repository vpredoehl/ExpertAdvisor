//
//  MarketData_iterator.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/6/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "MarketData_iterator.hpp"

MarketData_iterator::MarketData_iterator(MarketData::const_iterator s, const OffsetPP *o)
{
    curMD = s;
    if((ppo = o)) curO = o->cbegin();
}

auto MarketData_iterator::operator++(int) -> MarketData::const_iterator
{
    auto r = curMD;
    
    if(ppo == nullptr) return curMD++;
    curMD = std::next(curMD, *curO++);
    return r;
}

auto MarketData_iterator::operator++() -> MarketData::const_iterator
{
    if(ppo == nullptr) return ++curMD;
    return curMD = std::next(curMD, *curO++);
}

auto MarketData_iterator::operator=(const MarketData_iterator &mdi) -> MarketData_iterator
{
    curMD = mdi.curMD;
    curO = mdi.curO;
    ppo = mdi.ppo;
    return mdi;
}

