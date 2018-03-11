//
//  MarketData_iterator.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/6/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "MarketData_iterator.hpp"

MarketData_iterator::MarketData_iterator(MarketData::const_iterator s, OffsetPair p)
{
    auto seq = p.first;
    
    curMD = s;
    offsetIter = p.second;
    if((ppo = seq)) offsetIter = seq->cbegin();
}

auto MarketData_iterator::operator++(int) -> MarketData_iterator
{
    auto r = curMD;
    
    if(ppo == nullptr) return curMD++;
    curMD = std::next(curMD, *offsetIter++);
    return MarketData_iterator {r, {ppo, offsetIter}};
}

auto MarketData_iterator::operator++() -> MarketData_iterator
{
    if(ppo == nullptr) return ++curMD;
    return MarketData_iterator {curMD = std::next(curMD, *offsetIter++), { ppo, offsetIter } };
}

auto MarketData_iterator::operator=(const MarketData_iterator &mdi) -> MarketData_iterator
{
    curMD = mdi.curMD;
    offsetIter = mdi.offsetIter;
    ppo = mdi.ppo;
    return mdi;
}

