//
//  MarketData_iterator.hpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 3/6/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#ifndef MarketData_iterator_hpp
#define MarketData_iterator_hpp

#include "PricePoint.hpp"
#include <vector>
#include <iterator>

using OffsetPP = std::vector<MarketData::const_iterator::difference_type>;

class MarketData_iterator : public std::iterator<std::forward_iterator_tag, PricePoint>
{
    MarketData::const_iterator curMD;
    OffsetPP::const_iterator curO;
    const OffsetPP *ppo;
    
public:
    MarketData_iterator(MarketData::const_iterator, const OffsetPP* = nullptr);
    ~MarketData_iterator()  {   delete ppo; }

    auto operator++(int) -> MarketData::const_iterator;
    auto operator++() -> MarketData::const_iterator;
    auto operator-(const MarketData_iterator &mdi) -> OffsetPP::difference_type  {   return curMD - mdi.curMD;   }

    auto operator*() -> PricePoint  {   return *curMD;  }
    auto operator->() -> const PricePoint* const { return &*curMD;   }
    operator MarketData::const_iterator() const {   return curMD;   }
    auto operator=(const MarketData_iterator& i) -> MarketData_iterator;

    auto operator==(const MarketData_iterator &mdi) -> bool {   return curMD == mdi.curMD;  }
    auto operator!=(const MarketData_iterator &mdi) -> bool {   return curMD != mdi.curMD;  }
};

#endif /* MarketData_iterator_hpp */
