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
#include <iterator>

class Chart;
class MarketData_iterator : public std::iterator<std::forward_iterator_tag, PricePoint>
{
    MarketData::const_iterator curMD;
    NextCandleOffset::const_iterator offsetIter;
    const NextCandleOffset *ppo;
    
    using OffsetPair = std::pair<const NextCandleOffset*, NextCandleOffset::const_iterator>;
    
    friend class Chart;
public:
    MarketData_iterator(MarketData::const_iterator, OffsetPair = OffsetPair { nullptr, NextCandleOffset::const_iterator {}});

    auto operator++(int) -> MarketData_iterator;
    auto operator++() -> MarketData_iterator;
    auto operator-(const MarketData_iterator &mdi) -> NextCandleOffset::difference_type  {   return curMD - mdi.curMD;   }
    auto operator-(int o) -> MarketData_iterator    {   return curMD - o;    }

    auto operator*() -> PricePoint const  {   return *curMD;  }
    auto operator->() -> const PricePoint* const { return &*curMD;   }
    operator MarketData::const_iterator() const {   return curMD;   }
    auto operator=(const MarketData_iterator& i) -> MarketData_iterator;

    auto operator==(const MarketData_iterator &mdi) const -> bool {   return curMD == mdi.curMD;  }
    auto operator!=(const MarketData_iterator &mdi) const -> bool {   return curMD != mdi.curMD;  }
};

#endif /* MarketData_iterator_hpp */
