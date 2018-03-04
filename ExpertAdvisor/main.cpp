//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>

using SymbolData = std::map<std::string, MarketData>;

int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    PriceTP t;
    PricePoint pp;
    std::string l;
    
    csv >> l;   // header line

    SymbolData cd;
    while(csv >> pp)
    {
        MarketData &d = cd[PricePoint::sym];
        
        d.push_back(pp);
        std::cout << pp << std::endl;
    }
    
    std::cout.imbue(std::locale("en_US.UTF-8"));
    using namespace std::chrono;
    minutes candleDuration = minutes { 5 };
    
    std::for_each(cd.begin(), cd.end(), [candleDuration](const SymbolData::value_type &m)
                  {
                      const MarketData& priceD = m.second;
                      auto candleStartIter = priceD.cbegin(), candleEndIter = candleStartIter;
                      auto candleStartTime = candleStartIter->time;
                      auto candleEndTime = candleStartIter->time + candleDuration;
                      auto TimeNotInCandle = [&candleEndTime](const PricePoint &pp) -> bool    {   return pp.time >= candleEndTime;   };

                      while (priceD.end() != (candleEndIter = std::find_if(candleStartIter, priceD.cend(), TimeNotInCandle))) {
                          CandleStick cs(candleStartTime, candleStartIter, candleEndIter);
                          
                          std::cout << cs << std::endl;
                          candleStartIter = candleEndIter;
                          candleStartTime = candleEndTime;
                          candleEndTime += candleDuration;
                      }
                  });
    
    return 0;
}
