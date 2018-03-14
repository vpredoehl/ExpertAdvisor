//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "PricePoint.hpp"
#include "Chart.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>

using SymbolData = std::map<std::string, MarketPrice>;

int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    PriceTP t;
    PricePoint pp;
    std::string l;
    
    csv >> l;   // header line

    SymbolData symD;
    while(csv >> pp)
    {
        MarketPrice &d = symD[PricePoint::sym];
        
        d.push_back(pp);
    }
    
    using namespace std::chrono;
    std::vector<minutes> scanInterval = { minutes { 5 }, hours { 1 }, days { 1 }, weeks { 1 }, days { 30 } };
    std::for_each(symD.begin(), symD.end(), [](const SymbolData::value_type &m)
                  {
                      const MarketPrice &md = m.second;
                      Chart ch { md.cbegin(), md.cend() };
                      Chart min5FromScratch { md.cbegin(), md.cend(), minutes { 5 } };
                      Chart min5 { ch.cbegin(), ch.cend(), minutes {5}};
                      
                      std::cout << "Charts successfully constructed…" << std::endl << ch << std::endl;
                      std::cout << "5 min chart…" << min5 << std::endl;

                      if (min5FromScratch == min5)
                          std::cout << "5 Minute Charts Match!!!" << std::endl;
                  });
    
    return 0;
}
