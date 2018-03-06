//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "Chart.hpp"

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

    SymbolData symD;
    while(csv >> pp)
    {
        MarketData &d = symD[PricePoint::sym];
        
        d.push_back(pp);
        std::cout << pp << std::endl;
    }
    
    using namespace std::chrono;
    std::vector<minutes> scanInterval = { minutes { 5 }, hours { 1 }, days { 1 }, weeks { 1 }, days { 30 } };
    std::for_each(symD.begin(), symD.end(), [](const SymbolData::value_type &m)
                  {
                      Chart ch(m.second);
                      
                      std::cout << ch;
                  });
    
    return 0;
}
