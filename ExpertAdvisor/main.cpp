//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "PricePoint.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <string>

using ChartType = std::pair<std::string, TimeFrame>;
using MarketData = std::vector<PricePoint>;
using SymbolData = std::map<std::string, MarketData>;

int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    PriceTime t;
    PricePoint cs;
    std::string l;
    
    csv >> l;   // header line

    SymbolData cd;
    while(csv >> cs)
    {
        MarketData &d = cd[PricePoint::sym];
        
        d.push_back(cs);
        std::cout << cs << std::endl;
    }
    
    return 0;
}
