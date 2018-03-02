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
#include <vector>
#include <string>

using ChartType = std::pair<std::string, CandleStick::TimeFrame>;
using MarketData = std::vector<CandleStick>;
using ChartData = std::map<std::string, MarketData>;

int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    CandleStick::CandleTime t;
    CandleStick cs;
    std::string l;
    
    csv >> l;   // header line

    ChartData cd;
    while(csv >> cs)
    {
        MarketData &d = cd[CandleStick::sym];
        
        d.push_back(cs);
        std::cout << cs << std::endl;
    }
    
    return 0;
}
