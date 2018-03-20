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

auto ParseRawPriceData(std::ifstream csv)
{
    PriceTP t;
    PricePoint pp;
    std::string headerLine;
    SymbolData symD;

    csv >> headerLine;   // ignored
    while(csv >> pp)    symD[PricePoint::sym].push_back(pp);
    return symD;
}

int main(int argc, const char * argv[]) {
    SymbolData symD = ParseRawPriceData(std::ifstream { "COR_USD_Week3.csv", std::ios_base::in });
    
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

                      Chart min10FromScratch { md.cbegin(), md.cend(), minutes { 10 } };
                      Chart min10 { ch.cbegin(), ch.cend(), minutes {10}};
                      Chart min10From5  {   min5.cbegin(), min5.cend(), minutes {10}};

                      if(min10FromScratch == min10)
                          std::cout << "10 min charts match!!!" << std::endl;
                      if(min10 == min10From5)
                          std::cout << "min10From5 charts match!!!" << std::endl;
                      
                      Chart min15FromScratch { md.cbegin(), md.cend(), minutes { 15 } };
                      Chart min15 { ch.cbegin(), ch.cend(), minutes {15}};
                      Chart min15From5  {   min5.cbegin(), min5.cend(), minutes {15}};
                      Chart min30FromScratch { md.cbegin(), md.cend(), minutes { 30 } };
                      Chart min30 { ch.cbegin(), ch.cend(), minutes {30}};
                      Chart min30From5  {   min5.cbegin(), min5.cend(), minutes {30}};
                      Chart min30From15  {   min15.cbegin(), min15.cend(), minutes {30}};

                      Chart min45FromScratch { md.cbegin(), md.cend(), minutes { 45 } };
                      Chart min45 { ch.cbegin(), ch.cend(), minutes {45}};
                      Chart min45From5  {   min5.cbegin(), min5.cend(), minutes {45}};
                      Chart min45From15  {   min15.cbegin(), min15.cend(), minutes {45}};

                      if(min45FromScratch == min45)
                          std::cout << "45 min charts match!!!" << std::endl;
                      if(min45 == min45From5)
                          std::cout << "min45From15 charts match!!!" << std::endl;
                      if(min45 == min45From5)
                          std::cout << "min45From15 charts match!!!" << std::endl;

                      if(min15FromScratch == min15)
                          std::cout << "15 min charts match!!!" << std::endl;
                      if(min15 == min15From5)
                          std::cout << "min15From5 charts match!!!" << std::endl;
                      if(min30 == min30FromScratch)
                          std::cout << "min30 charts match!!!" << std::endl;
                      if(min30From5 == min30)
                          std::cout << "min30From5 charts match!!!" << std::endl;
                      if(min30From15 == min30)
                          std::cout << "min30From15 charts match!!!" << std::endl;
});
    
    return 0;
}
