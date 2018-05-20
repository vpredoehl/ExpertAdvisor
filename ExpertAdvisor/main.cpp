//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "DiskIO.hpp"

#include "experimental/filesystem" // Had to set User Header Search Path in Project->Build Settings and include as user headers to avoid conflict with released system headers and satisfy the lexical prepreocessor
#include <future>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <list>

using namespace std::experimental::filesystem;
using namespace std::chrono;

int main(int argc, const char * argv[]) {
    std::list<std::future<SymbolData>> parseFU;
    auto dirIter = recursive_directory_iterator( argc == 2 ?  argv[1] : forexPath );
    SymbolData allSyms;
    auto HasAvailTask = [&parseFU, &allSyms](auto maxTasks) -> bool
    {
        auto availFU = [](const std::future<SymbolData> &fut) -> bool
            {   return std::future_status::ready == fut.wait_for(std::chrono::milliseconds{10});    };
        auto availIter = std::find_if(parseFU.begin(), parseFU.end(), availFU);
        
        if (availIter != parseFU.end())
        {
            auto symD = availIter->get();
            parseFU.erase(availIter);
            std::for_each(symD.begin(), symD.end(), [&allSyms](SymbolData::value_type &s)
                          {
                              auto &allMarketD = allSyms[s.first];
                              auto &newData = s.second;
                              
                              allMarketD.splice(allMarketD.end(), newData);
                          });
            return true;
        }
        else if(parseFU.size() < maxTasks)  return true;
        return false;
    };

        //
        // parse .csv files
        //
    for(auto &f : dirIter)
    {
        if(f.path().extension() != ".csv")  continue;
        std::cout << f.path() << std::endl;
        parseFU.push_front(std::async(std::launch::async, ParseRawPriceData, std::ifstream { f.path(), std::ios_base::in }));

        while(!HasAvailTask(maxTasks))
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
    
    do {
        HasAvailTask(0);
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    } while (parseFU.size() > 0);

    //
    // append parsed data to file(s)
    //
    std::for_each(allSyms.cbegin(), allSyms.cend(), [](auto &sd)
                  {
                      auto sym = sd.first;
                      auto rawMarketPrice = sd.second;
                      
                      rawMarketPrice.sort([](PricePoint a, PricePoint b) -> bool {   return a.time < b.time; });
                      WriteMarketData(sym, rawMarketPrice);
                      rawMarketPrice.clear();
                  });
    return 0;
}

    //        //
    //        // compare to data read from file
    //        //
    //    std::for_each(allSyms.begin(), allSyms.end(), [](auto &sd)
    //                  {
    //                      std::string sym = sd.first;
    //                      auto symData = sd.second;
    //                      auto rmp = ReadMarketData(sym);
    //
    //                      symData.sort([](PricePoint a, PricePoint b) -> bool {   return a.time < b.time; });
    //                      symData.unique([](PricePoint a, PricePoint b)    {   return a.time == b.time && a == b;   });
    //
    //                      if(rmp == symData)
    //                          std::cout << sym + ": File matches!!!" << std::endl;
    //                      else
    //                      {
    //                          std::cout << sym + ": File DOES NOT match!!!" << std::endl;
    //                          std::cout << "parsed: " << symData.size() << " read from file: " << rmp.size() << std::endl;
    //                      }
    //                  });
