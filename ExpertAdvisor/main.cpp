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
    constexpr short maxWriteThreads = 6;

    std::list<std::future<SymbolData>> parseFU;
    std::list<std::thread> saveToDB;
    auto dirIter = recursive_directory_iterator( argc == 2 ?  argv[1] : forexPath );
    SymbolData allSyms;
    auto HasAvailTask = [&parseFU, &saveToDB, &allSyms](auto maxTasks) -> bool
    {
        auto availFU = [](const std::future<SymbolData> &fut) -> bool
            {   return std::future_status::ready == fut.wait_for(std::chrono::milliseconds{10});    };
        auto availIter = std::find_if(parseFU.begin(), parseFU.end(), availFU);
        
        if (availIter != parseFU.end())
        {
            auto symD = availIter->get();
            parseFU.erase(availIter);
            std::for_each(symD.begin(), symD.end(), [&saveToDB](SymbolData::value_type &s)
                          {
//                              WriteMarketData(s.first, s.second);
                              saveToDB.push_back(std::thread { WriteMarketData, s.first, std::move(s.second) });
                          });
            if(saveToDB.size() > maxWriteThreads)   {   saveToDB.front().join();    saveToDB.pop_front(); }
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

        // wait for all syms to save to databsae
    std::for_each(saveToDB.begin(), saveToDB.end(), [](std::thread &th){    th.join(); });
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
