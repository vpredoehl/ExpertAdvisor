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
#include <algorithm>
#include <pqxx/pqxx>

using namespace std::experimental::filesystem;
using namespace std::chrono;

int main(int argc, const char * argv[]) {
    constexpr short maxWriteThreads = 6;

    std::list<std::future<SymbolData>> parseFU;
    std::list<std::thread> saveToDB;
    auto dirIter = recursive_directory_iterator( argc == 2 ?  argv[1] : forexPath );
    SymbolData allSyms;
    auto HasAvailTask = [&parseFU, &saveToDB, &allSyms](auto maxTasks, std::string fileName = "") -> bool
    {
        auto availFU = [](const std::future<SymbolData> &fut) -> bool
            {   return std::future_status::ready == fut.wait_for(std::chrono::milliseconds{10});    };
        auto availIter = std::find_if(parseFU.begin(), parseFU.end(), availFU);
        
        if (availIter != parseFU.end())
        {
            auto symD = availIter->get();
            parseFU.erase(availIter);
            std::for_each(symD.begin(), symD.end(), [&saveToDB, &fileName](SymbolData::value_type &s)
                          { saveToDB.push_back(std::thread { WriteMarketData, s.first, std::move(s.second), fileName });    });
            if(saveToDB.size() > maxWriteThreads)   {   saveToDB.front().join();    saveToDB.pop_front(); }
            return true;
        }
        else if(parseFU.size() < maxTasks)  return true;
        return false;
    };

        //
        // parse .csv files
        //
    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result fileList = w.exec("select * from parsedfiles;");
    void PrintResult(const pqxx::result &r);

    PrintResult(fileList);
    for(auto &f : dirIter)
    {
        if(f.path().extension() != ".csv")  continue;
        if(fileList.cend() != std::find_if(fileList.cbegin(), fileList.cend(), [&f](auto fn) -> bool
                                           {     return f.path() == fn[0].c_str();  }))
            {   std::cout << "Skipping: " << f.path() << std::endl;   continue;   }
        
        std::cout << f.path() << std::endl;
        parseFU.push_front(std::async(std::launch::async, ParseRawPriceData, std::ifstream { f.path(), std::ios_base::in }));

        while(!HasAvailTask(maxTasks, f.path()))
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
