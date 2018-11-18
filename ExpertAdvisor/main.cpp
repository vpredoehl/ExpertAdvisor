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
#include <set>
#include <list>
#include <algorithm>
#include <pqxx/pqxx>

using namespace std::experimental::filesystem;
using namespace std::chrono;

std::set<std::string> pairs { "AUD/CAD", "AUD/CHF", "AUD/NZD", "AUD/JPY", "AUD/USD", "CAD/CHF", "CAD/JPY", "CHF/JPY",
    "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/GBP", "EUR/JPY", "EUR/NZD", "EUR/USD", "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/NZD",
    "GBP/USD", "GBP/JPY", "NZD/CAD", "NZD/CHF", "NZD/JPY", "NZD/USD", "USD/CAD", "USD/CHF", "USD/JPY" };
constexpr char fileSepChar = '_', pairSepChar = '/';

    // args:
    //    1.  Directory to parse
    //    2.  Symbol pairs separated by a '/'
    //    3.  max parse jobs
int main(int argc, const char * argv[]) {
    constexpr short maxWriteThreads = 18;
    auto maxTasks = defaultMaxTasks;

    using FutureAndFileNameP = std::pair<std::future<SymbolData>, std::string>;
    std::list<FutureAndFileNameP> parseFU;
    std::list<std::thread> saveToDB;
    auto dirIter = recursive_directory_iterator( argc >= 2 ?  argv[1] : forexPath );
    SymbolData allSyms;
    auto HasAvailTask = [&parseFU, &saveToDB, &allSyms](auto maxTasks) -> bool
    {
        auto availFU = [](auto &futAndFile) -> bool
            {   return std::future_status::ready == futAndFile.first.wait_for(std::chrono::milliseconds{10});    };
        auto availIter = std::find_if(parseFU.begin(), parseFU.end(), availFU);
        
        if (availIter != parseFU.end())
        {
            auto symD = availIter->first.get();
            std::string fileName = availIter->second;

            parseFU.erase(availIter);
            std::for_each(symD.begin(), symD.end(), [&saveToDB, &fileName](SymbolData::value_type &s)
                          { saveToDB.push_back(std::thread { WriteMarketData, s.first, std::move(s.second), fileName });    });
            if(saveToDB.size() > maxWriteThreads)   {   saveToDB.front().join();    saveToDB.pop_front(); }
            return true;
        }
        else if(parseFU.size() < maxTasks)  return true;
        return false;
    };

    if(argc >= 3)   {   pairs.clear();  pairs = { argv[2] }; }
    if(argc >= 4)   maxTasks = std::stoi(argv[3]);

    std::cout << "Mximum Parse Tasks: " << maxTasks << std::endl;

        //
        // parse .csv files
        //
    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result AlreadyParsedFiles = w.exec("select * from parsedfiles;");

    for(auto &f : dirIter)
    {
        if(f.path().extension() != ".csv")  continue;

        std::string fileN = f.path().filename();
        auto posn = fileN.find(fileSepChar);

            // check if file is in pairs list
        fileN.replace(posn, 1, &pairSepChar);
        fileN.erase(fileN.find(fileSepChar));   // remove _Week*.csv
        if(pairs.find(fileN) == pairs.end()) continue;

        if(AlreadyParsedFiles.cend() != std::find_if(AlreadyParsedFiles.cbegin(), AlreadyParsedFiles.cend(), [&f](pqxx::row fn) -> bool
                                           {    return f.path() == fn[0].c_str();   }))
            {   std::cout << "Skipping: " << f.path() << std::endl;   continue;   }
        std::cout << f.path() << std::endl;
        parseFU.push_front(FutureAndFileNameP { std::async(std::launch::async, ParseRawPriceData, std::ifstream { f.path(), std::ios_base::in }), f.path() });

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
