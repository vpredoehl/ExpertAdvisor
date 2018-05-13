//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "PricePoint.hpp"
#include "Chart.hpp"

#include "experimental/filesystem" // Had to set User Header Search Path in Project->Build Settings and include as user headers to avoid conflict with released system headers and satisfy the lexical prepreocessor
#include <future>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <list>

using namespace std::experimental::filesystem;

using SymbolData = std::map<std::string, RawMarketPrice>;
using ChartsForSym = std::map<std::string, Chart>;


auto ParseRawPriceData(std::ifstream csv)
{
    extern thread_local std::string lastParsedSym;
    PricePoint pp;
    std::string headerLine;
    SymbolData symD;

    csv >> headerLine;   // ignored
    while(csv >> pp)    symD[lastParsedSym].push_back(pp);
    return symD;
}

const std::string savePath = "/Volumes/Forex Data/";
auto ReadMarketData(std::string sym) -> RawMarketPrice
{
    auto posn = sym.find("/");

    if(posn != std::string::npos)    sym.replace(posn, 1, "-");
    std::ifstream f { savePath + sym, std::ios_base::binary | std::ios_base::in };
    PricePoint pp;
    RawMarketPrice rmp;

    while (f.read(reinterpret_cast<char*>(&pp), sizeof(pp)))    rmp.push_back(pp);
    return rmp;
}

void WriteMarketData(std::string sym, const RawMarketPrice &rmp)
{
    sym.replace(sym.find("/"), 1, "-");
    std::ofstream f { savePath + sym,  std::ios_base::binary | std::ios_base::out | std::ios_base::ate };
    
    std::for_each(rmp.cbegin(), rmp.cend(), [&f](const PricePoint &pp)
                  {   f.write(reinterpret_cast<const char*>(&pp), sizeof(pp));  });
}

auto SymsFromDirectory(std::string dirPath)
{
    SymbolData symD;
    auto fileIter = directory_iterator(dirPath);
    
    for (auto f : fileIter)
    {
        std::string sym = f.path().filename();
        auto posn = sym.find("-");
        
        if(is_directory(f)) continue;
        if(sym == ".DS_Store")  continue;

        auto rmp = ReadMarketData(sym);
        
        std::cout << "Loaded: " << sym << std::endl;
        if(posn != std::string::npos)    sym.replace(posn, 1, "/");
        symD[sym] = rmp;
    }
    return symD;
}

using namespace std::chrono;
using ChartInterval = std::vector<minutes>;
using ChartForSym = std::vector<Chart>;
auto MakeTestCharts(const RawMarketPrice &rmp, ChartInterval timeFrames = { minutes { 5 }, hours { 1 }, days { 1 }, weeks { 1 }, days { 30 } }) -> ChartForSym
{
    ChartForSym charts;
    Chart ch { rmp.cbegin(), rmp.cend(), minutes { 1 } };
    Chart min5FromScratch { rmp.cbegin(), rmp.cend(), minutes { 5 } };
    Chart min5 { ch.cbegin(), ch.cend(), minutes {5}};
    
    charts.push_back(ch);   charts.push_back(min5FromScratch);  charts.push_back(min5);
    
    Chart min10FromScratch { rmp.cbegin(), rmp.cend(), minutes { 10 } };
    Chart min10 { ch.cbegin(), ch.cend(), minutes {10}};
    Chart min10From5  {   min5.cbegin(), min5.cend(), minutes {10}};

    charts.push_back(min10FromScratch);   charts.push_back(min10);  charts.push_back(min10From5);

    Chart min15FromScratch { rmp.cbegin(), rmp.cend(), minutes { 15 } };
    Chart min15 { ch.cbegin(), ch.cend(), minutes {15}};
    Chart min15From5  {   min5.cbegin(), min5.cend(), minutes {15}};
    
    charts.push_back(min15FromScratch);   charts.push_back(min15);  charts.push_back(min15From5);

    Chart min30FromScratch { rmp.cbegin(), rmp.cend(), minutes { 30 } };
    Chart min30 { ch.cbegin(), ch.cend(), minutes {30}};
    Chart min30From5  {   min5.cbegin(), min5.cend(), minutes {30}};
    Chart min30From15  {   min15.cbegin(), min15.cend(), minutes {30}};

    charts.push_back(min30FromScratch);   charts.push_back(min30);  charts.push_back(min30From5);   charts.push_back(min30From15);

    Chart min45FromScratch { rmp.cbegin(), rmp.cend(), minutes { 45 } };
    Chart min45 { ch.cbegin(), ch.cend(), minutes {45}};
    Chart min45From5  {   min5.cbegin(), min5.cend(), minutes {45}};
    Chart min45From15  {   min15.cbegin(), min15.cend(), minutes {45}};
    
    charts.push_back(min45FromScratch);   charts.push_back(min45);  charts.push_back(min45From5);   charts.push_back(min45From15);

    return charts;
}

const std::string forexPath = "/Volumes/Forex Data/ratedata.gaincapital.com/2018/03 March";
const auto maxTasks = 12;

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

    //
    // append parsed data to file(s)
    //
    std::for_each(allSyms.cbegin(), allSyms.cend(), [](auto &sd)
                  {
                      auto sym = sd.first;
                      auto rawMarketPrice = sd.second;
                      
                      WriteMarketData(sym, rawMarketPrice);
                      rawMarketPrice.clear();
                  });
        //
        // make test charts
        //
    std::for_each(allSyms.begin(), allSyms.end(), [](const SymbolData::value_type &m)
                  {
                      auto symCharts = MakeTestCharts(m.second);
                  });
    
    return 0;
}
