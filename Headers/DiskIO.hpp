//
//  DiskIO.hpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 5/13/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#ifndef DiskIO_hpp
#define DiskIO_hpp

#include "PricePoint.hpp"
#include "Chart.hpp"

#include <map>

using SymbolData = std::map<std::string, RawMarketPrice>;
using ChartsForSym = std::map<std::string, Chart>;
using ChartInterval = std::vector<std::chrono::minutes>;
using ChartForSym = std::vector<Chart>;

auto ParseRawPriceData(std::ifstream csv) -> SymbolData;
auto ReadMarketData(std::string sym) -> RawMarketPrice;
void WriteMarketData(std::string sym, const RawMarketPrice&, std::string fileName);
auto SymsFromDirectory(std::string dirPath) -> SymbolData;

const std::string forexPath = "/Volumes/Forex Data/ratedata.gaincapital.com/2018/03 March";
const std::string savePath = "/Volumes/Forex Data/";
const auto maxTasks = 12;

#endif /* DiskIO_hpp */
