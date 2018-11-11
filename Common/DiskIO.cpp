//
//  DiskIO.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 5/13/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "DiskIO.hpp"

#include "experimental/filesystem" // Had to set User Header Search Path in Project->Build Settings and include as user headers to avoid conflict with released system headers and satisfy the lexical prepreocessor
#include <iostream>
#include <fstream>


auto ParseRawPriceData(std::ifstream csv) -> SymbolData
{
    extern thread_local std::string lastParsedSym;
    PricePoint pp;
    std::string headerLine;
    SymbolData symD;
    
    csv >> headerLine;   // ignored
    while(csv >> pp)    symD[lastParsedSym].push_back(pp);
    return symD;
}

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
    std::ofstream f { savePath + sym,  std::ios_base::binary | std::ios_base::out | std::ios_base::app };
    
    std::for_each(rmp.cbegin(), rmp.cend(), [&f](const PricePoint &pp)
                  {   f.write(reinterpret_cast<const char*>(&pp), sizeof(pp));  });
}

auto SymsFromDirectory(std::string dirPath) -> SymbolData
{
    SymbolData symD;
    auto fileIter = std::experimental::filesystem::directory_iterator(dirPath);
    
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
