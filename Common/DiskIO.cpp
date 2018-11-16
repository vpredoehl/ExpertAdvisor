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
#include <pqxx/pqxx>


void PrintResult(const pqxx::result &r)
{
    const auto num_rows = r.size();
    for (int rownum = 0; rownum < num_rows; ++rownum) {
        auto row = r[rownum];
        const int num_cols = row.size();
        for (int colnum=0; colnum < num_cols; ++colnum)
        {
            const pqxx::field field = row[colnum];
            std::cout << field.c_str() << '\t';
        }
        std::cout << std::endl;
    }
}

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

void WriteMarketData(std::string sym, const RawMarketPrice &rmp, std::string fileName)
{
    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    std::string tableName = sym;

    sym.replace(sym.find("/"), 1, "-"); tableName.replace(tableName.find("/"), 1, "");

    std::string insertRMP = "insert into " + tableName + "RMP ( time, ask, bid ) values \r";
    std::string createTable { "create table if not exists " + tableName + "RMP (\
        time timestamp not null,\
        bid decimal ( 15,5 ),\
        ask decimal ( 15,5 ) );" };


    try
    {
        pqxx::work w { c };
            // create the table
        pqxx::result  r = w.exec(createTable);
        w.commit();
            //        r = w.exec("create index if not exists time_frame on " + tableName + "(interval)");
    }
    catch (pqxx::sql_error e)   {   std::cerr << "postgresql sql_error: Create Table  " << e.what() << std::endl << "Query: " << e.query() << std::endl; }
    catch (pqxx::usage_error e)   {   std::cerr << "postgresql usage_error: Create Table  " << e.what() << std::endl; }

    try
    {
        pqxx::work w { c };

        std::for_each(rmp.cbegin(), rmp.cend(), [&w, &insertRMP](const PricePoint &pp)
                      {
                          std::time_t tp = std::chrono::system_clock::to_time_t(pp.time);
                          std::tm tp_tm = *std::localtime(&tp);
                          char buf[30];

                          strftime(buf, sizeof(buf), "'%F %T'", &tp_tm);
                          std::string values = std::string { " ( " }
                          + std::string { buf } + ", "
                          + std::string { std::to_string(pp.ask) } + ", "
                          + std::string { std::to_string(pp.bid) }
                          + std::string { ");\r" };

                          pqxx::result r = w.exec(insertRMP + values);
                          PrintResult(r);
                      });
            // add filename to list of parsef files
        pqxx::result  r = w.exec("insert into parsedfiles ( filename ) values ( '" + fileName + "' );");
        w.commit();
    }
    catch (pqxx::sql_error e)   {   std::cerr << "postgresql sql_error: Insert " << e.what() << std::endl << "Query: " << e.query() << std::endl; }
    catch (pqxx::usage_error e)   {   std::cerr << "postgresql usage_error: Insert " << e.what() << std::endl; }
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
