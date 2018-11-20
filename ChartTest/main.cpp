//
//  main.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 5/13/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "DiskIO.hpp"
#include "Chart.hpp"

#include "experimental/filesystem"
#include <set>
#include <vector>
#include <sstream>
#include <pqxx/pqxx>

using namespace std::chrono;
using TestTF = std::pair< unsigned short, unsigned short >; // to time frame / from time frame

std::ostream& operator<<(std::ostream& o, TestTF p)
{
    o << p.first << ":" << p.second;
    return o;
}

auto MakeTestCharts(std::string sym, const RawMarketPrice &rmp, TestTF ttf) -> ChartForSym
{
    ChartForSym charts;

    Chart ch { rmp.cbegin(), rmp.cend(), minutes { 1 } };
    Chart from1Min { ch.cbegin(), ch.cend(), minutes { ttf.first } };
    Chart fromScratch { rmp.cbegin(), rmp.cend(), minutes { ttf.first } };

    charts.push_back(ch);   charts.push_back(from1Min);  charts.push_back(fromScratch);

    return charts;
}

extern std::set<std::string> pairs;

int main(int argc, const char * argv[])
{
    if(argc < 3) {  std::cout << "Usage:\r\tChartTest fromDate toDate SYM/SYM toTF:fromTF …\r";  return 0;   }

    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result tables = w.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };
    std::vector<TestTF> testTimeFrame;

        // get time frame tests from command line
    for(auto t = 3; t < argc; ++t)
    {
        std::string chartTime { argv[t] };
        auto posn = chartTime.find(":");
        std::string fromTimeFrame { chartTime.substr(posn+1) }, toTimeFrame { chartTime.erase(posn) };

        testTimeFrame.push_back({ std::stoi(toTimeFrame), std::stoi(fromTimeFrame) });
    }

    for( auto p : tables )
    {
        std::string rawPriceTableName { p[0].c_str() };
        pqxx::result ppR = w.exec("select * from " + rawPriceTableName + " where time between '" + fromDate + "' and '" + toDate + "' order by time;");
        RawMarketPrice rmp;

            // take price points from db and put them into RawMarketPrice
        for( auto row : ppR )
        {
            auto bid { row["bid"] }, ask { row["ask"] };
            std::istringstream time { row["time"].c_str() };
            PricePoint pp;

            time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
            rmp.push_back(pp);
        }
        for(auto ttf: testTimeFrame)
        {
            auto charts = MakeTestCharts(rawPriceTableName, rmp, ttf);

            std::cout << ttf << (charts[0] == charts[1] ? " match" : " don't match") << std::endl;
            std::cout << ttf << (charts[0] == charts[2] ? " match" : " don't match") << std::endl;
            std::cout << ttf << (charts[1] == charts[2] ? " match" : " don't match") << std::endl;
        }
    }
    return 0;
}
 
