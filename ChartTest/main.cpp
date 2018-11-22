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
using TestTimeFrame = std::pair< unsigned short /* to time frame */, unsigned short /* from time frame */ >;
using TestChartData = std::pair<TestTimeFrame, Chart>;

std::ostream& operator<<(std::ostream& o, TestTimeFrame p)
{
    o << p.first << ":" << p.second;
    return o;
}

auto MakeTestCharts(std::string sym, const RawMarketPrice &rmp, std::vector<TestTimeFrame> testParams) -> std::vector<TestChartData>
{
    std::vector<TestChartData> charts;
    Chart ch { rmp.cbegin(), rmp.cend(), minutes { 1 } };

    for( auto tP : testParams )
    {
        unsigned short toTimeFrame = tP.first, fromTimeFrame = tP.second;

        if(charts.cend() == std::find_if(charts.cbegin(), charts.cend(), [toTimeFrame](const TestChartData &tc) {   return tc.first.first == toTimeFrame && tc.first.second == 0; }))
            charts.push_back({ { toTimeFrame, 0 }, ch });


            // find fromTimeFrame to use as source chart
        auto sourceTestToUse = std::find_if(charts.cbegin(), charts.cend(), [fromTimeFrame](const TestChartData &ch) {   return ch.first.first == fromTimeFrame; });
        
        if(sourceTestToUse == charts.cend())
        {
            Chart sourceChartFromScratch  { rmp.cbegin(), rmp.cend(), minutes { toTimeFrame } };

            charts.push_back({ { toTimeFrame, 0 }, sourceChartFromScratch }); // save for possible use later
            charts.push_back({ { toTimeFrame, fromTimeFrame }, { sourceChartFromScratch.cbegin(), sourceChartFromScratch.cend(), minutes { toTimeFrame } } });
        }
        else
            charts.push_back({ { toTimeFrame, fromTimeFrame }, { sourceTestToUse->second.cbegin(), sourceTestToUse->second.cend(), minutes { toTimeFrame } } });
    }

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
    std::vector<TestTimeFrame> testParams;

        // get time frame tests from command line
    for(auto t = 3; t < argc; ++t)
    {
        std::string chartTime { argv[t] };
        auto posn = chartTime.find(":");
        std::string fromTimeFrame { chartTime.substr(posn+1) }, toTimeFrame { chartTime.erase(posn) };

        testParams.push_back({ std::stoi(toTimeFrame), std::stoi(fromTimeFrame) });
    }
    std::sort(testParams.begin(), testParams.end(), [](auto &t1, auto &t2)   {   return t1.first == t2.first ? t1.second < t2.second : t1.first < t2.first;   });

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

        auto charts = MakeTestCharts(rawPriceTableName, rmp, testParams);
        TestChartData *scratchChart = nullptr;

            // check results of test
        std::sort(charts.begin(), charts.end(), [](auto &t1, auto &t2)  {    return t1.first.first == t2.first.first ? t1.first.second < t2.first.second : t1.first.first < t2.first.first;      });
        for( auto &testResult : charts )
        {
            unsigned short fromTimeFrame = testResult.first.second, toTimeFrame = testResult.first.first;

            if(fromTimeFrame == 0)  {   scratchChart = &testResult;  continue;    }   // skip charts from scratch

                // check equality to chart from scratch
            if(scratchChart && toTimeFrame == scratchChart->first.first)
                std::cout << testResult.first << " / " << scratchChart->first << ": "
                << (testResult.second == scratchChart->second ? "match" : "don't match") << std::endl;
        }
    }
    return 0;
}
 
