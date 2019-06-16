//
//  main.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 5/13/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "DiskIO.hpp"
#include "Chart.hpp"
#include "rmp_result_iterator.hpp"

#include <filesystem>
#include <set>
#include <vector>
#include <sstream>
#include <pqxx/pqxx>

using namespace std::chrono;
using TestTimeFrame = std::pair< unsigned short /* to time frame */, unsigned short /* from time frame */ >;
using TestChartData = std::pair<TestTimeFrame, Chart>;
using TestResultData = std::pair<bool, TestChartData>;


std::ostream& operator<<(std::ostream& o, TestTimeFrame p)
{
    o << p.first << ":" << p.second;
    return o;
}

auto MakeTestCharts(std::string sym, rmp_result_iterator cb, rmp_result_iterator ce, std::vector<TestTimeFrame> testParams) -> std::vector<TestResultData>
{
    std::vector<TestChartData> charts;
    std::vector<TestResultData> results;

    for( auto tP : testParams )
    {
        unsigned short toTimeFrame = tP.first, fromTimeFrame = tP.second;

            // find chart with toTimeFrame made from scratch - add if not found
            // need to have one of these for each toTimeFrame as basis for comparison
        auto fromScratchIter = std::find_if(charts.cbegin(), charts.cend(), [toTimeFrame](const TestChartData &tc) {   return tc.first.first == toTimeFrame && tc.first.second == 0; });
        if(charts.cend() == fromScratchIter)
        {
            charts.push_back({ { toTimeFrame, 0 }, { cb, ce, minutes { toTimeFrame } } });
            fromScratchIter = charts.end()-1;
        }

            // make test chart from base chart made from raw market price
        TestChartData lastTestChart { { toTimeFrame, fromTimeFrame }, { fromScratchIter->second.cbegin(), fromScratchIter->second.cend(), minutes { toTimeFrame } } };
        if(toTimeFrame == fromScratchIter->first.first)
        {
            bool passed = lastTestChart.second == fromScratchIter->second;

            results.push_back({ passed, std::move(lastTestChart) });
                // test passed - don't save chart ( if it was not created from raw market price )
            if(passed && results.back().second.first.second != 0) results.back().second.second.clear();
        }
        else
        {
            std::cout << lastTestChart.first << +": No chart made from raw market price to compare to.\r";
            results.push_back({ false, std::move(lastTestChart) });
        }
    }

    return results;
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
        Cursor cur(w, "select * from " + rawPriceTableName + " where time between '" + fromDate + "' and '" + toDate + "' order by time;", "mycursor", false);
        RawMarketPrice rmp;

        unsigned long idx = 0;
        const auto step = 1000;

//        do
//        {
//            try { ppR = cur.retrieve(idx, idx += step); }
        try
        {
            pqxx::result ppR = cur.retrieve(0, __UINT32_MAX__);

            std::cout << "rmp size: " << rmp.size() << std::endl;
            std::cout << "ppR size: " << ppR.size() << std::endl;

                // take price points from db and put them into RawMarketPrice
            auto results = MakeTestCharts(rawPriceTableName, ppR.cbegin(), ppR.cend(), testParams);
            for( auto tR : results )    std::cout << tR.second.first << (tR.first ? " passed" : " failed") << std::endl;
        }
        catch(pqxx::range_error e) { std::cout << "range exception: " << e.what(); break; }

//            for( auto row : ppR )
//            {
//                auto bid { row["bid"] }, ask { row["ask"] };
//                std::istringstream time { row["time"].c_str() };
//                PricePoint pp;
//
//                time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
//                rmp.push_back(pp);
//            }
//        } while(!ppR.empty());
    }
    return 0;
}
 
