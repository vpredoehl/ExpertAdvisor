//
//  main.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 5/13/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "DiskIO.hpp"
#include "Chart.hpp"
#include "rmp_cursor_iterator.hpp"

#include <filesystem>
#include <set>
#include <vector>
#include <sstream>
#include <future>
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

auto MakeTestCharts(std::string sym, std::string query, std::vector<TestTimeFrame> testParams) -> std::vector<TestResultData>
{
    std::vector<TestChartData> charts;
    std::vector<TestResultData> results;

    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    rmp_cursor_stream cur { w, query, sym + "_stream" };
    rmp_forward_iterator cb = cur.cbegin(), ce = cur.cend();

    for( auto tP : testParams )
    {
        unsigned short toTimeFrame = tP.first, fromTimeFrame = tP.second;

            // find chart with toTimeFrame made from scratch - add if not found
            // need to have one of these for each toTimeFrame as basis for comparison
        auto chartFromScratchIter = std::find_if(charts.cbegin(), charts.cend(), [toTimeFrame](const TestChartData &tc) {   return tc.first.first == toTimeFrame && tc.first.second == 0; });
        if(charts.cend() == chartFromScratchIter)
        {
            charts.push_back({ { toTimeFrame, 0 }, { sym, cb, ce, minutes { toTimeFrame } } });
            chartFromScratchIter = charts.end()-1;
        }

            // make test chart from base chart made from raw market price
        TestChartData lastTestChart { { toTimeFrame, fromTimeFrame }, { sym, chartFromScratchIter->second.cbegin(), chartFromScratchIter->second.cend(), minutes { toTimeFrame } } };
        if(toTimeFrame == chartFromScratchIter->first.first)
        {
            bool passed = lastTestChart.second == chartFromScratchIter->second;

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

    std::list<std::future<std::vector<TestResultData>>> taskR;
    try
    {
        for( auto p : tables )
        {
            std::string rawPriceTableName { p[0].c_str() };

                // take price points from db and put them into RawMarketPrice
            taskR.push_front(std::async(std::launch::async, MakeTestCharts, rawPriceTableName, "select * from " + rawPriceTableName + " where time between '" + fromDate + "' and '" + toDate + "' order by time;", testParams));
        }
    }
    catch(pqxx::failure e)  {   std::cout << "pqxx::failure: " << e.what() << std::endl;    }

    do
        try
        {
            auto FutureReady = [](auto &fu) -> bool
            {   return std::future_status::ready == fu.wait_for(std::chrono::milliseconds{100});    };
again:
            auto iter = std::find_if(taskR.begin(), taskR.end(), FutureReady);
            if(iter == taskR.cend())    goto again;

            std::vector<TestResultData> results = iter->get();  // or throw exception passed from thread

            taskR.erase(iter);
            for( auto tR : results )    std::cout << "Test symbol: " << tR.second.second.sym << "  Chart: " << tR.second.first << (tR.first ? " passed" : " failed") << std::endl;

        }
        catch(pqxx::range_error e) { std::cout << "range exception: " << e.what(); break; }
    while (taskR.size() > 0);

    return 0;
}
 
