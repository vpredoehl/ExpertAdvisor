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

using namespace std::chrono;

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


int main(int argc, const char * argv[])
{
    SymbolData allSyms = SymsFromDirectory(savePath);
        //
        // make test charts
        //
    std::for_each(allSyms.begin(), allSyms.end(), [](const SymbolData::value_type &m)
                  {
                      auto symCharts = MakeTestCharts(m.second);
                  });

    return 0;
}
