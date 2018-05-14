//
//  main.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 5/13/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include <iostream>

int main(int argc, const char * argv[]) {

        //
        // make test charts
        //
    std::for_each(allSyms.begin(), allSyms.end(), [](const SymbolData::value_type &m)
                  {
                      auto symCharts = MakeTestCharts(m.second);
                  });
    
    return 0;
}
