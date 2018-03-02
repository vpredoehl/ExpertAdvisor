//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include "CandleStick.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>


int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    CandleStick::CandleTime t;
    CandleStick cs;
    std::string l;
    time_t tt;
    
    csv >> l;   // header line

    csv >> cs;
    
    std::cout << cs << std::endl;
    
    return 0;
}
