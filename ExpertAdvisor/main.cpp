//
//  main.cpp
//  ExpertAdvisor
//
//  Created by Vincent Predoehl on 2/28/18.
//  Copyright © 2018 Vincent Predoehl. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace std::chrono;
using tp = time_point<system_clock, seconds>;


std::istream& operator>>(std::istream& i, struct tm &t)
{
    i >> std::get_time(&t, "%F %R:%S");
    return i;
}

int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    std::string l;
    struct tm t;
    
    csv >> l;   // header line
    
    std::getline(csv, l, ',');
    std::getline(csv, l, ',');
    std::getline(csv, l, ',');
    csv >> t;
    std::cout << std::put_time(&t, "%Y:%m %R");
    std::getline(csv, l, ',');

    csv >> l;
    
    return 0;
}
