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
using CandleTime = time_point<system_clock, minutes>;


std::istream& operator>>(std::istream& i, CandleTime &t)
{
    struct tm ct;

    i >> std::get_time(&ct, "%F %R:%S");
    t = time_point_cast<minutes>(std::chrono::system_clock::from_time_t(mktime(&ct)));
    return i;
}

int main(int argc, const char * argv[]) {
    std::ifstream csv { "COR_USD_Week3.csv", std::ios_base::in };
    std::string l;
    CandleTime t;
    time_t tt;
    
    csv >> l;   // header line
    
    std::getline(csv, l, ',');
    std::getline(csv, l, ',');
    std::getline(csv, l, ',');
    csv >> t;
    tt = system_clock::to_time_t(t);
    std::cout << std::put_time(localtime(&tt), "%F");
    std::getline(csv, l, ',');

    csv >> l;
    
    return 0;
}
