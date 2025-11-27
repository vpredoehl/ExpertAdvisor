//
//  main.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 11/27/25.
//  Copyright © 2025 Vincent Predoehl. All rights reserved.
//

#include "DiskIO.hpp"
#include "rmp_cursor_iterator.hpp"

#include <iostream>
#include <string>
#include <pqxx/pqxx>

int main(int argc, const char * argv[])
{
    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result tables = w.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };

    try
    {
        for(auto t : tables)
        {
            std::string rawPriceTableName { t[0].c_str() };
            std::string query ="select * from " + rawPriceTableName + " where time between '" + fromDate + "' and '" + toDate + "' order by time;";
            rmp_cursor_stream cur { w, query, rawPriceTableName + "_stream" };
            rmp_forward_iterator cb = cur.cbegin(), ce = cur.cend();

            std::cout << "Showing table: " << rawPriceTableName << std::endl;
            while(cb != ce) std::cout << *(cb++) << std::endl;
        }
    }
    catch(pqxx::failure e)  {   std::cout << "pqxx::failure: " << e.what() << std::endl; return 1;   }

    return 0;
}
