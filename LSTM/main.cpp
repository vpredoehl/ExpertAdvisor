//
//  main.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 11/27/25.
//  Copyright © 2025 Vincent Predoehl. All rights reserved.
//


#include <iostream>
#include <string>
#include <pqxx/pqxx>

#include "db_cursor_iterator.hpp"
#include "Tensor.hpp"

const std::string dbName = "forex";


int main(int argc, const char * argv[])
{
    pqxx::connection c { "hostaddr=127.0.0.1 dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result tables = w.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };
    
    Batch b;

    try
    {
        for(auto tbl : tables)
        {
            std::string rawPriceTableName { tbl[0].c_str() };
            std::string query ="select * from " + rawPriceTableName + " where time between '" + fromDate + "' and '" + toDate + "' order by time;";
            db_cursor_stream<PricePoint> cur { w, query, rawPriceTableName + "_stream" };
            
                // print query for candlesticks
            query ="select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" +  fromDate + "', '" + toDate + "') order by dt;";
            db_cursor_stream<Feature> cs_cur { w, query, rawPriceTableName + "_candlestick_stream" };
            db_forward_iterator csb = cs_cur.cbegin(), cse = cs_cur.cend();

//            std::cout << "Candlesticks for table: " << rawPriceTableName << std::endl;
//            while(csb != cse)   std::cout << *(csb++) << std::endl;
            
            Tensor t { rawPriceTableName };
            std::cout << "Building tensor for table: " << rawPriceTableName << std::endl;
            while(csb != cse)
            {
                std::cout << *csb << std::endl;
                t.Add(*csb++);
            }
        }
    }
    catch(pqxx::failure e)  {   std::cout << "pqxx::failure: " << e.what() << std::endl; return 1;   }
    
    return 0;
}
