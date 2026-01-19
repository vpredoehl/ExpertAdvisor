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

#include <device_tags.h>
#include <tensor.h>

#include "db_cursor_iterator.hpp"
#include "Tensor.hpp"
#include "LSTM.hpp"

const std::string dbName = "forex";


int main(int argc, const char * argv[])
{
    pqxx::connection c { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result tables = w.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };
    
    EA::LSTM l { 1, 0 };
    for ( auto m : l.param)   printMatrix("l", m);

    try
    {
        pqxx::result tables = w.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
        std::string fromDate{ argv[1] }, toDate{ argv[2] };
        for (auto tbl : tables)
        {
            std::string rawPriceTableName{ tbl[0].c_str() };
            std::string query = "select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" + fromDate + "', '" + toDate + "') order by dt;";
            db_cursor_stream<Feature> cs_cur{ w, query, rawPriceTableName + "_candlestick_stream" };
            db_forward_iterator csb = cs_cur.cbegin(), cse = cs_cur.cend();

            Tensor t{ rawPriceTableName, 5 };
            std::cout << "Building tensor for table: " << rawPriceTableName << std::endl;
            while (csb != cse) t.Add(*csb++);

            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> A(4, 4);
            printMatrix("A", A);
        }
    }
    catch (const pqxx::broken_connection& e)
    {
        std::cerr << "Broken connection: " << e.what() << "\n";
        return 1;
    }
    catch (const pqxx::failure& e)
    {
        std::cerr << "pqxx::failure: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "std::exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

