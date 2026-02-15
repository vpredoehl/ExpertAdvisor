//
//  main.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 11/27/25.
//  Copyright © 2025 Vincent Predoehl. All rights reserved.
//

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <pqxx/pqxx>

#include <device_tags.h>
#include <tensor.h>
#include <permute.h>
#include <conv.h>

#include "db_cursor_iterator.hpp"
#include "Tensor.hpp"
#include "LSTM.hpp"
#include "PgModelIO.hpp"

#include <MetaNN/operation/math/sigmoid.h>
#include <MetaNN/operation/math/tanh.h>
#include <MetaNN/operation/tensor/reshape.h>
#include <MetaNN/operation/tensor/slice.h>
#include "scalable_tensor.h"

const std::string dbName = "forex";
const std::string dbModelName = "LSTM";


int main(int argc, const char * argv[])
{
    pqxx::connection c_forex { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::connection c_LSTM { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbModelName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w_forex { c_forex }, w_LSTM { c_LSTM };
    pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };
    

    w_LSTM.exec("SET TRANSACTION READ WRITE;");
    try
    {
        pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
        std::string fromDate{ argv[1] }, toDate{ argv[2] };
        for (auto tbl : tables)
        {
            std::string rawPriceTableName{ tbl[0].c_str() };
            std::string query = "select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" + fromDate + "', '" + toDate + "') order by dt;";
            db_cursor_stream<Feature> cs_cur{ w_forex, query, rawPriceTableName + "_candlestick_stream" };
            db_forward_iterator csb = cs_cur.cbegin(), cse = cs_cur.cend();

            Tensor t{ rawPriceTableName };
            std::cout << "Building tensor for table: " << rawPriceTableName << std::endl;
            while (csb != cse) t.Add(*csb++);
  
            EA::LSTM l { t, 1, 0 };
            try {   DBIO::PgModelIO::loadAll(w_LSTM, 1, l); }
            catch(std::exception& e)    {   std::cout << e.what() << " using default params" << std::endl;    }
            // Iterate all batches (including trailing partial batch) and process each via CalculateBatch
            t.ForEachBatch( [&](auto b)
            {
                float pc = l.CalculateBatch(b);
                auto lastFeat = b.end();
//                printMatrix("lastFeat: ", *lastFeat);
//                std::cout << "lastFeat: " << (*lastFeat)(0,1) << "  predicted_close: " << pc << std::endl;
            } );
            printMatrix("params", l.param);

            // Persist trained model parameters to DB
            try
            {
                // Use a dedicated read-write transaction on the LSTM database to save

                long long modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                DBIO::PgModelIO::saveAll(w_LSTM, modelId, l);
                w_LSTM.commit();
                std::cout << "Saved model with model_id=" << modelId << std::endl;
            }
            catch (const std::exception& e) { std::cerr << "Model save/load error: " << e.what() << std::endl;    }

            break;
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

