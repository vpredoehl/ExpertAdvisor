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
#include <optional>
#include <pqxx/pqxx>

#include <device_tags.h>
#include <tensor.h>
#include <permute.h>
#include <conv.h>

#include "db_cursor_iterator.hpp"
#include "Tensor.hpp"
#include "LSTM.hpp"
#include "PgModelIO.hpp"
#include "BuildConfig.hpp"

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

            // Track whether we started from scratch (no model loaded)
            std::optional<long long> loadedModelId;
            bool startedFromScratch = true;

            // Optionally load the latest model parameters from the LSTM DB
#if LSTM_LOAD_LATEST
            try
            {
                pqxx::result r = w_LSTM.exec("SELECT max(model_id) FROM model;");
                if (!r.empty() && !r[0][0].is_null())
                {
                    loadedModelId = r[0][0].as<long long>();
                    DBIO::PgModelIO::loadAll(w_LSTM, *loadedModelId, l);
                    startedFromScratch = false;
                    std::cout << "Loaded model_id=" << *loadedModelId << std::endl;
                }
                else {  std::cout << "No models found; using default-initialized parameters" << std::endl;  }
            }
            catch (const std::exception& e) {   std::cout << "Load latest failed: " << e.what() << "; using default params" << std::endl;   }
#else
            std::cout << "LSTM_LOAD_LATEST=0; using default-initialized parameters" << std::endl;
#endif

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
#if LSTM_SAVE_ENABLE
                // Ensure this transaction is read-write for saving
                w_LSTM.exec("SET TRANSACTION READ WRITE;");

                long long modelId = -1;
                if (startedFromScratch)
                {
                #if LSTM_SAVE_OVERWRITE
                    // Overwrite the latest model if one exists; otherwise create a new snapshot
                    try {
                        pqxx::result rLatest = w_LSTM.exec("SELECT max(model_id) FROM model;");
                        if (!rLatest.empty() && !rLatest[0][0].is_null()) {
                            modelId = rLatest[0][0].as<long long>();
                            std::cout << "Overwriting latest model_id=" << modelId << " (started from scratch, overwrite enabled)" << std::endl;
                        } else {
                            modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            std::cout << "Created new model_id=" << modelId << " (no existing model to overwrite)" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cout << "Fetch latest model_id failed (" << e.what() << "); creating new snapshot" << std::endl;
                        modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                    }
                #else
                    // Create a new snapshot when saving (do not overwrite existing)
                    modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                    std::cout << "Created new model_id=" << modelId << " (started from scratch)" << std::endl;
                #endif
                }
                else
                {
                #if LSTM_SAVE_OVERWRITE
                    if (loadedModelId.has_value())
                    {
                        modelId = *loadedModelId;
                        std::cout << "Overwriting existing model_id=" << modelId << std::endl;
                    }
                    else
                    {
                        modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                        std::cout << "Created new model_id=" << modelId << " (no prior model to overwrite)" << std::endl;
                    }
                #else
                    // Create a new snapshot when saving (do not overwrite existing)
                    modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                    std::cout << "Created new model_id=" << modelId << std::endl;
                #endif
                }
                DBIO::PgModelIO::saveAll(w_LSTM, modelId, l);
                w_LSTM.commit();
                std::cout << "Saved model with model_id=" << modelId << std::endl;
#else
                (void)l; // unused in inference-only save-disabled builds
                std::cout << "LSTM_SAVE_ENABLE=0; skipping model save" << std::endl;
#endif
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

