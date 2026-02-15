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
#include <limits>
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


static void ProcessBatchPredict(EA::LSTM& l, const Window& b)
{
    // 1) Get raw head outputs sequence (as trained)
    auto predsRaw = l.RollingPredictNextLogReturn(b, /*resetAtStart=*/true);

    // 2) Transform raw head outputs back to raw return according to metadata
    //    If model was trained with z-score on the target, invert it first.
    std::vector<float> predRel; predRel.reserve(predsRaw.size());
    for (float yhat : predsRaw)
    {
        float t = l.targetUseZScore ? (yhat * l.targetStd + l.targetMean) : yhat;
        float raw = (t - l.targetBias) / std::max(l.targetScale, 1e-12f);
        if (l.targetType == EA::LSTM::TargetType::LogReturn)
            predRel.push_back(std::exp(raw) - 1.0f);  // convert log-return to percent move
        else
            predRel.push_back(raw);                    // already percent move (fraction)
    }

    // 3) Build ground-truth relative moves for comparison
    std::vector<float> actRel; // (close_next - close_T) / close_T
    actRel.reserve(predRel.size());
    constexpr size_t closeCol = 1;
    for (auto it = b.begin(); it + window_size < b.end(); ++it)
    {
        float close_T    = (*(it + (window_size - 1)))(0, closeCol);
        float close_next = (*(it + window_size))(0, closeCol);
        actRel.push_back((close_next - close_T) / close_T);
    }

    // 4) Print prediction distribution stats to diagnose saturation
    auto stats = [](const auto& v){
        struct S { double min, max, mean, std, uniq; } s{};
        if (v.empty()) return s;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        double sum = 0.0, sumsq = 0.0;
        std::unordered_map<long long, int> buckets;
        buckets.reserve(v.size());
        for (float x : v)
        {
            mn = std::min(mn, static_cast<double>(x));
            mx = std::max(mx, static_cast<double>(x));
            sum += x; sumsq += static_cast<double>(x) * static_cast<double>(x);
            // simple bucketing by rounding to 1e-6
            long long key = static_cast<long long>(std::llround(static_cast<double>(x) * 1e6));
            ++buckets[key];
        }
        double n = static_cast<double>(v.size());
        double mean = sum / n;
        double var = std::max(0.0, sumsq / n - mean * mean);
        s.min = mn; s.max = mx; s.mean = mean; s.std = std::sqrt(var); s.uniq = static_cast<double>(buckets.size());
        return s;
    };

    auto s_raw = stats(predsRaw);
    auto s_rel = stats(predRel);
    std::cout << "pred_raw stats: min=" << s_raw.min << " max=" << s_raw.max
              << " mean=" << s_raw.mean << " std=" << s_raw.std
              << " uniq~=" << s_raw.uniq << std::endl;
    std::cout << "pred_rel stats: min=" << s_rel.min << " max=" << s_rel.max
              << " mean=" << s_rel.mean << " std=" << s_rel.std
              << " uniq~=" << s_rel.uniq << std::endl;

    // 5) Using the mapping, compute price movement metrics (predicted vs actual)
    std::vector<float> predMove; predMove.reserve(predRel.size());
    std::vector<float> actualMove; actualMove.reserve(predRel.size());
    size_t idx = 0;
    for (auto it = b.begin(); it + window_size < b.end(); ++it)
    {
        float close_T    = (*(it + (window_size - 1)))(0, closeCol);
        float close_next = (*(it + window_size))(0, closeCol);
        float rel = predRel[idx++];
        float pred_close_next = close_T * (1.0f + rel);
        predMove.push_back(pred_close_next - close_T);
        actualMove.push_back(close_next - close_T);
    }

    size_t N = std::min(predMove.size(), actualMove.size());
    const size_t toPrint = std::min<size_t>(N, 10);
    for (size_t i = 0; i < toPrint; ++i)
    {
        double diff = static_cast<double>(predMove[i]) - static_cast<double>(actualMove[i]);
        std::cout << "i=" << i
                  << " pred_move=" << predMove[i]
                  << " act_move=" << actualMove[i]
                  << " diff=" << diff
                  << std::endl;
    }
    double maeMove = 0.0; size_t correctDir = 0;
    for (size_t i = 0; i < N; ++i)
    {
        maeMove += std::abs(static_cast<double>(predMove[i]) - static_cast<double>(actualMove[i]));
        bool predUp = predMove[i] >= 0.0f; bool actUp = actualMove[i] >= 0.0f; if (predUp == actUp) ++correctDir;
    }
    if (N > 0)
    {
        std::cout << "Batch MAE (price movement): " << (maeMove / static_cast<double>(N))
                  << " | Direction accuracy: " << (static_cast<double>(correctDir) / static_cast<double>(N) * 100.0)
                  << "% over " << N << " predictions" << std::endl;
    }
}


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
            if constexpr (load_latest || inference_only)
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
            else    std::cout << "load_latest=false; using default-initialized parameters" << std::endl;

            
            // Iterate all batches (including trailing partial batch) and process each via CalculateBatch
            t.ForEachBatch( [&](auto b)
            {
                if constexpr (inference_only)   ProcessBatchPredict(l, b);
                else
                {
                    float pc = l.CalculateBatch(b);
                }
            } );
//            printMatrix("params", l.param);

            // Persist trained model parameters to DB
            try
            {
                if constexpr ( save_enable && !inference_only )
                {
                    // Ensure this transaction is read-write for saving
                    w_LSTM.exec("SET TRANSACTION READ WRITE;");
                    
                    long long modelId = -1;
                    if (startedFromScratch)
                        if constexpr (save_overwrite)
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
                            } catch (const std::exception& e)
                            {
                                std::cout << "Fetch latest model_id failed (" << e.what() << "); creating new snapshot" << std::endl;
                                modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            }
                        else
                        {
                            // Create a new snapshot when saving (do not overwrite existing)
                            modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            std::cout << "Created new model_id=" << modelId << " (started from scratch)" << std::endl;
                        }
                    else
                        if constexpr (save_overwrite)
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
                        else
                        {
                            // Create a new snapshot when saving (do not overwrite existing)
                            modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            std::cout << "Created new model_id=" << modelId << std::endl;
                        }

                    DBIO::PgModelIO::saveAll(w_LSTM, modelId, l);
                    w_LSTM.commit();
                    std::cout << "Saved model with model_id=" << modelId << std::endl;
                }
                else
                    if constexpr (!(save_enable && !inference_only))
                        std::cout << "save_enable=false; skipping model save" << std::endl;
                    else if constexpr (inference_only)
                        std::cout << "inference_only=true; skipping model save" << std::endl;
                    else
                        std::cout << "skipping model save (unknown reason)" << std::endl;
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

