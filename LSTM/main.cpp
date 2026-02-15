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
    // 1) Get raw head outputs in log-return space (as trained)
    auto predsLogRet = l.RollingPredictNextLogReturn(b, /*resetAtStart=*/true);

    // 2) Build ground-truth relative moves for comparison
    std::vector<float> actRel; // (close_next - close_T) / close_T
    actRel.reserve(predsLogRet.size());
    constexpr size_t closeCol = 1;
    for (auto it = b.begin(); it + window_size < b.end(); ++it)
    {
        float close_T    = (*(it + (window_size - 1)))(0, closeCol);
        float close_next = (*(it + window_size))(0, closeCol);
        actRel.push_back((close_next - close_T) / close_T);
    }

    // 3) Evaluate candidate inverse mappings on relative move
    auto corr = [](const auto& a, const auto& b){
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
        double meanA=0, meanB=0; size_t N=a.size();
        for (size_t i=0;i<N;++i){ meanA+=a[i]; meanB+=b[i]; }
        meanA/=N; meanB/=N;
        double num=0, denA=0, denB=0;
        for (size_t i=0;i<N;++i){ double da=a[i]-meanA, db=b[i]-meanB; num+=da*db; denA+=da*da; denB+=db*db; }
        return (denA>0 && denB>0) ? (num / std::sqrt(denA*denB)) : 0.0;
    };
    auto mae = [](const auto& a, const auto& b){
        if (a.empty() || b.empty() || a.size() != b.size()) return std::numeric_limits<double>::infinity();
        double s=0; for (size_t i=0;i<a.size();++i) s += std::abs((double)a[i] - (double)b[i]); return s/a.size();
    };

    enum class TargetType { LogReturn, PercentReturn };
    struct Candidate { const char* name; TargetType type; float scale; float bias; std::vector<float> predRel; double mae; double corr; };
    std::vector<Candidate> cands;
    cands.reserve(4);

    // A) Assume head predicted 100 * log-return
    {
        Candidate c{"100x logret", TargetType::LogReturn, 100.0f, 0.0f, {}, 0.0, 0.0};
        c.predRel.reserve(predsLogRet.size());
        for (float yhat : predsLogRet) c.predRel.push_back(std::exp(yhat / 100.0f) - 1.0f);
        c.mae  = mae(actRel, c.predRel);
        c.corr = corr(actRel, c.predRel);
        cands.push_back(std::move(c));
    }
    // B) Assume head predicted raw log-return
    {
        Candidate c{"1x logret", TargetType::LogReturn, 1.0f, 0.0f, {}, 0.0, 0.0};
        c.predRel.reserve(predsLogRet.size());
        for (float yhat : predsLogRet) c.predRel.push_back(std::exp(yhat) - 1.0f);
        c.mae  = mae(actRel, c.predRel);
        c.corr = corr(actRel, c.predRel);
        cands.push_back(std::move(c));
    }
    // C) Assume head predicted 100 * percent return
    {
        Candidate c{"100x pct", TargetType::PercentReturn, 100.0f, 0.0f, {}, 0.0, 0.0};
        c.predRel.reserve(predsLogRet.size());
        for (float yhat : predsLogRet) c.predRel.push_back(yhat / 100.0f);
        c.mae  = mae(actRel, c.predRel);
        c.corr = corr(actRel, c.predRel);
        cands.push_back(std::move(c));
    }
    // D) Assume head predicted raw percent return (fraction)
    {
        Candidate c{"1x pct", TargetType::PercentReturn, 1.0f, 0.0f, {}, 0.0, 0.0};
        c.predRel.reserve(predsLogRet.size());
        for (float yhat : predsLogRet) c.predRel.push_back(yhat);
        c.mae  = mae(actRel, c.predRel);
        c.corr = corr(actRel, c.predRel);
        cands.push_back(std::move(c));
    }

    // 4) Pick best by MAE (break ties by higher correlation)
    auto best = std::min_element(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b){
        if (a.mae == b.mae) return a.corr > b.corr; // prefer higher correlation on tie
        return a.mae < b.mae;
    });

    if (best != cands.end())
    {
        std::cout << "Auto-selected target mapping: " << best->name
                  << "  (MAE=" << best->mae << ", corr=" << best->corr << ")" << std::endl;
        // Configure model's target transform for downstream PredictNextClose
        // Skipped: EA::LSTM does not expose SetTargetTransform/TargetType in this build.
        // We will continue using the locally selected TargetType (best->type) and scale to invert predictions below.
        (void)best; // suppress unused warning if configuration is skipped
    }

    // 5) Using the selected mapping, compute price movement metrics (predicted vs actual)
    std::vector<float> predMove; predMove.reserve(predsLogRet.size());
    std::vector<float> actualMove; actualMove.reserve(predsLogRet.size());
    size_t idx = 0;
    for (auto it = b.begin(); it + window_size < b.end(); ++it)
    {
        float close_T    = (*(it + (window_size - 1)))(0, closeCol);
        float close_next = (*(it + window_size))(0, closeCol);
        float yhat = predsLogRet[idx++];

        // Invert affine transform: target_raw = (yhat - bias) / scale
        float target_raw = (yhat - 0.0f) / std::max(best->scale, 1e-12f);
        float pred_close_next = 0.0f;
        if (best->type == TargetType::LogReturn)
            pred_close_next = std::exp(target_raw) * close_T;
        else
            pred_close_next = close_T * (1.0f + target_raw);

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
                else std::cout << "save_enabled=false; skipping model save" << std::endl;
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

