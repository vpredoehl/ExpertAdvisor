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

#ifndef EARLY_STOP_PATIENCE
#define EARLY_STOP_PATIENCE 10
#endif

#include <MetaNN/operation/math/sigmoid.h>
#include <MetaNN/operation/math/tanh.h>
#include <MetaNN/operation/tensor/reshape.h>
#include <MetaNN/operation/tensor/slice.h>
#include "scalable_tensor.h"


static void ProcessBatchPredict(EA::LSTM& l, const Window& b)
{
    auto stats = [](const auto& v)
    {
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

    // 1) Get predicted log-returns (already de-normalized / de-affined inside PredictNextReturn)
    auto predLogRet = l.RollingPredictNextLogReturn(b, /*resetAtStart=*/true);
    
#if LSTM_DEBUG_PRINTS
    std::cout << "predLogRet samples: ";
    for (size_t i = 0; i < std::min<size_t>(10, predLogRet.size()); ++i) std::cout << predLogRet[i] << " ";
    std::cout << "\n";
#endif
    
    // 2) Convert predicted log-return -> relative move fraction
    std::vector<float> predRel; predRel.reserve(predLogRet.size());
    for (float logret : predLogRet) predRel.push_back(std::exp(logret) - 1.0f);

    // 3) Build ground-truth relative moves for comparison from feature log-return
    size_t gtOutliers = 0;
    std::vector<float> actRel, actLogRet; // percent move derived from log-return
    actRel.reserve(predRel.size()); actLogRet.reserve(predRel.size());
    for (auto it = b.begin(); it + window_size - 1 + prediction_horizon < b.end(); ++it)
    {
        const float v_scaled = (*(it + window_size - 1 + prediction_horizon))(0, closeCol);
        const float v_unscaled = v_scaled / EA::LSTM::kFeatScale;   // <-- MUST unscale to raw log-return
        
        if(std::fabs(v_unscaled) > 0.02f)    gtOutliers++;  // 2% log-return is already huge for many FX horizons
//        if (actRel.size() < 5)   std::cout << "GT raw feature v_logret=" << v_unscaled << " exp(v_logret)-1=" << (std::exp(v_unscaled) - 1.0f) << std::endl;
        actLogRet.push_back(v_unscaled);                    // <-- MUST be this exact v
        actRel.push_back(std::exp(v_unscaled) - 1.0f);      // <-- derived from same v
//        if (actRel.size() < 10) std::cout << "GT v_logret=" << v_unscaled
//                      << " sign=" << (v_unscaled>=0 ? "+" : "-")
//                      << " abs=" << std::fabs(v_unscaled) << std::endl;
    }
    auto s_gt = stats(actLogRet);
//    std::cout << "actLogRet stats: min=" << s_gt.min << " max=" << s_gt.max
//              << " mean=" << s_gt.mean << " std=" << s_gt.std
//              << " uniq~=" << s_gt.uniq << std::endl << "GT outliers |v|>0.02: " << gtOutliers
//              << " of " << actLogRet.size() << std::endl;
    assert(predRel.size() == actRel.size());    // DEBUG


//#if LSTM_DEBUG_PRINTS
    // 4) Print prediction distribution stats to diagnose saturation
    auto s_raw = stats(predLogRet);
    auto s_rel = stats(predRel);
    std::cout << "pred_raw stats: min=" << s_raw.min << " max=" << s_raw.max
              << " mean=" << s_raw.mean << " std=" << s_raw.std
              << " uniq~=" << s_raw.uniq << std::endl;
    std::cout << "std ratio (pred_raw/actLogRet): " << (s_gt.std > 0.0 ? (s_raw.std / s_gt.std) : 0.0) << std::endl;
    std::cout << "means: pred_raw=" << s_raw.mean << " actLogRet=" << s_gt.mean << std::endl;

    std::cout << "pred_rel stats: min=" << s_rel.min << " max=" << s_rel.max
              << " mean=" << s_rel.mean << " std=" << s_rel.std
              << " uniq~=" << s_rel.uniq << std::endl;
//#endif
    
    // 5) Compare predicted vs actual relative moves directly (fractions)
    std::vector<float> predMove = predRel; // alias copy
    
    // DIAG: sign flip test (set to 1 to test)
    #if 0
        for (auto& x : predMove) x = -x;
    #endif

    std::vector<float> actualMove = actRel; // alias copy
    size_t N = std::min(predMove.size(), actualMove.size());
    const size_t toPrint = std::min<size_t>(N, 10);

    // Direction accuracy in log-return domain with an acted threshold
    size_t actedLog = 0;
    size_t correctLog = 0;
    const float actedThrLog = 2e-4;
    for (size_t i = 0; i < N && i < predLogRet.size() && i < actLogRet.size(); ++i)
    {
        if (std::fabs(predLogRet[i]) < actedThrLog) continue; // only act on confident predictions
        ++actedLog;
        bool predUp = predLogRet[i] >= 0.0f;
        bool actUp  = actLogRet[i]  >= 0.0f;
        if (predUp == actUp) ++correctLog;
    }
    double accLog = actedLog ? (static_cast<double>(correctLog) / static_cast<double>(actedLog) * 100.0) : 0.0;
    double covLog = N ? (static_cast<double>(actedLog) / static_cast<double>(N) * 100.0) : 0.0;
    std::cout << "Direction accuracy (log-return): " << accLog
              << "% over " << actedLog << " acted (of " << N << ")"
              << " thr=" << actedThrLog
              << " coverage=" << covLog << "%" << std::endl;

//#if LSTM_DEBUG_PRINTS
    const size_t M = std::min<size_t>(5, std::min(predLogRet.size(), actLogRet.size()));
    for (size_t i = 0; i < M; ++i)  std::cout << "align i=" << i << " predLogRet=" << predLogRet[i]  << " actLogRet(unscaled)=" << actLogRet[i] << " actRel=" << (std::exp(actLogRet[i]) - 1.0f) << "\n";
    for (size_t i = 0; i < toPrint; ++i)
    {
        auto sgn = [](float x){ return (x > 0) - (x < 0); };
        int sp = sgn(predMove[i]);
        int sa = sgn(actualMove[i]);
        std::cout << "i=" << i
            << " pred_rel=" << predMove[i]
            << " act_rel=" << actualMove[i]
            << " sp=" << sp
            << " sa=" << sa
            << " match=" << (sp == sa)
            << " |pred|=" << std::abs(predMove[i])
            << " |act|=" << std::abs(actualMove[i])
            << "\n";
    }
//#endif
    double maeMove = 0.0;
    size_t correctDir = 0;
    size_t acted = 0;

    // Ignore tiny predictions (no-signal zone)
    const float predThr = 1e-4f;
    const float actThr  = 1e-4f;   // optional: ignore tiny true moves too

    for (size_t i = 0; i < N; ++i)
    {
        maeMove += std::abs(static_cast<double>(predMove[i]) -
                            static_cast<double>(actualMove[i]));

        if (std::fabs(predMove[i]) < predThr) continue;
        if (std::fabs(actualMove[i]) < actThr) continue;

        ++acted;

        bool predUp = predMove[i] >= 0.0f;
        bool actUp  = actualMove[i] >= 0.0f;
        if (predUp == actUp) ++correctDir;
    }

    if (N > 0)  std::cout << "Batch MAE (relative move fraction): "
                  << (maeMove / static_cast<double>(N))
                  << " | Direction accuracy: "
                  << (acted ? (static_cast<double>(correctDir) /
                               static_cast<double>(acted) * 100.0)
                            : 0.0)
                  << "% over " << acted << " acted (of " << N << ")"
                  << " predThr=" << predThr
                  << " actThr=" << actThr
                  << " coverage=" << (static_cast<double>(acted) / static_cast<double>(N) * 100.0) << "%"
                  << std::endl;
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
            
            std::cout << "Candlestick query: " << query << "\n";
            std::cout << "Building tensor for table: " << rawPriceTableName << std::endl;
            while (csb != cse) t.Add(*csb++);
  
            thread_local EA::LSTM l { t, 1, 0 };

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
            std::cout << std::setprecision(15);
            for(auto e = 0; e < epoch_count; e++)
            t.ForEachBatch( [&](auto b)
            {
                if constexpr (inference_only)   ProcessBatchPredict(l, b);
                else
                {
                    auto l2 = [](const auto& m){
                        auto low = MetaNN::LowerAccess(m);
                        const float* p = low.RawMemory();
                        size_t len = m.Shape()[0]*m.Shape()[1];
                        double s=0; for(size_t i=0;i<len;++i){ double v=p[i]; s += v*v; }
                        return std::sqrt(s);
                    };

                    double p0 = l2(l.param);
                    double b0 = l2(l.bias);
                    double hw0 = l2(l.returnHeadWeight);
                    double hb0 = l2(l.returnHeadBias);
                    double dhw0 = l2(l.returnHeadDirWeight);
                    double dhb0 = l2(l.returnHeadDirBias);

                    auto [ loss, _, _]  = l.CalculateBatch(b);

                    double p1 = l2(l.param);
                    double b1 = l2(l.bias);
                    double hw1 = l2(l.returnHeadWeight);
                    double hb1 = l2(l.returnHeadBias);
                    double dhw1 = l2(l.returnHeadDirWeight);
                    double dhb1 = l2(l.returnHeadDirBias);

                    std::cout << "epoch " << (e+1)
                              << " loss=" << loss
                              << " ||param|| " << p0  << " -> " << p1
                              << " ||bias|| "  << b0  << " -> " << b1;
                    if (l.targetType == EA::LSTM::TargetType::BinaryReturn) std::cout << " ||dirHeadW|| " << dhw0 << " -> " << dhw1 << " ||dirHeadB|| " << dhb0 << " -> " << dhb1 << std::endl;
                    else std::cout << " ||headW|| " << hw0 << " -> " << hw1 << " ||headB|| " << hb0 << " -> " << hb1 << std::endl;
                }
            } );

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

