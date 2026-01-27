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

#include <MetaNN/operation/math/sigmoid.h>
#include <MetaNN/operation/math/tanh.h>
#include <MetaNN/operation/tensor/reshape.h>
#include <MetaNN/operation/tensor/slice.h>
#include "scalable_tensor.h"

const std::string dbName = "forex";


int main(int argc, const char * argv[])
{
    pqxx::connection c { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w { c };
    pqxx::result tables = w.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };
    

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
  
            EA::LSTM l { 1, 0 };
            Window w = t.GetWindow(0);
            
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> h_prev(1, hidden_size);
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> c_prev(1, hidden_size);
            for (size_t j = 0; j < hidden_size; ++j) {
                h_prev.SetValue(0, j, 0.0f);
                c_prev.SetValue(0, j, 0.0f);
            }
            
            for (const auto& f : w)
            {
                printMatrix("Feature", f);
                /*
                Debugging guidance: You can comment out the following for quick structural checks:
                - The affine transform and gates: lines from `auto yExpr = MetaNN::Dot(z, l.param);` down to the gate activation prints.
                - The LSTM state update: lines computing `c_t_expr`, `h_t_expr`, their reshapes, evaluation, and assignments back to `c_prev`/`h_prev`.
                If you comment those out, keep the `z` construction and `printMatrix` calls to verify inputs and previous states.
                */
                {
                    const size_t featWidth = f.Shape()[1];
                    // Concatenate [x_t, h_{t-1}] into a contiguous row without element-wise loops
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> z(1, n_in);
                    {
                        auto low_z = MetaNN::LowerAccess(z);
                        float* z_mem = low_z.MutableRawMemory();

                        auto low_f = MetaNN::LowerAccess(f);
                        const float* f_mem = low_f.RawMemory();
                        std::copy(f_mem, f_mem + featWidth, z_mem);

                        auto low_h = MetaNN::LowerAccess(h_prev);
                        const float* h_mem = low_h.RawMemory();
                        std::copy(h_mem, h_mem + hidden_size, z_mem + featWidth);
                    }
                    printMatrix("Concat z [x_t|h_{t-1}]", z);

                    // Heavy compute: affine transform. Comment out along with subsequent gate logic if you only need to verify data flow.
                    auto yExpr = MetaNN::Dot(z, l.param);

                    // Split gates as a (4 x gateWidth) view
                    const size_t gateWidth = yExpr.Shape()[1] / 4;
                    auto gates2D = MetaNN::Reshape(yExpr, MetaNN::Shape(4, gateWidth));

                    // Gate activations directly on 1D slices
                    auto i = MetaNN::Sigmoid(gates2D[0]);
                    auto f = MetaNN::Sigmoid(gates2D[1]);
                    auto g = MetaNN::Tanh   (gates2D[2]);
                    auto o = MetaNN::Sigmoid(gates2D[3]);

                    // Debug: evaluate and print gate activations
                    {
                        auto i_h = i.EvalRegister();
                        auto f_h = f.EvalRegister();
                        auto g_h = g.EvalRegister();
                        auto o_h = o.EvalRegister();
                        MetaNN::EvalPlan::Inst().Eval();
                        auto i_m = MetaNN::Reshape(i_h.Data(), MetaNN::Shape(1, gateWidth));
                        auto f_m = MetaNN::Reshape(f_h.Data(), MetaNN::Shape(1, gateWidth));
                        auto g_m = MetaNN::Reshape(g_h.Data(), MetaNN::Shape(1, gateWidth));
                        auto o_m = MetaNN::Reshape(o_h.Data(), MetaNN::Shape(1, gateWidth));
                        printMatrix("Gate i (input)", i_m);
                        printMatrix("Gate f (forget)", f_m);
                        printMatrix("Gate g (cell candidate)", g_m);
                        printMatrix("Gate o (output)", o_m);
                    }

                    printMatrix("h_prev", h_prev);
                    printMatrix("c_prev", c_prev);

                    // View c_prev as 1D
                    auto c_prev_1d = MetaNN::Reshape(c_prev, MetaNN::Shape(gateWidth));

                    // Vectorized LSTM updates
                    auto c_t_expr = f * c_prev_1d + i * g;                 // 1D
                    auto h_t_expr = o * MetaNN::Tanh(c_t_expr);            // 1D

                    // Reshape results to (1 x gateWidth) and evaluate once
                    auto c_t_2d_expr = MetaNN::Reshape(c_t_expr, MetaNN::Shape(1, gateWidth));
                    auto h_t_2d_expr = MetaNN::Reshape(h_t_expr, MetaNN::Shape(1, gateWidth));
                    auto c_t_h = c_t_2d_expr.EvalRegister();
                    auto h_t_h = h_t_2d_expr.EvalRegister();
                    MetaNN::EvalPlan::Inst().Eval();

                    // Materialize evaluated handles into concrete matrices for printing
                    auto c_t_m = MetaNN::Reshape(c_t_h.Data(), MetaNN::Shape(1, gateWidth));
                    auto h_t_m = MetaNN::Reshape(h_t_h.Data(), MetaNN::Shape(1, gateWidth));
                    printMatrix("c_t", c_t_m);
                    printMatrix("h_t", h_t_m);

                    // Assign back to persistent states
                    c_prev = c_t_h.Data();
                    h_prev = h_t_h.Data();
                }
            }
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

