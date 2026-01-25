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
                {
                    const auto& pShape = l.param.Shape();
                    std::cout << "Param shape: rows=" << pShape[0]
                              << " cols=" << pShape[1] << std::endl;

                    printMatrix("W_i (n_out x n_in)", l.gateMatrix(0));
                    printMatrix("W_f (n_out x n_in)", l.gateMatrix(1));
                    printMatrix("W_g (n_out x n_in)", l.gateMatrix(2));
                    printMatrix("W_o (n_out x n_in)", l.gateMatrix(3));

                    // Build z_t = [x_t, h_{t-1}] with h_{t-1} = 0 for now.
                    // z has shape (1 x n_in) = (1 x (feature_size + hidden_size))
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> z(1, n_in);
                    const size_t featWidth = f.Shape()[1];
                    // Copy feature part
                    for (size_t j = 0; j < featWidth; ++j)
                    {
                        z.SetValue(0, j, f(0, j));
                    }
                    // Append previous hidden state (size hidden_size)
                    for (size_t j = 0; j < hidden_size; ++j)
                    {
                        z.SetValue(0, featWidth + j, h_prev(0, j));
                    }

                    // Compute (1 x n_in) · (n_in x 4*n_out) = (1 x 4*n_out)
                    auto yOp = MetaNN::Dot(z, l.param);
                    auto y = Evaluate(yOp);
                    printMatrix("Y (concat gates)", y);

                    const size_t totalCols = y.Shape()[1];
                    const size_t gateWidth = totalCols / 4;

                    // Split into four gates: i, f, g, o each (1 x n_out)
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> gate_i(1, gateWidth);
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> gate_f(1, gateWidth);
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> gate_g(1, gateWidth);
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> gate_o(1, gateWidth);

                    for (size_t j = 0; j < gateWidth; ++j)
                    {
                        gate_i.SetValue(0, j, y(0, j));
                        gate_f.SetValue(0, j, y(0, gateWidth + j));
                        gate_g.SetValue(0, j, y(0, 2 * gateWidth + j));
                        gate_o.SetValue(0, j, y(0, 3 * gateWidth + j));
                    }

                    printMatrix("Gate i", gate_i);
                    printMatrix("Gate f", gate_f);
                    printMatrix("Gate g", gate_g);
                    printMatrix("Gate o", gate_o);

                    // Apply activations using MetaNN ops
                    auto i_expr = MetaNN::Sigmoid(gate_i);
                    auto f_expr = MetaNN::Sigmoid(gate_f);
                    auto g_expr = MetaNN::Tanh(gate_g);
                    auto o_expr = MetaNN::Sigmoid(gate_o);

                    auto i_h = i_expr.EvalRegister();
                    auto f_h = f_expr.EvalRegister();
                    auto g_h = g_expr.EvalRegister();
                    auto o_h = o_expr.EvalRegister();
                    MetaNN::EvalPlan::Inst().Eval();

                    const auto& i_t = i_h.Data();
                    const auto& f_t = f_h.Data();
                    const auto& g_t = g_h.Data();
                    const auto& o_t = o_h.Data();

                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> c_t(1, gateWidth);
                    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> h_t(1, gateWidth);

                    for (size_t j = 0; j < gateWidth; ++j)
                    {
                        float c_val = f_t(0, j) * c_prev(0, j) + i_t(0, j) * g_t(0, j);
                        c_t.SetValue(0, j, c_val);
                    }

                    auto ct_tanh_expr = MetaNN::Tanh(c_t);
                    auto ct_tanh_h = ct_tanh_expr.EvalRegister();
                    MetaNN::EvalPlan::Inst().Eval();
                    const auto& ct_tanh = ct_tanh_h.Data();

                    for (size_t j = 0; j < gateWidth; ++j)
                    {
                        h_t.SetValue(0, j, o_t(0, j) * ct_tanh(0, j));
                    }

                    // Update persistent states
                    for (size_t j = 0; j < gateWidth; ++j)
                    {
                        c_prev.SetValue(0, j, c_t(0, j));
                        h_prev.SetValue(0, j, h_t(0, j));
                    }
                }
            }
            break;
            //            for ( auto m : l.param)   printMatrix("l", m);
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

