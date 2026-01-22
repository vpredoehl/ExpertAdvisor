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
            
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> w_feat(4, l.param[0].Shape()[0]);
            {
                const size_t rowsW = l.param[0].Shape()[0]; // n_out
                for (size_t k = 0; k < 4; ++k)
                {
                    for (size_t j = 0; j < rowsW; ++j)
                    {
                        w_feat.SetValue(k, j, l.param[0](j, k));
                    }
                }
            }
            
            for (const auto& f : w)
            {
                MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> feature(1, 4);
                feature.SetValue(0, 0, f.open);
                feature.SetValue(0, 1, f.close);
                feature.SetValue(0, 2, f.high);
                feature.SetValue(0, 3, f.low);
                printMatrix("Feature", feature);
                printMatrix("Param", l.param[0]);
                // Compute (1x4) · (4xn_out) = (1xn_out) via Dot
                auto outOp = MetaNN::Dot(feature, w_feat);
                auto out = Evaluate(outOp);
                printMatrix("Out", out);
                std::cout << f << std::endl;
                
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

