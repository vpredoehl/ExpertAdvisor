//
//  LSTM.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef LSTM_hpp
#define LSTM_hpp

#include <MetaNN/meta_nn.h>
#include <array>

using std::array;

constexpr auto hidden_size = 8;
constexpr auto feature_size = 4;
constexpr auto n_in = feature_size + hidden_size;
constexpr auto n_out = hidden_size;

namespace EA
{
    class LSTM
    {
        using W = MetaNN::Matrix<float, MetaNN::DeviceTags::CPU>;
        
        float Forget()  // calculate forget module
        {
            return 0;
        }
        
    public:
        float long_term, short_term, in;
        array<W,4> param;
        
        LSTM(float lt, float st);
        LSTM() = delete;
    };
}

#endif /* LSTM_hpp */
