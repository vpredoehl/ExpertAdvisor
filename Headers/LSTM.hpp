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
#include <cassert>

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
        W param; // Combined gate weights matrix with shape [n_in x 4*n_out]

        struct GateMatrixView {
            W& m;
            size_t colOffset;
            inline size_t rows() const { return static_cast<size_t>(n_out); }
            inline size_t cols() const { return static_cast<size_t>(n_in); }
            inline MetaNN::Shape<2> Shape() const { return MetaNN::Shape<2>(rows(), cols()); }
            inline float operator()(size_t r, size_t c) const { return m(c, colOffset + r); }
            inline void SetValue(size_t r, size_t c, float v) { m.SetValue(c, colOffset + r, v); }
        };

        struct ConstGateMatrixView {
            const W& m;
            size_t colOffset;
            inline size_t rows() const { return static_cast<size_t>(n_out); }
            inline size_t cols() const { return static_cast<size_t>(n_in); }
            inline MetaNN::Shape<2> Shape() const { return MetaNN::Shape<2>(rows(), cols()); }
            inline float operator()(size_t r, size_t c) const { return m(c, colOffset + r); }
        };

        inline GateMatrixView gateMatrix(size_t gateIndex) {
            assert(gateIndex < 4);
            return GateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
        }
        inline ConstGateMatrixView gateMatrix(size_t gateIndex) const {
            assert(gateIndex < 4);
            return ConstGateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
        }
        
        LSTM(float lt, float st);
        LSTM() = delete;
    };
}

#endif /* LSTM_hpp */

