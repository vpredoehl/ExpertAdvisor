//
//  Tensor.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/3/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include "Tensor.hpp"

void Tensor::Add(Feature pp)
{
    short posn = seq_size++ - 1;
    
    for(auto &w : window_seq)
    {
        w[posn] = pp;
        if(posn-- == 0) // first window is complete
        {
            if(seq_size == window_size + 1)
            {
                b.push_back(window_seq.front());    // add window to batch
                window_seq.pop_front();
                window_seq.emplace_back(Window{});
                seq_size = window_size;
            }
            break;
        }
    }
}
