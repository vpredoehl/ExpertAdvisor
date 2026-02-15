#pragma once

// Centralized build-time configuration for the LSTM project.
// Adjust these defaults in one place or override via target build settings
// (Preprocessor Macros) as needed.

// 1: compile out training logic and run forward-only inference paths
#ifndef LSTM_INFERENCE_ONLY
#define LSTM_INFERENCE_ONLY 1
#endif

// 1: load latest model from DB at startup; 0: start from scratch
#ifndef LSTM_LOAD_LATEST
#define LSTM_LOAD_LATEST 1
#endif

// Default: enable save in training builds, disable in inference-only builds
#ifndef LSTM_SAVE_ENABLE
  #if LSTM_INFERENCE_ONLY
    #define LSTM_SAVE_ENABLE 0
  #else
    #define LSTM_SAVE_ENABLE 1
  #endif
#endif

// 1: overwrite the loaded/latest model_id when saving; 0: create a new model snapshot
#ifndef LSTM_SAVE_OVERWRITE
#define LSTM_SAVE_OVERWRITE 0
#endif
