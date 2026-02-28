#pragma once

// Centralized build-time configuration for the LSTM project.
// Adjust these defaults in one place or override via target build settings
// (Preprocessor Macros) as needed.

// 1: compile out training logic and run forward-only inference paths
#ifndef LSTM_INFERENCE_ONLY
#define LSTM_INFERENCE_ONLY 0
#endif

#define LSTM_DEBUG_PRINTS 0

constexpr bool inference_only = (LSTM_INFERENCE_ONLY != 0);
constexpr bool save_enable = true;
constexpr bool reset_state_per_window = true;

// true: load latest model from DB at startup; false: start from scratch
    //not needed to be set for inference_only = true
constexpr bool load_latest = false;


// 1: overwrite the loaded/latest model_id when saving; 0: create a new model snapshot
constexpr bool save_overwrite = false;

