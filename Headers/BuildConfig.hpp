#pragma once

// Centralized build-time configuration for the LSTM project.
// Adjust these defaults in one place or override via target build settings
// (Preprocessor Macros) as needed.

#define LSTM_DEBUG_PRINTS 0
#define LSTM_DEBUG_INTERNAL_PRINTS 0

#ifndef LSTM_DISABLE_UPDATES
#define LSTM_DISABLE_UPDATES 0
#endif

constexpr bool default_runtime_inference_mode = false;
constexpr bool save_enable = true;
constexpr bool reset_state_per_window = true;

// true: load latest model from DB at startup; false: start from scratch
constexpr bool load_latest = false;


// 1: overwrite the loaded/latest model_id when saving; 0: create a new model snapshot
constexpr bool save_overwrite = false;

// Gate/state execution mode:
//   0 = CPU/reference only
//   1 = CPU/reference + validate fused Metal against CPU
//   2 = fused Metal only (no CPU fallback)
#ifndef LSTM_GATESTATE_MODE
#define LSTM_GATESTATE_MODE 2
#endif
