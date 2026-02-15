#pragma once

// Centralized build-time configuration for the LSTM project.
// Adjust these defaults in one place or override via target build settings
// (Preprocessor Macros) as needed.

// 1: compile out training logic and run forward-only inference paths
constexpr bool inference_only = true;
constexpr bool save_enable = true;

// true: load latest model from DB at startup; false: start from scratch
constexpr bool load_latest = true;


// 1: overwrite the loaded/latest model_id when saving; 0: create a new model snapshot
constexpr bool save_overwrite = true;
