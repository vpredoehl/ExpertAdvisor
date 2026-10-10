#!/usr/bin/env bash

# Canonical LSTM build environment.
# Source this file from build and validation scripts.

export LSTM_REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

export LSTM_DERIVED_DATA="${LSTM_DERIVED_DATA:-$LSTM_REPO_ROOT/DerivedData/ExpertAdvisor}"

export LSTM_RELEASE_PRODUCTS="$LSTM_DERIVED_DATA/Build/Products/Release"

export LSTM_RELEASE_BINARY="$LSTM_RELEASE_PRODUCTS/LSTM_Release"

export LSTM_TEST_PRODUCTS_DIR="$LSTM_RELEASE_PRODUCTS"
