#!/bin/bash
set -euo pipefail

BIN="./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release"

if [ "${1:-}" != "RUN" ]; then
    echo "Prepared only. No experiments queued."
    echo "Run explicitly with:"
    echo "  $0 RUN"
    exit 0
fi

COMMON=(
  --queue-experiment
  --symbol=cadchfrmp
  --prediction-horizon=4
  --target-epochs=20
  --threshold=0.0008
  --core-lr-mult=120
  --head-lr-mult=25
  --checkpoint-interval=20
  --train-start=2010-01-01
  --train-end=2025-01-01
  --infer-start=2025-01-01
  --infer-end=2026-01-01
  --donchian20-mode=enabled
  --donchian-lookback=20
  --feature-warmup-scope=legacy_cold_boundary
)

echo "=== CONTROL ==="
"$BIN" "${COMMON[@]}"

echo "=== ABLATE directional_efficiency ==="
"$BIN" "${COMMON[@]}" \
  --ablate-features=directional_efficiency

echo "=== ABLATE return_sign_persistence ==="
"$BIN" "${COMMON[@]}" \
  --ablate-features=return_sign_persistence

echo "=== ABLATE return_direction_imbalance ==="
"$BIN" "${COMMON[@]}" \
  --ablate-features=return_direction_imbalance

echo "=== ABLATE directional_adverse_excursion ==="
"$BIN" "${COMMON[@]}" \
  --ablate-features=directional_adverse_excursion

echo "=== ABLATE multi_bar_range_pressure ==="
"$BIN" "${COMMON[@]}" \
  --ablate-features=multi_bar_range_pressure

echo "=== ABLATE rolling_range_expansion ==="
"$BIN" "${COMMON[@]}" \
  --ablate-features=rolling_range_expansion
