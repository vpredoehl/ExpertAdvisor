#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 file1.txt [file2.txt ...]"
  exit 1
fi

for f in "$@"; do
  out="${f%.txt}_analysis.txt"

  awk '
    BEGIN {
      keep = 0
    }

    /INFERENCE_CONFIG_RESOLVED/ ||
    /RUNTIME_CONFIG/ ||
    /MODEL_TRAIN_CONFIG/ ||
    /MODEL_TRAIN_SYMBOL_META/ ||
    /CHECKPOINT_SAVE_BEGIN/ ||
    /CHECKPOINT_SAVE_DONE/ ||
    /Created new model_id/ ||
    /Saved model with model_id/ ||
    /INFER_ALL_BEGIN/ ||
    /INFER_ALL_RESUME/ ||
    /INFER_ALL_MODEL_BEGIN/ ||
    /INFER_ALL_MODEL_DONE/ ||
    /INFER_ALL_DONE/ ||
    /^model_id,name,completed_epochs/ ||
    /^[0-9]+,[^,]+,[0-9]+,/ ||
    /Overall 3-class accuracy/ ||
    /Overall 3-class confusion matrix/ ||
    /MODEL_ACCEPTANCE/ {
      print
      next
    }

    /Epoch [0-9]+\/[0-9]+/ {
      if ($0 ~ /Epoch (20|40|60|80|100|120|140|160|180|200|220|240)\//) {
        print
      }
      next
    }

    /Validation/ ||
    /VALIDATION/ ||
    /Actual distribution/ ||
    /Pred distribution/ ||
    /3-class accuracy:/ ||
    /3-class confusion matrix/ {
      if (keep < 40) {
        print
        keep++
      }
      next
    }

    /^$/ {
      keep = 0
    }
  ' "$f" > "$out"

  echo "wrote $out ($(wc -c < "$out") bytes)"
done
