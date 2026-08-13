#!/usr/bin/env bash
set -euo pipefail

out="${1:-lstm_results.csv}"
shift || true

if [ "$#" -lt 1 ]; then
  echo "usage: $0 output.csv *_analysis.txt"
  exit 1
fi

echo "source_file,model_id,name,completed_epochs,accuracy,accept_model,reject_reason,pred_down,pred_neutral,pred_up" > "$out"

for f in "$@"; do
  awk -v file="$f" '
    /^model_id,name,completed_epochs,accuracy,accept_model,reject_reason,pred_down,pred_neutral,pred_up/ { in_table=1; next }
    in_table && /^[0-9]+,/ {
      print file "," $0
      next
    }
  ' "$f" >> "$out"
done

echo "wrote $out"
