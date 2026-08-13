#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-CodexReviewOutput.txt}"
PROMPT="${2:-CodexReviewPrompt.txt}"

codex exec < "$PROMPT" 2>&1 | tee "$OUTPUT"

