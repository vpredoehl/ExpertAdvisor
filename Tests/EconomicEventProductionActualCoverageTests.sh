#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m unittest -v Tests/EconomicEventProductionActualCoverageTests.py
printf 'PHASE12_PRODUCTION_COVERAGE=PASS\n'
