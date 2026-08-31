#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m unittest -v Tests/BlsEconomicEventReleaseActualImporterTests.py
node Tests/BlsReleaseAcquisitionTests.mjs

printf 'PHASE17_BLS_FOCUSED_TESTS=PASS\n'
