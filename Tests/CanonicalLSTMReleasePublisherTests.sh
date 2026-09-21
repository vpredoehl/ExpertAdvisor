#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/usr/bin/python3 "${repo_root}/Tests/CanonicalLSTMReleasePublisherTests.py"
printf '%s\n' 'CanonicalLSTMReleasePublisherTests passed'
