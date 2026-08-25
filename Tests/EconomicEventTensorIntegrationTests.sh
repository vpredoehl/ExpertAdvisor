#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/economic_event_tensor_integration_tests"
BIN="$BUILD_DIR/EconomicEventTensorIntegrationTests"
PRODUCTS_DIR="${LSTM_TEST_PRODUCTS_DIR:-$ROOT/DerivedData/Development/Build/Products/Debug}"

LIBPQXX_PREFIX="$(brew --prefix libpqxx@7.10.1)"
LIBPQ_PREFIX="$(brew --prefix libpq)"

mkdir -p "$BUILD_DIR"

for required in libMetaNN.a libMetalBuffer.a default.metallib MetaNN_metal.metallib; do
    if [[ ! -f "$PRODUCTS_DIR/$required" ]]; then
        printf 'missing test dependency: %s\n' "$PRODUCTS_DIR/$required" >&2
        exit 2
    fi
done

include_flags=()
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find "$ROOT/MetaNN" -type d -print)

# Tensor/MetaNN diagnostic hook is normally supplied by the full LSTM
# executable. This standalone integration test supplies a deterministic
# test-only implementation instead of linking production main.cpp.
cat > "$BUILD_DIR/LstmRuntimeDiagnosticLoggingStub.cpp" <<'STUB'
extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}
STUB

xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 \
    -O1 -Wall -Wextra -Werror \
    -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
    -Wno-unused-but-set-variable -Wno-format -Wno-ignored-qualifiers \
    -Wno-reorder-ctor -Wno-sign-compare \
    -I"$ROOT/Headers" \
    -I"$ROOT/Sources" \
    -I"$LIBPQXX_PREFIX/include" \
    -I"$LIBPQ_PREFIX/include" \
    "${include_flags[@]}" \
    "$ROOT/Tests/EconomicEventTensorIntegrationTests.cpp" \
    "$ROOT/LSTM/Tensor.cpp" \
    "$ROOT/Common/PricePoint.cpp" \
    "$ROOT/Sources/EconomicEventFeatures.cpp" \
    "$BUILD_DIR/LstmRuntimeDiagnosticLoggingStub.cpp" \
    -L"$PRODUCTS_DIR" \
    -L"$LIBPQXX_PREFIX/lib" \
    -L"$LIBPQ_PREFIX/lib" \
    -lMetaNN \
    -lMetalBuffer \
    -lpqxx \
    -lpq \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -framework Foundation \
    -o "$BIN"

cp "$PRODUCTS_DIR/default.metallib" "$BUILD_DIR/default.metallib"
cp "$PRODUCTS_DIR/MetaNN_metal.metallib" "$BUILD_DIR/MetaNN.metallib"

(cd "$BUILD_DIR" && "$BIN")
