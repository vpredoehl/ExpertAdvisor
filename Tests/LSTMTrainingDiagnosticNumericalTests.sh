#!/usr/bin/env bash
set -euo pipefail

# Invoke only after checking for active scheduler/training/inference workers.
# Compiles a small database-free fixture; never launches LSTM_Release.
# Optional argument is a read-only pre-change LSTM.cpp for exact A/B comparison.
# Always compares current metann/combined paths, including diagnostics on/off.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_phase25a_numerical.XXXXXX)"
cleanup() {
    local result=$?
    if [[ "${result}" -eq 0 ]]; then
        rm -rf -- "${test_dir}"
    else
        printf 'failed fixture evidence retained: %s\n' "${test_dir}" >&2
    fi
    return "${result}"
}
trap cleanup EXIT
products_dir="${LSTM_TEST_PRODUCTS_DIR:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release}"
for required in libMetaNN.a libMetalBuffer.a default.metallib MetaNN_metal.metallib; do
    test -f "${products_dir}/${required}"
done
include_flags=("-I${repo_root}/MetaNN/MetaNN")
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find -L "${repo_root}/MetaNN/MetaNN/MetaNN" -type d -print)
read -r -a pqxx_compile_flags <<< "$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link_flags <<< "$(pkg-config --libs libpqxx)"

compile_fixture() {
    local source="$1" output="$2"
    xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 \
        -O3 -DNDEBUG -fobjc-arc -DLSTM_NUMERICAL_TEST_OBSERVERS -Wall -Wextra -Werror \
        -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
        -Wno-unused-but-set-variable -Wno-format -Wno-ignored-qualifiers \
        -Wno-reorder-ctor -Wno-sign-compare \
        -I"${repo_root}/Headers" -I"${repo_root}/Tests" "${include_flags[@]}" "${pqxx_compile_flags[@]}" \
        "${repo_root}/Tests/LSTMTrainingDiagnosticNumericalTests.cpp" \
        "${source}" "${repo_root}/LSTM/Tensor.cpp" \
        "${repo_root}/Sources/EconomicEventFeatures.cpp" "${repo_root}/Common/PricePoint.cpp" \
        "${repo_root}/LSTM/MetalForwardAffine.mm" \
        -L"${products_dir}" -lMetaNN -lMetalBuffer "${pqxx_link_flags[@]}" \
        -framework Metal -framework MetalPerformanceShaders -framework Foundation \
        -o "${output}"
}

compile_fixture "${repo_root}/LSTM/LSTM.cpp" "${test_dir}/current"
if [[ $# -eq 1 ]]; then
    compile_fixture "$1" "${test_dir}/baseline"
elif [[ $# -ne 0 ]]; then
    printf '%s\n' 'usage: LSTMTrainingDiagnosticNumericalTests.sh [baseline-LSTM.cpp]' >&2
    exit 2
fi
cp "${products_dir}/default.metallib" "${test_dir}/default.metallib"
cp "${products_dir}/MetaNN_metal.metallib" "${test_dir}/MetaNN.metallib"

(
    cd "${test_dir}"
    for mode in legacy auxiliary log percent; do
        for enabled in on off; do
            EA_LSTM_FORWARD_AFFINE=metann ./current "${mode}" "${enabled}" "current_${mode}_${enabled}.bin" \
                > "current_${mode}_${enabled}.log"
            EA_LSTM_FORWARD_AFFINE=combined ./current "${mode}" "${enabled}" "combined_${mode}_${enabled}.bin" \
                > "combined_${mode}_${enabled}.log"
            cmp "current_${mode}_${enabled}.bin" "combined_${mode}_${enabled}.bin"
            test -s "current_${mode}_${enabled}.bin.tensors"
            cmp "current_${mode}_${enabled}.bin.tensors" "combined_${mode}_${enabled}.bin.tensors"
            if [[ "${enabled}" == on ]]; then
                rg -q '^DIAG_LSTM_FORWARD_AFFINE,path=metann$' "current_${mode}_${enabled}.log"
                rg -q '^DIAG_LSTM_FORWARD_AFFINE,path=combined$' "combined_${mode}_${enabled}.log"
            else
                ! rg -q '^DIAG_LSTM_FORWARD_AFFINE,' "current_${mode}_${enabled}.log"
                ! rg -q '^DIAG_LSTM_FORWARD_AFFINE,' "combined_${mode}_${enabled}.log"
            fi
            # Compare gradient/clipping/optimizer diagnostics, excluding only
            # the new path label and process-specific buffer addresses.
            for variant in current combined; do
                sed -E '/^DIAG_LSTM_FORWARD_AFFINE,/d; /^DIAG_GATESTATE_BUFFERS,/s/0x[[:xdigit:]]+/<address>/g' \
                    "${variant}_${mode}_${enabled}.log" > "${variant}_affine_output.txt"
            done
            cmp current_affine_output.txt combined_affine_output.txt
            if [[ -x ./baseline ]]; then
                EA_LSTM_FORWARD_AFFINE=metann ./baseline "${mode}" "${enabled}" "baseline_${mode}_${enabled}.bin" \
                    > "baseline_${mode}_${enabled}.log"
                cmp "baseline_${mode}_${enabled}.bin" "current_${mode}_${enabled}.bin"
            fi
        done
        cmp "current_${mode}_on.bin" "current_${mode}_off.bin"
        cmp "combined_${mode}_on.bin" "combined_${mode}_off.bin"
        cmp "current_${mode}_on.bin.tensors" "current_${mode}_off.bin.tensors"
        cmp "combined_${mode}_on.bin.tensors" "combined_${mode}_off.bin.tensors"
        # Compare every targeted enabled diagnostic line, including preclip,
        # postclip and clipping-effect statistics for all 64 eligible updates.
        rg '^DIAG_(GRAD_PRECLIP_FULL_|GRAD_POSTCLIP_FULL_|GRAD_CLIP_EFFECT_)' \
            "current_${mode}_on.log" > current_diagnostics.txt
        [[ "$(wc -l < current_diagnostics.txt)" -eq 768 ]]
        if [[ "${mode}" == log || "${mode}" == percent ]]; then
            rg -q '^DIAG_GRAD_CLIP_EFFECT_.*clipped_count=[1-9]' current_diagnostics.txt
        fi
        ! rg -q '^DIAG_(GRAD_PRECLIP_FULL_|GRAD_POSTCLIP_FULL_|GRAD_CLIP_EFFECT_)' \
            "current_${mode}_off.log"
        if [[ -x ./baseline ]]; then
            rg '^DIAG_(GRAD_PRECLIP_FULL_|GRAD_POSTCLIP_FULL_|GRAD_CLIP_EFFECT_)' \
                "baseline_${mode}_on.log" > baseline_diagnostics.txt
            cmp baseline_diagnostics.txt current_diagnostics.txt
            # Buffer addresses naturally differ between fresh processes. Keep
            # every diagnostic field and value except those address literals.
            for variant in baseline current; do
                sed -E '/^DIAG_LSTM_FORWARD_AFFINE,/d; /^DIAG_GATESTATE_BUFFERS,/s/0x[[:xdigit:]]+/<address>/g' \
                    "${variant}_${mode}_on.log" > "${variant}_output.txt"
            done
            cmp baseline_output.txt current_output.txt
        fi
        printf 'PASS %s: 66 updates, metann/combined bitwise parity, diagnostic on/off parity, gradient/optimizer diagnostics, matrix restore continuation' "${mode}"
        if [[ -x ./baseline ]]; then
            printf ', baseline/current bitwise parity and enabled diagnostic parity'
        fi
        printf '\n'
        for enabled in on off; do
            for path in metann combined; do
                EA_LSTM_FORWARD_AFFINE="${path}" ./current "${mode}" "${enabled}" "production_${mode}_${path}_${enabled}.bin" production \
                    > "production_${mode}_${path}_${enabled}.log"
            done
            cmp "production_${mode}_metann_${enabled}.bin" "production_${mode}_combined_${enabled}.bin"
            cmp "production_${mode}_metann_${enabled}.bin.tensors" "production_${mode}_combined_${enabled}.bin.tensors"
        done
        cmp "production_${mode}_metann_on.bin" "production_${mode}_metann_off.bin"
        cmp "production_${mode}_combined_on.bin" "production_${mode}_combined_off.bin"
        cmp "production_${mode}_metann_on.bin.tensors" "production_${mode}_metann_off.bin.tensors"
        cmp "production_${mode}_combined_on.bin.tensors" "production_${mode}_combined_off.bin.tensors"
        printf 'PASS %s: H=64 T=64, 3 updates + restored continuation, bitwise forward cache and pre/postclip gradients\n' "${mode}"
    done
)
