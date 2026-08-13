#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

PROMPT=""
OUTPUT=""
TRANSCRIPT=""
TITLE=""
REASONING="high"
MODEL=""
APPROVAL_POLICY="never"
SANDBOX_MODE="danger-full-access"
DOCUMENT_TYPE="architecture review"
STATUS_VALUE="final"
FORCE=0

usage() {
    cat <<EOF
Usage:
  $SCRIPT_NAME --prompt PROMPT.md [options]

Required:
  --prompt FILE              Prompt file to send to codex.

Optional:
  --output FILE              Clean final response file.
                             Default: PROMPT basename with _Prompt replaced by _Output.
  --transcript FILE          Full terminal transcript.
                             Default: PROMPT basename with _Prompt replaced by _Transcript.txt.
  --title TEXT               Markdown title.
                             Default: derived from the output filename.
  --reasoning LEVEL          model_reasoning_effort value.
                             Default: high
  --model MODEL              Explicit Codex model. Omit to use the configured default.
  --approval-policy POLICY   Codex approval policy.
                             Default: never
  --sandbox-mode MODE        Codex sandbox mode.
                             Default: danger-full-access
  --document-type TEXT       YAML front-matter document_type.
                             Default: architecture review
  --status TEXT              YAML front-matter status.
                             Default: final
  --force                    Allow overwriting existing output/transcript files.
  -h, --help                 Show this help.

Example:
  $SCRIPT_NAME \\
    --prompt CampaignOperations_Final_Verification_CEE_Prompt.md \\
    --output CampaignOperations_Final_Verification_CEE_Output.md \\
    --transcript CampaignOperations_Final_Verification_CEE_Transcript.txt \\
    --title "Campaign Operations Final Verification CEE" \\
    --reasoning xhigh
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

derive_output_name() {
    local prompt="$1"
    local dir base stem

    dir="$(dirname "$prompt")"
    base="$(basename "$prompt")"
    stem="${base%.*}"

    if [[ "$stem" == *"_Prompt" ]]; then
        stem="${stem%_Prompt}_Output"
    else
        stem="${stem}_Output"
    fi

    printf '%s/%s.md\n' "$dir" "$stem"
}

derive_transcript_name() {
    local prompt="$1"
    local dir base stem

    dir="$(dirname "$prompt")"
    base="$(basename "$prompt")"
    stem="${base%.*}"

    if [[ "$stem" == *"_Prompt" ]]; then
        stem="${stem%_Prompt}_Transcript"
    else
        stem="${stem}_Transcript"
    fi

    printf '%s/%s.txt\n' "$dir" "$stem"
}

derive_title() {
    local output="$1"
    local base stem

    base="$(basename "$output")"
    stem="${base%.*}"
    stem="${stem%_Output}"
    stem="${stem//_/ }"

    printf '%s\n' "$stem"
}

yaml_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "$value"
}

while (($# > 0)); do
    case "$1" in
        --prompt)
            (($# >= 2)) || die "--prompt requires a value"
            PROMPT="$2"
            shift 2
            ;;
        --output)
            (($# >= 2)) || die "--output requires a value"
            OUTPUT="$2"
            shift 2
            ;;
        --transcript)
            (($# >= 2)) || die "--transcript requires a value"
            TRANSCRIPT="$2"
            shift 2
            ;;
        --title)
            (($# >= 2)) || die "--title requires a value"
            TITLE="$2"
            shift 2
            ;;
        --reasoning)
            (($# >= 2)) || die "--reasoning requires a value"
            REASONING="$2"
            shift 2
            ;;
        --model)
            (($# >= 2)) || die "--model requires a value"
            MODEL="$2"
            shift 2
            ;;
        --approval-policy)
            (($# >= 2)) || die "--approval-policy requires a value"
            APPROVAL_POLICY="$2"
            shift 2
            ;;
        --sandbox-mode)
            (($# >= 2)) || die "--sandbox-mode requires a value"
            SANDBOX_MODE="$2"
            shift 2
            ;;
        --document-type)
            (($# >= 2)) || die "--document-type requires a value"
            DOCUMENT_TYPE="$2"
            shift 2
            ;;
        --status)
            (($# >= 2)) || die "--status requires a value"
            STATUS_VALUE="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$PROMPT" ]] || die "--prompt is required"
[[ -f "$PROMPT" ]] || die "prompt file not found: $PROMPT"
command -v codex >/dev/null 2>&1 || die "codex command not found"
command -v tee >/dev/null 2>&1 || die "tee command not found"

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$(derive_output_name "$PROMPT")"
fi

if [[ -z "$TRANSCRIPT" ]]; then
    TRANSCRIPT="$(derive_transcript_name "$PROMPT")"
fi

if [[ -z "$TITLE" ]]; then
    TITLE="$(derive_title "$OUTPUT")"
fi

[[ "$OUTPUT" != "$PROMPT" ]] || die "output must differ from prompt"
[[ "$TRANSCRIPT" != "$PROMPT" ]] || die "transcript must differ from prompt"
[[ "$OUTPUT" != "$TRANSCRIPT" ]] || die "output and transcript must differ"

if ((FORCE == 0)); then
    [[ ! -e "$OUTPUT" ]] || die "output already exists: $OUTPUT (use --force to overwrite)"
    [[ ! -e "$TRANSCRIPT" ]] || die "transcript already exists: $TRANSCRIPT (use --force to overwrite)"
fi

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$TRANSCRIPT")"

RAW_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/run_cee_review.output.XXXXXX")"
HEADER_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/run_cee_review.header.XXXXXX")"

cleanup() {
    rm -f "$RAW_OUTPUT" "$HEADER_OUTPUT"
}
trap cleanup EXIT

CODEX_ARGS=(
    exec
    -c "approval_policy=${APPROVAL_POLICY}"
    -c "sandbox_mode=${SANDBOX_MODE}"
    -c "model_reasoning_effort=\"${REASONING}\""
    --output-last-message "$RAW_OUTPUT"
)

if [[ -n "$MODEL" ]]; then
    CODEX_ARGS+=(--model "$MODEL")
fi

printf 'Prompt:          %s\n' "$PROMPT"
printf 'Output:          %s\n' "$OUTPUT"
printf 'Transcript:      %s\n' "$TRANSCRIPT"
printf 'Title:           %s\n' "$TITLE"
printf 'Reasoning:       %s\n' "$REASONING"
printf 'Model:           %s\n' "${MODEL:-default}"
printf 'Approval policy: %s\n' "$APPROVAL_POLICY"
printf 'Sandbox mode:    %s\n' "$SANDBOX_MODE"
printf '\n'

set +e
/usr/bin/time -l codex "${CODEX_ARGS[@]}" \
    < "$PROMPT" \
    2>&1 | tee "$TRANSCRIPT"

# PIPESTATUS is replaced after every command, including a variable assignment.
# Copy the complete array immediately after the pipeline before reading elements.
PIPELINE_STATUS=("${PIPESTATUS[@]}")
CODEX_STATUS="${PIPELINE_STATUS[0]:-1}"
TEE_STATUS="${PIPELINE_STATUS[1]:-1}"
set -e

if ((CODEX_STATUS == 130 || TEE_STATUS == 130)); then
    printf '\nReview interrupted by user (Ctrl-C).\n' >&2
    printf 'Codex exit status: %d\n' "$CODEX_STATUS" >&2
    printf 'tee exit status:   %d\n' "$TEE_STATUS" >&2
    exit 130
fi

if ((CODEX_STATUS != 0)); then
    printf '\nCodex failed.\n' >&2
    printf 'Codex exit status: %d\n' "$CODEX_STATUS" >&2
    printf 'tee exit status:   %d\n' "$TEE_STATUS" >&2
    exit "$CODEX_STATUS"
fi

if ((TEE_STATUS != 0)); then
    printf '\nTranscript capture failed.\n' >&2
    printf 'Codex exit status: %d\n' "$CODEX_STATUS" >&2
    printf 'tee exit status:   %d\n' "$TEE_STATUS" >&2
    exit "$TEE_STATUS"
fi

[[ -s "$RAW_OUTPUT" ]] || die "codex succeeded but produced no final output"

ESCAPED_TITLE="$(yaml_escape "$TITLE")"
ESCAPED_DOCUMENT_TYPE="$(yaml_escape "$DOCUMENT_TYPE")"
ESCAPED_STATUS="$(yaml_escape "$STATUS_VALUE")"
ESCAPED_PROMPT="$(yaml_escape "$PROMPT")"

cat > "$HEADER_OUTPUT" <<EOF
---
title: "$ESCAPED_TITLE"
document_type: "$ESCAPED_DOCUMENT_TYPE"
status: "$ESCAPED_STATUS"
generated_from: "$ESCAPED_PROMPT"
reasoning_effort: "$(yaml_escape "$REASONING")"
model: "$(yaml_escape "${MODEL:-default}")"
---

# $TITLE

EOF

cat "$RAW_OUTPUT" >> "$HEADER_OUTPUT"

# Normalize trailing whitespace and ensure exactly one final newline.
perl -pi -e 's/[ \t]+$//' "$HEADER_OUTPUT"
perl -0777 -pi -e 's/\n+\z/\n/' "$HEADER_OUTPUT"

mv "$HEADER_OUTPUT" "$OUTPUT"

printf '\nCompleted successfully.\n'
printf 'Codex exit status: 0\n'
printf 'tee exit status:   0\n'
printf 'Markdown output:   %s\n' "$OUTPUT"
printf 'Transcript:        %s\n' "$TRANSCRIPT"
