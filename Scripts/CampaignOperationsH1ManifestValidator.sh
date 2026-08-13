#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest_dir="$repo_root/Database/manifests"
migration="$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql"
while (($#)); do
    case "$1" in
        --manifest-dir) manifest_dir="${2:-}"; shift 2 ;;
        --migration) migration="${2:-}"; shift 2 ;;
        *) echo "usage: $0 [--manifest-dir DIR] [--migration FILE]" >&2; exit 64 ;;
    esac
done

inventory="$manifest_dir/055_campaign_operations_h1_object_inventory.tsv"
explicit_acl="$manifest_dir/055_campaign_operations_h1_explicit_acl.tsv"
default_acl="$manifest_dir/055_campaign_operations_h1_default_acl.tsv"
column_acl="$manifest_dir/055_campaign_operations_h1_column_acl.tsv"
acl_sql="$manifest_dir/055_campaign_operations_h1_acl_manifest.sql"
digest_file="$manifest_dir/055_campaign_operations_h1_manifest.sha256"
manifest_version="h1-manifest-set-v1"

fail_manifest() {
    local code="$1" key="$2" detail="$3"
    printf 'SQLSTATE=55000 diagnostic=%s key=%s stage=manifest-validation manifest_version=%s detail=%s\n' \
        "$code" "$key" "$manifest_version" "$detail" >&2
    exit 1
}

for required_file in "$inventory" "$explicit_acl" "$default_acl" \
    "$column_acl" "$acl_sql" "$digest_file" "$migration"; do
    [[ -s "$required_file" ]] || fail_manifest H1A009 "$required_file" missing-file
done

validate_tsv() {
    local file="$1" fields="$2" version="$3" expected_rows="$4" key_column="$5"
    awk -F '\t' -v fields="$fields" -v version="$version" \
        -v expected_rows="$expected_rows" -v key_column="$key_column" '
        NR == 1 { next }
        NF != fields { failed=1; printf "malformed:%d:field-count-%d", NR, NF; exit 20 }
        $1 != version { failed=1; printf "version:%d:%s", NR, $1; exit 21 }
        {
            for (i=1; i<=NF; ++i) if ($i == "") {
                failed=1; printf "malformed:%d:empty-field-%d", NR, i; exit 22
            }
            key=$key_column;
            if (++seen[key] > 1) { failed=1; printf "duplicate:%s", key; exit 23 }
            rows++
        }
        END {
            if (!failed && rows != expected_rows) {
                printf "row-count:%d:expected-%d", rows, expected_rows; exit 24
            }
        }
    ' "$file"
}

run_shape_check() {
    local file="$1" fields="$2" version="$3" rows="$4" key_column="$5"
    local finding="" status=0
    finding="$(validate_tsv "$file" "$fields" "$version" "$rows" "$key_column")" || status=$?
    [[ "$status" == 0 ]] || {
        case "$finding" in
            duplicate:*) fail_manifest H1A009 "${finding#duplicate:}" duplicate-primary-key ;;
            version:*) fail_manifest H1A009 "$file" "$finding" ;;
            row-count:*) fail_manifest H1A009 "$file" "$finding" ;;
            *) fail_manifest H1A009 "$file" "${finding:-malformed-tsv}" ;;
        esac
    }
}

run_shape_check "$inventory" 9 h1-object-inventory-v1 93 2
run_shape_check "$explicit_acl" 8 h1-explicit-acl-v1 70 2
run_shape_check "$default_acl" 7 h1-default-acl-v1 27 2
run_shape_check "$column_acl" 8 h1-column-acl-v1 7 2

finding="$(awk -F '\t' '
    NR==FNR && FNR>1 {
        identity[$3 SUBSEP $5]=1;
        if (($3 SUBSEP $5) in owner && owner[$3 SUBSEP $5] != $6)
            { print "conflicting-owner:" $3 ":" $5; exit }
        if (($3 SUBSEP $5) in origin && origin[$3 SUBSEP $5] != $8)
            { print "conflicting-origin:" $3 ":" $5; exit }
        owner[$3 SUBSEP $5]=$6; origin[$3 SUBSEP $5]=$8; next
    }
    FNR==1 { next }
    !(($3 SUBSEP $4) in identity) { print "unknown-object:" $3 ":" $4; exit }
' "$inventory" "$explicit_acl")"
case "$finding" in
    conflicting-*) fail_manifest H1A009 "${finding#*:}" "${finding%%:*}" ;;
    unknown-object:*) fail_manifest H1A009 "${finding#unknown-object:}" unknown-explicit-acl-object ;;
esac

finding="$(awk -F '\t' '
    NR==FNR { if (FNR>1 && $3=="table") table[$5]=1; next }
    FNR==1 { next }
    !($3 in table) { print $3; exit }
' "$inventory" "$column_acl")"
[[ -z "$finding" ]] || fail_manifest H1A009 "$finding" unknown-column-acl-table

finding="$(awk -F '\t' '
    NR==1 { next }
    $4!="<global>" && $4!="public" { print $2 ":unknown-scope"; exit }
    $5!="f" && $5!="r" && $5!="S" && $5!="T" && $5!="n" {
        print $2 ":unknown-object-kind"; exit
    }
    $6!="explicit" && $6!="null" { print $2 ":unknown-state"; exit }
    {
        owner[$3]=1; state[$3 SUBSEP $4 SUBSEP $5]=1
    }
    END {
        if (length($0) && length(owner) != 3) print "owners:incomplete-owner-set"
    }
' "$default_acl")"
[[ -z "$finding" ]] || fail_manifest H1A009 "$finding" invalid-default-acl-state

# The digest makes omission, addition, or modification fail before any catalog
# comparison.  The migration carries the same digest marker so its separately
# embedded allowlists cannot silently drift from the deployment manifest.
expected_digest="$(awk 'NF==2 && $1=="h1-manifest-set-v1" {print $2}' "$digest_file")"
[[ "$expected_digest" =~ ^[0-9a-f]{64}$ ]] || \
    fail_manifest H1A009 "$digest_file" malformed-authoritative-digest
actual_digest="$(shasum -a 256 "$inventory" "$explicit_acl" "$default_acl" \
    "$column_acl" "$acl_sql" | awk '{print $1}' | shasum -a 256 | awk '{print $1}')"
[[ "$actual_digest" == "$expected_digest" ]] || \
    fail_manifest H1A009 "$actual_digest" "manifest-digest-mismatch-expected-$expected_digest"
embedded_digest="$(sed -n 's/^-- H1_MANIFEST_DIGEST_SHA256: //p' "$migration" | head -n 1)"
[[ "$embedded_digest" == "$expected_digest" ]] || \
    fail_manifest H1A009 "${embedded_digest:-missing}" \
        "embedded-manifest-drift-expected-$expected_digest"

printf 'H1_MANIFEST_V1_OK version=%s digest=%s inventory_rows=93 explicit_acl_rows=70 default_acl_states=27 column_acl_rows=7\n' \
    "$manifest_version" "$actual_digest"
