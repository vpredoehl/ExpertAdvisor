#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${H1_TRACEABILITY_MANIFEST:-$repo_root/Tests/fixtures/CampaignOperationsH1Traceability.tsv}"
lock_manifest="${H1_LOCK_MANIFEST:-$repo_root/Tests/fixtures/CampaignOperationsH1LockPathMatrix.tsv}"
mode="${1:---validate}"
artifact_root="${2:-}"
requested_run_id="${3:-}"
runtime_results="${H1_RUNTIME_RESULTS:-${artifact_root:+$artifact_root/h1-runtime-results.tsv}}"
format_version="h1-runtime-result-v2"

fail_trace() {
    local code="$1" key="$2" detail="$3"
    printf 'H1T%s key=%s stage=runtime-traceability-reconciliation detail=%s\n' \
        "$code" "$key" "$detail" >&2
    exit 1
}

[[ -s "$manifest" ]] || fail_trace 001 "$manifest" missing-traceability-manifest
[[ -s "$lock_manifest" ]] || fail_trace 001 "$lock_manifest" missing-lock-manifest

shape_finding="$(awk -F '\t' '
  NR==1 {
    if (NF!=13 || $1!="requirement_id" || $13!="expected_status") {
      print "header:" NR; exit
    }
    next
  }
  NF!=13 { print "fields:" NR ":" NF; exit }
  {
    for(i=1;i<=13;i++) if($i==""){print "empty:" NR ":" i;exit}
    if(++requirement[$1]>1){failed=1;print "duplicate-requirement:" $1;exit}
    if(++fixture[$4]>1){failed=1;print "duplicate-fixture:" $4;exit}
    if($13!="SUCCESS" && $13!="EXPECTED_FAILURE" &&
       $13!="OBSERVED_BLOCKING"){print "status:" $1 ":" $13;exit}
    rows++
  }
  END { if(!failed && rows<60) print "row-count:" rows }
' "$manifest")"
[[ -z "$shape_finding" ]] || fail_trace 002 "$shape_finding" invalid-traceability-manifest

lock_finding="$(awk -F '\t' '
  NR==1 { if(NF!=14 || $1!="test_id" || $14!="seam_disposition"){
    print "header";exit}; next }
  NF!=14 {print "fields:" NR;exit}
  {if(++id[$1]>1){failed=1;print "duplicate:" $1;exit}; if($1!~ /^H1LOCK[0-9][0-9][0-9]$/){failed=1;print "id:" $1;exit}; rows++}
  END{if(!failed && rows<1)print "row-count:" rows}
' "$lock_manifest")"
[[ -z "$lock_finding" ]] || fail_trace 003 "$lock_finding" invalid-lock-matrix

[[ "$mode" == "--validate" || "$mode" == "--report" || \
   "$mode" == "--check-report" ]] || \
    fail_trace 001 "$mode" unknown-mode
[[ -n "$artifact_root" && -d "$artifact_root" ]] || \
    fail_trace 004 "${artifact_root:-missing}" missing-artifact-root
[[ -s "$runtime_results" ]] || fail_trace 004 "$runtime_results" missing-runtime-results

if [[ -z "$requested_run_id" ]]; then
    [[ -s "$artifact_root/run-id" ]] || fail_trace 005 run-id missing-run-id
    requested_run_id="$(tr -d '\r\n' < "$artifact_root/run-id")"
fi
[[ "$requested_run_id" =~ ^h1-[A-Za-z0-9._-]+$ ]] || \
    fail_trace 005 "$requested_run_id" malformed-run-id

runtime_finding="$(awk -F '\t' -v version="$format_version" -v run="$requested_run_id" '
  NR==1 {
    current="result_format_version\trun_id\tfixture_id\trequirement_id\ttest_source\ttimestamp\tactual_status\tsqlstate\tdiagnostic\tobject_identity\tstage\toperation_id\tartifact_path\tartifact_digest\tlock_outcome\tcycle_detected\tcleanup_result\tgenerator_id\tgenerator_version\tgenerator_implementation\tgenerator_entry_point\temitted_runtime_record_id\toutput_artifact_id\trecord_digest";
    if($0!=current){print "header";exit}; fields=NF; next
  }
  NF!=fields {print "fields:" NR ":" NF;exit}
  {
    for(i=1;i<=NF;i++)if($i==""){print "empty:" NR ":" i;exit}
    if($1!=version){print "version:" $1;exit}
    if($2!=run){print "stale-run:" $2;exit}
    if($7!="SUCCESS" && $7!="EXPECTED_FAILURE" &&
       $7!="OBSERVED_BLOCKING"){print "status:" $7;exit}
    if($8!~ /^[0-9A-Z]{5}$/){print "sqlstate:" $8;exit}
    if($14!~ /^[0-9a-f]{64}$/){print "digest:" $14;exit}
    if($16!="true" && $16!="false"){print "cycle:" $16;exit}
    key=$4 SUBSEP $3;
    if(++seen[key]>1){print "duplicate:" $4 ":" $3;exit}
  }
' "$runtime_results")"
[[ -z "$runtime_finding" ]] || fail_trace 006 "$runtime_finding" malformed-runtime-result

while IFS=$'\t' read -r _ _ fixture requirement _ _ _ _ _ _ _ _ artifact digest _ _ _ _ _ _ _ _ _ _; do
    [[ "$fixture" == "fixture_id" ]] && continue
    [[ "$artifact" != /* && "$artifact" != *".."* ]] || \
        fail_trace 007 "$requirement:$fixture" unsafe-artifact-path
    artifact_file="$artifact_root/$artifact"
    [[ -s "$artifact_file" ]] || fail_trace 007 "$requirement:$fixture" missing-artifact
    actual_digest="$(shasum -a 256 "$artifact_file" | awk '{print $1}')"
    [[ "$actual_digest" == "$digest" ]] || \
        fail_trace 008 "$requirement:$fixture" artifact-digest-mismatch
done < "$runtime_results"

reconciliation="$(awk -F '\t' '
  NR==FNR {
    if(FNR==1)next;
    key=$1 SUBSEP $4;
    expected[key]=$13 SUBSEP $8 SUBSEP $9 SUBSEP $10 SUBSEP $11 SUBSEP $12;
    expected_key[key]=1; next
  }
  FNR==1 {next}
  {
    key=$4 SUBSEP $3;
    if(!(key in expected_key)){failed=1;print "unexpected-runtime:" $4 ":" $3;exit}
    split(expected[key],want,SUBSEP);
    split($13,path,"/"); artifact=path[length(path)];
    actual=$7 SUBSEP $8 SUBSEP $9 SUBSEP $10 SUBSEP $11 SUBSEP artifact;
    if(actual!=expected[key]){
      failed=1;print "disagreement:" $4 ":" $3 ":expected=" expected[key] ":actual=" actual;exit
    }
    if($16=="true" && $9=="no-deadlock-cycle"){
      failed=1;print "cycle:" $4 ":" $3;exit
    }
    observed[key]=1
  }
  END {
    if(!failed)for(key in expected_key)if(!(key in observed)){
      split(key,p,SUBSEP);print "missing-runtime:" p[1] ":" p[2];exit
    }
  }
' "$manifest" "$runtime_results")"
[[ -z "$reconciliation" ]] || fail_trace 009 "$reconciliation" runtime-disagreement

lock_reconciliation="$(awk -F '\t' '
  NR==FNR {if(FNR>1){pair[$1]=$2; requirement[$1]=$10;
    permitted[$1]=$7}; next}
  FNR==1{next}
  $3~/^H1LOCK/ {
    if(!($3 in pair)){failed=1;print "unknown-lock:" $3;exit}
    if($4!=requirement[$3]){failed=1;print "requirement:" $3;exit}
    if($12!=$3){failed=1;print "operation-id:" $3 ":" $12;exit}
    if($10!=pair[$3]){failed=1;print "operation-pair:" $3 ":" $10;exit}
    if($15!=permitted[$3]){failed=1;print "wait-direction:" $3 ":" $15;exit}
    if($16!="false"){failed=1;print "cycle:" $3;exit}
    observed[$3]=1
  }
  END{if(!failed)for(id in pair)if(!(id in observed)){print "missing-lock:" id;exit}}
' "$lock_manifest" "$runtime_results")"
[[ -z "$lock_reconciliation" ]] || \
    fail_trace 011 "$lock_reconciliation" lock-matrix-runtime-disagreement

render_report() {
    python3 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" \
      generate "$artifact_root" "$requested_run_id" >/dev/null
    cat "$artifact_root/CampaignOperationsH1Traceability.md"
}

if [[ "$mode" == "--report" ]]; then
    render_report
    exit 0
fi
if [[ "$mode" == "--check-report" ]]; then
    checked_report="${4:-$repo_root/docs/CampaignOperationsH1Traceability.md}"
    [[ -s "$checked_report" ]] || fail_trace 010 "$checked_report" missing-generated-report
    python3 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" validate \
      "$artifact_root" "$requested_run_id" "$checked_report" >/dev/null || \
      fail_trace 010 "$checked_report" stale-generated-report
    printf 'H1_TRACEABILITY_REPORT_CURRENT run_id=%s\n' "$requested_run_id"
    exit 0
fi

printf 'H1_TRACEABILITY_RECONCILIATION_OK run_id=%s records=%s\n' \
    "$requested_run_id" "$(awk 'END{print NR-1}' "$runtime_results")"
