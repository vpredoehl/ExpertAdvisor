#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 2 && -s "$1" ]] || { echo "usage: $0 RUNTIME RUN_ID" >&2; exit 64; }
finding="$(awk -F '\t' -v run="$2" '
  NR==FNR {if(FNR==1)next; expected[$1]=$2 SUBSEP $3 SUBSEP $4 SUBSEP $5 SUBSEP $9;next}
  FNR==1 {if(NF!=9||$1!="version"){print "header";exit};next}
  {if($1!="h1-uniqueness-invariant-runtime-v1"||$2!=run){print "stale:"$3;exit}
   if(++seen[$3]>1){print "duplicate:"$3;exit}
   actual=$4 SUBSEP $5 SUBSEP $6 SUBSEP $7 SUBSEP $8
   if(actual!=expected[$3]){print "mapping:"$3;exit}
   if($9!="PASS"){print "cleanup:"$3;exit}}
  END{for(id in expected)if(!(id in seen)){print "missing:"id;exit}}
' "$repo_root/Tests/fixtures/CampaignOperationsH1UniquenessInvariants.tsv" "$1" | head -n 1)"
[[ -z "$finding" ]] || { echo "H1I001 key=$finding stage=uniqueness-invariant-reconciliation" >&2; exit 1; }
echo "H1_UNIQUENESS_INVARIANT_RECONCILIATION_OK rows=1"
