#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --host HOST --port PORT --user USER --database DATABASE" >&2
    exit 64
}

audit_host="" audit_port="" audit_user="" audit_database=""
while (($#)); do
    case "$1" in
        --host) audit_host="${2:-}"; shift 2 ;;
        --port) audit_port="${2:-}"; shift 2 ;;
        --user) audit_user="${2:-}"; shift 2 ;;
        --database) audit_database="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$audit_host" && -n "$audit_port" && -n "$audit_user" && \
   -n "$audit_database" ]] || usage

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$repo_root/Scripts/CampaignOperationsH1ManifestValidator.sh" >/dev/null
inventory="$repo_root/Database/manifests/055_campaign_operations_h1_object_inventory.tsv"
sql_file="$(mktemp /tmp/h1-restore-acl-origin.XXXXXX.sql)"
trap 'rm -f "$sql_file"' EXIT

{
    echo 'BEGIN;'
    while IFS=$'\t' read -r version _ object_class _ object_identity owner \
        _ expected_origin _; do
        [[ "$version" == "version" ]] && continue
        [[ "$expected_origin" == "explicit" ]] || continue
        case "$object_class" in
            schema)
                printf 'GRANT ALL PRIVILEGES ON SCHEMA %s TO %s;\n' \
                    "$object_identity" "$owner" ;;
            table|view|materialized_view|partitioned_table|foreign_table)
                printf 'GRANT ALL PRIVILEGES ON TABLE %s TO %s;\n' \
                    "$object_identity" "$owner" ;;
            sequence)
                printf 'GRANT USAGE ON SEQUENCE %s TO %s;\n' \
                    "$object_identity" "$owner" ;;
            function)
                printf 'GRANT ALL PRIVILEGES ON FUNCTION %s TO %s;\n' \
                    "$object_identity" "$owner" ;;
            procedure)
                printf 'GRANT ALL PRIVILEGES ON PROCEDURE %s TO %s;\n' \
                    "$object_identity" "$owner" ;;
            aggregate)
                printf 'GRANT ALL PRIVILEGES ON FUNCTION %s TO %s;\n' \
                    "$object_identity" "$owner" ;;
            *)
                printf 'SQLSTATE=55000 diagnostic=H1A009 key=%s stage=restore-acl-origin detail=unsupported-object-class\n' \
                    "$object_class" >&2
                exit 1 ;;
        esac
    done < "$inventory"
    echo 'COMMIT;'
} > "$sql_file"

psql -X -q -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    -h "$audit_host" -p "$audit_port" -U "$audit_user" \
    "$audit_database" -f "$sql_file"
echo "H1_RESTORE_ACL_ORIGIN_V1_OK stage=post-database-restore-materialization"
