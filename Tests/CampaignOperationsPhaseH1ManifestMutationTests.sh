#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
validator="$repo_root/Scripts/CampaignOperationsH1ManifestValidator.sh"
source_manifest="$repo_root/Database/manifests"
source_migration="$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql"
scratch="$(mktemp -d /tmp/ea-h1-manifest-mutations.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT

run_mutation() {
    local label="$1" expected_code="$2" expected_key="$3" mutation="$4"
    local case_root="$scratch/$label" manifest_copy="$scratch/$label/manifests"
    mkdir -p "$manifest_copy"
    cp "$source_manifest"/055_campaign_operations_h1_* "$manifest_copy/"
    cp "$source_migration" "$case_root/migration.sql"
    eval "$mutation"
    local log="$case_root/result.log"
    if "$validator" --manifest-dir "$manifest_copy" \
        --migration "$case_root/migration.sql" >"$log" 2>&1; then
        echo "manifest mutation $label unexpectedly passed" >&2
        exit 1
    fi
    rg -q "SQLSTATE=55000 diagnostic=${expected_code}" "$log" || {
        cat "$log" >&2; echo "manifest mutation $label reached wrong diagnostic" >&2; exit 1;
    }
    rg -q 'key=[^ ]+ stage=manifest-validation' "$log" && \
      rg -q "$expected_key" "$log" || {
        cat "$log" >&2; echo "manifest mutation $label omitted exact key/stage" >&2; exit 1;
    }
    printf 'H1_MUTATION_CASE\tmanifest-%s\tDatabase/manifests\t%s:%s\tmanifest-validation\t%s:%s\tmanifest-validation\tPASS\n' \
      "$label" "$expected_code" "$expected_key" "$expected_code" "$expected_key"
}

run_mutation remove_object H1A009 object_inventory \
    "perl -i -ne 'print unless /table:public\\.campaign_operations_production_transition_context/' \"\$manifest_copy/055_campaign_operations_h1_object_inventory.tsv\""
run_mutation remove_acl H1A009 explicit_acl \
    "perl -i -ne 'print unless /production_enablement_event:campaign_operations_production_reader:SELECT/' \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\""
run_mutation remove_default H1A009 default_acl \
    "perl -i -ne 'print unless /campaign_operations_owner:<global>:r/' \"\$manifest_copy/055_campaign_operations_h1_default_acl.tsv\""
run_mutation duplicate_object H1A009 table:public.campaign_operations_production_transition_context \
    "rg 'table:public\\.campaign_operations_production_transition_context' \"\$manifest_copy/055_campaign_operations_h1_object_inventory.tsv\" >> \"\$manifest_copy/055_campaign_operations_h1_object_inventory.tsv\""
run_mutation duplicate_acl H1A009 production_enablement_event \
    "rg 'production_enablement_event:campaign_operations_production_reader:SELECT' \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\" >> \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\""
run_mutation duplicate_default H1A009 campaign_operations_owner:public:f \
    "rg 'campaign_operations_owner:public:f' \"\$manifest_copy/055_campaign_operations_h1_default_acl.tsv\" >> \"\$manifest_copy/055_campaign_operations_h1_default_acl.tsv\""
run_mutation unreferenced_extra_acl H1A009 table:public.h1_unknown \
    "perl -i -pe 'if (/production_enablement_event:campaign_operations_production_reader:SELECT/) { s/public\\.campaign_operations_production_enablement_event/public.h1_unknown/g; s/table:public\\.campaign_operations_production_enablement_event/table:public.h1_unknown/g }' \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\""
run_mutation unreferenced_column_grant H1A009 public.h1_unknown \
    "perl -i -pe 'if (/^h1-column-acl-v1.*experiment_scheduler_protocol/) { s/public\\.experiment_scheduler_protocol/public.h1_unknown/ }' \"\$manifest_copy/055_campaign_operations_h1_column_acl.tsv\""
run_mutation change_owner H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/table:public\\.campaign_operations_production_transition_context/) { s/campaign_operations_h1_boundary_authority/campaign_operations_owner/ }' \"\$manifest_copy/055_campaign_operations_h1_object_inventory.tsv\""
run_mutation change_grantee H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/production_enablement_event:campaign_operations_production_reader:SELECT/) { s/campaign_operations_production_reader/campaign_operations_reader/g }' \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\""
run_mutation change_privilege H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/production_enablement_event:campaign_operations_production_reader:SELECT/) { s/SELECT/UPDATE/g }' \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\""
run_mutation add_grant_option H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/production_enablement_event:campaign_operations_production_reader:SELECT/) { s/:false/:true/; s/\\tfalse\\t/\\ttrue\\t/ }' \"\$manifest_copy/055_campaign_operations_h1_explicit_acl.tsv\""
run_mutation change_origin H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/table:public\\.campaign_operations_production_transition_context/) { s/\\texplicit\\t/\\tnull\\t/ }' \"\$manifest_copy/055_campaign_operations_h1_object_inventory.tsv\""
run_mutation change_signature H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/production_disable_replay_v1/) { s/\\(text,bigint,text,integer,text,text\\)/(text,bigint,text,integer,text,integer)/g }' \"\$manifest_copy/055_campaign_operations_h1_object_inventory.tsv\""
run_mutation add_column_privilege H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/experiment_scheduler_protocol:pqxx:UPDATE/) { s/updated_at/updated_at,singleton/ }' \"\$manifest_copy/055_campaign_operations_h1_column_acl.tsv\""
run_mutation remove_column_privilege H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/experiment_scheduler_protocol:pqxx:UPDATE/) { s/,updated_at// }' \"\$manifest_copy/055_campaign_operations_h1_column_acl.tsv\""
run_mutation change_default_scope H1A009 'campaign_operations_owner:<global>:r' \
    "perl -i -pe 'if (/campaign_operations_owner:public:r/) { s/:public:/:<global>:/; s/\\tpublic\\t/\\t<global>\\t/ }' \"\$manifest_copy/055_campaign_operations_h1_default_acl.tsv\""
run_mutation unsafe_builtin_default H1A009 manifest-digest-mismatch \
    "perl -i -pe 'if (/campaign_operations_owner:<global>:f/) { s/\\texplicit\\t/\\tnull\\t/ }' \"\$manifest_copy/055_campaign_operations_h1_default_acl.tsv\""
run_mutation stale_embedded_copy H1A009 stale-embedded \
    "perl -i -pe 's/H1_MANIFEST_DIGEST_SHA256: [0-9a-f]{64}/H1_MANIFEST_DIGEST_SHA256: stale-embedded/' \"\$case_root/migration.sql\""

echo "Campaign Operations H1 manifest mutation tests passed cases=19"
