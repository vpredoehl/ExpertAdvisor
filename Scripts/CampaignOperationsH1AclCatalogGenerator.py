#!/usr/bin/env python3
"""Capture PostgreSQL ACL/default catalogs without consulting expected manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from CampaignOperationsH1AclCatalog import OBSERVED_FIELDS

CAPTURE_VERSION = "h1-acl-catalog-capture-v1"
QUERY_ID = "pg-catalog-acl-query-v3"


def fail(detail: str) -> None:
    print(f"H1ACL502 key=acl-catalog-v3 stage=independent-catalog-capture detail={detail}",
          file=sys.stderr)
    raise SystemExit(1)


def run_psql(connection: list[str], database: str, sql: str) -> tuple[list[list[str]], dict[str, object]]:
    executable = shutil.which("psql")
    if executable is None:
        fail("psql-not-found")
    command = [executable, *connection, "-X", "-A", "-t", "-F", "\t", "-q", "-v", "ON_ERROR_STOP=1",
               "-d", database, "-c", sql]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    receipt = {
        "executable": executable,
        "executable_digest": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        "command": command, "exit_status": completed.returncode,
        "stdout_digest": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_digest": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        fail("catalog-query-failed")
    lines = list(csv.reader(completed.stdout.splitlines(), delimiter="\t"))
    if lines and len(lines[-1]) == 1 and re.fullmatch(r"\([0-9]+ rows?\)", lines[-1][0]):
        lines.pop()
    return lines, receipt


CATALOG_SQL = r"""
WITH object_catalog AS (
 SELECT 'schema'::text object_class, quote_ident(n.nspname) object_identity,
        pg_get_userbyid(n.nspowner) owner, n.nspacl acl, 'n'::"char" object_kind
 FROM pg_namespace n WHERE n.nspname='public'
 UNION ALL
 SELECT CASE c.relkind WHEN 'S' THEN 'sequence' WHEN 'v' THEN 'view'
          WHEN 'm' THEN 'materialized_view' WHEN 'p' THEN 'partitioned_table'
          WHEN 'f' THEN 'foreign_table' ELSE 'table' END,
        quote_ident(n.nspname)||'.'||quote_ident(c.relname),pg_get_userbyid(c.relowner),c.relacl,
        CASE WHEN c.relkind='S' THEN 'S'::"char" ELSE 'r'::"char" END
 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
 WHERE n.nspname='public' AND c.relkind IN ('r','p','S','v','m','f')
 UNION ALL
 SELECT CASE p.prokind WHEN 'p' THEN 'procedure' WHEN 'a' THEN 'aggregate' ELSE 'function' END,
        quote_ident(n.nspname)||'.'||p.oid::regprocedure::text,pg_get_userbyid(p.proowner),p.proacl,'f'::"char"
 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public'
 UNION ALL
 SELECT CASE WHEN t.typtype='d' THEN 'domain' ELSE 'type' END,
        quote_ident(n.nspname)||'.'||quote_ident(t.typname),pg_get_userbyid(t.typowner),t.typacl,'T'::"char"
 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace WHERE n.nspname='public'
), object_rows(tuple_kind,object_class,object_identity,owner,raw_acl,origin,
              default_acl_source,grantee,privilege_type,is_grantable,object_kind) AS (
 SELECT 'object_acl'::text tuple_kind,o.object_class,o.object_identity,o.owner,
        coalesce(o.acl::text,'') raw_acl,CASE WHEN o.acl IS NULL THEN 'null' ELSE 'explicit' END origin,
        CASE WHEN o.acl IS NULL THEN 'acldefault('||o.object_kind::text||','||o.owner||')' ELSE 'none' END default_acl_source,
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END grantee,
        acl.privilege_type,acl.is_grantable,o.object_kind
 FROM object_catalog o
 CROSS JOIN LATERAL aclexplode(coalesce(o.acl,acldefault(o.object_kind,(SELECT oid FROM pg_roles WHERE rolname=o.owner)))) acl
), column_rows(tuple_kind,object_class,object_identity,owner,raw_acl,origin,
              default_acl_source,grantee,privilege_type,is_grantable,object_kind) AS (
 SELECT 'column_acl','column',quote_ident(n.nspname)||'.'||quote_ident(c.relname)||'.'||quote_ident(a.attname),
        pg_get_userbyid(c.relowner),a.attacl::text,'explicit','none',
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
        acl.privilege_type,acl.is_grantable,'r'::"char"
 FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace
 CROSS JOIN LATERAL aclexplode(a.attacl) acl
 WHERE n.nspname='public' AND a.attnum>0 AND NOT a.attisdropped AND a.attacl IS NOT NULL
), role_scope_kind AS (
 SELECT r.oid,r.rolname owner,s.scope_name,s.namespace_oid,k.object_kind
 FROM pg_roles r
 CROSS JOIN (SELECT '<global>'::text scope_name,0::oid namespace_oid
             UNION ALL SELECT nspname,oid FROM pg_namespace) s
 CROSS JOIN (VALUES ('f'::"char"),('r'::"char"),('S'::"char"),('T'::"char"),('n'::"char")) k(object_kind)
 WHERE r.rolname LIKE 'campaign_operations_%'
), default_rows(tuple_kind,object_class,object_identity,owner,raw_acl,origin,
               default_acl_source,grantee,privilege_type,is_grantable,object_kind,scope_name) AS (
 SELECT 'default_acl','default',r.owner||':'||r.scope_name||':'||r.object_kind::text,r.owner,
        coalesce(d.defaclacl::text,''),CASE WHEN d.oid IS NULL THEN 'null' ELSE 'explicit' END,
        'acldefault('||r.object_kind::text||','||r.owner||')',
        CASE acl.grantee WHEN 0 THEN 'PUBLIC' ELSE pg_get_userbyid(acl.grantee) END,
        acl.privilege_type,acl.is_grantable,r.object_kind,r.scope_name
 FROM role_scope_kind r LEFT JOIN pg_default_acl d ON d.defaclrole=r.oid
  AND d.defaclnamespace=r.namespace_oid AND d.defaclobjtype=r.object_kind
 CROSS JOIN LATERAL aclexplode(coalesce(d.defaclacl,acldefault(r.object_kind,r.oid))) acl
)
SELECT tuple_kind,object_class,object_identity,owner,raw_acl,origin,default_acl_source,
       grantee,privilege_type,CASE WHEN is_grantable THEN 'true' ELSE 'false' END,
       ''::text default_scope,object_kind::text
FROM object_rows
UNION ALL SELECT tuple_kind,object_class,object_identity,owner,raw_acl,origin,default_acl_source,
       grantee,privilege_type,CASE WHEN is_grantable THEN 'true' ELSE 'false' END,'',object_kind::text
FROM column_rows
UNION ALL SELECT tuple_kind,object_class,object_identity,owner,raw_acl,origin,default_acl_source,
       grantee,privilege_type,CASE WHEN is_grantable THEN 'true' ELSE 'false' END,scope_name,object_kind::text
FROM default_rows
ORDER BY 1,2,3,4,8,9,10,11,12
"""


def main() -> None:
    if len(sys.argv) != 9 or sys.argv[1] != "generate":
        raise SystemExit("usage: ... generate HOST PORT USER DATABASE RUN_ID CAPTURE_TSV PAYLOAD_JSON")
    execution_id = os.environ.get("H1_TRUSTED_GENERATOR_EXECUTION_ID", "")
    if not re.fullmatch(r"GEXEC-.+", execution_id):
        fail("trusted-generator-execution-required")
    host, port, user, database, run_id = sys.argv[2:7]
    capture_path, payload_path = map(Path, sys.argv[7:9])
    connection = ["-h", host, "-p", port, "-U", user]
    identity_rows, identity_receipt = run_psql(
        connection, database,
        "SELECT system_identifier::text||':'||current_database()||':'||oid::text "
        "FROM pg_control_system(),pg_database WHERE datname=current_database()")
    if len(identity_rows) != 1 or len(identity_rows[0]) != 1:
        fail("unobservable-cluster-identity")
    cluster_id = identity_rows[0][0]
    query_execution_id = f"QEXEC-{run_id}-{uuid.uuid4().hex}"
    catalog_rows, query_receipt = run_psql(connection, database, CATALOG_SQL)
    if not catalog_rows or any(len(row) != 12 for row in catalog_rows):
        fail("invalid-catalog-row-shape")
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    with capture_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OBSERVED_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for values in catalog_rows:
            body = dict(zip(OBSERVED_FIELDS[5:], values))
            writer.writerow({"capture_version": CAPTURE_VERSION, "query_id": QUERY_ID,
                             "cluster_id": cluster_id, "run_id": run_id,
                             "query_execution_id": query_execution_id, **body})
    payload = {
        "query_id": QUERY_ID, "catalog_rows": catalog_rows,
        "owner": sorted({row[3] for row in catalog_rows}),
        "raw_acl": [row[4] for row in catalog_rows],
        "origin": [row[5] for row in catalog_rows],
        "default_acl_source": [row[6] for row in catalog_rows],
        "expanded_acl_rows": catalog_rows,
        "cluster_id": cluster_id, "query_execution_id": query_execution_id,
        "identity_query_receipt": identity_receipt, "catalog_query_receipt": query_receipt,
    }
    payload_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
