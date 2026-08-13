# Campaign Operations — Post-Phase-H Direct SQL Readiness Boundary Independent Reverification

## 1. Executive disposition

DIRECT_SQL_READINESS_BOUNDARY_REVERIFICATION_FAILED

The migration-059 correction closes direct `EXECUTE` on the raw
`transition_campaign_operations_request_dispatch_production_v2(...)` function,
but it does **not** close the original direct-SQL readiness bypass under the
stated threat model.

The decisive reason is architectural: the same production Manager/dispatcher
login that was previously able to call the raw transition is now intentionally
granted `EXECUTE` on
`campaign_operations_production_dispatch_authorized_v3(...)`. That wrapper is
therefore directly callable from ordinary SQL by the same adversarial runtime
principal.

The wrapper calls a database-side readiness gate, but that gate can only verify
database-observable evidence. In particular, it accepts the
`approved_build_contract_canonical_value` as caller input and compares it with
the persisted approved build canonical. It cannot prove that the caller is the
reviewed C++ executable or that the actual running executable/build passed
`CaptureActualManagerBuildContract` / the C++ readiness evaluator.

Therefore a direct SQL caller holding the intended production Manager login can
read or otherwise obtain the persisted approved build canonical, pass it to the
wrapper, satisfy the SQL-visible predicates, and reach the production V2
transition without passing the application-only actual-running-build preflight.

That is precisely the class of bypass the prior focused security review warned
against: a wrapper executable by the same Manager role is forgeable for this
threat and is not a sealed service boundary.

## 2. Threat boundary

The reviewed threat model assumes an adversarial caller can:

- connect directly using the intended production Manager login;
- inherit or `SET ROLE` to its legitimately granted capability roles;
- call any function on which those roles have `EXECUTE`;
- choose arbitrary function arguments;
- use SQL directly instead of the CLI.

It does not assume superuser, database-owner, migration-owner, or compromised
administrator authority.

Under that threat model, the production Manager login has the exact ability
needed to invoke the new migration-059 wrapper directly.

## 3. SQL privilege / reachability result

Migration 059 correctly removes direct dispatcher access to the raw production
transition:

- `PUBLIC` is revoked;
- `pqxx` is revoked;
- `campaign_operations_production_dispatcher` is revoked;
- the raw function remains executable by its sealed owner,
  `campaign_operations_h1_boundary_authority`.

The migration also makes
`campaign_operations_production_dispatch_readiness_gate_v1(text)` private to
the sealed owner.

However, migration 059 then grants:

`EXECUTE` on
`campaign_operations_production_dispatch_authorized_v3(...)`
to `campaign_operations_production_dispatcher`.

The H2 production Manager login is intentionally a member of that dispatcher
role. Consequently this call graph remains directly reachable from ordinary SQL:

production Manager login
  -> campaign_operations_production_dispatcher
  -> campaign_operations_production_dispatch_authorized_v3(...)
  -> campaign_operations_production_dispatch_readiness_gate_v1(...)
  -> transition_campaign_operations_request_dispatch_production_v2(...)

The raw-function ACL bypass is fixed; the stronger service/readiness bypass is
not.

## 4. SQL gate versus C++ readiness

The SQL gate materially improves protection. It checks the database-observable
subset of readiness, including:

- migration-055 identity/checksum;
- scheduler contract/generation/cutover/evidence completeness;
- enablement contract and effective enable head;
- independent verification reference;
- approved build canonical equality;
- production role graph;
- completion nested-V2 proof;
- reconciliation blockers;
- enablement history integrity;
- existing Admission V1 canonical/hash/version evidence;
- existing production Attempt V2 canonical/hash/version/link evidence.

Those checks are useful and should be preserved.

But they are not equivalent to the complete C++ readiness contract. The
application additionally obtains the **actual running build identity** from the
real executable and working tree. The SQL wrapper receives only a
caller-supplied canonical string. Equality between that string and the
persisted approved canonical proves possession of the string, not execution by
the approved binary.

Thus a direct SQL caller can bypass at least the application-only
`actual_manager_build_contract` / `manager_build_contract_mismatch` boundary.

This is not hypothetical in the architecture: the earlier focused security
review explicitly concluded that a caller-supplied build canonical is not an
attestation and that “a wrapper executable by the same Manager role is not a
correction.”

## 5. Genesis-empty preservation

The earlier genesis-readiness correction is preserved in the C++ evaluator:

- explicit evidence counts distinguish zero Admission rows from malformed
  evidence;
- explicit evidence counts distinguish zero production Attempt V2 rows from
  malformed evidence;
- zero relevant rows are rendered/accepted as `genesis-empty`;
- once evidence exists, the required versions are enforced.

Migration 059's SQL gate also permits truly empty Admission and production
Attempt families because its integrity checks are `IF EXISTS (...)` scans. It
does not require a row merely to prove version 1/2.

So migration 059 does **not** reintroduce the original circular genesis
bootstrap defect.

## 6. Existing evidence fail-closed behavior

The SQL gate remains fail-closed for the structural conditions it actually
inspects. Existing Admission rows are rejected on wrong version, canonical
mismatch, or hash mismatch. Production Attempt rows in the gate's selected
scope are rejected on wrong version, canonical/hash mismatch, or missing
Admission linkage.

The C++ hydration path is stronger and inspects a broader V2-shape presence
surface, including nullable production-only columns. The SQL gate's production
Attempt scope currently begins with:

- `attempt_contract_version = 2`, or
- non-null `request_production_admission_id`, or
- non-null `production_enablement_event_id`.

That is narrower than the C++ readiness loader's explicit production-evidence
presence predicate, which also treats other production-only canonical/hash,
operation-key, actor, principal, approved-build and capability columns as
evidence. The database constraints may prevent many partial shapes in normal
operation, but the gate is not textually equivalent to the C++ scope. This is a
secondary defense-depth discrepancy, not the primary failure.

## 7. Tests

`Tests/CampaignOperationsPhaseHDirectSQLBoundaryTests.sh` is properly isolated:
it runs the H2 workflow in a disposable PostgreSQL cluster and creates
disposable database clones for hostile readiness mutations.

It proves useful properties:

- direct raw transition is denied to the Manager login;
- inherited dispatcher membership does not restore raw transition access;
- PUBLIC/pqxx lack raw access;
- the authorized wrapper succeeds for an approved replay;
- scheduler, enablement, canonical corruption, completion-proof and
  reconciliation blockers reject wrapper calls;
- build-canonical mismatch rejects.

But the positive wrapper test also demonstrates the unresolved threat:
`h2_manager_login` itself directly executes
`campaign_operations_production_dispatch_authorized_v3(...)`.

There is no way for that test to prove the real approved executable is running,
because the SQL call is deliberately made directly through `psql`.

Accordingly, the tests prove the new database gate works; they do not prove the
original C++-service-boundary bypass is closed.

## 8. Migration / ACL / manifest review

Migration 059 is appropriately forward/additive and guarded on exact migration
058 identity. It uses a sealed SECURITY DEFINER owner and fixed
`search_path = pg_catalog, public`, revokes the raw transition from runtime
roles, adds postcondition checks, and supplies an explicit ACL manifest.

The manifest correctly captures the intended final raw/wrapper ACL tuples.

Those mechanics are internally coherent, but they encode the wrong final
authority model for the stronger security requirement: the dispatcher remains
a direct SQL caller of the wrapper.

Applying the migration would improve the database boundary, but it would also
make this incomplete authority model durable in `schema_migrations`.

## 9. Concurrency / atomicity

The wrapper calls the readiness gate and raw transition in one transaction.
The raw transition retains the authoritative Phase-H/Phase-E mutation locks and
revalidation. This is materially better than an external preflight followed by
an unrelated SQL mutation.

I did not find enough packaged evidence to independently prove every race
between the gate's non-locking readiness observations and all mutable readiness
families. That question is secondary here because the wrapper already fails
the primary caller-authentication/service-boundary requirement.

A corrected design should bind the readiness authorization to the same
transaction while also ensuring the runtime SQL client cannot manufacture that
authorization.

## 10. Documentation consistency

The modified documentation accurately describes what migration 059 implements:
the Manager cannot call the raw V2 transition and instead calls a
readiness-gated service function.

However, calling that function a “sealed” service boundary overstates the
security property if the same adversarial Manager login has direct `EXECUTE`.
The earlier security review explicitly required a distinct trusted boundary or
database-verifiable unforgeable attestation.

Documentation should not claim the original direct-SQL service/readiness bypass
is closed until that stronger property exists.

## 11. Findings

| Severity | Finding |
|---|---|
| **High / Blocker** | The production Manager/dispatcher login can directly execute `campaign_operations_production_dispatch_authorized_v3(...)`. The wrapper is therefore not a distinct trusted service boundary and can be invoked without the C++ actual-running-build preflight. |
| Medium | SQL gate production-Attempt evidence scope is narrower than the C++ explicit V2-presence scope; review should align them or prove database constraints make the difference unreachable. |
| Medium | No packaged concurrency proof establishes that every mutable SQL-readiness predicate is protected against gate-to-transition races; this should be verified after the primary authority model is corrected. |
| Positive | Raw V2 transition ACL is correctly removed from dispatcher/PUBLIC/pqxx. |
| Positive | Genesis-empty Admission/Attempt behavior is preserved. |
| Positive | SQL-visible scheduler, enablement, role, completion-proof, reconciliation and structural evidence checks are substantially hardened. |

## 12. Narrow correction required

Do not discard migration 059's SQL gate work. The narrow next correction is to
change who is allowed to invoke the readiness-approved mutation boundary.

The production Manager login must not itself possess direct SQL `EXECUTE` on
the final service function that can reach the raw transition.

Use one of the previously reviewed acceptable patterns:

1. a distinct trusted service principal/boundary used only by the reviewed C++
   service after application readiness succeeds; or
2. an unforgeable database-verifiable, same-transaction attestation/context
   that the ordinary Manager SQL principal cannot synthesize.

A caller-supplied build canonical, GUC, string token, or wrapper executable by
the same Manager role is insufficient.

Preserve:

- raw-transition revocation;
- the SQL-visible readiness gate;
- genesis-empty semantics;
- immutable Admission/Attempt checks;
- lock ordering;
- replay/recovery behavior;
- emergency disable separation.

## 13. Apply / stage recommendation

**Do not apply migration 059 to the production LSTM database yet.**

**Do not stage/commit the current correction as final closure yet.**

The correction is directionally valuable but does not satisfy the threat model
that motivated the direct-SQL finding.

READY_FOR_TARGETED_CORRECTION
