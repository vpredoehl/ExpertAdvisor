# Volume I — Foundation and Engineering Constitution

Status: Authoritative
Version: 1.0.0
Last revised: 2026-07-15
Applies to: All LSTM Research Platform software, schemas, tests, commands, and architecture documents

## 1. Authority and interpretation

### 1.1 Purpose

This volume establishes the permanent engineering constitution of the LSTM
Research Platform. It governs how the platform represents research intent,
owns state, performs work, evolves schemas, exposes commands, and evaluates
changes. Domain volumes and ADRs refine these rules but do not silently weaken
them.

### 1.2 Precedence

When sources disagree, maintainers MUST identify and resolve the disagreement;
they MUST NOT select whichever source is convenient. The intended precedence is:

1. explicit safety and operator instructions for the current operation;
2. accepted ADRs that explicitly amend named architecture sections;
3. this volume;
4. authoritative domain-volume contracts;
5. executable migrations and public interface specifications;
6. implementation and tests as evidence of current behavior;
7. historical phase documents and informal notes.

Tests prove behavior but do not independently create architecture. A test that
conflicts with accepted architecture is either exposing a defect or is itself
incorrect.

### 1.3 Normative language

- **MUST** and **MUST NOT** express mandatory requirements.
- **SHOULD** and **SHOULD NOT** express defaults that require documented
  justification to depart from.
- **MAY** expresses a permitted option.
- **Authoritative** means the value or component decides the contract.
- **Advisory** means information supports a human or later policy decision but
  cannot itself authorize execution.

### 1.4 Architectural scope

Architecture governs behavior, ownership, identity, compatibility,
transactions, concurrency, operational safety, public interfaces, and durable
data. Purely local refactoring that preserves these properties is an
implementation concern, though it remains subject to coding and testing rules.

## 2. Project philosophy

### 2.1 Research platform, not a single training program

The platform exists to produce inspectable, reproducible research evidence.
Training a model is one stage in a larger lifecycle that includes data
preparation, labels, experiment identity, scheduling, checkpoints, inference,
analysis, continuation, recommendations, review, and future profitability
evaluation.

### 2.2 Evidence before automation

The system SHOULD first make evidence explicit, durable, and inspectable. New
automation MUST be layered on authoritative evidence and explicit policy. It
MUST NOT infer execution authority from a score, rank, review status, or
operator convenience.

### 2.3 Explicit state over implicit convention

Material state MUST be represented in typed code, durable schema, canonical
text, or an explicit interface. Filenames, process presence, log wording,
ordering accidents, and undocumented defaults MUST NOT become hidden sources
of truth.

### 2.4 Conservative compatibility

When historical state cannot be mapped authoritatively, the platform MUST
preserve it and fail closed for operations requiring missing provenance. It
MUST NOT backfill, guess, or reinterpret identity merely to make new features
apply to legacy rows.

### 2.5 Separation of proposal, decision, and execution

Research proposals, evaluation, human review, experiment creation, queueing,
and worker execution are distinct capabilities. A component owning one stage
does not own later stages unless architecture explicitly assigns it.

## 3. Determinism

### 3.1 Deterministic functions

Given the same authoritative inputs and versioned policy, a deterministic
component MUST produce the same canonical output, ordering, reason codes, and
decision. It MUST NOT depend on locale, unordered-container iteration,
session timezone, wall-clock time, process ID, or database row-return order
unless that input is explicitly part of the contract.

### 3.2 Total ordering

Any persisted or externally displayed ranking, selection, or bounded list MUST
define a total order. Semantic tie criteria and final deterministic tie-breakers
MUST be documented. Database queries whose result order matters MUST use an
explicit `ORDER BY`.

### 3.3 Numeric determinism

Canonical numeric representations MUST be locale-independent, finite, and
round-trip stable. NaN and infinities MUST be rejected unless a domain volume
defines a specific representation. Negative zero SHOULD normalize to zero
when the distinction has no domain meaning.

### 3.4 Controlled nondeterminism

Randomness, parallel reduction, hardware-specific kernels, and time-dependent
behavior MUST be treated as explicit inputs or documented limitations. Seeds,
algorithm versions, and execution environment provenance SHOULD be persisted
where exact replay depends on them. Claims of determinism MUST match what the
platform can actually reconstruct.

## 4. Reproducibility

### 4.1 Reproducibility record

A reproducibility record SHOULD identify, as applicable:

- semantic and invocation configuration;
- source data and date boundaries;
- feature, label, normalization, model, optimizer, and scoring versions;
- code/build/schema provenance;
- lineage and resume source;
- random seeds and known unpersisted state;
- resulting model and evaluation evidence.

### 4.2 Honest reproducibility

The platform MUST distinguish exact replay, semantic reproduction, compatible
re-execution, and approximate comparison. Missing data snapshots, RNG state,
or implementation contracts MUST be reported rather than silently inferred.

### 4.3 Immutable evidence

Historical results, score components, review events, and equivalent audit
records SHOULD be append-only through normal runtime interfaces. Corrections
use new records, explicit supersession, or migrations; they do not rewrite
history invisibly.

## 5. Identity

### 5.1 Identity categories

The platform distinguishes at least:

- **row identity**: a database primary key locating one record;
- **semantic identity**: the research question or configuration being studied;
- **invocation identity**: semantic identity plus execution-affecting lineage
  and operational inputs required to distinguish runs;
- **provenance identity**: origin, implementation, data, and lineage evidence;
- **runtime state**: mutable lifecycle facts that do not redefine the original
  research question.

These categories MUST NOT be conflated.

### 5.2 Canonical text

Semantic identities MUST have an authoritative, version-tagged canonical
representation. Canonicalization MUST specify field order, null handling,
numeric formatting, collection ordering, symbols, dates, and version prefixes.
Equality is determined by canonical content, not database ID or hash alone.

### 5.3 Hashes

Hashes are lookup accelerators and compact diagnostics unless an accepted ADR
defines a collision-resistant identity scheme with different guarantees. A
hash match MUST be followed by authoritative canonical comparison. Hash
collisions MUST be observable and MUST NOT merge distinct identities.

### 5.4 Identity evolution

Adding or removing identity fields, changing canonical grammar, or changing
equivalence semantics is an architectural change. It requires a version bump,
compatibility analysis, migration strategy, affected-volume update, and ADR.

### 5.5 Experiment identity

An experiment row ID identifies one lifecycle record; it does not by itself
identify a unique research question. Duplicate detection MUST state whether it
uses semantic, invocation, lineage, or another named equivalence. Operational
state such as worker PID, timestamps, logs, or status MUST NOT enter semantic
identity.

## 6. Component boundaries

### 6.1 Pure domain components

Pure domain components own parsing, canonicalization, validation, deterministic
calculation, transition rules, and stable reason codes. They MUST NOT access
PostgreSQL, start processes, inspect scheduler state, or perform hidden I/O.

### 6.2 Repositories

Repositories own SQL, typed row mapping, database existence checks, narrow
transactional persistence primitives, and database-level concurrency
mechanisms. Raw database rows MUST NOT escape repository boundaries.
Repository APIs MUST validate narrow structural preconditions such as positive
identifiers and valid limits when callers can invoke them directly.

Business policy SHOULD remain in domain or service code unless the database
must enforce the same invariant at its boundary.

### 6.3 Services

Services orchestrate domain logic and repositories. They own use-case order,
transaction selection when spanning repository steps, structured outcomes,
and mapping failures to stable service results. They MUST NOT duplicate SQL or
scatter transition logic across presentation code.

### 6.4 CLI and presentation

CLI code owns argument parsing, command mutual exclusion, early validation,
dispatch, and presentation. It MUST NOT embed persistence SQL or become the
only enforcement point for durable invariants.

### 6.5 Scheduler

The scheduler owns only explicitly scheduled lifecycle work. A new command or
service does not become scheduler-managed merely because it can run unattended.
Scheduler integration requires an accepted ownership contract under §10.

## 7. Database ownership

### 7.1 PostgreSQL as durable source of truth

PostgreSQL is the authoritative store for durable experiments, models,
evaluations, lifecycle state, policies, recommendations, reviews, and audit
history represented by the current schema. In-memory values and logs may cache
or explain this state but MUST NOT silently supersede it.

### 7.2 Schema invariants

The database SHOULD enforce referential integrity, valid state shapes,
uniqueness, ownership, immutability, and value domains where doing so is safe
for legacy data and concurrency. Application validation improves errors but
does not replace an appropriate database boundary.

### 7.3 Least privilege

Runtime roles receive only privileges required by their responsibilities.
Append-only audit tables MUST NOT grant ordinary runtime roles `UPDATE` or
`DELETE`. Test cleanup uses rollback, isolated schemas, or an explicit owner
connection; it MUST NOT weaken runtime permissions.

### 7.4 Migrations

Schema evolution is additive and ledgered. A checksum-recorded migration MUST
NOT be rewritten. Corrections use the next migration. Migrations MUST preserve
valid rows, avoid guessed provenance, be repeatable under the project runner,
and document compatibility effects.

Illustrative SQL in architecture documents is non-executable. Only files in
the migration system alter schema.

### 7.5 Legacy data

Legacy rows remain valid when their original contract allowed them. New
operations MAY exclude them when required provenance is missing, but MUST do so
explicitly and observably. Backfills require authoritative inputs and a
separate reviewed migration or data-repair procedure.

## 8. Transaction rules

### 8.1 Atomic invariants

State changes that form one logical fact MUST commit or roll back together.
Examples include status plus audit event, score plus ordered components, and a
decision plus its immutable provenance.

### 8.2 Transaction size

Write transactions SHOULD be short and should not contain expensive model
work, external processes, user interaction, network calls, or large in-memory
computation. Read, calculate, and persist phases SHOULD be separated when the
contract permits, followed by an authoritative recheck inside the write
transaction.

### 8.3 Read-only behavior

Inspection and evidence-loading paths SHOULD use read-only transactions.
Validation failure before persistence MUST create no durable row unless an
explicit attempt/audit record is part of the architecture.

### 8.4 Failure behavior

Failure between related writes MUST leave no partial state. Transaction errors
MUST surface stable outcomes; they MUST NOT be silently treated as success or
idempotency. Finalization failures SHOULD preserve the original error.

### 8.5 External work

Database transactions MUST NOT remain open while launching or waiting for
workers. Process launch and database state transitions require an explicit
claim/attempt protocol owned by the scheduler volume.

## 9. Concurrency philosophy

### 9.1 Narrow serialization

Concurrency control SHOULD serialize only records or semantic keys that can
conflict. Global locks require exceptional justification.

### 9.2 Database authority

Correctness MUST rely on database-visible mechanisms such as row locks,
constraints, unique indexes, transaction-scoped advisory locks, or compare-and-
set updates—not process-local mutexes alone.

### 9.3 Defined outcomes

Every concurrent mutation contract MUST define winners, losers, retry
behavior, idempotency, and resulting row counts. A conflict MUST be observable
and deterministic; it MUST NOT create partial or duplicate history.

### 9.4 Idempotency

Idempotency is explicit, not assumed. A retry is idempotent only when the
system can prove the request and persisted immutable result are equivalent.
Otherwise, repeated terminal transitions fail or require an authoritative
request identifier.

### 9.5 Lock order

Operations that lock multiple rows or tables MUST define a stable lock order
and test deadlock-sensitive paths. Locks are released by transaction end, not
by ad hoc process conventions.

## 10. Scheduler interaction rules

### 10.1 Scheduler ownership

The scheduler owns experiment lifecycle polling, capacity allocation, worker
claiming, launch, recovery, and persisted operational transitions assigned in
Volume XI. It does not own pure research policy merely because policy results
may later influence work.

### 10.2 Explicit integration

A subsystem enters scheduler polling only through an accepted architectural
change defining:

- trigger and eligibility;
- durable claim and attempt state;
- capacity and fairness;
- retry and orphan recovery;
- idempotency and concurrency;
- operator controls and dry-run behavior;
- observability and shutdown; and
- regression boundaries for existing work.

### 10.3 Advisory isolation

Recommendation generation, scoring, ranking, and review are advisory unless a
later accepted ADR defines a separate conversion boundary. Approval alone is
not execution authorization, scheduler eligibility, queueing, or experiment
creation.

### 10.4 Operational safety

Development and verification MUST NOT disturb running experiments. Tests use
isolated schemas, rollback, mocks, or exact disposable IDs. Process signals,
scheduler launch, experiment creation, and live-row mutation require explicit
task authorization.

## 11. Testing philosophy

### 11.1 Test layers

Material features SHOULD include, as applicable:

- pure parsing, validation, identity, and calculation tests;
- schema and migration tests;
- repository mapping and transaction tests;
- concurrency tests with independent connections;
- service orchestration and structured-output tests;
- CLI parsing, mutual-exclusion, and dispatch tests;
- controlled integration tests with disposable fixtures; and
- regression tests for adjacent subsystems.

### 11.2 Invariant-first tests

Tests SHOULD prove externally meaningful invariants, including unchanged state
after rejection, rollback after partial failure, exact SQLSTATE where the
database contract matters, deterministic ordering, and absence of unauthorized
side effects.

### 11.3 Test isolation

Tests MUST NOT rely on or mutate genuine research rows. Cleanup uses exact IDs
and appropriate privileges. Test-only convenience MUST NOT weaken runtime
constraints or permissions.

### 11.4 Build quality

Changed C++ components SHOULD compile with `-Wall -Wextra -Wpedantic -Werror`.
The Release configuration MUST build before a major capability is complete.
Formatting and `git diff --check` MUST pass.

### 11.5 Regression scope

Verification MUST be proportional to affected ownership boundaries. A change
to persistence or scheduler interaction demands broader regression coverage
than a pure formatting change. Reports distinguish tests actually run from
tests inferred or deferred.

## 12. Coding standards

### 12.1 General standards

Code MUST favor explicit types, narrow APIs, deterministic behavior, stable
error codes, and readable ownership. Hidden global state and duplicated policy
logic SHOULD be avoided.

### 12.2 C++ boundaries

- Domain headers SHOULD expose typed values rather than database or CLI types.
- Raw `pqxx::row` values remain inside repositories.
- Optional data uses explicit optional types, not magic values.
- Enums exposed durably require stable text mappings and strict parsing.
- Exhaustive switches SHOULD fail visibly for invalid values.
- Numeric and date parsing MUST be locale-independent and strict.

### 12.3 Errors and reason codes

Machine reason codes are stable lowercase identifiers. Human explanation is
stored or rendered separately. Exceptions MAY carry stable internal errors,
but public commands MUST map them predictably.

### 12.4 Scope discipline

Implementation changes MUST preserve unrelated worktree changes and avoid
opportunistic redesign. A discovered adjacent defect is reported and fixed
only when authorized or required for the requested invariant.

## 13. Documentation standards

### 13.1 Required distinctions

Documents MUST distinguish current behavior, normative requirements, future
extensions, and illustrative examples. Future text MUST NOT imply delivery.

### 13.2 Stable references

Architecture headings use explicit section numbers. Filenames, volume numbers,
ADR numbers, and durable terminology SHOULD remain stable. References use the
form `Volume VIII §7.3` and `ADR-0003`.

### 13.3 Revision history

Every volume and ADR contains version/date history. A behavioral architecture
change names the accepting ADR. Editorial corrections may be grouped but must
not conceal a contract change.

### 13.4 Implementation documentation

Detailed phase documents and runbooks MAY remain near implementation. They
SHOULD link back to architecture, while architecture references them for exact
current details rather than duplicating volatile command or schema inventories.

## 14. CLI standards

### 14.1 Explicit commands

Mutating commands MUST be explicit, target a bounded object or operation, and
validate arguments before persistence when possible. Unrelated command families
MUST be mutually exclusive.

### 14.2 Machine output

Machine records use stable event names, field names, reason codes, explicit
null representation, deterministic field order, and documented escaping.
Untrusted text MUST NOT inject delimiters or lines.

### 14.3 Human output

Human summaries SHOULD be concise and readable. Ordinary spaces and punctuation
remain readable, while control bytes and terminal escape sequences are rendered
safely. Human text does not replace machine events.

### 14.4 Exit status

Exit codes MUST distinguish success, validation failure, conflict, and system
failure when callers need that distinction. Repeated terminal operations MUST
not be silently reported as success.

### 14.5 Safety language

Advisory commands MUST state their non-executing boundary where confusion is
plausible. Approval, score, rank, evaluation, and profitability evidence MUST
not be described as guarantees.

## 15. Versioning policy

### 15.1 Architecture versions

Architecture uses semantic versions:

- major: incompatible constitutional or domain-contract change;
- minor: backward-compatible architectural capability or material extension;
- patch: clarification or correction that does not change behavior.

### 15.2 Contract versions

Canonical identities, persisted policy grammars, schemas, serialized models,
machine-output records, and other durable contracts carry their own explicit
versions where compatibility requires them. Architecture version does not
replace those versions.

### 15.3 Database versions

Database versioning is the ordered migration ledger. Migration filenames and
checksums are permanent after application. Architecture records intent but
does not renumber or rewrite history.

### 15.4 CLI compatibility

Existing command meanings and machine events SHOULD remain backward-compatible.
Breaking changes require an ADR, migration/deprecation plan, version impact,
and updates to tests and documentation.

## 16. Architecture governance

### 16.1 Changes requiring an ADR

An ADR is required for material changes to:

- identity or canonicalization;
- durable ownership or schema strategy;
- transaction or concurrency semantics;
- scheduler responsibilities;
- experiment lifecycle or status transitions;
- recommendation-to-experiment conversion;
- model/data compatibility contracts;
- public CLI or machine-output compatibility;
- security or runtime privilege boundaries; or
- this constitution.

### 16.2 Decision lifecycle

ADRs progress through the statuses in `adr/README.md`. Accepted ADRs are
immutable records; later decisions supersede rather than rewrite them, apart
from clearly marked editorial corrections.

### 16.3 Review and approval

Architecture review evaluates alternatives, compatibility, migrations,
operational risk, observability, testing, and rollback. Implementation SHOULD
not begin while a required ADR remains merely proposed.

### 16.4 Emergency corrections

An urgent safety fix MAY precede complete documentation when delaying it would
increase harm. The change MUST remain narrowly scoped, preserve evidence, and
receive a follow-up ADR and volume update promptly.

## 17. Future extension policy

### 17.1 Capability gates

Future capabilities begin as documented proposals. They become executable only
after authoritative inputs, identity, persistence, transaction behavior,
concurrency, CLI, tests, and operational controls are defined.

### 17.2 No authority by implication

The presence of a future volume, schema placeholder, status, score, or approved
ADR does not by itself enable runtime behavior. Implementation and operator
authorization remain separate gates.

### 17.3 Compatibility-first extension

Extensions SHOULD add new versioned fields, tables, commands, or adapters while
preserving existing contracts. If compatibility is impossible, the proposal
must define migration, coexistence, rollback, and observability explicitly.

### 17.4 Autonomous research

Autonomous experimentation, recommendation conversion, profitability-driven
selection, and self-modifying policies require explicit budgets, safety limits,
audit trails, kill switches, and scheduler ownership. They MUST NOT emerge by
connecting existing advisory components informally.

### 17.5 Data and model evolution

New features, labels, architectures, attention mechanisms, optimizers, and data
sources require versioned semantic contracts. Git commit or binary name alone
is insufficient as semantic identity.

## 18. Glossary

**ADR**
Architecture Decision Record: an immutable explanation of a consequential
decision, its context, and consequences.

**Advisory**
Evidence or disposition that does not authorize execution.

**Canonical text**
Versioned, deterministic, authoritative textual representation used for
semantic equality.

**Continuation**
An explicitly derived experiment invocation extending prior experiment/model
lineage under a persisted continuation policy.

**Durable state**
State that must survive process termination and is represented in PostgreSQL
or another explicitly accepted authoritative store.

**Experiment**
A persisted lifecycle record describing one configured research execution and
its operational state.

**Invocation identity**
Identity that distinguishes executions using semantic configuration plus
lineage or operational inputs defined by the domain contract.

**Legacy row**
A durable record created under an earlier schema or contract that may lack
newer provenance.

**Provenance**
Evidence describing origin, lineage, policy, data, implementation, operator,
or time without necessarily defining semantic equality.

**Recommendation**
A persisted advisory proposal for a research configuration. It is not an
experiment or scheduler job.

**Repository**
The component owning SQL, typed persistence mapping, and narrow database
transactions.

**Runtime state**
Mutable lifecycle and operational facts such as status, phase, worker PID, or
timestamps.

**Semantic identity**
The authoritative equivalence relation for the research question or
configuration under study.

**Service**
The component orchestrating domain rules and repositories for one use case.

**Source of truth**
The authoritative durable representation from which competing state is
reconciled.

**Structural rank**
Deterministic ordering metadata. It is not inherently a score or quality claim.

**Worker**
A process executing bounded training, inference, analysis, or another
scheduler-owned operation.

## 19. References

- [Architecture index](README.md)
- [ADR framework](adr/README.md)
- [ADR-0001: PostgreSQL as source of truth](adr/ADR-0001-postgresql-source-of-truth.md)
- [ADR-0002: Deterministic experiment identity](adr/ADR-0002-deterministic-experiment-identity.md)
- [ADR-0003: Advisory recommendation evaluation](adr/ADR-0003-advisory-recommendation-evaluation.md)
- [ADR-0004: Scheduler ownership boundaries](adr/ADR-0004-scheduler-ownership-boundaries.md)
- [Phase 4 recommendation foundation](../Phase4AExperimentRecommendationFoundation.rst)
- [Phase 4 recommendation persistence](../Phase4AExperimentRecommendationPersistence.rst)
- [Phase 4 recommendation scoring](../Phase4AExperimentRecommendationScoring.rst)
- [Phase 4 recommendation review](../Phase4AExperimentRecommendationReview.rst)

## 20. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-07-15 | Established the platform constitution, governance, boundaries, glossary, and future-extension policy. |
