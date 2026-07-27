# LSTM Research Platform Architecture

Status: Authoritative documentation framework
Architecture version: 1.1.0
Last revised: 2026-07-24

## 1. Purpose and authority

This directory is the durable architecture reference for the LSTM Research
Platform. It explains the system's intended boundaries, invariants, ownership,
and extension rules. It complements the source code, database migrations,
tests, and operational runbooks; it does not replace them.

[Volume I](Volume_I_Foundation.md) is constitutional. Its rules apply across
all platform components unless an accepted Architecture Decision Record (ADR)
explicitly amends them. Volumes II–XII specify individual domains. ADRs record
why consequential decisions were made and how later changes relate to them.

Normative words such as **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have the
meaning defined in Volume I §1.3.

## 2. Document organization

| Volume | Domain | Current status |
|---|---|---|
| [I](Volume_I_Foundation.md) | Foundation and engineering constitution | Authoritative |
| [II](Volume_II_Data_Pipeline.md) | Market-data ingestion, transformation, and feature inputs | Foundation outline |
| [III](Volume_III_Label_Generation.md) | Targets, labels, horizons, and label provenance | Foundation outline |
| [IV](Volume_IV_Model_Architecture.md) | Model topology, compatibility, and serialization | Foundation outline |
| [V](Volume_V_Training_Engine.md) | Training, optimization, checkpoints, and resumption | Foundation outline |
| [VI](Volume_VI_Inference_Evaluation.md) | Inference, evaluation, analysis, and evidence | Foundation outline |
| [VII](Volume_VII_Experiment_Lifecycle.md) | Experiment identity, state, lineage, and continuation | Foundation outline |
| [VIII](Volume_VIII_Recommendation_Engine.md) | Advisory recommendation generation, scoring, persistence, and review | Foundation outline |
| [IX](Volume_IX_Trading_Profitability.md) | Trading simulation and profitability evidence | Reserved outline |
| [X](Volume_X_Research_Automation.md) | Controlled research automation and bounded Campaign Operations | Campaign Operations authoritative; broader automation reserved |
| [XI](Volume_XI_Scheduler.md) | Scheduler ownership, workers, recovery, and capacity | Foundation with accepted claim hardening |
| [XII](Volume_XII_Database.md) | PostgreSQL ownership, schema, migrations, and data integrity | Foundation aligned through Campaign Operations authority |

The reusable [volume template](VOLUME_TEMPLATE.md) fixes the standard section
order for Volumes II–XII. Stable headings permit references such as
“Volume VIII §7.3.” Sections may grow subordinate headings without renumbering
the standard top-level sections.

The [ADR index](adr/README.md) defines the ADR lifecycle and numbering scheme.

## 3. Architecture and implementation

Architecture describes intended contracts. Implementation demonstrates the
currently shipped behavior. Neither silently overrides the other:

1. An implementation change MUST conform to accepted architecture.
2. A discovered mismatch MUST be classified as an implementation defect, a
   documentation defect, or a proposed architectural change.
3. A material architectural change MUST be accepted in an ADR and reflected
   in the affected volume before or with its implementation.
4. Database migrations and tests remain executable evidence. Illustrative SQL
   in these documents is non-executable unless explicitly identified as a
   migration file.
5. Historical design documents remain evidence for the phases they describe.
   This hierarchy supplies the durable cross-phase organization.

Architecture documents MUST distinguish among:

- current implemented behavior;
- required invariants;
- planned but unimplemented behavior; and
- rejected or deliberately deferred behavior.

An outline or future-extension section is not implementation authorization.

## 4. Proposing an architectural change

An architectural proposal follows this sequence:

1. Identify the affected volume sections and existing ADRs.
2. State the problem and evidence without presupposing a solution.
3. Draft a new numbered ADR with status `Proposed`.
4. Describe compatibility, data migration, concurrency, operational safety,
   rollback, testing, and documentation effects.
5. Review the proposal against Volume I.
6. Mark the ADR `Accepted`, `Rejected`, or retain it as `Proposed`.
7. Update affected volume text and revision histories.
8. Implement only the accepted scope, with tests that demonstrate its stated
   invariants.

Editorial corrections that do not alter behavior, ownership, identity,
transactions, concurrency, or compatibility do not require an ADR. They still
update the affected document's revision history when material.

## 5. Review expectations

Architecture review MUST ask:

- Is the authoritative owner of each state transition clear?
- Is semantic identity defined independently of row IDs and hash shortcuts?
- Are transaction boundaries and concurrency outcomes explicit?
- Does the database enforce every invariant it can enforce safely?
- Are scheduler and worker responsibilities separated?
- Are machine interfaces stable, escaped, and testable?
- Are legacy rows and migration ledgers preserved deliberately?
- Is deferred behavior clearly non-executable?
- Can tests prove the contract without disturbing real experiments?

## 6. Relationship to Codex implementation prompts

Future Codex prompts SHOULD cite exact architecture sections and relevant ADRs,
for example:

> Implement the approved change in Volume VIII §7.3 and ADR-0012. Preserve
> Volume I §§5–10 and do not expand the scope described by ADR-0012.

Prompts MUST NOT redefine settled architecture casually. If a prompt conflicts
with an accepted volume or ADR, the conflict must be reported before code or
schema changes begin. A prompt may request an architectural amendment, but the
ADR and volume update are part of that request and precede behavioral work.

Codex implementation reports SHOULD identify:

- architecture sections implemented or preserved;
- ADRs followed or introduced;
- migrations and compatibility decisions;
- transaction and concurrency verification;
- regression boundaries; and
- deliberately deferred behavior.

Architecture references constrain implementation scope; they do not grant
permission to operate schedulers, mutate production data, create experiments,
or perform external actions unless the prompt separately authorizes them.

## 7. Document maintenance

- Markdown is the canonical format for this hierarchy.
- Filenames and volume numbers are stable public references.
- Headings use explicit section numbers.
- Links SHOULD be relative within this directory.
- Every volume and ADR includes a revision history.
- Renaming or splitting a volume requires an accepted ADR and compatibility
  aliases or clear replacement references.
- Superseded text remains discoverable through Git history and ADR links.

## 8. Existing implementation references

The Phase 4 recommendation documents remain detailed implementation references:

- [Recommendation identity and candidate generation](../Phase4AExperimentRecommendationFoundation.rst)
- [Recommendation persistence](../Phase4AExperimentRecommendationPersistence.rst)
- [Recommendation scoring](../Phase4AExperimentRecommendationScoring.rst)
- [Recommendation review](../Phase4AExperimentRecommendationReview.rst)

The accepted
[Campaign Operations specification](../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
is the detailed normative refinement of Volume X under ADR-0010 through
ADR-0017. Its traceability matrix identifies the exact ADR authority for each
major feature. Review prompts, transcripts, and earlier candidate outputs
remain historical evidence only.

Implemented Campaign Operations increments:

- [Phase 3 durable dispatch](CampaignOperations_Phase3_Durable_Dispatch.md)
- [Phase 4 controls, cancellation, and reconciliation](CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md)

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-07-15 | Established the permanent architecture hierarchy, governance workflow, and implementation-prompt relationship. |
| 1.1.0 | 2026-07-24 | Accepted the bounded Campaign Operations refinement, its ADR authority chain, and scheduler production-dispatch gate. |
| 1.2.0 | 2026-07-25 | Recorded implemented Campaign Operations Phase E durable dispatch and Phase F controls, cancellation, and reconciliation references. |
