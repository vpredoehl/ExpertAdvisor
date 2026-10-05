# RepositoryAgent / Qwen Field Evaluation

## Purpose

This document records a field evaluation of the ExpertAdvisor RepositoryAgent
and its local Qwen semantic-verification path.

The purpose of the evaluation was not to demonstrate that an LLM could freely
search, understand, or refactor the repository. The objective was narrower:
determine whether a bounded, deterministic repository interface combined with
a constrained semantic verifier can support trustworthy repository
investigation and behavior-preserving refactoring decisions.

The evaluation therefore emphasized:

- deterministic repository discovery and evidence selection;
- exact, bounded source evidence;
- explicit relationship evidence;
- fail-closed behavior when evidence was insufficient or malformed;
- semantic verification against selected evidence;
- rejection of unsupported architectural conclusions;
- preservation of a restricted `codex_assisted` trust boundary.

A successful evaluation did not require finding code to refactor. A negative
refactoring result was considered useful when the system could establish why a
plausible refactoring hypothesis was unsupported or already represented by an
existing abstraction.


## Trust Model

RepositoryAgent and Qwen serve different roles.

RepositoryAgent is responsible for deterministic repository operations:
catalog discovery, exact identity resolution, bounded evidence selection,
structural relationships, and evidence rereads.

Qwen is used as a constrained semantic verifier over evidence selected by
RepositoryAgent. It is not intended to be the repository search mechanism and
is not trusted to invent repository identities, perform unrestricted
repository exploration, or independently establish source relationships.

The intended flow is:

    deterministic discovery
      -> exact repository identity
      -> bounded evidence selection
      -> exact evidence reread
      -> constrained semantic verification

The design is intentionally fail closed. Missing identities, ambiguous
relationships, malformed structural metadata, evidence outside admitted
bounds, or insufficient semantic support should result in rejection or a
not-established outcome rather than expansion into unrestricted search.

RepositoryAgent provides separate tool profiles. The `full` profile retains
the complete RepositoryAgent surface. The `codex_assisted` profile exposes
only bounded planning-facing operations. Raw operations such as unrestricted
file listing, source excerpts, source search, and lower-level index traversal
remain unavailable to an assisted caller.

The same profile authorization mechanism governs both MCP tool listing and
tool invocation so a hidden operation cannot be invoked merely by knowing its
name.


## Bounded Investigation Capabilities Evaluated

The field work exercised or built upon several RepositoryAgent capabilities.

### Source claims

Bounded source-claim verification selects an exact source range, rereads that
range, and asks the semantic verifier whether the supplied source supports a
specific claim.

### Source bundle claims

Multiple exact ranges can be combined when a claim cannot be established from
one source range. Evidence remains explicitly bounded rather than becoming an
open-ended search context.

### Relationship claims

RepositoryAgent can represent and verify structural relationships using
evidence for the relevant source and destination identities rather than asking
the semantic verifier to infer repository relationships from arbitrary text.

### Relationship chains and sets

Bounded relationship-chain and relationship-set investigation allows larger
structural claims to be decomposed into individually evidenced relationships.

### Exact symbol investigation

`investigate_symbol` selects the exact indexed extent of a known symbol and
admits only bounded direct structural relationships associated with that
symbol.

### Subsystem investigation

`investigate_subsystem` provides bounded multi-file investigation over an
explicitly admitted subsystem/file set. Selection remains deterministic and
subject to fixed inventory, relationship, range, and source-line limits.

### Relationship-aware semantic verification

Relationship verification associates semantic evidence with individual
relationships instead of treating a collection of source ranges as a generic
text bundle. This reduces the opportunity for the verifier to support a
relationship using evidence belonging to a different edge.

### Operation binding and invocation

Scheduler investigation exposed an important limitation of ordinary lexical
call relationships: production behavior is sometimes connected through
operation fields and callbacks rather than direct function calls.

RepositoryAgent was extended to distinguish structural relationship kinds
including:

- `direct_invocation`;
- `operation_binding`;
- `operation_invocation`;
- callback/member invocation classifications that are not promoted to direct
  calls without sufficient structural proof.

This allowed scheduler execution paths to be represented without falsely
labeling operation bindings as ordinary direct calls.

### Positional aggregate operation bindings

Production scheduler composition also uses positional aggregate initialization
for operation sets. A dedicated verifier was introduced for these cases.

It rereads both:

1. the aggregate type declaration establishing field order; and
2. the complete aggregate initializer establishing the positional binding.

The resulting semantic proof is tied to the aggregate type, declared field
sequence, operation field, caller/callee identities, and exact source
evidence. This avoids treating positional correspondence as an unevidenced
assumption.

These capabilities formed the evidence infrastructure used during the
subsequent refactoring evaluation.


## SchedulerCore Evaluation

A read-only structural evaluation of `SchedulerCore` screened five plausible
behavior-preserving refactoring candidates.

The goal was to require evidence of actual duplication, responsibility mixing,
obsolete structure, or another concrete structural deficiency before changing
working scheduler code.


### Candidate 1: final-phase operation composition

The initial broad audit suggested extracting typed final-phase operation
composition around `RunFinalExperimentPhase`.

Deeper investigation reversed that conclusion.

Repository evidence showed that `FinalExperimentDispatchOperations` already
provides the named typed dependency/composition abstraction that the proposed
refactoring would have introduced. `FinalExperimentDispatchService` consumes
that abstraction, while `RunFinalExperimentPhase` binds its operation fields.

The initial candidate was therefore a false positive produced by insufficient
evidence depth.

**Conclusion:** insufficient benefit / already represented by an existing
abstraction.


### Candidate 2: selected-worker to reservation mapping

Investigation considered whether worker selection and reservation passed
admission-dependent information through an insufficiently explicit or
duplicated representation.

The repository already contains the relevant reservation abstraction,
including `ReservedWorkerAttempt` and the reservation/launch boundaries.

No duplicated transfer mechanism requiring another abstraction was
established.

**Conclusion:** already abstracted.


### Candidate 3: worker-attempt lifecycle

Investigation considered whether worker-attempt lifecycle behavior was spread
across production scheduler code and should be consolidated.

`WorkerAttemptLifecycleService` already provides the named lifecycle
abstraction.

No second competing lifecycle implementation was established.

**Conclusion:** already abstracted.


### Candidate 4: pending-candidate query and row mapping

Investigation considered separating pending-candidate query construction from
database-row mapping.

Evidence showed the responsibilities already reside behind
`SchedulerAdmissionService` and `PostgresSchedulerRepository`, with database
query/mapping behavior located at the repository boundary.

No useful additional separation was established.

**Conclusion:** already appropriately separated.


### Candidate 5: TRAIN semantic-selection delta

Investigation considered making TRAIN-specific semantic selection behavior
more explicit.

The repository already separates shared semantic admission from phase-specific
worker selection through the relevant semantic-admission and
training/inference-selection operations.

No actual duplicated semantic-admission implementation was established.

**Conclusion:** already abstracted.


### SchedulerCore conclusion

The final result was:

    NO CURRENT SCHEDULERCORE REFACTOR JUSTIFIED

This negative result is important.

The investigation began with several structurally plausible refactoring ideas.
Deeper bounded evidence showed that the repository already contained the
abstractions those proposals would have attempted to introduce.

In particular, Candidate 1 demonstrated why evidence depth matters: a
reasonable-looking refactoring recommendation was withdrawn after the
existing typed composition boundary was established.

Avoiding an unnecessary structural change to working scheduler code is a
successful outcome of the evaluation.


## Discovery Limitation Outside SchedulerCore

The first attempt to extend the evaluation beyond `SchedulerCore` exposed a
different problem.

`discover_catalog_targets` was effective when the caller already possessed a
sufficiently narrow valid catalog scope. At broad scope `Sources`, however,
reasonable conceptual queries could exceed existing catalog-population or
candidate-symbol caps.

The restricted interface did not provide a deterministic way to enumerate
valid direct child scopes below `Sources`.

This created a practical dead end:

    broad scope
      -> discovery cap
      -> no valid narrower scope identity
      -> caller guesses names
      -> guessed identities do not exist
      -> no evidence
      -> no Qwen verification

An early non-SchedulerCore attempt consequently guessed several nonexistent
symbols. Those investigations correctly failed, but they did not constitute a
useful evaluation of Qwen because Qwen never received source evidence.

This was classified as a RepositoryAgent discovery/navigation deficiency, not
a Qwen semantic-verification failure.

The appropriate response was not to weaken discovery caps or expose raw search.
Instead, the restricted caller needed a deterministic way to navigate catalog
metadata until it reached an appropriately narrow scope.


## Bounded Catalog Navigation

RepositoryAgent was extended with:

    list_catalog_children

The operation provides deterministic metadata-only navigation through the
indexed repository catalog.


### Behavior

`list_catalog_children`:

- resolves scope through the existing canonical scope rules;
- returns direct child scopes and direct source-file identities;
- excludes deeper descendants from a direct-child page;
- uses deterministic ordering;
- supports bounded pagination;
- defaults to 16 results;
- rejects requested limits above 32;
- returns a continuation cursor when additional children remain.


### Cursor integrity

Pagination cursors are authenticated and process-local.

A cursor is bound to the relevant scope and catalog identity so it cannot be
silently reused against another scope or an incompatible catalog state.

Malformed, cross-scope, stale, or otherwise incompatible cursors fail closed.


### Trust-boundary behavior

Catalog navigation is metadata only.

It does not return:

- source text;
- source excerpts;
- semantic search results;
- source evidence;
- semantic conclusions.

It performs no Qwen inference.

The operation was exposed to `codex_assisted` without exposing the raw
file/source/search operations that the profile intentionally hides.

Existing `discover_catalog_targets` caps were not weakened. Catalog navigation
instead gives the caller a deterministic method for narrowing a broad scope
before using existing bounded discovery.


### Tests

Focused testing covered:

- real direct children of `Sources`;
- nested scope navigation;
- exclusion of deeper descendants;
- deterministic ordering;
- complete pagination without duplicates or omissions;
- bounded limits;
- invalid scopes;
- malformed cursors;
- cross-scope cursor reuse;
- stale/incompatible cursor handling;
- absence of source/excerpt leakage;
- assisted-profile exposure;
- continued rejection of hidden raw operations.


### Live acceptance

The feature was tested against the real indexed repository.

The restricted caller navigated:

    Sources
      -> Sources/ModelInputPreparation

The narrowed scope returned exact source identities including:

    Sources/ModelInputPreparation/ModelInputPreparation.cpp
    Sources/ModelInputPreparation/ModelInputPreparation.hpp

Existing `discover_catalog_targets` then returned exact identities within that
scope.

One discovered source identity was passed into a bounded source
investigation. The configured restricted RepositoryAgent service selected one
evidence range and invoked Qwen once.

The semantic verdict was supported.

The complete pipeline therefore succeeded:

    catalog navigation
      -> exact scope
      -> catalog discovery
      -> exact identity
      -> bounded evidence read
      -> Qwen semantic verification

This directly resolved the discovery failure that had blocked the earlier
non-SchedulerCore evaluation.


## Non-SchedulerCore Evaluation

After catalog navigation passed live acceptance, the refactoring experiment
was repeated outside `SchedulerCore`.

Four real non-SchedulerCore scopes were selected from catalog metadata rather
than guessed:

- `Sources/MarketDataCore`;
- `Sources/ModelInputPreparation`;
- `Sources/StrategyEvaluationAdapters`;
- `Sources/StrategyEvaluationCore`.

Exact identities were obtained through bounded catalog discovery before source
investigation.


### Candidate 1: ModelInputPreparation responsibility mixing

The hypothesis was that model-input preparation combined data loading, tensor
construction, and result/metadata assembly in a way that justified extracting
a separate structural responsibility.

Qwen was given bounded source evidence for the relevant implementation.

The verifier rejected the claim that the supplied evidence established a
clear, independently meaningful extraction seam.

The presence of multiple operations in a function was not treated as
sufficient evidence of inappropriate responsibility mixing.

**Classification:** `NOT ACTUAL RESPONSIBILITY MIXING`.


### Candidate 2: MarketDataCore / ModelInputPreparation duplication

The hypothesis was that `MarketDataCore` and `ModelInputPreparation`
substantially duplicated a data-to-model-input transformation.

A bounded source bundle supplied evidence from both implementations.

Qwen rejected the duplication hypothesis.

The evidence established distinct responsibilities:

- `MarketDataCore` provides candlestick-loading behavior;
- `ModelInputPreparation` uses loading behavior while adding economic-event
  handling and tensor construction.

The shared loading operation already exists as an explicit abstraction.

**Classification:** `NOT ACTUAL DUPLICATION`.


### Candidate 3: StrategyEvaluation provenance and identity construction

Two exact indexed identities were examined:

- `BuildStrategyEvaluationProvenance`;
- `BuildStrategyEvaluationIdentity`.

At first glance both participate in strategy-evaluation identity handling and
could appear to represent duplicate construction logic.

Bounded evidence established different transformations.

`BuildStrategyEvaluationProvenance` derives provenance information from the
authoritative market path.

`BuildStrategyEvaluationIdentity` consumes provenance together with strategy
identity information to construct the canonical evaluation identity/hash.

The repository already expresses these concepts through named provenance and
identity abstractions and dedicated builders.

**Classification:** `ALREADY ABSTRACTED`.


### Candidate 4: StrategyEvaluationAdapters

Metadata navigation established the real scope, but bounded catalog discovery
under the selected metadata-derived query returned no exact investigation
target.

The experiment did not broaden the query until something appeared and did not
guess a symbol.

No semantic verifier call was warranted without selected evidence.

**Classification:** `NOT ESTABLISHED`.


### Non-SchedulerCore conclusion

No candidate survived the required evidentiary threshold.

The result was:

    NO REFACTOR IMPLEMENTATION JUSTIFIED YET

As with the SchedulerCore evaluation, this is a useful negative result rather
than a failed search for work.

The evaluation demonstrated that plausible architectural hypotheses could be
screened against repository evidence without creating pressure to modify code
merely because a refactoring exercise had been initiated.


## Qwen Findings

The field evaluation supports a specific role for Qwen.

Qwen was useful as a narrow semantic verifier when supplied exact,
RepositoryAgent-selected evidence.

It was particularly useful when the question was not simply what syntax
exists, but what the supplied source actually establishes.


### Rejecting unsupported conclusions

Qwen rejected the proposed ModelInputPreparation responsibility-mixing claim.

It also rejected the claimed duplication between MarketDataCore and
ModelInputPreparation.

These are important results because both hypotheses were superficially
plausible. The semantic verifier did not merely confirm the proposed
refactoring narrative.


### Establishing useful semantic distinctions

Qwen established the semantic distinction between strategy-evaluation
provenance construction and strategy-evaluation identity construction.

That finding helped demonstrate that an apparent consolidation opportunity was
already represented by intentionally different transformations and named
abstractions.


### Deterministic work that did not require Qwen

Qwen was not necessary for:

- navigating catalog scopes;
- identifying exact source files;
- discovering exact indexed function identities;
- resolving symbol extents;
- enforcing evidence caps;
- classifying structural relationship types;
- rereading exact evidence;
- enforcing tool-profile restrictions.

These responsibilities are better handled deterministically.


### Qwen is not the discovery mechanism

The failed first non-SchedulerCore experiment demonstrated an important
boundary.

When deterministic discovery could not provide a valid scope or identity,
Qwen had nothing useful to verify.

Adding deterministic catalog navigation fixed the problem.

The resulting architecture is therefore complementary:

    RepositoryAgent:
        where is the exact repository object?
        what evidence is admissible?
        what structural relationship is established?

    Qwen:
        does this exact bounded evidence support the semantic claim?

Using Qwen to compensate for missing repository discovery would weaken this
separation and was not supported by the evaluation.


### Structured output

Most constrained semantic verification completed successfully.

Earlier field work observed occasional structured-output retry/failure cases,
which remain worth monitoring. The final non-SchedulerCore Phase 2C
evaluation had no structured-output failures.

No model change is justified by the evidence collected here.


### Ledger behavior

Real field investigations frequently reported zero semantic-ledger hits,
including the final non-SchedulerCore evaluation.

This does not establish a cache defect. Claims and evidence identities may
simply have differed sufficiently that reuse was inappropriate.

Ledger effectiveness should therefore be observed during future real
workloads rather than redesigned based on this evaluation alone.


## Remaining Limitations and Follow-up

The following issues were observed but were intentionally not repaired as part
of the catalog-navigation work.


### malformed_direct_relationship

Two `investigate_symbol` calls for StrategyEvaluation identities failed closed
with:

    malformed_direct_relationship

The exact indexed extents remained available, allowing subsequent
source-backed investigation without weakening the boundary.

The failure deserves separate investigation because valid symbol identities
should ideally produce usable relationship metadata or a more specific
structural explanation.

It did not prevent completion of the field evaluation.


### Legacy RepositoryAgent test-harness defects

Several committed tests contain pre-existing harness/dependency problems.

Observed examples include:

- `test_ledger.py`;
- `test_retrieval.py`;
- `test_verifier.py`;

which use relative imports for sibling modules not present in the empty
`Tools.RepositoryAgent.tests` package.

`test_structural.py` also depends on a missing oracle file:

    expertadvisor_repository_agent_ledger_hardened.py

These issues predate bounded catalog navigation and were not modified merely
to make a broad test loop appear green.

Runnable RepositoryAgent regression tests relevant to the implementation
passed under their established invocation conventions.

The legacy harness issues should be handled separately if those tests are to
be restored as part of a canonical test suite.


### Local Qwen environment discrepancy

A direct local RepositoryAgent-checkout acceptance attempt reached exact
source selection but could not initialize the Qwen runtime because that Python
environment lacked `mlx_lm`.

The already configured restricted RepositoryAgent MCP service, using the
intended Qwen environment, successfully completed semantic verification for
the same discovered identity.

This was therefore classified as an environment dependency discrepancy rather
than a catalog-navigation or semantic-verification defect.


### Catalog discovery caps

`discover_catalog_targets` continues to fail closed when its existing
population/candidate caps are exceeded.

Those caps were deliberately retained.

`list_catalog_children` provides deterministic scope narrowing instead of
turning broad discovery into unrestricted or silently truncated search.


### Ledger utilization

Low observed ledger-hit rates should continue to be measured during normal
repository investigations.

No ledger redesign is justified by the current evidence.


## Field Evaluation Metrics

The final non-SchedulerCore Phase 2C run reported:

- 12 MCP calls;
- 0 invalid MCP calls;
- 1 bounded discovery-cap outcome;
- 5 evidence reads;
- 3 semantic model calls;
- 1 supported semantic result;
- 2 rejected semantic results;
- 2 symbol investigations that failed closed before evidence selection due to
  malformed relationship metadata;
- 0 structured-output failures;
- 0 ledger hits.

The numbers are less important than the behavior they represent.

The semantic verifier both accepted and rejected claims, deterministic
discovery handled repository identity, and malformed structural evidence
failed closed rather than being silently bypassed.


## Final Assessment

The field evaluation supports continued use of the bounded RepositoryAgent
architecture.

The strongest result is not that the system can generate refactoring ideas.
It is that the system can prevent plausible but insufficiently evidenced
refactoring ideas from becoming code changes.

Several conclusions follow.

1. **Deterministic repository discovery is essential.**

   Semantic verification is useful only after exact repository identities and
   evidence have been established.

2. **Catalog navigation materially improved assisted-agent usability without
   broadening source access.**

   `list_catalog_children` solved a demonstrated navigation deficiency while
   retaining existing discovery caps and raw-operation restrictions.

3. **Qwen is useful primarily as a constrained semantic verifier.**

   It successfully rejected unsupported architectural hypotheses and
   established useful semantic distinctions from bounded evidence.

4. **Fail-closed behavior is valuable.**

   Missing identities, discovery caps, malformed relationship metadata, and
   insufficient semantic evidence did not trigger unrestricted fallback
   behavior.

5. **Existing abstractions must be established before introducing new ones.**

   The SchedulerCore evaluation showed that apparently reasonable refactoring
   candidates can disappear when deeper repository evidence reveals an
   existing named abstraction.

6. **Negative refactoring results are useful engineering results.**

   Neither the SchedulerCore investigation nor the sampled non-SchedulerCore
   investigation established sufficient evidence for a behavior-preserving
   structural refactor.

The final refactoring conclusions are therefore:

    NO CURRENT SCHEDULERCORE REFACTOR JUSTIFIED

and:

    NO REFACTOR IMPLEMENTATION JUSTIFIED YET

No production refactor should be implemented merely to produce a positive
outcome from this evaluation.

The combination of deterministic RepositoryAgent discovery/evidence selection
and constrained Qwen semantic verification demonstrated practical value as a
repository-investigation system, particularly by distinguishing what the
source actually establishes from what initially appears architecturally
plausible.
