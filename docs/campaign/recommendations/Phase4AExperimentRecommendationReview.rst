Phase 4A Step 5: explicit advisory recommendation review
========================================================

Step 5 adds an explicit operator-driven review state and immutable audit event
for persisted Step 3 recommendations.  Approval means only that an operator
reviewed an advisory recommendation and marked it approved for possible later
use.  It does not authorize execution, predict profitability or correctness,
create an experiment, enter a scheduler queue, or make the recommendation
eligible for automatic conversion.

Transitions and metadata
------------------------

Only ``proposed`` recommendations may transition, once, to ``approved``,
``rejected``, or ``expired``.  Those three states are terminal.  Reopening,
unapproval, reversal, superseding, and same-state retries are rejected with a
stable status conflict and do not add another event or alter timestamps.

Migration 032 adds ``approved_at`` and the immutable
``experiment_recommendation_review_event`` table.  Historical approved rows
that already use ``approved_experiment_id`` remain valid, but Step 5 approval
sets ``approved_at`` and deliberately leaves ``approved_experiment_id`` null.
Rejection reuses ``rejected_at`` and ``rejected_reason``; expiration reuses
``expired_at``.  Reviewer, stable reason code, optional note, optional score
reference, and identity/scan/source snapshots live in the event table.
Additive migration 033 removes ``UPDATE`` and ``DELETE`` privileges from the
normal ``pqxx`` runtime role; runtime access is limited to inserting and
reading events plus using the event-ID sequence.  Fixture cleanup therefore
uses a database-owner test connection rather than weakening audit history.

Each review uses one short transaction.  It locks only the target
recommendation row, verifies that the current status is ``proposed``, validates
complete identity and scan provenance, checks optional score ownership and a
completed score run, updates status metadata, inserts exactly one event, and
commits.  An event failure rolls back the status update.  Concurrent attempts
on the same recommendation produce one winner and one deterministic conflict;
reviews of unrelated recommendations do not share a global lock.

Scores are optional evidence, not policy.  Step 5 never selects the latest or
highest score implicitly, never applies a threshold, and never modifies score
runs, results, components, ranks, or explanations.

Reason and reviewer contract
----------------------------

Reject reason codes are ``operator_rejected``,
``duplicate_research_direction``, ``unsupported_research_priority``,
``insufficient_evidence``, ``excessive_scope_change``,
``superseded_by_manual_plan``, and ``other``.  Expiration codes are
``operator_expired``, ``stale_recommendation``, ``policy_obsolete``,
``source_evidence_obsolete``, ``research_window_closed``, and ``other``.
Reject and expire require nonempty reason text.  Approval records
``operator_approved`` and may carry an optional note or reason text.

Reviewer text is optional authoritative operator input.  It is stored
separately, is never guessed, and is not treated as authorization.  CLI and
database access controls remain external operational controls; Step 5 does not
invent an access-control or signature system.

Commands
--------

Review one recommendation::

  LSTM_Release --approve-experiment-recommendation=42 \
    --recommendation-reviewer=operator \
    --recommendation-review-score-id=91

  LSTM_Release --reject-experiment-recommendation=42 \
    --recommendation-review-reason-code=insufficient_evidence \
    --recommendation-review-reason="Source evidence is incomplete."

  LSTM_Release --expire-experiment-recommendation=42 \
    --recommendation-review-reason-code=research_window_closed \
    --recommendation-review-reason="The research window has closed."

Inspect immutable review history::

  LSTM_Release --list-experiment-recommendation-reviews \
    --recommendation-review-action=approve --recommendation-review-limit=100
  LSTM_Release --recommendation-review-status=17
  LSTM_Release --recommendation-review-history=42

Review mutation commands are mutually exclusive with recommendation
generation/scoring, scheduler control, continuation control, and experiment
lifecycle commands.  Review options without a review action fail before any
database write.  Machine fields use the existing percent-escaping contract.
Human confirmation lines preserve ordinary spaces and punctuation while
rendering control bytes visibly, preventing injected lines or terminal escape
sequences.

Deferred behavior
-----------------

Recommendation-to-experiment conversion, queueing, scheduler-managed review,
automatic score-based disposition, execution authorization, and experiment
creation remain deliberately deferred.
