LSTM append-only input-width expansion
======================================

Purpose
-------

Historical-width loading remains the default.  An ordinary
``--resume-model-id`` constructs the model at the persisted width and projects
today's Tensor row to the historical feature prefix.  Inference and ordinary
resume therefore retain their existing behavior.

Input-width expansion is a separate, explicit continuation experiment.  Queue
it with both ``--resume-model-id`` and ``--resume-expand-input-width``.  The
continuation is constructed at the current model input width and can learn
features appended after the source checkpoint was trained.

Safety contract
---------------

The persisted ``model_meta`` matrix is the authoritative source of input width
and hidden size.  Expansion accepts only widths registered by
``ModelInputContract.hpp`` whose Tensor feature layout is a strict historical
prefix of the current layout.  It rejects unknown, wider, non-current-target,
semantically incompatible, missing, or malformed source state.

Newly saved models also contain a 1x2 ``model_input_semantics_meta`` matrix:
``[metadata_schema_version, semantic_layout_version]``.  A semantic-layout
version identifies the complete registered feature layout at save time.  Each
append-only feature addition must retain the old registry entry, increment the
current version, and add a successor entry linked to the prior version with
its new maximum input width.  Expansion accepts a marker only when the
persisted version and input width are valid for an ancestor of the current
layout.  Unknown versions, widths introduced after the persisted version, and
versions outside the current append-only ancestry fail closed.  Marker-less
historical models retain the pre-marker known-width compatibility rule.

The fused LSTM parameter matrix is row-major ``(n_in + H) x (4H)`` with gate
columns ``[input | forget | candidate | output]``.  Its input rows consist of a
historical Tensor prefix followed by four multi-horizon return features.  To
expand it, the loader:

#. copies the historical Tensor rows without changing their indices;
#. inserts exactly zero rows for newly appended Tensor features;
#. relocates the four return-feature rows after the widened Tensor prefix; and
#. relocates the recurrent ``H`` rows after the widened input block.

Biases and both output heads are loaded unchanged.  SGD has no persisted moment
buffers; its update count and completed epoch count are restored normally.
Zero initialization preserves the source prediction before retraining even
when new feature values are nonzero.  It is trainable because backpropagation
forms each input-row gradient as ``x_new^T * d_gates``; that gradient does not
depend on the current input-weight value.

Persistence and retries
-----------------------

The source model is never updated.  Checkpoints and the final model are new
rows linked through the existing ``model.parent_model_id`` and
``model.experiment_id`` lineage.  Expanded descendants persist current-width
``model_meta``, the SGD update/epoch state, a model-input semantic layout
marker, and immutable expansion metadata containing:

* the original expansion source model id and input width;
* the expanded width;
* the newly introduced Tensor column interval and semantic names; and
* the ``zero`` initialization policy; and
* the semantic-layout generation at the time of that expansion event.

The canonical ASCII value is ordered exactly as follows (without whitespace):
``schema=1;source_model_id=...;source_input_width=...;expanded_input_width=...;new_tensor_columns=BEGIN:END;new_tensor_features=NAME|...;initialization=zero;semantic_layout=V``.
The column interval is half-open.  Parsing validates the event against the
registered layout generation recorded in ``V`` and then proves that generation
is in the current append-only ancestry.  It does not rewrite or require ``V``
to equal today's generation.  Unknown generations, a generation paired with
the wrong expanded width, incompatible ancestry, noncanonical text, incorrect
feature names or columns, and broken source/parent lineage all fail closed.

This permits a width-52 model expanded under V1 to be the source of a later
width-53 expansion under V2.  The width-52 model retains its V1 event; the new
width-53 descendant records a new V2 event whose source is the width-52 model.
The durable parent chain preserves the earlier history.

The experiment row persists ``resume_expand_input_width=true``.  Scheduler
retries forward the same opt-in.  A retry from an already widened checkpoint is
accepted only when that checkpoint contains valid expansion provenance, its
recorded event generation is a compatible ancestor, its model/source widths
agree with ``model_meta``, it belongs to the retrying experiment, and its
parent chain reaches the recorded source.
Ordinary resumes from a widened model behave as normal current-width resumes.

Feature ablation
----------------

Expansion never shrinks the model.  An ablated newly available feature remains
present at the current width and receives the normal zero ablation value.
Source ablations cannot be removed, and expansion may add ablations only for
features that were absent from the historical source width.

Scientific interpretation
-------------------------

Fresh ablation experiments ask whether a feature helps when training starts
from a new initialization.  Expanded continuation experiments ask whether an
appended feature improves a mature model while retaining its learned state.
They are distinct experiment classes and must not be compared as if their
initialization histories were equivalent.  Recommendation source loading marks
expanded continuations ineligible with
``input_width_expansion_experiment_class_excluded``; no recommendation schema
or evaluation-idempotency contract is changed.

Queue example
-------------

After applying migration ``071_resume_input_width_expansion.sql``, queue a
historical model for an absolute final epoch with:

::

   DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
     --queue-experiment \
     --resume-model-id=MODEL_ID \
     --resume-expand-input-width \
     --target-epochs=240

To ablate a newly appended feature while retaining current width, additionally
pass, for example, ``--ablate-features=historical_level_proximity``.
