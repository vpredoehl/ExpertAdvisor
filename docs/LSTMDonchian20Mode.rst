Donchian-20 Controlled Experiment Mode
======================================

The LSTM experiment configuration has one durable Donchian-20 mode.  The
supported values are ``enabled`` and ``zero_ablation``.  The default for new
experiments and legacy rows is ``enabled``.

Both modes retain the 34 base feature columns and the 38-column effective LSTM
input.  ``enabled`` writes the causal Donchian-20 values into columns 32 and
33.  ``zero_ablation`` writes exactly zero to those same columns.

New direct or scheduler-launched experiments may specify the mode with::

  --donchian20-mode=enabled
  --donchian20-mode=zero_ablation

The scheduler persists the mode on ``experiment.donchian20_mode`` and passes
it to training, final inference, and checkpoint inference.  Model metadata
persists the mode for every checkpoint and final model.  A model-mode mismatch
against an explicit runtime or experiment mode is rejected; feature width
alone is not sufficient for compatibility.

Recommendation conversion and campaign materialization preserve the mode in
their canonical experiment invocation.  Older invocation canonicals without
the field are interpreted as ``enabled`` for backward compatibility.

Campaign planning may explicitly request one or both matched arms with::

  --campaign-donchian20-arms=enabled
  --campaign-donchian20-arms=zero_ablation
  --campaign-donchian20-arms=enabled:zero_ablation

The default is to preserve the existing single-mode campaign behavior; an
unspecified arm option does not expand a campaign.  A paired request expands
each selected scientific candidate into distinct campaign members and
distinct experiment identities.  The requested arm is included in campaign,
recommendation, conversion, and materialization canonical identities, and
experiment/model metadata continues to enforce provenance downstream.

The schema change is migration ``060_donchian20_mode.sql``.
