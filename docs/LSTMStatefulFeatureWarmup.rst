LSTM Stateful Feature Warmup
============================

Semantic provenance
-------------------

Feature query scope is a persisted scientific configuration.  Migration 067
stores ``feature_warmup_scope`` on every experiment: historical rows are
``legacy_cold_boundary`` and new work is ``full_history_warmup``.  The same
value is saved in model metadata and in recommendation semantic configuration
v5.  Older recommendation semantic v3/v4 canonical forms are interpreted as
``legacy_cold_boundary``; unknown v5 values fail closed.

The LSTM runtime retrieves ``candlestick`` source rows from ``-infinity``
through the requested end date before constructing ``Tensor``.  Earlier rows
warm feature state only; they are not training or inference output rows.

The logical output start is the count of those same source rows whose ``dt`` is
strictly before the requested ``fromDate``.  Training, single-model inference,
and ``--infer-all`` begin batching at that index.  This preserves the requested
date contract while giving the first emitted row the same EMA, ATR, rolling,
and return history it has in a wider query.

Exact recursive EMA and ATR values require all predecessor observations
available from the authoritative ``candlestick`` source.  The runtime therefore
does not use a finite, undocumented warmup-bar constant.
