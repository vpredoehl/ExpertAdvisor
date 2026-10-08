# Release floating-point validation contract

`ModelInputPreparation`, `ProfitabilityCore`, `StrategyEvaluationCore`, and
`SchedulerCore` explicitly compile their Release code with
`GCC_OPTIMIZATION_LEVEL = 3` (`-O3`). These targets validate non-finite values,
canonicalize persisted numeric identities, or construct features used by
both dedicated training and inference workers. They must not inherit the
project's `fast` optimization setting: `-Ofast` permits the compiler to discard
required finite-value checks and changes the contract of infinity sentinels.
Replacing its spelling with `-O3 -ffast-math` does not restore that contract.

Keep the existing `std::isfinite` checks and each boundary's existing failure
behavior. Invalid configuration and canonicalization inputs throw or return a
configuration error; invalid profitability price pairs are excluded from
its actionable observations. Normal market data is not a substitute for
validating these boundaries. Do not add finite-only assumptions or fast-math
flags to these targets without independently establishing equivalent behavior.

This policy retains Release optimization. It changes no feature formulas,
registered semantic layouts, model input widths, deployment floors, transaction
boundaries, or worker composition. Dedicated TRAIN and INFER consume the same
`ModelInputPreparation` library. The legacy monolithic target and other targets
retain their existing settings; this document does not qualify their numerical
behavior.

Run `bash Tests/ReleaseFiniteValueValidationTests.sh` to check the target policy,
exercise runtime NaN and both infinity inputs under `-O3 -DNDEBUG`, verify finite
acceptance, reproduce the original `-Ofast` failure as a negative control, and
run the existing byte-for-byte training/inference feature-vector parity oracle
under optimized and unoptimized conforming compilation. The test asserts layout
13 and width 171. It uses only standalone, database-free CPU tests, with
disposable files under Rollover DerivedData; it does not invoke a worker.

A successful test or worker identity qualification does not approve production
deployment. The existing INFER macOS 26.2 floor must also be reconciled with its
required runtime library closure before publication/deployment.
