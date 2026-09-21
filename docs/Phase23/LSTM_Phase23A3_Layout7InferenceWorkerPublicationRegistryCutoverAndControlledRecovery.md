---
title: "Phase 23A3 — Corrected Layout-7 Inference Worker Publication, Registry Cutover, and Controlled Production Recovery"
status: "GO WITH PREREQUISITES"
---

# Phase 23A3 — Layout-7 Inference Worker Publication, Registry Cutover, and Controlled Recovery

## Outcome

**GO WITH PREREQUISITES.** The corrected standalone layout-7 inference worker was built, fully regression-validated, published as a new immutable artifact, and atomically made the sole current `layout=7, role=infer` registry binding.  Production recovery was intentionally not dispatched: the only live production scheduler has a startup-time registry snapshot of the former artifact and `max-infer-procs=0`; it also has two unrelated active train workers.  The repository exposes no supported experiment-scoped live registry reload or inference-capacity adjustment.  Requeueing 648/649 under those conditions would either not dispatch or violate the exact-new-worker selection gate.

No production inference/analyze capacity was enabled, no scheduler was restarted or added, and no active training was disturbed.

## Gate 0 — amended provenance

The validated pre-amend Phase 23A2 commit was `3b7134a9ca98496d0575d1025dc9d76674cde28d`.  The current amended baseline is `ef51c80b4543d8c1b6a2b0bf7af404b12f44b838` (`ef51c80 Validate deterministic managed inference A/B`), on `lstm-feature-development`.

Both commits have the same parent, `1e5228ce4d59c721e7269d67ac6209ccdd279179`; therefore they are sibling replacement commits, not ancestor/descendant commits.  Tree-to-tree comparison established that the only replacement delta was:

- addition of `LSTM_Phase23A2_DeterministicManagedInferenceEndToEndFixtureAndABValidation_Continuation_Output.md`;
- a 100% rename of the Phase 23A2 `_Output.md` report into `docs/Phase23/`.

No executable source, build setting, migration, test, publisher, registry, or runtime file changed.  Phase 23A2 behavioral evidence was carried forward, then freshly repeated at the amended HEAD under Gate 2.

The worktree was clean before publication.

## Gate 1 — pre-cutover production and operational state

Pre-cutover operational registry:

- path: `Builds/SemanticWorkers/registry.json`
- SHA-256: `23f8109ddba8cc3e0a4346b8c42d599dcceacd83587f404214368b3913d96d92`
- current layout-7 inference artifact: `Builds/SemanticWorkers/layout7/infer/bdb4d905b5badedfcaa3706bed9a00ec03762800/11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941/lstm-infer-worker`
- former inference artifact SHA-256: `11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941`
- former provenance: role `infer`, layout `7`, width `77`, source commit `bdb4d905b5badedfcaa3706bed9a00ec03762800`, manifest schema `2`, capability `[infer]`.

The training/reference layout-7 artifact was separately preserved at SHA-256 `ac5023fa912b93c34679e984c4fb5d76bc6d83f6d44c84d5429159da210f3a60`.

Immediately before the recovery decision:

| Item | Durable state |
| --- | --- |
| 648 | `failed:infer`, model 1914, no active attempt |
| 649 | `failed:infer`, model 1916, no active attempt |
| 1129 | 648 historical infer attempt, `failed`, exit `-5`, signal `5`, former immutable worker |
| 1130 | 649 historical infer attempt, `failed`, exit `-5`, signal `5`, former immutable worker |
| 650 | `running:train`, observed attempt 1131 |
| 651 | `running:train`, observed attempt 1132 |

The sole live scheduler was PID 96205, held the active authority lease, had `train=2, infer=0, analyze=0`, and was observing only train attempts 1131/1132.  Its command and the routing documentation establish that it loads the semantic-worker registry once at scheduler startup.  It must not be silently replaced, restarted, or supplemented with a second scheduler in this phase.

## Gate 2 — Release builds and regressions

All succeeded using the stable requested layout `DerivedData/ExpertAdvisor/Build/Products/Release`:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Infer Worker" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Scheduler Bundle" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

Build identities:

| Product | SHA-256 |
| --- | --- |
| `LSTM_Release` | `263f586791306c1c3e00ee14f036d62f2f3f9a7cf942237edf02721599667310` |
| `lstm-infer-worker` | `35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b` |
| `lstm-scheduler` | `3f6b020323bc5de1a9d2ff368aa868866c30483476acb3496c1826c5009a2734` |

`lstm-infer-worker --build-identity` reported artifact role `lstm-infer-worker`, source commit `ef51c80b4543d8c1b6a2b0bf7af404b12f44b838`, and the exact worker SHA above.  The Release runtime package included both `default.metallib` and `MetaNN_metal.metallib`.

Passed focused regressions:

- `Tests/LSTMPhase23A1StandaloneManagedInferenceSIGTRAPTests.sh`
- `Tests/SemanticWorkerPublisherTests.sh`
- `Tests/SemanticWorkerRegistryTests.sh`
- `Tests/SchedulerSemanticAdmissionTests.sh`
- `Tests/SchedulerInferenceOrphanResultRecoveryTests.sh DerivedData/ExpertAdvisor/Build/Products/Release/lstm-scheduler Builds/SemanticWorkers/registry.json`
- `Tests/LSTMPhase23A2ManagedInferenceEndToEndABTests.sh DerivedData/ExpertAdvisor/Build/Products/Release/lstm-scheduler DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker`

The fresh Phase 23A2 A/B ran two compatibility and two standalone paths through scheduler-created attempts, detached materialization, durable result and profitability persistence, and orphan reconciliation.  Every path produced the exact normalized durable semantic digest:

```text
60f42065747e58843ffc11763fd5c7da7c1ce53820c5735b7344f132bc6ef1f8
```

There was no standalone SIGTRAP.

## Gates 3–4 — immutable publication, registry cutover, and resolver verification

The supported and sole publisher was used:

```text
/usr/bin/python3 Scripts/PublishSemanticWorker.py \
  --repository-root /Volumes/Developer\ SSD/ExpertAdvisor \
  --built-executable /Volumes/Developer\ SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker \
  --worker-rule current
```

Published immutable artifact:

```text
Builds/SemanticWorkers/layout7/infer/ef51c80b4543d8c1b6a2b0bf7af404b12f44b838/35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b/lstm-infer-worker
```

- SHA-256 before publication, after publication, and in the manifest: `35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b`.
- manifest: immutable storage, schema `2`, role `infer`, layout `7`, width `77`, capability `[infer]`, provenance commit `ef51c80b4543d8c1b6a2b0bf7af404b12f44b838`.
- executable mode is `0555`; manifest mode is `0444`; required runtime links resolve to the validated content-addressed runtime package.

Post-cutover registry SHA-256 is `78ac5c4e906c43555f6d36c21ff3fe7b4f76e7952550c051efc4654340745c69`.  Its only changed mapping is `layout=7, role=infer`, now current and pointing to the new artifact.  The layout-7 train/reference entry remains at its prior path/SHA, layout-6 remains historical, and the former layout-7 inference artifact still exists with unchanged SHA `11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941`.

Independent resolver verification ran `Tests/SemanticWorkerRegistryTests.sh` with `EA_SEMANTIC_REGISTRY_UNDER_TEST` set to the operational registry.  It loaded the operational registry, selected the distinct current layout-7 inference artifact for a 77-wide layout-7 request, selected the separate train/reference artifact for training, and validated runtime resolution successfully.

No rollback was needed or exercised.

## Gates 5–8 — controlled recovery boundary

The supported lifecycle interface was independently dry-run only:

```text
DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release --requeue-inference=648 --dry-run
DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release --requeue-inference=649 --dry-run
```

Both commands reported the authorized transition `failed:infer -> pending:infer`; durable production state remained unchanged.

Applying either command is blocked for this phase because no supported mechanism can guarantee that the live, old-snapshot scheduler will select the new artifact while admitting only 648/649.  Raising its broad inference limit would risk unrelated dispatch, and a second scheduler or a restart would violate the operational constraints.  Consequently:

- no new attempts were created;
- 648/649 retain their pre-recovery durable state;
- 1129/1130 remain untouched historical failures;
- no inference result or profitability row was created for this phase;
- 650/651 remain active train-only work on attempts 1131/1132;
- no unrelated experiment was requeued or dispatched;
- the Phase 23A2 A/B run created no remaining database or process after its own cleanup.  Two pre-existing, unowned databases named `ea_phase23a2_probe_15701` and `ea_phase23a2_forex_probe_15701` were observed during the final audit and deliberately left untouched; this phase did not create, use, or delete them.

## Required prerequisite and next normal-production decision

Before 648/649 can be recovered, provide an approved operational handoff that guarantees all of the following atomically or in a demonstrably fenced sequence:

1. the existing scheduler is at a safe handoff boundary without disturbing attempts 1131/1132;
2. its registry snapshot is replaced by a freshly started/explicitly reloaded snapshot that resolves the published `35bdf9…c706b` artifact;
3. only experiments 648 and 649 are admitted for inference, with no unrelated pending inference/analyze release; and
4. inference capacity is returned to zero immediately after their two attempts complete.

At that time, apply the two supported `--requeue-inference=ID --yes` commands, capture their new attempt IDs and persisted worker paths, and require each to resolve the new immutable artifact SHA before allowing completion.  Only after both commits durable result plus profitability evidence should a separate decision consider normal production inference/analyze enablement.
