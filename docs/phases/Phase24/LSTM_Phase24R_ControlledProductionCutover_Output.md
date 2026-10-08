# Phase 24R — Controlled production dedicated-worker cutover

**Stage A and authorized initial Stage B: PASS at the recorded acceptance snapshot.** Experiment 713 reconciled to `pending/analyze` without duplicate inference. Experiment 714 completion remains **PENDING**; subsequent operator observations report its existing worker suspended and experiment 747 resumed (section 11). Initial acceptance evidence below is historical, not a current production-state assertion. No concurrency expansion is authorized.

## 1. Baselines and committed qualification evidence

Date: 2026-10-08, America/Chicago. Read AGENTS.md in both repositories, Phase 24Q report and retained evidence, and relevant Phase 24N/24O/24P reports; consulted the established semantic-worker publication and earlier controlled-cutover documentation.

| Repository | Branch | Baseline HEAD / state |
| --- | --- | --- |
| Development `/Volumes/Developer SSD/ExpertAdvisor-Rollover` | `dedicated-train-layout-rollover-squashed-v1` | `6e685f9c9b9f43d2f16c5d483ffd0ab3d3da7d15`; only the requested untracked Phase 24Q report |
| Production `/Volumes/Developer SSD/ExpertAdvisor` | `lstm-feature-development` | `b8cdfef03ccdb073caccbf93b0a4282070c0c4d3`; clean initially and finally |

Both configure `gitlab` (`git@gitlab.com:vpredoehl/ExpertAdvisor.git`), `local` (`/Volumes/Developer/ExpertAdvisor`), `network` (`/Volumes/TC3T HD/ExpertAdvisor.git`) and `origin` (`git@github.com:vpredoehl/ExpertAdvisor.git`) for fetch/push. Actual HEADs differ; no synchronization, merge, fetch or push was attempted.

Reviewed the Phase 24Q report against retained experiment/model/attempt/inference/protocol/lease exports, native ablation/restoration probe, cleanup record and artifact hashes. Committed **only that report** as **`baf2e7bc368075f20c568f8f0fa01caf9892087c`**, subject `Phase 24Q: Record successful native operational qualification`. Its reviewed SHA-256 is `96a8dda9e0e25c1333db945b62ca7719cb3c31d7c91049e71c15ce2798c2779c`. No generated log, credential, database archive, PostgreSQL storage or DerivedData file was staged. Rollover was clean after this commit and before supported publication. This Phase 24R report remains uncommitted.

No Phase 24Q workload, completed regression suite, TRAIN/INFER build, Xcode application build, Clean, dependency change or shared MetaNN edit was performed. The nested `MetaNN/MetaNN -> /Volumes/Developer SSD/ExpertAdvisor/MetaNN` link and shared worktree remain unchanged.

## 2. Stage A read-only production inventory

Explicit effective destination: **PostgreSQL 17.11, TCP 127.0.0.1:5432, database LSTM**. Runtime principal `pqxx`; existing backup principal `vjp`. Production system identifier **7499293406780276383**. Inventory used explicit endpoints and read-only transactions; no credential was printed or copied into Git. Existing production server PID **3876**, start Sep 29 22:15:10, data directory `/Volumes/Forex Data/forexdb`, remained present.

No production scheduler or active TRAIN/ANALYZE worker was found. No active INFER remained other than the intentionally suspended historical process. Full-host scheduler censuses preceded backup/publication/protocol and final verification. The suspended worker does not carry the `--schedule-experiments` dispatch flag and was not classified as a scheduler by the unchanged supported cutover guard.

| Protected experiment | Observed state | Preservation |
| --- | --- | --- |
| **713**, model **2044**, attempt **1397** | `running/infer`, stale recorded PID 46580; process exited after successful persistence | No restart, signal, status rewrite or reconciliation by Stage A |
| **747**, model **2089**, attempt **1394** | `paused/infer`; PID/PGID **5916**, PPID 1, start **Oct 7 18:06:54**, **Ts** | Exact production executable/arguments/start identity unchanged; no signal/resume/termination |
| **732, 733, 748, 749** | `paused/train`, no worker PID | Remained paused; not resumed |

Baseline counts: completed/done **468**, cancelled/train **17**, failed/train **1**, paused/infer **1**, paused/train **4**, pending/analyze **1**, pending/infer **18**, running/infer **1**. Historical abandoned attempts were inventoried without interpreting their stored PIDs as live workers or repairing them. Connections were inventoried; an existing interactive `psql` connection remained unrelated to scheduling. No process was terminated or signaled.

The deployed scheduler remains at `/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/lstm-scheduler`, embedded source **b8cdfef03ccdb073caccbf93b0a4282070c0c4d3**, SHA-256 **1a676739cf5d4e27a604e2c199b1a46276204da32d9ef5585da35e52ea4e2550**. It was not replaced or rebuilt. Its last durable invocation was `scheduler:fecf9d5eee74af5d7475162de7c8746d903ea0d9248b96f2`, released on graceful shutdown. Persisted phase priority remains train:infer:analyze, revision 3; global-control revision 338 is unchanged.

## 3. Verified backups and rollback readiness

Evidence and backups: **`DerivedData/ExpertAdvisor/Phase24R/`**, private directory mode 0700. Backup location: **`DerivedData/ExpertAdvisor/Phase24R/backup/`**. Nothing here is committed or published as source.

Preserved byte-verified original registry, original `current` relative link target, every production worker/runtime publication manifest, historical executable/resource hashes and modes/link inventory, deployed scheduler executable, and persisted launch/protocol/lease/phase/global configuration. Original registry SHA-256: **af1121e80d42af13b3933628954348550214b790f85cc9823b71c156b26e8e79**.

The supported `LSTM_Release --backup-database --backup-output=<backup/LSTM-pre-cutover.dump>` command produced a full schema/data custom archive using the existing `vjp` backup principal. Size **695,096,386 bytes**, SHA-256 **df5b4004aa7f8ad102f90acb05e7b455d05aae9ace07bd3e11815d9e5753d393**. Its project-generated manifest records schema **099**, models **1810**, experiments **511**, analyses **970**. Archive listing and **full decoding** with `pg_restore --file=/dev/null` both passed.

An initial attempt under runtime `pqxx` correctly failed on restricted campaign tables. Preserved the incomplete archive and failure log; retried the same supported workflow with the existing verified backup owner. No permission grant, credential modification or production write occurred.

Restored the exact archive entries for nine public scheduler/experiment/inference tables and their two experiment check-constraint functions into an independent SCRAM-authenticated test cluster, private storage, port **55483**, role `phase24r_restore`, database `ea_phase24r_restore`. All rows matched production exactly: protocol **1**, lease **1**, invocations **214**, phase policy **1**, global control **1**, attempts **1397**, experiments **511**, inference results **967**, profitability observations **182**. Retained row digests in `database-restore-rehearsal.json`. Initial partial-restore setup errors (duplicate archived schema names and two omitted check-function dependencies) were corrected through exact TOC selection; no constraints were removed. The cluster, data and generated credentials were stopped/removed after verification.

This targeted restore proves content recovery for relevant state; it does not claim a full production ownership/ACL/foreign-key or sealed campaign-role recovery rehearsal. The complete archive remains available for a separately approved broader recovery. **No database rollback is needed for this Stage A**, because protocol was already complete and all nine production table contents remained identical after the operation.

Before production publication, ran the supported paired refresh on a separate copied publication tree, then restored its original registry using the existing publisher's `atomic_write_json` under `.publish.lock` and `update_current_link_after_registry_commit`. Registry bytes/link and original TRAIN/INFER routing were restored exactly. New immutable generations remained retained. Rehearsal receipt: `registry-rollback-rehearsal.json`.

Documented operator rollback: `rollback-procedure.md`. Keep scheduling stopped; verify no competing scheduler/writer and the expected live post-publication registry hash; acquire the existing publication lock; validate backed-up registry against retained artifacts; restore it atomically with the established helpers; restore the recorded convenience link; revalidate original hash and routing. Retain all immutable artifacts and failure evidence. Never manually rewrite database protocol/experiment rows. Unexpected database changes require a separately approved recovery plan, not a Git reversal or automatic whole-database overwrite.

## 4. Qualified publication and runtime verification

Both qualified candidates retain **ARM64 / minimum macOS 27.0 / SDK 27.0 / semantic layout 13 / width 171**. Fresh early `--build-identity` queries match the independent SHA/source/role values. Mach-O inspection and strict signature verification passed. No native TRAIN/INFER workload was launched.

| Role | Source commit | SHA-256 | Immutable capabilities |
| --- | --- | --- | --- |
| TRAIN | `12e31c0299193e4198907701182ccbaad0dcf67a` | **018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9** | `train,train_feature_ablation_v1` |
| INFER | `d99449d44f4473639c7fab1b831f058d6ac4386b` | **a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35** | `infer` |

TRAIN's ablation claim rests on completed Phase 24Q actual input suppression/native training/continuation evidence, plus earlier contract qualification. It was not added merely to force selection.

Reverified **18 worker/dependency/conditional-module path references** against Phase 24N: exact hashes/resolutions unchanged, strict signatures valid. Existing shared runtime **6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e** was reused without overwriting resources:

- `MetaNN_metal.metallib`: **9c894e9e02b3dfafb69d639535ebc064c7b5ef30ce72edd5cf628f220e16f759**.
- `default.metallib`: **a13694e6940e8287c1b3ca696edbcc291e85de2fd514ade52e431054f1d537d5**.

Supported production publication command, from the clean Rollover checkout:

```text
python3 Scripts/RefreshSemanticWorkerGeneration.py
  --repository-root=/Volumes/Developer SSD/ExpertAdvisor-Rollover
  --training-executable=<stable Rollover Release>/lstm-train-worker
  --inference-executable=<stable Rollover Release>/lstm-infer-worker
  --artifact-root=/Volumes/Developer SSD/ExpertAdvisor/Builds/SemanticWorkers
  --source-commit=12e31c0299193e4198907701182ccbaad0dcf67a
  --inference-source-commit=d99449d44f4473639c7fab1b831f058d6ac4386b
  --train-feature-ablation-qualified
```

Exact quoted argument array/environment is retained in `commands.jsonl`; the displayed lines are an argument listing, not a directly pasted shell script. The publisher verified identity, staged/fsynced immutable artifacts/manifests, validated registry integrity and atomically committed both current bindings under its existing lock. Exit **0**, no recovery required.

Published paths, relative to production `Builds/SemanticWorkers`:

```text
layout13/train/12e31c0299193e4198907701182ccbaad0dcf67a/018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9/lstm-train-worker
layout13/infer/d99449d44f4473639c7fab1b831f058d6ac4386b/a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35/lstm-infer-worker
```

Registry schema **5**, artifact manifests schema **2**, runtime manifest schema **1**, current layout **13**, worker records **18 -> 19**. New registry SHA-256: **bd29849c3a0d8676548fd495401e7865f899dd432f6c380865db760b665b63e5**. New dedicated TRAIN priority **1**; outgoing TRAIN retained as historical priority **0** with all original capabilities. Outgoing layout-13 INFER singleton binding was retired while its immutable artifact remains untouched, including 747's executable. Layouts **5–12** retain their existing routing. `current` now points to the new INFER directory; the registry remains authoritative.

The actual committed registry exactly matches the rehearsed prospective registry. All historical executable/manifest/resource hashes, modes and existing links remain unchanged except the intended registry and convenience-link changes. No historical artifact, shared Metal resource, deployed scheduler or analyzer was replaced. Production Git remains clean because these are ignored operational publication files.

## 5. Production protocol and authority

Actual production baseline was **generation 52, state complete**, completed July 31; this was queried directly, not assumed from Phase 24Q. Lease **released**, fence **198**, no competing scheduler authority.

Invoked the unchanged supported production-target CLI:

```text
<stable Rollover Release>/LSTM_Release --complete-scheduler-protocol-cutover --yes
SCHEDULER_PROTOCOL_CUTOVER_COMPLETE,generation=52,result=already_complete
```

Exit **0**. This supported transaction used explicit production endpoint/database settings and the existing process/coordination guards. It needs normal transaction settings for its `FOR UPDATE` locks even on the idempotent path; ordinary inventory/backup/comparison connections remained explicitly read-only. No SQL protocol edit, force flag, bypass or guard weakening was used.

Before/after protocol and lease rows are exactly identical: **52 -> 52**, **complete -> complete**, **released/fence 198 unchanged**. No scheduler authority was acquired or scheduler invocation created. Source review confirms AlreadyComplete returns before the protocol update. Relevant full table-content digests also match the backup/restore baseline exactly.

## 6. Routing verification without production dispatch

Compiled a read-only C++20 probe with **Apple Clang, -Wall -Wextra -Werror**, using the real `SemanticWorkerRegistry`, `TrainingWorkerSelection`, `InferenceWorkerSelection` and canonical mask parser. Inspected production/Rollover source equivalence for these components and scheduler authority/daemon. No worker/scheduler binary was rebuilt. An initial sandbox temporary-file failure was corrected with a local writable TMPDIR; retained wrapper diagnostics are not represented as an application build.

The probe loads and validates the **actual production registry/artifact/manifests/runtime bytes**. **28 checks passed before publication and 28 afterward**, without executing any worker or opening a database connection:

| Case | Post-publication result |
| --- | --- |
| Persisted layout 13 / width 171 TRAIN, empty mask | Exact qualified dedicated TRAIN hash/path |
| Layout 13 / 171 canonical `relative_tick_volume` TRAIN | Same ablation-capable dedicated TRAIN |
| Layout 13 / 171 INFER | Exact qualified dedicated INFER hash/path |
| Historical INFER 5–12 and TRAIN/ablation 7–12 | Identical selected path/hash to baseline |
| Unsupported layout 999 TRAIN/INFER | Rejected |
| Incompatible layout 13 / width 103 TRAIN/INFER | Rejected |
| Required absent capability | Rejected |

Registry loader and publisher validate all immutable capability/manifest claims and canonical runtime paths. Missing capability is tested through the real selector's required-capability input, without editing production claims. Retained routing outputs: `baseline-routing-results.json`, `published-routing-results.json` and prospective/rollback logs. No experiment was queued, dispatched, resumed or reconciled.

## 7. Stage A experiment 713 reconciliation review and paused-worker preservation

Read-only production evidence independently confirms inference result **1025**, model **2044**, producer attempt **1397**, final/completed at **13:03:57**, plus its persisted profitability observation. The user-supplied success markers agree with those durable rows. At the Stage A checkpoint, experiment 713 remained **running/infer**, active attempt 1397 and stale recorded PID, pending ordinary scheduler reconciliation. Stage B later recovered that completion as recorded below.

Reviewed the preserved production scheduler's missing-process/exact-attempt reconciliation and `findAuthoritativeFinalInferenceResultForWorkerAttempt` repository contract. Ran **that exact SELECT predicate** read-only with experiment 713/attempt 1397; it returns **result 1025, model 2044, forced_rerun=false**. Its native missing-process path transitions authoritative completed inference to ANALYZE instead of relaunching INFER. No actual reconciliation was run in Stage A. Stage B must revalidate these facts immediately before activation and stop if current evidence would cause rerunning already-persisted inference.

All nine table-content comparisons prove Stage A did not modify experiment 713, other experiments, attempts or results. Historical **747 remains Ts**, same PID/group/start/executable/arguments; its immutable artifact and old manifest are retained despite current-layout singleton refresh. Other intentionally paused experiments remain unchanged. No signals were sent.

## 8. Stage A disposition and mandatory human checkpoint

**Stage A completed with production activation not yet started.** The user subsequently explicitly approved bounded Stage B with TRAIN 0 / INFER 1 / ANALYZE 0 and no automatic continuation queueing. That approval does not extend to workload expansion.

Prepared activation constraints: use the preserved supported production scheduler/configuration with **TRAIN 0, INFER 1, ANALYZE 0**, preserve all operator pauses, and suppress the old automatic continuation-queueing flags because new queues/expansion are not authorized. The old persisted launch arguments included TRAIN 2, INFER 1, ANALYZE 18 and auto-evaluate/auto-queue continuations; they are retained as configuration evidence and must not be replayed unchanged for the conservative activation. No production launch configuration was changed during Stage A. A concrete, unexecuted conservative argument proposal is retained in `proposed-stage-b-command.json`; automatic continuation options are omitted, matching their supported false defaults.

The approved Stage B prerequisites were: recheck no scheduler/competing authority, the exact deployed scheduler and new registry hashes, 747 identity/paused state, and 713 authoritative completion predicate; start only one scheduler; verify owner/fence, reconciliation-before-dispatch, correct dedicated/historical routing, no duplicate launch and expected lifecycle changes. ANALYZE activation, higher INFER capacity, paused TRAIN resumption and automatic queue expansion each require separate explicit authorization.

Remaining limits: publication routing is proven, but live post-cutover scheduler dispatch/reconciliation belongs to Stage B; short synthetic Phase 24Q qualification does not prove long-run/model-quality/peak concurrent Qwen behavior. Existing CLI/analyzer diagnostic debt remains documented. Backups are local and retained under ignored DerivedData; no remote backup was pushed. No database downgrade or full-database restore should be inferred from the registry rollback procedure.

## 9. Review output and retained evidence

Application behavioral/source changes: **none**. Operational change: current layout-13 TRAIN/INFER registry bindings now select the exact qualified dedicated workers; historical TRAIN routing and immutable generations/resources are preserved. Protocol/configuration/data unchanged. Only authorized Phase 24Q report commit occurred; no production source commit, merge or push.

Evidence root contains exact command arrays/environments/exits, baseline repository/remotes/process/connection inventory, publication/rollback rehearsal, backup archive/manifest/checksum/restore receipts, candidate Mach-O/identity/signature and dependency checks, original/final registry and routing results, protocol comparisons, nine table-content digests, suspended-worker preservation and final Stage A assertions. Private restore storage/credentials were removed; backups and evidence remain. No completed Phase 24Q regression suites were repeated.

Final development status:

```text
git status --short
?? docs/phases/Phase24/LSTM_Phase24R_ControlledProductionCutover_Output.md

git diff --stat
(empty: Phase 24R report is untracked; Phase 24Q report is committed)
```

Production `git status --short` and `git diff --stat`: empty. Shared MetaNN status: unchanged/clean. The Stage A human checkpoint was honored. Subsequent explicit Stage B approval and verified activation are recorded below; no workload expansion is authorized.


## 10. Explicitly authorized Stage B — initial activation acceptance

The user reviewed Stage A and explicitly approved production scheduler activation with **TRAIN 0 / INFER 1 / ANALYZE 0**, automatic continuation queueing disabled, and preservation of operator-paused experiments **732, 733, 747, 748, 749**. A later observation-limit instruction expressly accepted initial activation checks and required ending Codex observation without waiting for experiment 714 to complete. This report follows that instruction.

| Acceptance item | Actual result |
| --- | --- |
| **Scheduler activation** | **PASS at initial acceptance** — one active supported production daemon, PID **58062**, authority fence **200** |
| **Worker routing** | **PASS** — actual model-2048 layout-9/103 INFER selection used its immutable historical worker; current layout-13 dedicated selection validated read-only |
| **Concurrency enforcement** | **PASS** — native census and authoritative capacity count show **one active INFER**, **zero TRAIN**, **zero ANALYZE**; later candidates deferred on full capacity |
| **Experiment 713 reconciliation** | **PASS** — durable result recovered; **pending/analyze**, no new inference attempt, duplicate result or profitability row |
| **Experiment 714 inference completion** | **PENDING** — original worker **57354**, attempt **1398**, running normally; completion is not required for Stage B acceptance |
| **Operator pauses** | **PASS** — all five full experiment rows exactly equal the pre-activation snapshots; 747 remains suspended/untouched |
| **Automatic continuation queueing** | **Disabled** — runtime startup log confirms `auto_queue_continuations=0`; automatic evaluation also remains 0 |
| **Observation limit** | **Honored** — observation ended; no wait for inference completion, no scheduler/worker stop at final verification |

### Pre-activation prerequisites

All five checks passed before initial activation:

1. Full process census proved no production scheduler, while 747's exact suspended identity remained present.
2. Production registry SHA **bd29849c3a0d8676548fd495401e7865f899dd432f6c380865db760b665b63e5** matched Stage A. The established registry/manifests/runtime verifier and native selectors accepted the exact published artifacts; control/ablation TRAIN and INFER selected their qualified dedicated hashes.
3. Production protocol remained **52/complete**, authority released before acquisition.
4. Experiment 713's actual authoritative completion predicate returned **result 1025/model 2044/forced_rerun=false**, proving the supported missing-process recovery path would advance to ANALYZE rather than duplicate INFER.
5. The unchanged production scheduler binary, SHA **1a676739cf5d4e27a604e2c199b1a46276204da32d9ef5585da35e52ea4e2550**, successfully parsed the complete proposed argument list with early `--help`. No work or authority was started by that check. Its embedded source remains production **b8cdfef03ccdb073caccbf93b0a4282070c0c4d3**.

Used the project's existing **standalone native daemon CLI**, running in the production working directory, with detached session/stdio for persistence. No additional authority system, daemon auto-retry service or source change was introduced. Arguments are preserved in `StageB/authorized-command.json`, actual invocation rows and process census:

```text
<production Release>/lstm-scheduler
  --schedule-experiments
  --phase-priority=train:infer:analyze
  --max-train-procs=0
  --max-infer-procs=1
  --max-analyze-procs=0
  --scheduler-poll-seconds=30
  --semantic-worker-registry=<production>/Builds/SemanticWorkers/registry.json
  --analyze-worker=<production Release>/lstm-analyze-worker
  --scheduler-verbose
```

Automatic continuation flags are omitted, using supported false defaults. Effective database settings explicitly target **127.0.0.1:5432/LSTM**, and market-data reads target **127.0.0.1:5432/forex**, using existing production runtime credentials. No credentials or grants were changed. Scheduler stdout/stderr is retained under Rollover DerivedData; supported worker logs remain in production `experiment_logs`. No TRAIN/ANALYZE job or new experiment was queued by this phase.

### Actual reconciliation, dispatch and ownership

The first scheduler invocation **57281**, started **13:58:42**, acquired fence **199** and logged:

```text
SCHEDULER_WORKER_RESULT_RECOVERED,experiment_id=713,model_id=2044,phase=infer,reason=completed_inference_result
SCHEDULER_PHASE_TRANSITION,experiment_id=713,from_phase=infer,to_phase=analyze
SCHEDULER_WORKER_RECONCILED,worker_attempt_id=1397,experiment_id=713,result=completed_evidence
```

Actual experiment 713 is now **pending/analyze**, worker PID/active-attempt binding cleared. Attempt **1397** is **completed**, reconciliation result `process_missing_result_recovered`; its exit_code remains **NULL**, not a fabricated observed zero. Original result **1025** and the original profitability observation remain byte-equivalent to their pre-activation records. ANALYZE capacity stays zero, so no analysis worker was dispatched.

Exactly one new worker attempt was created: **1398**, experiment **714**, model **2048**, layout **9**, width **103**, worker PID/PGID **57354**, native start **Oct 8 13:58:42**, kernel identity **1791485922:908892**. Its selected historical INFER source is **e964fa9e335e9ae63918187a7ffee7aa77f32b4b**, SHA **dc3c10fecc1990a9eb4194265322820ac2d95a384639603c9070bdaf9e6ede8d**. The immutable path matches the actual registry singleton and persisted attempt. Worker metadata/native logs show registration, detached model materialization and inference input preparation; it remains **running/infer** with completion pending. No new inference-result row had been committed at the final acceptance snapshot.

All 18 initially eligible INFER experiments were historical layout **9/103**. There was no eligible layout-13 production job. Live dispatch therefore proves historical routing; **dedicated layout-13 routing is PASS through actual production-registry selection/validation, not claimed as a live Stage B dispatch**. No new layout-13 experiment was created and no paused experiment was resumed merely to obtain live dedicated dispatch evidence. Phase 24Q's existing native dedicated execution evidence remains the qualification authority.

### Monitoring defect and controlled correction

The Codex observer initially tested the whole `ps` line for ` T`, incorrectly matching the weekday **Thu** as a suspended-state marker. This caused its initial verification timeout despite database/census evidence already showing correct operation. The observer requested **SIGTERM only to its owned scheduler PID 57281**; the supported handler logged graceful shutdown, exit **0**, and authority release at fence **199**. No worker or process group was signaled. This was an observation-harness defect, not a native scheduler/worker defect or an unexpected production dispatch. Original failure/receipts are retained.

Corrected the parser to inspect the actual state column and reconstructed the original 60 census observations from their retained records. They show at most **one active INFER** throughout; original flawed receipts were preserved separately. After verifying the first scheduler was absent, lease released, registry/binary unchanged, all paused rows unchanged, and 713 already safely recovered, restored the same approved one-scheduler configuration. The two invocations were **sequential, never concurrent**; this correction is disclosed rather than characterized as a single total process launch.

Current scheduler **58062**, PGID **58062**, start **Oct 8 14:02:20**, kernel identity **1791486140:124505**, invocation **scheduler:cb77733da82ab0079216995d519665f0ed8246c47e2e22a7**, holds active fence **200**. It observed/adopted the still-running **same** attempt **1398/PID 57354**, consumed one INFER slot, and did **not** launch a duplicate worker. The restored scheduler log contains no `SCHEDULER_CHILD_LAUNCHED`; the original log contains exactly one launch for 1398. Worker 714's original PID/group/start/executable identity remained unchanged across scheduler handoff.

At the user's observation limit, performed one final native identity census and read-only database snapshot. The observer had ended by the final observer-only check; no signal was sent to current scheduler **58062**, worker **57354**, or historical **5916**. Both production scheduler and experiment 714 worker were deliberately left running. No ongoing Codex observation loop remains.

### Preservation and scope at the initial acceptance snapshot

All five full operator-paused experiment rows equal the pre-activation snapshots exactly. Historical 747 remains **Ts**, PID/PGID **5916**, start **Oct 7 18:06:54**, same model **2089**, attempt **1394**, executable and arguments. Its attempt observation timestamps can be refreshed by ordinary scheduler observation; its experiment pause and native process state were preserved without signal or resume.

Protocol rows, registry hash, deployed scheduler hash, persisted phase/global policy and maximum experiment ID remain unchanged. Authority/invocation heartbeat and the supported 713/714 lifecycle/attempt transitions are the expected authorized production mutations. No automatic continuation queue, paused TRAIN resumption, ANALYZE activation or concurrency expansion occurred. No duplicate inference persistence was observed; experiment 714 has not yet completed, so its eventual result is not claimed here.

Retained Stage B evidence: `StageB/baseline.json`, native configuration-help/identity and read-only routing logs, `StageB/activation.json`, initial scheduler/transition logs, original and corrected census observations, graceful-shutdown receipt, `StageB/RestoredActivation/activation.json` and scheduler log, `StageB/user-directed-final-state.json`, `StageB/user-directed-final-identities.json`, `StageB/stage-b-acceptance.json`, observer-end receipt, and the initial worker log snapshot. Exact command/environment records remain in `commands.jsonl`. No generated evidence is committed.

**Initial Stage B recommendation: GO for the approved bounded configuration at that acceptance snapshot. Inference completion for 714 remains PENDING; see subsequent observations below.** Leave production scheduler and its INFER worker running. Stop Codex work here; no workload expansion, pause resumption, merge or push is authorized.


## 11. Subsequent operator observations recorded during Phase 24S

These observations were supplied by the operator after the initial Stage B acceptance window. Phase 24S performed no live production inspection or mutation and does not independently establish a later completion time or current process state.

| Experiment / process | Subsequent reported observation | Completion evidence |
| --- | --- | --- |
| **713**, model **2044**, attempt **1397** | Inference transaction committed, worker exited; scheduler reconciled to **pending/analyze** without rerun | **Completed inference / reconciliation PASS**, independently evidenced in section 10; ANALYZE completion is not claimed |
| **714**, model **2048**, attempt **1398**, historical layout **9**, PID **57354** | Later **pending/infer**, existing PID retained, OS **Ts** (suspended) | **Inference completion PENDING / unverified**; no final result is claimed |
| **747**, model **2089**, attempt **1394**, historical layout **13**, PID **5916** | Operator manually resumed its pre-existing worker; later **running/infer**, OS **Rs** | **Inference completion unverified**; no final result is claimed |
| Production scheduler **58062** | Reported still running | No new activation, signal or configuration change by Phase 24S |
| **715** | Reported pending | Dispatch/completion not asserted |

Initial preservation of all five operator pauses was verified only for the original acceptance window. The subsequent operator-authorized resumption of 747 changes that observation; Phase 24S did not resume it or any paused experiment. No later status is invented for 732, 733, 748 or 749.

These facts establish displacement-shaped states, not the component that issued a stop signal, exact scheduling priorities, FIFO behavior, immediate recovery, starvation, duplicate launch or a scheduler defect. Phase 24S reproduces supported priority displacement and delayed same-attempt recovery privately and records separate adversarial limits in `LSTM_Phase24S_Scheduler_Displacement_Recovery_Output.md`. The initial Stage B PASS does not qualify a future TRAIN concurrency expansion or an unconditional cap against external SIGCONT.


Development archival status at Phase 24T entry: this completed historical report and the preserved Phase 24S tests are included in the independent Phase 24S baseline commit. Statements above about uncommitted state describe their original phase closeout. Retained private evidence was reviewed; no later production state was inspected or asserted.
