# Phase 23A3 run output

Status: **GO WITH PREREQUISITES**.

- Amended baseline: `ef51c80b4543d8c1b6a2b0bf7af404b12f44b838`; sibling-tree comparison with pre-amend Phase 23A2 `3b7134a9ca98496d0575d1025dc9d76674cde28d` proved documentation-only change.
- Release standalone worker: `35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b`.
- Old immutable layout-7 inference worker preserved: `Builds/SemanticWorkers/layout7/infer/bdb4d905b5badedfcaa3706bed9a00ec03762800/11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941/lstm-infer-worker`, SHA `11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941`.
- New immutable worker: `Builds/SemanticWorkers/layout7/infer/ef51c80b4543d8c1b6a2b0bf7af404b12f44b838/35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b/lstm-infer-worker`, same SHA as the validated build.
- Registry cutover: `23f8109d…d96d92` -> `78ac5c4e…745c69`; resolver passed against the operational registry.
- Fresh Phase 23A2 A/B passed, exact digest `60f42065747e58843ffc11763fd5c7da7c1ce53820c5735b7344f132bc6ef1f8` on all four paths.
- 648/649 stay `failed:infer`, models 1914/1916; 1129/1130 stay historical failed signal-5 attempts; no new recovery attempts or result/profitability rows.
- 650/651 stay active train attempts 1131/1132.  No unrelated work was dispatched.  Rollback was not needed.  This run left no new disposable DB/process; two pre-existing unowned `ea_phase23a2*_probe_15701` databases were preserved.

Prerequisite: an approved, experiment-scoped scheduler handoff/reload that proves resolution to the new artifact and admits only 648/649, without releasing broad inference/analyze capacity.
