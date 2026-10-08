# Phase 24H — MetaNN Xcode Reference Repair

## Outcome

The MetaNN dependency-resolution fault is repaired with **one parent-project path change**:

`MetaNN/MetaNNXC.xcodeproj` → `MetaNN/MetaNN/MetaNNXC.xcodeproj`

The existing symlink is valid and was preserved. Xcode now loads the ExpertAdvisor, MetaNNXC, and MetalSwift projects; every dependency in the INFER closure has a resolved target GUID, including the shared target `MetaNN` producing `libMetaNN.a`, `MetaNN_metal`, and `MetalBuffer`. Effective Release output settings for the INFER target and all eight dependencies remain inside Rollover-specific DerivedData. No target was compiled, linked, or executed.

**GO for dependency resolution and isolated-output preparation. NO-GO for an immediate Release build in the current dirty worktree or for native execution without explicit authorization.** The project repair and this report must remain uncommitted as requested, so the existing clean-source Release provenance check still blocks a build. The repaired configuration is ready for a later expressly authorized native-validation phase once clean, authoritative source provenance is established without bypasses.

Only the reviewed Phase 24F/G reports were committed, as **`cec957b41082171e8894f36869f31c12e5424218`** — `Phase 24F/G: Document native validation preparation`. The Phase 24H implementation/report remain unstaged and uncommitted.

## 1. Initial dependency-resolution evidence

Initial HEAD was `817a674044beb59a9703aaa0960f6817183f9529`. Worktree status contained only the untracked Phase 24F and Phase 24G reports; there was no existing tracked diff.

The current filesystem arrangement was inspected without changing it:

```text
MetaNN/                              local directory
MetaNN/MetaNN                        valid symlink to /Volumes/Developer SSD/ExpertAdvisor/MetaNN
MetaNN/MetaNN/MetaNNXC.xcodeproj      accessible project
MetaNN/MetaNNXC.xcodeproj             absent
```

Phase 24G's retained Xcode PIF graph was copied into the Phase 24H evidence directory before editing. It loaded only the parent ExpertAdvisor project. Its INFER external dependency record was `{"name": "libMetaNN (from MetaNNXC.xcodeproj)"}` without a target GUID; no shared MetaNN/MetalBuffer target was loaded. The parent project bytes and SHA-256 snapshots of both shared dependency project files were also retained before editing.

Initial evidence is retained in `DerivedData/ExpertAdvisor/Phase24H/parent-project-before.pbxproj`, `resolved-build-graph-before.json`, and `shared-project-sha256-before.json`, alongside the original Phase 24G metadata logs.

## 2. Root cause and previous-build discrepancy

`ExpertAdvisor.xcodeproj/project.pbxproj:712` used a group-relative project path missing the second `MetaNN` component. Its enclosing main group (`084505502047962100A0B88F`) has no path prefix, and `PBXProject` has empty `projectDirPath`/`projectRoot`. Therefore Xcode expected the absent root-relative `MetaNN/MetaNNXC.xcodeproj`, not the accessible nested project.

The shared target/product IDs referenced by the parent all exist in the accessible MetaNN project. The external archive target is currently named `MetaNN`, with product `libMetaNN.a`; legacy `remoteInfo`/dependency labels such as `libMetaNN` are descriptive labels and do not invalidate the matching remote IDs. No target-ID repair or search-path change was needed.

The path-only correction caused Xcode to load both external projects and resolve the existing dependency GUIDs, with the same scheme/configuration and the same Rollover DerivedData root used in Phase 24G. This before/after evidence establishes a current path-resolution fault rather than a need to initialize MetaNN or modify shared sources. No cache was deleted and no clean operation was used.

The user's prior successful build is accepted as historical context. No retained prior Rollover product, build log, or `.xcactivitylog` was found in the inspected Rollover DerivedData tree; it contained Phase 24G metadata, not compiled products. Phase 24G had found no repository-local DerivedData before its inspection. The parent project history was inspected, and the current shared schemes/target IDs were compared. That evidence does **not** establish whether the earlier success used cached archives, another DerivedData root, a different scheme/configuration, or an earlier effective filesystem/project state. Those possibilities remain unconfirmed; none is asserted as the historical cause.

## 3–4. Files and reference-chain change

Implementation file modified:

- `ExpertAdvisor.xcodeproj/project.pbxproj:712`: changes only the path on existing `PBXFileReference` `08F4FCB32F11F5DF00260473`.

The complete reference chain remains intact:

| Link | Existing identity and verification |
| --- | --- |
| Project file reference | `08F4FCB32F11F5DF00260473`; path now points through the existing nested symlink; `sourceTree = <group>` retained. |
| `PBXProject.projectReferences` | Associates that ProjectRef with ProductGroup `08F4FCB42F11F5DF00260473` (`project.pbxproj:2065–2071`). |
| Archive target container proxy | `08A334CD2F5FCBAC00BE2DCE`, type 1, same container portal; remote target `08A331D42F5FCA0200BE2DCE` exists and is the shared `MetaNN` archive target (`:521–527`). |
| Archive target dependency | `08A334CE2F5FCBAC00BE2DCE` retains that targetProxy and is still an INFER dependency (`:1943–1964`, `:2899–2903`). |
| Archive product container proxy | `08A334D92F5FCBAC00BE2DCE`, type 2, same portal; remote product `08A331D52F5FCA0200BE2DCE` exists and is `libMetaNN.a` (`:542–548`). |
| Archive reference proxy | `08A334DA2F5FCBAC00BE2DCE` still refers through that product proxy, with path `libMetaNN.a` and `BUILT_PRODUCTS_DIR` (`:2108–2114`). |
| INFER framework membership | Build-file `0F8000153800000100AAA001` still uses that archive proxy (`:491`, `:1185–1201`). |
| Metal target/product proxies | `08A334CB2F5FCBAC00BE2DCE` targets `08A32F9F2F5FC20900BE2DCE` (`MetaNN_metal`); `08A334D72F5FCBAC00BE2DCE` targets product `08A32FA02F5FC20900BE2DCE` (`MetaNN_metal.metallib`). Existing dependency/reference proxies retained. |

The shared archive target depends on `MetaNN_metal`, which depends on the external MetalBuffer target `08A32A292F5D090F00BE2DCE` in `MetalSwift/MetalSwift.xcodeproj`. Xcode's loaded graph now confirms this entire chain.

INFER search paths are unchanged (`project.pbxproj:3795–3827`): source headers under `$(PROJECT_DIR)/Headers`, `Sources`, and `DERIVED_FILE_DIR`; recursive shared MetaNN headers under `$(PROJECT_DIR)/MetaNN/**`; Homebrew libpqxx/libpq/libomp include/library paths; `$(PROJECT_DIR)` and inherited library paths. Shared MetaNN/MetalSwift absolute header search paths were inspected but not modified. Source-path reading is separate from output placement; no production archive was copied in as a fallback.

A parsed before/after project comparison changed the single file-reference path in the baseline object and then required complete equality. It passed: all IDs, targets, dependency relationships, framework memberships, schemes, scripts, and other settings remain unchanged. Every parent MetaNN container proxy's remote ID was checked against the shared project with the correct target/product object type.

## 5. Non-compiling verification commands and results

| Executed check | Result |
| --- | --- |
| `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj` | Passed. |
| Parsed project comparison and proxy/remote-ID consistency checks | Passed; exactly one path changed. |
| `xcodebuild -list` for the parent project | Exit 0; now lists external `libMetaNN`, `MetaNN_metal`, and `MetalBuffer` schemes. |
| INFER `xcodebuild -showBuildSettings` | Exit 0; Rollover output paths. |
| Latest workspace PIF project/target closure validation | Passed; three projects, nine relevant targets, no unresolved dependency in the INFER closure. |
| Eight dependency-scheme `-showBuildSettings` inspections | All exit 0; relevant generated-output settings resolve under Rollover DerivedData; `DEPLOYMENT_LOCATION = NO`. |
| `bash Tests/ReleaseWorkerBuildConfigurationTests.sh` | Passed. |
| `bash Tests/DedicatedTrainingWorkerArchitectureTests.sh` | Passed. |
| `git diff --check` and documentation staged-diff check | Passed. |
| Shared project hashes, unchanged symlink, production/shared Git status | Passed; no observed shared-source modification. |

Exact parent metadata commands:

```bash
/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -list

/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -scheme 'LSTM Infer Worker' \
  -configuration Release \
  -derivedDataPath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor' \
  -showBuildSettings
```

The second command was also executed with each of these scheme names in place of `LSTM Infer Worker`: `libMetaNN`, `MetaNN_metal`, `MetalBuffer`, `SchedulerCore`, `ProfitabilityCore`, `StrategyEvaluationCore`, `ModelInputPreparation`, and `MarketDataCore`. No output-setting overrides were supplied, including no `CONFIGURATION_BUILD_DIR` override. Metadata commands used the approved filesystem escalation because Phase 24G had demonstrated that the sandbox blocked Xcode PIF-cache writes. No automatic approval-review rejection occurred.

Logs and evidence are retained under `DerivedData/ExpertAdvisor/Phase24H/`: `xcode-list.log`, `infer-build-settings.log`, eight dependency build-settings logs and exit-status files, `dependency-output-settings.json`, and `resolved-build-graph-after.json`. The latter identifies the latest workspace cache generation and its complete INFER dependency closure; old unresolved cache generations can coexist and were not mistaken for current evidence.

`-showBuildSettings` alone was not used as proof of resolution. The loaded PIF workspace explicitly contains canonical project paths for Rollover ExpertAdvisor, shared MetaNNXC, and shared MetalSwift. INFER's dependency record now has a GUID resolving to `MetaNN` and its `libMetaNN.a` product; transitively `MetaNN_metal` and `MetalBuffer` also have loaded GUIDs/products. Every dependency GUID in the nine-target closure resolves within the loaded projects.

## 6. DerivedData isolation

Stable metadata/build-output root:

`/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor`

For INFER and all eight dependency targets, effective `SYMROOT`, `OBJROOT`, `CONFIGURATION_BUILD_DIR`, `TARGET_BUILD_DIR`, `DERIVED_FILE_DIR`, `TARGET_TEMP_DIR`, `SHARED_PRECOMPS_DIR`, and `MODULE_CACHE_DIR` resolve inside that root. Shared `PROJECT_DIR` values correctly name production's MetaNN/MetalSwift **input** directories; their products and intermediates resolve under Rollover.

Examples:

- All Release products: `Build/Products/Release` beneath that root.
- MetaNN derived sources: `Build/Intermediates.noindex/MetaNNXC.build/Release/MetaNN.build/DerivedSources`.
- MetaNN_metal derived sources: `Build/Intermediates.noindex/MetaNNXC.build/Release/MetaNN_metal.build/DerivedSources`.
- MetalBuffer derived sources: `Build/Intermediates.noindex/MetalSwift.build/Release/MetalBuffer.build/DerivedSources`.

`DEPLOYMENT_LOCATION = NO`; no install/archive/localization action was requested. The effective normal-build outputs do not target production DerivedData, production binaries, or historical semantic-worker storage. No build products were generated by these metadata checks.

## 7. Shared dependency safety and documentation closure

Read-only inspection covered the relevant parent and external targets' build phases/rules/configurations. The INFER target's only shell phase generates provenance into Rollover `DERIVED_FILE_DIR`. Local library dependencies have no shell phases. Shared MetaNN, MetaNN_metal, and MetalBuffer have no shell phases or custom build rules writing to their source directories; relevant header phases are empty. The canonical publication phase is outside the INFER dependency closure. No symlink, shared project, Git submodule configuration, publication script, registry, or worker artifact was altered.

Shared MetaNN remains at gitlink revision `a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`. SHA-256 snapshots of both shared dependency project files match before/after inspection. Production and shared MetaNN `git status --short` were empty with optional Git locks disabled. These checks establish the observed input/configuration state and operation scope; they are not a blanket audit of every ignored production file or a native compiler result.

Phase 24F/G reports were reviewed before their separate documentation-only commit. Phase 24F received a correction distinguishing the missing parent-referenced path from the accessible shared project and marking its original independent-checkout/output-override proposal as superseded by later user instructions. Phase 24G received an explicit historical-snapshot note. Their executed-test results, pre-repair observations, and lack of native validation remain accurately recorded.

Documentation commit: `cec957b41082171e8894f36869f31c12e5424218`, two report files, 412 insertions. The staged path list was checked before commit; it did not include `project.pbxproj`. This authorized commit updates Rollover's linked-worktree Git metadata/branch only; production checkout content was not modified.

## 8–9. Remaining risks and native-build recommendation

1. Dependency resolution and relevant output isolation now pass. Native compilation/linking, actual identity output, artifact SHA-256, embedded provenance, runtime resources, and publisher qualification remain unverified; no native acceptance or publication claim is made.
2. The repaired project and this report remain dirty/uncommitted by instruction. `Scripts/GenerateBuildProvenance.py:40–49` will reject a Release build until a later authorized clean-provenance state is established. Committing Phase 24F/G alone does not remove that gate. Do not remove reports, hide changes, or bypass provenance checks.
3. Shared sources and host CPU/memory/I/O remain shared even though generated outputs are isolated. Native compilation may contend with historical experiments; no worker/resource intervention occurred here. Pre-main library behavior must still be considered before a later identity query.
4. Shared dependency absolute include paths and the Xcode 27.0 versus documented 26.5 toolchain difference remain unchanged. Metadata resolution does not prove native ABI/header compatibility or warning-free compilation.
5. The historical successful build's exact scheme/cache/configuration remains unknown. The current path-only before/after evidence is sufficient for this repair without inventing a historical explanation.

**GO for the repaired dependency graph as a prerequisite to a later explicitly authorized Phase 24G native build. NO-GO for building now with dirty provenance, executing workers, or publishing/deploying artifacts.** Stop here; do not compile/link, execute INFER, or commit the Phase 24H implementation/report.

## Final review state

Current HEAD: `cec957b41082171e8894f36869f31c12e5424218`. Index empty.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
?? docs/phases/Phase24/LSTM_Phase24H_MetaNNXcodeReferenceRepair_Output.md
```

`git diff --stat` (untracked report excluded):

```text
 ExpertAdvisor.xcodeproj/project.pbxproj | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

The two Phase 24F/G reports are committed; Phase 24H remains uncommitted. No compilation or worker execution occurred.
