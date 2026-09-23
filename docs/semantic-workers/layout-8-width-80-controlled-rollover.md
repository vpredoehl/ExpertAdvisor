---
title: "Layout 8 / Width 80 semantic-worker controlled rollover"
document_type: "engineering and operator procedure"
status: "implementation validation; production cutover requires operator approval"
---

# Layout 8 / Width 80 semantic-worker controlled rollover

## Motivation and starting state

The source semantic contract is layout 8 / width 80, while the authoritative
operational registry remains at layout 7 / width 77. Its layout-7 bindings are
deliberately distinct: the immutable `LSTM_Release` training/reference artifact
has `[train,infer,analyze]`, while `lstm-infer-worker` is infer-only. Layout 6
remains an inference-only historical archive. A current-layout advance cannot
be safely expressed as an inference publication because scheduler startup
requires both role bindings for its current layout.

## Supported mechanism

`Scripts/RollSemanticWorkerLayout.py` is a coordinated, operator-invoked
publication operation. It is separate from the infer-only
`Scripts/PublishSemanticWorker.py`, whose same-layout inference replacement
behavior is unchanged. It also does not invoke or alter canonical
`LSTM_Release` publication.

The command derives layout and width from `Headers/ModelInputExpansion.hpp` and
requires clean source `HEAD`. It accepts only two absolute product paths:

- `--training-executable`: `LSTM_Release`, embedded exact `HEAD`, published as
  role `train` with `[train,infer,analyze]` and manifest schema 1;
- `--inference-executable`: `lstm-infer-worker`, embedded exact `HEAD` and
  matching its `--build-identity` SHA-256/source-commit contract, published as
  role `infer` with `[infer]` and manifest schema 2.

The two Release products must carry byte-identical `default.metallib` and
`MetaNN_metal.metallib` resources. They are copied once into the established
content-addressed runtime package and each worker directory receives the same
deterministic relative resource links. Training/reference artifacts retain the
legacy compatible path:

```text
layout8/<commit>/<training-sha256>/LSTM_Release
```

Inference artifacts use the role-aware path:

```text
layout8/infer/<commit>/<inference-sha256>/lstm-infer-worker
```

## Atomic transition and failure behavior

The operation holds the existing publisher lock and first validates every prior
registry artifact, manifest, hash, runtime package, and resource link. It then
stages immutable runtime/artifacts. Before replacing the registry, it constructs
and validates a prospective v4 registry with exactly these current bindings:

| Before | After |
| --- | --- |
| layout 7 / train / current | layout 7 / train / historical |
| layout 7 / infer / current | layout 7 / infer / historical |
| — | layout 8 / train / current |
| — | layout 8 / infer / current |

Layout 6 / infer remains historical and unchanged. The operation rejects a
target layout that is already registered, a malformed/missing/corrupt prior
artifact, missing candidates, role/name mismatch, provenance mismatch, runtime
mismatch, staging collision, and any prospective registry that fails the
two-current-role invariant. Only one atomic `registry.json` replacement makes
the result authoritative. A failure before it leaves the old registry usable;
newly staged immutable paths can be unreachable but are never partial current
state. The convenience `current` symlink is updated only afterward.

There is no train-only advance command and the old infer-only publisher still
cannot advance a layout. Historical artifacts are never deleted or overwritten.

## Validation evidence

`Tests/SemanticWorkerRolloverTests.sh` uses a disposable artifact root and
injects missing/wrong-role candidates, corrupt prior artifacts, staged conflicts,
dirty-source/build-identity preflight failures, malformed registries, and a
registry-replacement failure. It proves the old layout-7 registry bytes remain
unchanged for failures before authority replacement. Its C++ registry check
proves exact routing after success:

| requested identity | selected binding |
| --- | --- |
| 77 / 6 / infer | historical layout-6 infer |
| 77 / 7 / train | historical layout-7 train |
| 77 / 7 / infer | historical layout-7 infer |
| 80 / 8 / train | current layout-8 train |
| 80 / 8 / infer | current layout-8 infer |

It also rejects width/layout and role mismatches, so no historical layout-7
request can resolve to layout 8 and no layout-8 request can resolve to layout 7.

An isolated clean-worktree validation at
`66b2613fb7df0bf8064df02e95a62cbf0afdad91` produced and successfully rolled
these disposable candidates:

| product | SHA-256 |
| --- | --- |
| `LSTM_Release` training/reference | `5718d86eff7a0ca4b0ad9b3bac5fc5cfecdbd119b8f387870f55949308bb2b07` |
| `lstm-infer-worker` infer-only | `a5f5f4dc6c753879ab339c3c1c4b0c5d2b130dde2b181754f06e7b912acc76d6` |
| `default.metallib` | `a13694e6940e8287c1b3ca696edbcc291e85de2fd514ade52e431054f1d537d5` |
| `MetaNN_metal.metallib` | `9c894e9e02b3dfafb69d639535ebc064c7b5ef30ce72edd5cf628f220e16f759` |

The standalone worker reported that exact commit and SHA through
`--build-identity`; `LSTM_Release` contained that exact embedded commit. The
disposable post-state preserved the operational fixture's layout 6 / infer and
layout 7 / train+infer hashes while adding the two layout-8 paths above.

## Production operator procedure

This procedure is intentionally not automatic.

1. Confirm a clean checkout at the approved layout-8 commit, inspect the live
   scheduler/process state read-only, and record the registry SHA-256 and all
   current layout-7 bindings.
2. Build isolated clean Release products from that exact commit:

   ```bash
   xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
     -configuration Release -derivedDataPath <clean-derived-data> build
   xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Infer Worker" \
     -configuration Release -derivedDataPath <clean-derived-data> build
   ```

3. Record each SHA-256, verify `strings LSTM_Release` contains the approved
   commit, and verify `lstm-infer-worker --build-identity` reports that same
   commit and SHA. Confirm both runtime resource hashes match.
4. After an explicit publication approval, run:

   ```bash
   /usr/bin/python3 Scripts/RollSemanticWorkerLayout.py \
     --repository-root /Volumes/Developer\ SSD/ExpertAdvisor \
     --training-executable <clean-derived-data>/Build/Products/Release/LSTM_Release \
     --inference-executable <clean-derived-data>/Build/Products/Release/lstm-infer-worker \
     --source-commit <approved-40-hex-commit>
   ```

5. Read the resulting registry and independently load it with the registry
   resolver. Verify the five-row routing matrix above. Do not restart the
   scheduler in this publication step.

If the command fails before registry replacement, retain layout 7 as the
authoritative state and investigate; do not hand-edit the registry or any
immutable artifact. A scheduler activation/restart is a separately approved
operational handoff because scheduler processes use startup snapshots. The
first controlled TG4 ablation additionally requires that later approval plus
its own experiment/admission plan; this rollover itself queues no work.
