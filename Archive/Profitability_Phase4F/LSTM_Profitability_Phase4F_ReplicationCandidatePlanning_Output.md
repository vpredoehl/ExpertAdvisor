---
title: "LSTM Profitability Phase 4F Replication Candidate Planning"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4F_ReplicationCandidatePlanning_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4F Replication Candidate Planning

Implemented Phase 4F as a pure, deterministic, in-memory advisory planner.

Key deliverables:

- [ReplicationCandidatePlanning.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ReplicationCandidatePlanning.hpp>)
- [ReplicationCandidatePlanning.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ReplicationCandidatePlanning.cpp>)
- [ReplicationCandidatePlanningTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ReplicationCandidatePlanningTests.cpp>)
- [ReplicationCandidatePlanningTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/ReplicationCandidatePlanningTests.sh>)
- [Phase 4F report](</Volumes/Developer SSD/ExpertAdvisor/LSTM_Profitability_Phase4F_ReplicationCandidatePlanning_Output.md>)
- Xcode project wiring updated.

Behavior includes deterministic scientific identity, exact-rerun exclusion, candidate deduplication, allowlist validation, diversity ranking, structured MIXED-reason targeting, candidate limits, deterministic rendering, and explicit no-action/insufficient-diversity statuses.

Validation:

- Phase 4F tests: PASS
- Phase 4E regression tests: PASS
- Phase 4D regression tests: PASS
- TrainingObjective tests: PASS
- Shell syntax: PASS
- Xcode project lint: PASS
- `git diff --check`: PASS
- Release build: expected provenance-gate failure because the worktree is dirty; focused compilation passed with `-Wall -Wextra -Werror`.

Remaining risk: the full Release target must be rebuilt after commit when the clean-worktree provenance requirement is satisfied.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
?? LSTM_Profitability_Phase4F_ReplicationCandidatePlanning_Output.md
?? Sources/ReplicationCandidatePlanning.cpp
?? Sources/ReplicationCandidatePlanning.hpp
?? Tests/ReplicationCandidatePlanningTests.cpp
?? Tests/ReplicationCandidatePlanningTests.sh
```

`git diff --stat`:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj | 8 ++++++++
 1 file changed, 8 insertions(+)
```

Untracked files are not included by `git diff --stat`.

LIVE_DATABASE_ACCESSED=false
LSTM_DATABASE_ACCESSED=false
FOREX_DATABASE_ACCESSED=false
PRODUCTION_DATABASE_MUTATED=false
PRODUCTION_SCHEDULER_TOUCHED=false
PRODUCTION_DERIVEDDATA_TOUCHED=false
DISPOSABLE_DATABASE_USED=none

Experiments 599/600 were not inspected. Nothing was staged or committed.