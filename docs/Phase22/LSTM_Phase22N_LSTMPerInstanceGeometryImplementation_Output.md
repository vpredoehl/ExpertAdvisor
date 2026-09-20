---
title: "Phase 22N EA::LSTM Per-Instance Geometry Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22N_LSTMPerInstanceGeometryImplementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# Phase 22N EA::LSTM Per-Instance Geometry Implementation

Implemented Phase 22N and left all changes unstaged.

Report: [LSTM_Phase22N_LSTMPerInstanceGeometryImplementation_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase22N_LSTMPerInstanceGeometryImplementation_Output.md)

Key result: `EA::LSTM` now owns immutable `hiddenSize_`; all member structural geometry, including Metal dispatch K dimensions, derives from it rather than global `hidden_size`/`n_out`.

Validation:

- Debug stable-DerivedData build passed (RC 0).
- `git diff --check` passed.
- No executable or DB-backed tests were run because active scheduler/training workers were detected.
- No scheduler, persistence API, schema, registry, or publisher changes.

Final status:

```text
 M Headers/LSTM.hpp
 M LSTM/LSTM.cpp
 M LSTM/main.cpp
?? LSTM_Phase22N_LSTMPerInstanceGeometryImplementation_Output.md
```