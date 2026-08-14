# LSTM H4 Build-Identity CWD-Independence Independent Reverification

## Verdict

PASS — approved for integration.

The implementation corrects the production H4 build-identity defect in which
CaptureActualManagerBuildContract() depended on the runtime current working
directory to execute `git rev-parse HEAD` and `git status --porcelain`.

## Findings

The corrected implementation removes runtime Git/CWD discovery from
CaptureActualManagerBuildContract(). Source identity is obtained from build-time
embedded provenance (`EXPERTADVISOR_SOURCE_COMMIT`), while executable identity
continues to be derived from the explicitly supplied executable path.

The build provenance generator captures the source commit from the repository at
build time and rejects dirty Release source trees. The production C++ path fails
closed when embedded provenance is absent or malformed.

The existing `campaign_operations_manager_build_v1` canonical contract remains
unchanged.

The Xcode Release target integrates generation of
`GeneratedBuildProvenance.hpp` before source compilation and makes the generated
header available through the derived-file include path.

Independent focused testing of the provenance generator confirmed:

- clean Release provenance captures the exact 40-hex source HEAD;
- dirty Release source state is rejected;
- failed dirty-tree generation does not replace previously valid generated
  provenance;
- Debug provenance does not establish a production source identity;
- repository discovery occurs explicitly at build time rather than from the
  production process runtime CWD.

The supplied H4 supervisor regression suite passed 31 tests.

## Residual verification item

The newly added C++ build-identity regression was not demonstrated as linked and
executed in the supplied implementation evidence because that isolated test
environment lacked the pqxx dependency. This is an execution-evidence gap rather
than an identified implementation defect. It should be exercised through the
normal repository/build environment before production activation.

## Production safety

No production H4 start, production enablement refresh, or replacement activation
was performed as part of this reverification.

The previously staged replacement binary predates this correction and must not be
used for production authorization. A fresh clean Release build must be produced
after this correction is committed, followed by CWD-independent runtime identity
verification and regeneration of the self-contained runtime stage.

## Conclusion

The build-identity CWD-independence correction is approved for integration.

Next boundary:

commit correction -> clean Release rebuild -> focused C++/runtime verification ->
regenerate self-contained stage -> verify final staged SHA-256 -> production
authorization refresh.
