# Research-only TG3 AB capacity counterfactual observer

Schema: `causal-ab-capacity-counterfactual-v1`.

Opt in with `--ab-capacity-audit-dir NEW_EXTERNAL_DIRECTORY` on the standalone
research runner. This option does not authorize a historical run. Build-only
validation and the synthetic test script never execute a database study.

The immutable completed-bar callback runs after primary evaluation. Real TG1A
and TG3 mirrors verify confirmations, birth order, retained snapshot order,
ATR, and lifecycle counters. The TG3 mirror expires normally, then receives
confirmed fractals one at a time in native sorted order. Before/after snapshots
capture only capacity removals. The eviction count must reconcile exactly.
No birth or suppressed fractal anchor is restored. TG3 construction reads only
bounded fractal history; AB retention never feeds later anchor selection.

Victims expire when current bar minus availability bar exceeds 2048. At every
observed scheduled hourly :45 slot, ACTUAL uses the supplied frozen retained
ABs and lines. COUNTERFACTUAL adds surviving capacity victims, preserving native
AB order, and uses the identical lines and ATR. The existing geometry evaluator
performs natural deduplication, complete aggregation, three frozen ratios,
0.25 ATR thresholds, and frozen cohort precedence. Its optional diagnostic
input/work budgets leave the history audit's original defaults unchanged.
The counterfactual is an AB-retention intervention, not a new detector protocol.

Complete/evaluated primary geometry must match ACTUAL. Unevaluated primary
capacity-invalid predicates are never used as geometry evidence. Warmup,
embargo, horizon eligibility and primary status remain recorded exclusions;
computed geometry cohorts do not promote those slots to primary observations.
Known empty geometry is INELIGIBLE_NO_GEOMETRY with UNKNOWN predicates. Missing
slots are UNOBSERVABLE_MISSING_SLOT. Missing ATR or any reconstruction failure
makes the diagnostic incomplete; totals of changed predicates/cohorts become
UNKNOWN. Verified partial change counts are explicitly separate.

Default diagnostic budgets: 8192 live AB records, 64 MiB conservative memory
reservation, 256 MiB decoded output, 16,777,216 combined actual/counterfactual
pairs per decision, and 1,073,741,824 pairs per invocation. The memory forecast
includes native mirrors, vector spare capacity, stream buffers, transient
copies, and both geometry maps; it is not an RSS measurement. A 16 KiB output
reserve permits incomplete metadata after streaming output saturation. These
limits are diagnostic budgets; TG3 stays 512, TG1A stays 4096, and histories
stay 128/64. Victims are age-pruned, not accumulated forever. No partial pair
universe is accepted. A budget or parity failure stops diagnostic reconstruction,
records UNKNOWN subsequent decisions where output space permits, and reports
unpublished decision counts otherwise. Primary callbacks continue unchanged.

Artifacts are deterministic uncompressed CSV/JSON: audit_protocol.json,
audit_provenance.json, source_evictions.csv, restored_candidates.csv,
decisions.csv, summary.csv, audit_checksums.sha256. Per-candidate rows describe
presence, age, direction, eligible AB/line pairs, Fibonacci proximity and
conjunction; decision rows separate candidate contribution from aggregate
predicate/cohort changes. Source summaries include prefix/scoring counts and
first eviction bar/UTC timestamp. On failure, event counters cover the verified
prefix only. They do not assert complete historical counts.

Publication closes CSV streams and validates parent checksums, fingerprint,
source/candle provenance and complete primary geometry gzip members. A parent
INVALID is accepted as a linkage state without removing or renaming it. An
AUDIT_COMPLETE marker is atomically renamed only after diagnostic completeness,
parity, checksum verification and parent integrity. It explicitly states
primary_validity=UNCHANGED and parent_status=INVALID or COMPLETE. Publication
faults never create a completion marker. Independently validate with
ABCapacity::ValidateSidecar(directory, parentPrimary).

Tests use native candle mirrors for detector parity and native TG3 interface
fixtures for geometry effects and AB-retention/birth independence. Interface
fixtures alone do not establish candle detector parity. No January 2015
counterfactual has been executed. Its changed-predicate and cohort counts remain
UNKNOWN. A separately authorized replay is needed to measure them.
