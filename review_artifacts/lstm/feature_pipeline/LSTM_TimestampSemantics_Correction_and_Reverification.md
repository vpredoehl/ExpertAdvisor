# LSTM Timestamp Semantics Correction and Reverification

## Status

**CLOSED / VERIFIED**

## Scope

This review covers the timestamp-semantics issue identified while inventorying
the existing LSTM feature pipeline before introducing any new feature increment.

The active PostgreSQL feature ingestion path is implemented in:

- `Common/db_cursor.cpp`

The correction is independent of the classification-inference clamp parity
correction in `LSTM/LSTM.cpp`.

## Finding

The active timestamp parser previously converted PostgreSQL candlestick
timestamps using `std::mktime(&ct)`.

The source database is `forex`. Database inspection established:

- PostgreSQL `TimeZone` is `America/Chicago`.
- `eurusdrmp.time` is `timestamp without time zone`.
- `candlestick(...).dt` is `timestamp without time zone`.
- The `candlestick` function accepts `timestamp without time zone` boundaries
  and returns `SETOF cst`.

Therefore PostgreSQL does not attach timezone semantics to the returned civil
timestamp. Timestamp interpretation occurs in the C++ ingestion path.

## Source-Data Provenance

The historical RMP tables were populated from DAT_NT-style CSV files through
the `csvtemp` import/trigger workflow.

The CSV timestamps are naive civil timestamps without an explicit UTC offset.

Historical EURUSD data showed the Sunday market reopening at approximately:

    2010-01-03 17:00:12

This is consistent with U.S. Eastern civil time:

    2010-01-03 17:00:12 EST
    = 2010-01-03 22:00:12 UTC

The evidence therefore shows that interpreting the stored timestamp directly
as UTC would be incorrect. Interpreting it according to the host's
America/Chicago timezone would also be incorrect.

## Defect

The historical implementation used:

    std::mktime(&ct)

This made timestamp interpretation dependent on the process/host timezone.

On a host configured for America/Chicago, an RMP timestamp representing
17:00 Eastern would instead be interpreted as 17:00 Central.

That changes the absolute instant and consequently can alter downstream
timestamp-derived LSTM features.

## Correction

`Common/db_cursor.cpp` now performs deterministic conversion from the
historical America/New_York civil-time convention to UTC.

The implementation:

- does not modify the process-global `TZ` environment;
- applies EST (UTC-5) during standard time;
- applies EDT (UTC-4) during daylight-saving time;
- uses the post-2007 U.S. DST rules applicable to this 2009+ dataset;
- rejects the nonexistent spring-forward 02:00-02:59 interval;
- rejects the ambiguous fall-back 01:00-01:59 interval;
- preserves strict timestamp parsing.

The resulting UTC instant is stored in `PriceTP`, allowing the existing
UTC-based downstream time decomposition to operate deterministically.

## Release Build Verification

The modified source was built using:

    xcodebuild -project ExpertAdvisor.xcodeproj \
      -scheme "LSTM Release" \
      -configuration Release \
      build

Result:

    ** BUILD SUCCEEDED **
    build exit=0

## Focused Runtime Reverification

A temporary standalone C++ harness under `/tmp` executed the same
America/New_York civil-time conversion algorithm used by
`Common/db_cursor.cpp`.

Observed results:

    PASS  2010-01-03 17:00:12 NY -> 2010-01-03 22:00:12 UTC
    PASS  2010-07-01 17:00:00 NY -> 2010-07-01 21:00:00 UTC
    PASS  2010-03-14 01:59:59 NY -> 2010-03-14 06:59:59 UTC
    PASS  2010-03-14 03:00:00 NY -> 2010-03-14 07:00:00 UTC
    PASS  2010-11-07 00:59:59 NY -> 2010-11-07 04:59:59 UTC
    PASS  2010-11-07 02:00:00 NY -> 2010-11-07 07:00:00 UTC
    PASS  DST exceptional civil time 2010-03-14 02:00 rejected
    PASS  DST exceptional civil time 2010-11-07 01:00 rejected

    TIMESTAMP_CONVERSION_TEST=PASS
    timestamp test exit=0

The temporary test source and executable were removed after execution.

## Reverification Conclusion

The corrected conversion:

- reproduces the expected EST-to-UTC historical mapping;
- reproduces the expected EDT-to-UTC mapping;
- transitions correctly across both DST boundaries;
- explicitly rejects civil times that cannot be deterministically interpreted
  without additional source metadata;
- is independent of the host timezone;
- compiles successfully in the Release target.

**Timestamp-semantics correction: VERIFIED.**

## Related Baseline-Integrity Correction

The same feature-pipeline review identified a separate preprocessing mismatch:
training clamps LSTM feature values to `[-10,+10]`, while classification
inference previously did not apply the equivalent clamp.

`LSTM/LSTM.cpp` was corrected so classification inference applies the same
clamp.

Together these corrections establish the baseline before introducing the next
LSTM feature increment.
