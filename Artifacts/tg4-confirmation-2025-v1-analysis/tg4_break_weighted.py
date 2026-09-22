import csv
import math
from collections import defaultdict

p = "/Volumes/Developer SSD/ExpertAdvisor/Artifacts/tg4-confirmation-2025-v1/observations.csv"

# key = (symbol, break_bar)
# value = [state, outcome]
breaks = {}

with open(p, newline="") as f:
    for r in csv.DictReader(f):

        if r["direction"] != "UTL":
            continue

        if r["paired_outer"] != "true":
            continue

        if r["outer_target_state"] not in ("succeeded", "failed"):
            continue

        state = r["tg3_confluence_state"]

        if state not in ("confluent", "non_confluent"):
            continue

        key = (r["symbol"], r["break_bar"])
        value = (state, r["outer_target_state"])

        old = breaks.get(key)

        if old is not None and old != value:
            raise RuntimeError(
                f"mixed break cluster {key}: {old} versus {value}"
            )

        breaks[key] = value


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))

    p = successes / n
    z2 = z * z

    center = (p + z2 / (2 * n)) / (1 + z2 / n)

    half = (
        z
        * math.sqrt(
            p * (1 - p) / n
            + z2 / (4 * n * n)
        )
        / (1 + z2 / n)
    )

    return center - half, center + half


def summarize(rows):
    n = len(rows)
    successes = sum(
        outcome == "succeeded"
        for _, outcome in rows
    )

    rate = successes / n
    lo, hi = wilson(successes, n)

    return n, successes, rate, lo, hi


groups = defaultdict(list)
symbols = defaultdict(lambda: defaultdict(list))

for (symbol, break_bar), value in breaks.items():
    state, outcome = value

    groups[state].append(value)
    symbols[symbol][state].append(value)


print("TG4A BREAK-WEIGHTED ANALYSIS")
print("============================")
print("unique break clusters:", len(breaks))
print()

results = {}

for state in ("confluent", "non_confluent"):
    n, success, rate, lo, hi = summarize(groups[state])
    results[state] = rate

    print(state)
    print("  breaks     :", n)
    print("  successes :", success)
    print("  failures  :", n - success)
    print("  rate       :", f"{rate:.6f}")
    print("  Wilson 95% :", f"[{lo:.6f}, {hi:.6f}]")
    print()

diff = results["confluent"] - results["non_confluent"]

print(
    "absolute rate difference:",
    f"{diff:+.6f}",
    f"({diff * 100:+.2f} percentage points)"
)

print()
print("PER-SYMBOL BREAK-WEIGHTED RESULTS")
print("=================================")

for symbol in sorted(symbols):

    print()
    print(symbol)

    rates = {}

    for state in ("confluent", "non_confluent"):
        rows = symbols[symbol][state]

        if not rows:
            print(" ", state, ": no observations")
            continue

        n, success, rate, lo, hi = summarize(rows)
        rates[state] = rate

        print(
            f"  {state:14s}"
            f" n={n:5d}"
            f" success={success:5d}"
            f" rate={rate:.6f}"
            f" 95%=[{lo:.6f},{hi:.6f}]"
        )

    if (
        "confluent" in rates
        and "non_confluent" in rates
    ):
        d = rates["confluent"] - rates["non_confluent"]

        print(
            "  difference     ",
            f"{d:+.6f}",
            f"({d * 100:+.2f} pp)"
        )

outer_groups = defaultdict(
    lambda: {
        "confluent": 0,
        "non_confluent": 0,
        "succeeded": 0,
        "failed": 0,
    }
)

with open(p, newline="") as f:
    for r in csv.DictReader(f):

        if r["direction"] != "UTL":
            continue

        if r["paired_outer"] != "true":
            continue

        if r["outer_target_state"] not in ("succeeded", "failed"):
            continue

        state = r["tg3_confluence_state"]

        if state not in ("confluent", "non_confluent"):
            continue

        key = (
            r["symbol"],
            r["outer_candidate_identity"],
        )

        outer_groups[key][state] += 1
        outer_groups[key][r["outer_target_state"]] += 1


mixed_state = 0
mixed_outcome = 0
only_confluent = 0
only_nonconfluent = 0

for g in outer_groups.values():

    if g["confluent"] and g["non_confluent"]:
        mixed_state += 1
    elif g["confluent"]:
        only_confluent += 1
    else:
        only_nonconfluent += 1

    if g["succeeded"] and g["failed"]:
        mixed_outcome += 1


print()
print("OUTER-CANDIDATE STRUCTURE")
print("=========================")
print("outer candidates             :", len(outer_groups))
print("confluent only               :", only_confluent)
print("non-confluent only           :", only_nonconfluent)
print("mixed confluence             :", mixed_state)
print("mixed outcome                :", mixed_outcome)

# ------------------------------------------------------------
# Within-outer matched comparison, using unique break clusters.
# Each break receives one vote. Only outer candidates containing
# BOTH confluent and non-confluent breaks participate.
# ------------------------------------------------------------

outer_breaks = defaultdict(dict)

with open(p, newline="") as f:
    for r in csv.DictReader(f):

        if r["direction"] != "UTL":
            continue
        if r["paired_outer"] != "true":
            continue
        if r["outer_target_state"] not in ("succeeded", "failed"):
            continue

        state = r["tg3_confluence_state"]

        if state not in ("confluent", "non_confluent"):
            continue

        outer = (
            r["symbol"],
            r["outer_candidate_identity"],
        )

        brk = (
            r["symbol"],
            r["break_bar"],
        )

        value = (
            state,
            r["outer_target_state"],
        )

        old = outer_breaks[outer].get(brk)

        if old is not None and old != value:
            raise RuntimeError(
                f"inconsistent break {brk}: {old} vs {value}"
            )

        outer_breaks[outer][brk] = value


matched = []

for outer, breaks_for_outer in outer_breaks.items():

    values = list(breaks_for_outer.values())

    c = [
        outcome
        for state, outcome in values
        if state == "confluent"
    ]

    n = [
        outcome
        for state, outcome in values
        if state == "non_confluent"
    ]

    if not c or not n:
        continue

    c_rate = sum(x == "succeeded" for x in c) / len(c)
    n_rate = sum(x == "succeeded" for x in n) / len(n)

    matched.append(
        (outer, len(c), len(n), c_rate, n_rate, c_rate - n_rate)
    )


positive = sum(x[5] > 0 for x in matched)
negative = sum(x[5] < 0 for x in matched)
ties = sum(x[5] == 0 for x in matched)

mean_diff = sum(x[5] for x in matched) / len(matched)

weighted_c_success = 0
weighted_c_n = 0
weighted_n_success = 0
weighted_n_n = 0

for outer, nc, nn, cr, nr, d in matched:
    weighted_c_success += cr * nc
    weighted_c_n += nc
    weighted_n_success += nr * nn
    weighted_n_n += nn

weighted_c_rate = weighted_c_success / weighted_c_n
weighted_n_rate = weighted_n_success / weighted_n_n


print()
print("WITHIN-OUTER MATCHED BREAK ANALYSIS")
print("===================================")
print("mixed-confluence outer candidates :", len(matched))
print("positive within-outer differences :", positive)
print("negative within-outer differences :", negative)
print("ties                              :", ties)
print(
    "equal-outer mean difference       :",
    f"{mean_diff:+.6f}",
    f"({mean_diff * 100:+.2f} pp)"
)
print()
print("pooled breaks within matched outers")
print(
    " confluent     :",
    weighted_c_n,
    f"rate={weighted_c_rate:.6f}"
)
print(
    " non-confluent :",
    weighted_n_n,
    f"rate={weighted_n_rate:.6f}"
)
print(
    " difference    :",
    f"{weighted_c_rate - weighted_n_rate:+.6f}",
    f"({(weighted_c_rate - weighted_n_rate) * 100:+.2f} pp)"
)
from math import comb

informative = positive + negative
k = min(positive, negative)

p_one_tail = sum(
    comb(informative, i)
    for i in range(k + 1)
) / (2 ** informative)

p_two_sided = min(1.0, 2 * p_one_tail)

diffs = sorted(x[5] for x in matched)

def quantile(values, p):
    i = min(len(values) - 1, int((len(values) - 1) * p))
    return values[i]

print()
print("WITHIN-OUTER DIRECTIONAL ROBUSTNESS")
print("===================================")
print("informative outer candidates :", informative)
print("positive                     :", positive)
print("negative                     :", negative)
print("positive fraction            :", f"{positive/informative:.6f}")
print("exact sign-test p (2-sided)  :", f"{p_two_sided:.12g}")
print("difference Q25               :", f"{quantile(diffs,.25):+.6f}")
print("difference median            :", f"{quantile(diffs,.50):+.6f}")
print("difference Q75               :", f"{quantile(diffs,.75):+.6f}")

from datetime import datetime

# ------------------------------------------------------------
# Temporal robustness:
# one rate comparison per symbol x ISO calendar week.
# Uses unique break clusters so duplicate observation rows
# cannot increase the weight of a break event.
# Only blocks containing BOTH confluence states participate.
# ------------------------------------------------------------

weekly_breaks = defaultdict(dict)

with open(p, newline="") as f:
    for r in csv.DictReader(f):

        if r["direction"] != "UTL":
            continue
        if r["paired_outer"] != "true":
            continue
        if r["outer_target_state"] not in ("succeeded", "failed"):
            continue

        state = r["tg3_confluence_state"]

        if state not in ("confluent", "non_confluent"):
            continue

        # break_timestamp is ISO-8601 UTC in the TG4 artifact.
        ts = r["break_timestamp"].replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)

        iso = dt.isocalendar()

        block = (
            r["symbol"],
            iso.year,
            iso.week,
        )

        brk = (
            r["symbol"],
            r["break_bar"],
        )

        value = (
            state,
            r["outer_target_state"],
        )

        old = weekly_breaks[block].get(brk)

        if old is not None and old != value:
            raise RuntimeError(
                f"inconsistent break {brk}: {old} vs {value}"
            )

        weekly_breaks[block][brk] = value


weekly_results = []

for block, break_map in weekly_breaks.items():

    values = list(break_map.values())

    c = [
        outcome
        for state, outcome in values
        if state == "confluent"
    ]

    n = [
        outcome
        for state, outcome in values
        if state == "non_confluent"
    ]

    # Matched temporal block: both conditions must occur.
    if not c or not n:
        continue

    c_rate = sum(x == "succeeded" for x in c) / len(c)
    n_rate = sum(x == "succeeded" for x in n) / len(n)

    weekly_results.append(
        (
            block,
            len(c),
            len(n),
            c_rate,
            n_rate,
            c_rate - n_rate,
        )
    )


positive_week = sum(x[5] > 0 for x in weekly_results)
negative_week = sum(x[5] < 0 for x in weekly_results)
tied_week = sum(x[5] == 0 for x in weekly_results)

informative_week = positive_week + negative_week

week_diffs = sorted(x[5] for x in weekly_results)

equal_week_mean = (
    sum(week_diffs) / len(week_diffs)
    if week_diffs else float("nan")
)


# Exact sign test among non-tied symbol-weeks.
if informative_week:
    k = min(positive_week, negative_week)

    p_one_tail = sum(
        comb(informative_week, i)
        for i in range(k + 1)
    ) / (2 ** informative_week)

    week_sign_p = min(1.0, 2 * p_one_tail)
else:
    week_sign_p = float("nan")


# Pooled unique breaks, restricted to matched symbol-weeks.
c_success = 0
c_n = 0
n_success = 0
n_n = 0

for block, nc, nn, cr, nr, diff in weekly_results:
    c_success += round(cr * nc)
    c_n += nc

    n_success += round(nr * nn)
    n_n += nn

c_pooled_rate = c_success / c_n
n_pooled_rate = n_success / n_n


print()
print("SYMBOL-WEEK TEMPORAL ROBUSTNESS")
print("===============================")
print("matched symbol-weeks           :", len(weekly_results))
print("informative symbol-weeks       :", informative_week)
print("positive                       :", positive_week)
print("negative                       :", negative_week)
print("ties                           :", tied_week)

if informative_week:
    print(
        "positive fraction             :",
        f"{positive_week / informative_week:.6f}"
    )

print(
    "exact sign-test p (2-sided)    :",
    f"{week_sign_p:.12g}"
)

print(
    "equal-symbol-week mean diff    :",
    f"{equal_week_mean:+.6f}",
    f"({equal_week_mean * 100:+.2f} pp)"
)

print(
    "difference Q25                 :",
    f"{quantile(week_diffs, .25):+.6f}"
)

print(
    "difference median              :",
    f"{quantile(week_diffs, .50):+.6f}"
)

print(
    "difference Q75                 :",
    f"{quantile(week_diffs, .75):+.6f}"
)

print()
print("pooled breaks in matched weeks")
print(
    " confluent                     :",
    c_n,
    f"rate={c_pooled_rate:.6f}"
)
print(
    " non-confluent                 :",
    n_n,
    f"rate={n_pooled_rate:.6f}"
)
print(
    " difference                    :",
    f"{c_pooled_rate - n_pooled_rate:+.6f}",
    f"({(c_pooled_rate - n_pooled_rate) * 100:+.2f} pp)"
)

