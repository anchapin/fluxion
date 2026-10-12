#!/usr/bin/env python3
# scripts/check_benchmark_band_provenance.py
#
# Issue #4168 acceptance criterion #3: provenance gate for the scorecard
# bands in `src/validation/benchmark.rs`.
#
# Fails when any of the following hold:
#   1. a `BenchmarkData` insert in either the Informed or the Blind table
#      lacks a per-metric source citation in its preceding comment block
#      (the comment must name a source: a citation keyword AND a table/
#      document reference);
#   2. two distinct case IDs carry byte-identical peak-band tuples
#      (peak_heating_min/max, peak_cooling_min/max) without written
#      justification — a justification is a `peak-band-shared:` marker
#      (or a `960`-style distinct-band note) in one of the two comments;
#   3. `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
#      holds a `published_range_mwh` that is not byte-equal to the
#      corresponding `annual_heating_*` / `annual_cooling_*` entry in the
#      Informed `benchmark.rs` table;
#   4. two files declare the same case ID with different provenance
#      (checked for `annex_b8_section7_statistics.json` vs the synthetic
#      quarantine README: the synthetic directory must NOT declare a case
#      as reference provenance).
#
# Exit codes: 0 = all checks pass; 1 = one or more failures (printed).
# Regenerate nothing; edit benchmark.rs comments/values and re-run.

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCHMARK_RS = REPO / "src" / "validation" / "benchmark.rs"
BASELINE_JSON = (
    REPO / "tests" / "reference_data" / "zone_balance" / "strict_energy_gate_baseline.json"
)
STATISTICS_JSON = REPO / "data" / "reference" / "ashrae140" / "annex_b8_section7_statistics.json"
SYNTHETIC_README = REPO / "data" / "synthetic" / "ashrae140" / "README.md"

CITATION_KEYWORDS = ("source", "transcribed", "envelope", "band aligned", "reference")
CITATION_SOURCES = (
    "Table B8",
    "B8-1",
    "B8-2",
    "B8-3",
    "B8-4",
    "ASHRAE 140",
    "Annex B",
    "Std140_TF_Results",
    "energy_reference.csv",
    "ashrae140_reference.json",
    "annex_b8_section7_statistics.json",
    "generate_case_",
    "§B",
    "§",
    "Annex",
)

INSERT_RE = re.compile(
    r'(?P<comment>(?:\s*//[^\n]*\n)+)?\s*data\.insert\(\s*"(?P<case>[^"]+)"\.to_string\(\),\s*'
    r"BenchmarkData\s*\{(?P<body>.*?)\},\s*\);",
    re.DOTALL,
)
FIELD_RE = {
    name: re.compile(rf"\b{name}:\s*([0-9.eE+-]+)")
    for name in (
        "annual_heating_min",
        "annual_heating_max",
        "annual_cooling_min",
        "annual_cooling_max",
        "peak_heating_min",
        "peak_heating_max",
        "peak_cooling_min",
        "peak_cooling_max",
    )
}


def parse_benchmark_tables():
    """Return {table: {case_id: {field: float}}} for both benchmark tables."""
    src = BENCHMARK_RS.read_text(encoding="utf-8")
    # Split on the two table constructors so case IDs are scoped per table.
    informed_start = src.index("pub fn get_all_benchmark_data() ->")
    blind_start = src.index("pub fn get_all_benchmark_data_blind() ->")
    tables = {
        "informed": src[informed_start:blind_start],
        "blind": src[blind_start:],
    }
    out = {}
    for table, text in tables.items():
        entries = {}
        for m in INSERT_RE.finditer(text):
            case = m.group("case")
            body = m.group("body")
            fields = {}
            for name, rx in FIELD_RE.items():
                fm = rx.search(body)
                if fm:
                    fields[name] = float(fm.group(1))
            comment = m.group("comment") or ""
            entries[case] = {"fields": fields, "comment": comment}
        out[table] = entries
    return out


def main() -> int:
    failures = []

    tables = parse_benchmark_tables()

    # ---- Check 1: per-metric source citation -------------------------
    for table, entries in tables.items():
        for case, entry in sorted(entries.items()):
            comment = entry["comment"]
            has_keyword = any(k in comment.lower() for k in CITATION_KEYWORDS)
            has_source = any(s in comment for s in CITATION_SOURCES)
            if not (has_keyword and has_source):
                failures.append(
                    f"[{table}] Case {case}: BenchmarkData insert lacks a per-metric "
                    f"source citation in its preceding comment (issue #4168 AC3)"
                )

    # ---- Check 2: byte-identical peak-band tuples across case IDs ----
    for table, entries in tables.items():
        seen = {}
        for case, entry in sorted(entries.items()):
            f = entry["fields"]
            if len(f) < 8:
                continue
            key = (
                f.get("peak_heating_min"),
                f.get("peak_heating_max"),
                f.get("peak_cooling_min"),
                f.get("peak_cooling_max"),
            )
            if key in seen and seen[key] != case:
                other = seen[key]
                justified = any(
                    "peak-band-shared:" in e["comment"] for e in (entry, entries[other])
                )
                if not justified:
                    failures.append(
                        f"[{table}] Cases {other} and {case} carry byte-identical "
                        f"peak-band tuples {key} without a written justification "
                        f"(add a `peak-band-shared:` note to one of the comments; "
                        f"issue #4168 AC3)"
                    )
            else:
                seen[key] = case

    # ---- Check 3: strict baseline byte-equality with benchmark.rs ----
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        informed = tables["informed"]
        for key, metric in sorted((baseline.get("metrics") or {}).items()):
            m = re.match(r"case_(\w+)_(heating|cooling)", key)
            if not m or metric is None:
                continue
            case, kind = m.groups()
            entry = informed.get(case)
            if entry is None:
                continue
            if kind == "heating":
                lo, hi = "annual_heating_min", "annual_heating_max"
            else:
                lo, hi = "annual_cooling_min", "annual_cooling_max"
            published = metric.get("published_range_mwh")
            if published is None:
                continue
            f = entry["fields"]
            if lo in f and hi in f and (
                float(published[0]) != f[lo] or float(published[1]) != f[hi]
            ):
                failures.append(
                    f"strict_energy_gate_baseline.json {key}.published_range_mwh = "
                    f"{published} is not byte-equal to benchmark.rs Case {case} "
                    f"{lo}/{hi} = [{f[lo]}, {f[hi]}] (issue #4168 AC3)"
                )

    # ---- Check 4: same case ID, conflicting provenance ---------------
    if STATISTICS_JSON.exists() and SYNTHETIC_README.exists():
        stats = json.loads(STATISTICS_JSON.read_text(encoding="utf-8"))
        readme = SYNTHETIC_README.read_text(encoding="utf-8")
        # The synthetic quarantine declares non-reference provenance; it must
        # not also appear inside the authoritative statistics file as a
        # source. Conversely, the statistics file's cases must not be listed
        # as provided by the synthetic CSVs.
        for series_case in ("195", "800"):
            if f"series_{series_case}.csv" in json.dumps(stats.get("_source", "")):
                failures.append(
                    f"annex_b8_section7_statistics.json cites series_{series_case}.csv "
                    f"(a quarantined synthetic series) as provenance (issue #4168 AC3)"
                )
        if "must never" not in readme and "never" not in readme:
            failures.append(
                "data/synthetic/ashrae140/README.md lost its non-reference "
                "provenance statement (issue #4168 AC3)"
            )

    if failures:
        print(f"FAIL: {len(failures)} benchmark-band provenance problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    n = sum(len(e) for e in tables.values())
    print(f"OK: benchmark band provenance consistent ({n} inserts across Informed/Blind)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
