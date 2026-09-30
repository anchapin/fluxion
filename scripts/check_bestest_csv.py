#!/usr/bin/env python3
"""Validate a Section 7 results CSV against the BESTEST-GSR populate-script contract.

The consumer (NatLabRockies/BESTEST-GSR results/bestest_populate_report.rb) parses
timestamps by character offset and re-splits packed arrays on commas. Both fail
silently: a mis-padded day or a short monthly array produces a wrong number in the
Standard Output Report with no error raised anywhere. This script is the guard.

Usage:
    python3 scripts/check_bestest_csv.py workflow_results.csv
    python3 scripts/check_bestest_csv.py workflow_results.csv --contract path/to/contract.json

Exit code 0 when the file is safe to hand to the populate script, 1 otherwise.
"""
import argparse, csv, json, os, re, sys

DEFAULT_CONTRACT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir,
    "data", "reference", "ashrae140", "bestest_gsr_csv_contract.json")

MONTHS = {"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"}
TS = re.compile(r"^(\d{2})-([A-Z]{3})-(\d{2}):(\d{2})$")


def check_timestamp(value):
    """Return a list of problems with one DD-MMM-HH:MM timestamp."""
    problems = []
    if not TS.match(value):
        return ["does not match DD-MMM-HH:MM (consumer parses by character offset, "
                "so padding and position are load bearing)"]
    day, mon, hour, minute = TS.match(value).groups()
    if mon not in MONTHS:
        problems.append("month %r is not a three-letter uppercase abbreviation" % mon)
    if not 1 <= int(day) <= 31:
        problems.append("day %s out of range" % day)
    if int(hour) > 24:
        problems.append("hour %s out of range" % hour)
    if int(minute) > 59:
        problems.append("minute %s out of range" % minute)
    if int(minute) > 0 and int(hour) == 24:
        problems.append("minutes nonzero at hour 24: the consumer increments the hour "
                        "and would emit 25")
    return problems


def check_array(value, expected_len=None, min_len=None):
    parts = [p for p in value.split(",")]
    if expected_len is not None and len(parts) != expected_len:
        return ["has %d comma-separated values, expected %d" % (len(parts), expected_len)]
    if min_len is not None and len(parts) < min_len:
        return ["has %d comma-separated values, expected at least %d" % (len(parts), min_len)]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--allow-missing-fields", action="store_true",
                    help="report absent columns as warnings rather than errors")
    args = ap.parse_args()

    with open(args.contract) as fh:
        c = json.load(fh)
    prefix = c["field_prefix"]
    f = c["fields"]
    errors, warnings = [], []

    with open(args.csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("FAIL: no data rows")
        return 1
    headers = list(rows[0].keys())

    key_header = c["key_column"]["header"]
    if headers[0] != key_header:
        errors.append("column 1 header is %r, must be %r" % (headers[0], key_header))

    expected = ["%s.%s" % (prefix, n) for group in
                ("scalar", "timestamp", "monthly_array", "day_profile_array", "other_packed")
                for n in f[group]]
    present = set(headers)
    for col in expected:
        if col not in present:
            (warnings if args.allow_missing_fields else errors).append("missing column %r" % col)
    for col in headers:
        if col.startswith(prefix) and not col.startswith(prefix + "."):
            errors.append("column %r drops the dot after the prefix; emit the dotted form, "
                          "not the Ruby symbol form" % col)

    seen = {}
    for i, row in enumerate(rows, start=2):
        raw = (row.get(key_header) or "").strip()
        if not raw:
            errors.append("row %d: empty %s" % (i, key_header))
            continue
        key = raw.split(" ")[0]
        if key in seen:
            errors.append("row %d: case key %r already used on row %d; the consumer keeps the "
                          "last one silently" % (i, key, seen[key]))
        seen[key] = i

        for name in f["timestamp"]:
            v = (row.get("%s.%s" % (prefix, name)) or "").strip()
            if v:
                for p in check_timestamp(v):
                    errors.append("row %d case %s: %s = %r %s" % (i, key, name, v, p))
        for name in f["monthly_array"]:
            v = (row.get("%s.%s" % (prefix, name)) or "").strip()
            if v:
                for p in check_array(v, expected_len=c["array_format"]["monthly_length"]):
                    errors.append("row %d case %s: %s %s" % (i, key, name, p))
        for name in f["day_profile_array"]:
            v = (row.get("%s.%s" % (prefix, name)) or "").strip()
            if v:
                for p in check_array(v, min_len=24):
                    errors.append("row %d case %s: %s %s" % (i, key, name, p))

    for w in warnings:
        print("WARN: %s" % w)
    for e in errors:
        print("FAIL: %s" % e)
    print("\n%d rows, %d case keys, %d errors, %d warnings"
          % (len(rows), len(seen), len(errors), len(warnings)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
