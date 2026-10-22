#!/usr/bin/env python3
"""Free-float energy-balance audit (loop round 86, FLX_FF_EB_AUDIT dump).

Reads a `cargo test ... free_float_strict_gate -- --nocapture` log that
carries the engine's FFAUDIT_P/_Z/_C/_D lines (env-gated, read-only dump in
`step_physics_5r1c`) and reconstructs the hourly component energy balance of
the air node and the mass node on the 5R1C per-class path (600FF / 650FF;
the 900/950FF pair runs the 9R4C model family and emits no FFAUDIT lines).

Emits per case:
- closure residual stats (air node, mass node) — the convention error of the
  hourly trapezoid reconstruction, expected small against the sub-step CN
  integrator's own exactness;
- class-storage cross-check (sum of FFAUDIT_D vs cm·ΔT from FFAUDIT_Z) —
  validates the instrumentation itself;
- annual integral per component bucket, and the top swing days' integrated
  buckets + hour-by-hour table — the attribution surface for the missing
  diurnal swing.

Usage: python3 scripts/audit_free_float_balance.py <ff_audit.log> [--case 600FF]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict

DT_S = 3600.0
CASE_MARK = re.compile(r"\[#4170 Case (\S+) strict\]")


def parse_kv(tokens: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = float(v)
    return out


def parse_log(path: str) -> tuple[dict[str, dict[int, dict]], dict[str, dict[int, dict[int, float]]]]:
    """Walk the log; commit each case's rows at its end-of-run marker.

    Z/P rows: last write per timestep wins (warm-up passes repeat 0..8759;
    the final pass is the reported year). P rows merge into the Z row of the
    same timestep (P is emitted earlier inside the step). D rows keep the
    per-class storage deltas (kJ).
    """
    cases: dict[str, dict[int, dict]] = {}
    zrows: dict[int, dict] = {}
    prows: dict[int, dict] = {}
    drows: dict[int, dict[int, float]] = {}
    dcase: dict[str, dict[int, dict[int, float]]] = {}

    def commit(name: str) -> None:
        merged: dict[int, dict] = {}
        for ts, row in zrows.items():
            merged[ts] = dict(row)
        for ts, phi in prows.items():
            merged.setdefault(ts, {"ts": ts}).update(phi)
        cases[name] = merged
        dcase[name] = drows

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = CASE_MARK.search(line)
            if m:
                commit(m.group(1))
                zrows, prows, drows = {}, {}, {}
                continue
            if "FFAUDIT" not in line:
                continue
            t = line.split()
            if t[0] == "FFAUDIT_Z":
                ts = int(t[1])
                row = parse_kv(t[3:])
                row["ts"] = ts
                prev = prows.pop(ts, None)
                if prev:
                    row.update(prev)
                zrows[ts] = row
            elif t[0] == "FFAUDIT_P":
                ts = int(t[1])
                phi = {k: float(v.split(",")[0]) for k, v in
                       (kv.split("=", 1) for kv in t[2:] if "=" in kv)}
                if ts in zrows:
                    zrows[ts].update(phi)
                else:
                    prows[ts] = phi
            elif t[0] == "FFAUDIT_D":
                ts, z, c = int(t[1]), int(t[2]), int(t[3])
                if z == 0:
                    drows.setdefault(ts, {})[c] = float(t[4]) / 1e3
    return cases, dcase


def balance(rows: list[dict]) -> list[dict]:
    """Hourly component energies (kJ) with trapezoid mid-state fluxes."""
    recs: list[dict] = []
    prev: dict | None = None
    for r in rows:
        rec = dict(r)
        rec["hod"] = r["ts"] % 24
        if prev is None:
            rec.update(storage_air=0.0, storage_m=0.0, vent_air=0.0, ext_air=0.0,
                       ground_air=0.0, to_mass=0.0, gains_air=r["phi_ia"] * DT_S / 1e3,
                       ext_m=0.0, vent_m=0.0, from_air=0.0, gains_m=r["phi_m"] * DT_S / 1e3)
            recs.append(rec)
            prev = r
            continue
        t_air_mid = 0.5 * (r["t_air"] + prev["t_air"])
        t_m_mid = 0.5 * (r["t_m_new"] + prev["t_m_new"])
        # Basis note: derived_h_ext / derived_h_tr_3 (and the night-vent
        # addition) are physical W/K — the step multiplies num/den by
        # term_rest_1 AFTER them. derived_ground_coeff is added to the
        # already-scaled numerator, so it is the one coefficient that
        # carries the term_rest_1 factor.
        tr1 = r["term_rest_1"]
        h_ext_phys = r["h_ext"]
        h_am_phys = r["h_air_mass"]
        g_phys = r["g_coeff"] / tr1
        h_ext_vent = r["h_ve_night"] if r["nv"] > 0.5 else 0.0
        h_ext_base = h_ext_phys - h_ext_vent
        rec.update(
            vent_air=h_ext_vent * DT_S * (r["t_out"] - t_air_mid) / 1e3,
            ext_air=h_ext_base * DT_S * (r["t_out"] - t_air_mid) / 1e3,
            ground_air=g_phys * DT_S * (r["t_g"] - t_air_mid) / 1e3,
            to_mass=-h_am_phys * DT_S * (t_air_mid - t_m_mid) / 1e3,
            gains_air=r["phi_ia"] * DT_S / 1e3,
            ext_m=r["h_tr_em"] * DT_S * (r["t_sol_air"] - t_m_mid) / 1e3,
            # The mass-side vent coupling rides h3 into the CN call, whose
            # supply temperature is the sub-step zone air (contract 7 pins
            # the routing; the BE/FE fallback branches of `advance_node`
            # would drive it to outdoor — not taken for realistic masses).
            vent_m=r["h_vent_mass"] * DT_S * (t_air_mid - t_m_mid) / 1e3,
            from_air=h_am_phys * DT_S * (t_air_mid - t_m_mid) / 1e3,
            gains_m=r["phi_m"] * DT_S / 1e3,
            storage_air=r["c_air"] * (r["t_air"] - prev["t_air"]) / 1e3,
            storage_m=r["cm"] * (r["t_m_new"] - r["tm_start"]) / 1e3,
        )
        recs.append(rec)
        prev = r
    return recs


def table(headers: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), "-|-".join("-" * w for w in widths)]
    out += [fmt.format(*[str(c) for c in r]) for r in rows]
    return "\n".join(out)


BUCKETS = ("gains_air", "gains_m", "vent_air", "vent_m", "ext_air", "ext_m",
           "ground_air", "to_mass", "storage_air", "storage_m")


def report(case: str, by_ts: dict[int, dict], drows: dict[int, dict[int, float]]) -> str:
    rows = [by_ts[ts] for ts in sorted(by_ts) if "t_air" in by_ts[ts] and "phi_ia" in by_ts[ts]]
    recs = balance(rows)
    full = recs[1:] or recs

    res_air = [r["storage_air"] - (r["vent_air"] + r["ext_air"] + r["ground_air"]
                                   + r["to_mass"] + r["gains_air"]) for r in full]
    res_m = [r["storage_m"] - (r["ext_m"] + r["vent_m"] + r["from_air"] + r["gains_m"])
             for r in full]

    def stats(vals: list[float]) -> tuple[float, float, float]:
        a = [abs(x) for x in vals]
        return (min(a), statistics.median(a), max(a)) if a else (0.0, 0.0, 0.0)

    sa, sm = stats(res_air), stats(res_m)
    # scale-relative closure: |res| as a share of the hour's mean |flux|
    def rel_res(recs_list: list[dict], keys: tuple[str, ...], res: list[float]) -> list[float]:
        out = []
        for r, e in zip(recs_list, res):
            scale = statistics.mean(abs(r[k]) for k in keys)
            out.append(abs(e) / scale if scale > 1e-9 else 0.0)
        return out

    rel_air = rel_res(full, ("vent_air", "ext_air", "ground_air", "to_mass", "gains_air"), res_air)
    rel_m = rel_res(full, ("ext_m", "vent_m", "from_air", "gains_m"), res_m)
    ra, rm_ = stats(rel_air), stats(rel_m)
    xchk = [abs(sum(drows.get(r["ts"], {}).values()) - r["storage_m"])
            for r in full if drows.get(r["ts"])]
    xchk_max = max(xchk) if xchk else float("nan")

    lines = [f"## Case {case}", ""]
    lines += [table(
        ["closure node", "|res| med (kJ/h)", "|res| max (kJ/h)", "med |res|/mean|flux|", "max |res|/mean|flux|"],
        [["air", f"{sa[1]:.2f}", f"{sa[2]:.2f}", f"{ra[1]*100:.1f}%", f"{ra[2]*100:.1f}%"],
         ["mass", f"{sm[1]:.2f}", f"{sm[2]:.2f}", f"{rm_[1]*100:.1f}%", f"{rm_[2]*100:.1f}%"]]),
        "",
        "(air/mass reconstructions are hourly trapezoid integrals over the committed end-of-hour",
        "states; the residual is the convention gap against the sub-step CN integrator, reported",
        "per node both absolute and relative to the hour's mean flux magnitude.)", ""]
    lines += [f"class-storage cross-check: max |sum(FFAUDIT_D) - cm*dT| = {xchk_max:.6f} kJ "
              f"over {len(xchk)} hours", ""]

    tot = {b: sum(r[b] for r in full) / 1e3 for b in BUCKETS}
    lines += ["### Annual integral (MJ)", "",
              table(["bucket", *BUCKETS], [["MJ", *[f"{tot[b]:.1f}" for b in BUCKETS]]]), ""]

    day: dict[int, list[dict]] = defaultdict(list)
    for r in full:
        day[r["ts"] // 24].append(r)
    by_range = max(day.items(), key=lambda kv: max(x["t_air"] for x in kv[1]) - min(x["t_air"] for x in kv[1]))
    by_peak = max(day.items(), key=lambda kv: max(x["t_air"] for x in kv[1]))
    swing = [by_range, by_peak]

    for d, drecs in swing:
        rng = max(x["t_air"] for x in drecs) - min(x["t_air"] for x in drecs)
        tag = "highest peak" if drecs is by_peak[1] else "largest range"
        lines += [f"### Swing day {d} (day {d + 1} of year, {tag}): T_air range {rng:.2f} K "
                  f"(min {min(x['t_air'] for x in drecs):.2f}, max {max(x['t_air'] for x in drecs):.2f})",
                  "",
                  table(["bucket (MJ)", "air node", "mass node"],
                        [["exterior (sol-air/dry-bulb)", f"{sum(x['ext_air'] for x in drecs)/1e3:.2f}",
                          f"{sum(x['ext_m'] for x in drecs)/1e3:.2f}"],
                         ["vent exchange", f"{sum(x['vent_air'] for x in drecs)/1e3:.2f}",
                          f"{sum(x['vent_m'] for x in drecs)/1e3:.2f}"],
                         ["gains (phi)", f"{sum(x['gains_air'] for x in drecs)/1e3:.2f}",
                          f"{sum(x['gains_m'] for x in drecs)/1e3:.2f}"],
                         ["air<->mass exchange", f"{sum(x['to_mass'] for x in drecs)/1e3:.2f}",
                          f"{sum(x['from_air'] for x in drecs)/1e3:.2f}"],
                         ["storage change", f"{sum(x['storage_air'] for x in drecs)/1e3:.2f}",
                          f"{sum(x['storage_m'] for x in drecs)/1e3:.2f}"],
                         ["ground", f"{sum(x['ground_air'] for x in drecs)/1e3:.2f}", "-"],
                         ["phi_st (surface pools, MJ)", f"{sum(x['phi_st'] for x in drecs)/1e3:.2f}", "-"]]),
                  "",
                  table(["hod", "t_out", "t_air", "t_m", "sol-air", "phi_ia", "phi_m", "phi_st",
                         "stor_m", "stor_air"],
                        [[f"{x['hod']:d}", f"{x['t_out']:.1f}", f"{x['t_air']:.2f}", f"{x['t_m_new']:.2f}",
                          f"{x['t_sol_air']:.1f}", f"{x['phi_ia']:.0f}", f"{x['phi_m']:.0f}",
                          f"{x['phi_st']:.0f}", f"{x['storage_m']:.0f}", f"{x['storage_air']:.0f}"]
                         for x in drecs]),
                  ""]

    top = swing[0][1]
    lines += ["### Class storage split on the first swing day (from FFAUDIT_D, kJ)",
              "",
              table(["class", "kJ (day total)"],
                    [[f"class {c}", f"{sum(drows.get(x['ts'], {}).get(c, 0.0) for x in top):.1f}"]
                     for c in (0, 1, 2)]) if drows else ["(no D rows)", "-"], ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--case", default=None, help="only this case")
    args = ap.parse_args()

    cases, dcase = parse_log(args.log)
    if args.case:
        cases = {k: v for k, v in cases.items() if k == args.case}
    if not cases:
        print("no FFAUDIT rows found", file=sys.stderr)
        return 1
    for name in sorted(cases):
        print(report(name, cases[name], dcase.get(name, {})))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
