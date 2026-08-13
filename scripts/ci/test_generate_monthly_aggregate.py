"""
Tests for ``scripts/generate_monthly_aggregate.py`` -- Issue #2748.

The aggregator consumes hourly EnergyPlus CSV exports (produced by
``generate_case_600_900_energy.py`` and friends) and emits monthly heating
/cooling rollups in the format consumed by the Phase D ±10% gate in
``tests/ashrae_140_blind_validation.rs::test_monthly_energy_validation_baseline``.

The tests below pin:

* ``hour_to_month_idx`` -- the hour-of-year -> calendar-month mapping that
  must agree with ``MONTH_START_HOUR`` in the Rust harness
  (``tests/ashrae_140_blind_validation.rs:100``). Boundary hours are the
  load-bearing cases (off-by-one = wrong month bucket = wrong annual sum).
* ``acceptance_window`` -- the ±10% Phase D band. Zero midpoints collapse
  to (0, 0); the Rust harness skips them via
  ``if ref_mid <= 1e-6 { continue; }``.
* ``parse_hourly_csv`` -- tolerant comment / header / multi-zone handling.
  A real E+ 25.2.0 export has 8760 rows (single-zone) or 17520 rows
  (multi-zone); the parser must accept both and the aggregator must sum
  across zones for Cases 950/960.
* ``aggregate_monthly`` -- sum identity (Σ month == annual), non-negativity
  clamp, and the per-month tolerance window.
* ``write_monthly_csv`` -- format round-trip: the written CSV must be
  re-loadable and produce the same totals.

The fixtures live under ``tmp_path``; we do NOT touch the real
``tests/reference_data/ashrae140/monthly/`` files. The end-to-end
``regenerate_case`` path is exercised separately against the in-repo
Case 920 hourly export to confirm the round-trip works against a real
file (this is a one-line sanity check, not a gate).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_monthly_aggregate.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module():
    """Load generate_monthly_aggregate.py as a fresh module (per the
    ``scripts/ci/conftest.py::load_script`` pattern, but we need the module
    in two flavours — one for unit tests, one for the repo-rooted path
    constants — so inline it here instead of using ``load_script``).

    Registers the module in ``sys.modules`` so ``@dataclass`` can look it
    up (it uses ``sys.modules[cls.__module__].__dict__`` to resolve
    type-name conflicts across re-imports of the same file).
    """
    name = "generate_monthly_aggregate"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def agg():
    return _load_module()


# ---------------------------------------------------------------------------
# hour_to_month_idx
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hour,expected_month,expected_label",
    [
        (0, 0, "Jan"),
        (1, 0, "Jan"),
        (743, 0, "Jan"),
        (744, 1, "Feb"),
        (745, 1, "Feb"),
        (1415, 1, "Feb"),
        (1416, 2, "Mar"),
        (2160, 3, "Apr"),
        (2880, 4, "May"),
        (3624, 5, "Jun"),
        (4344, 6, "Jul"),
        (5088, 7, "Aug"),
        (5832, 8, "Sep"),
        (6552, 9, "Oct"),
        (7296, 10, "Nov"),
        (8016, 11, "Dec"),
        (8759, 11, "Dec"),
    ],
)
def test_hour_to_month_idx_boundaries(agg, hour, expected_month, expected_label):
    assert agg.hour_to_month_idx(hour) == expected_month
    assert agg.MONTH_LABELS[agg.hour_to_month_idx(hour)] == expected_label


@pytest.mark.parametrize("bad_hour", [-1, 8760, 9000, 100_000])
def test_hour_to_month_idx_out_of_range_raises(agg, bad_hour):
    with pytest.raises(ValueError, match="outside 0..8759"):
        agg.hour_to_month_idx(bad_hour)


def test_hour_to_month_idx_first_hour_of_each_month(agg):
    """Each MONTH_START_HOUR boundary should map to its own month."""
    for m, start in enumerate(agg.MONTH_START_HOUR[:-1]):
        assert agg.hour_to_month_idx(start) == m, (
            f"hour {start} (start of {agg.MONTH_LABELS[m]}) mapped to "
            f"month {agg.hour_to_month_idx(start)} instead of {m}"
        )


# ---------------------------------------------------------------------------
# acceptance_window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mid,expected",
    [
        (0.0, (0.0, 0.0)),  # zero midpoint collapses (structurally zero months)
        (1e-12, (0.0, 0.0)),  # sub-noise midpoint also collapses
        (1e-3, (9e-4, 1.1e-3)),  # small but non-zero midpoint
        (1.0, (0.9, 1.1)),
        (5.075, (4.5675, 5.5825)),
    ],
)
def test_acceptance_window(agg, mid, expected):
    lo, hi = agg.acceptance_window(mid)
    assert lo == pytest.approx(expected[0])
    assert hi == pytest.approx(expected[1])


def test_acceptance_window_phase_d_default_is_10pct(agg):
    """Issue #668 / #1165: Phase D tolerance is ±10%. Pin it so a future
    drift (e.g. changing the default to ±15%) is caught by the test rather
    than silently relaxing the gate.
    """
    assert agg.PHASE_D_TOLERANCE == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# parse_hourly_csv
# ---------------------------------------------------------------------------


def _write_hourly_csv(path: Path, rows: list[tuple[int, float, float]]) -> None:
    """Write a synthetic case_<id>_energy_hourly.csv with the canonical header."""
    with path.open("w", newline="") as f:
        f.write(
            "# EnergyPlus Version: 25.2.0,# Synthetic test fixture,# "
            "EPW: USA_CO_Golden-NREL.724666_TMY3.epw,# Columns: hour,T_zone(C),T_out(C),Q_heat(W),Q_cool(W)\n"
        )
        f.write("hour,T_zone(C),T_out(C),Q_heat(W),Q_cool(W)\n")
        for hour, qh, qc in rows:
            f.write(f"{hour},20.0,-5.0,{qh},{qc}\n")


def test_parse_hourly_csv_strips_comments_and_header(agg, tmp_path):
    rows = [(h, 100.0, 0.0) for h in range(1, 11)]
    p = tmp_path / "hourly.csv"
    _write_hourly_csv(p, rows)
    parsed = agg.parse_hourly_csv(p)
    assert [r.hour for r in parsed] == list(range(1, 11))
    assert all(r.q_heat_w == 100.0 for r in parsed)
    assert all(r.q_cool_w == 0.0 for r in parsed)


def test_parse_hourly_csv_clamps_negative_to_zero(agg, tmp_path):
    """Tiny negative numbers from E+ round-off must not poison the monthly sum."""
    rows = [(1, 1e-12, -5e-3), (2, -1e-3, 100.0), (3, 0.0, 0.0)]
    p = tmp_path / "hourly.csv"
    _write_hourly_csv(p, rows)
    parsed = agg.parse_hourly_csv(p)
    assert [r.q_heat_w for r in parsed] == [1e-12, 0.0, 0.0]
    assert [r.q_cool_w for r in parsed] == [0.0, 100.0, 0.0]


def test_parse_hourly_csv_multizone_accepts_17520_rows(agg, tmp_path):
    """Cases 950/960 are 17520 rows (8760 × 2 zones); the parser must
    accept all rows and the aggregator must sum across zones.
    """
    rows = [(h, 50.0, 0.0) for h in range(1, 8761)]
    rows += [(h, 50.0, 0.0) for h in range(1, 8761)]  # zone 2
    p = tmp_path / "multi.csv"
    _write_hourly_csv(p, rows)
    parsed = agg.parse_hourly_csv(p)
    assert len(parsed) == 17520
    totals = agg.aggregate_monthly(parsed)
    # Each hour contributes 50 W heating for 1 h = 50 Wh; 17520 × 50 Wh
    # = 876000 Wh = 0.876 MWh.
    assert totals.annual_heating_mwh == pytest.approx(0.876)
    assert totals.annual_cooling_mwh == pytest.approx(0.0)


def test_parse_hourly_csv_rejects_missing_data(agg, tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        agg.parse_hourly_csv(p)


def test_parse_hourly_csv_rejects_short_row(agg, tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "# hdr\nhour,T_zone(C),T_out(C),Q_heat(W),Q_cool(W)\n1,20,5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected >="):
        agg.parse_hourly_csv(p)


# ---------------------------------------------------------------------------
# aggregate_monthly
# ---------------------------------------------------------------------------


def _full_year_constant_power(agg, q_heat_w: float, q_cool_w: float) -> list:
    """Return a synthetic 8760-hour series with constant Q_heat / Q_cool."""
    return [
        agg.HourlyRow(hour=h, q_heat_w=q_heat_w, q_cool_w=q_cool_w)
        for h in range(1, 8761)
    ]


def test_aggregate_monthly_constant_heating_sums_to_annual(agg):
    # 1000 W × 8760 h = 8.76 MWh
    rows = _full_year_constant_power(agg, 1000.0, 0.0)
    totals = agg.aggregate_monthly(rows)
    assert totals.annual_heating_mwh == pytest.approx(8.76, rel=1e-6)
    # Each month holds (hours_in_month × 1000 W) / 1e6 MWh. The months
    # are 28-31 days so the per-month total varies (Feb=0.672 MWh,
    # others 0.720-0.744 MWh); verify per-month against MONTH_START_HOUR
    # rather than naively dividing 8.76/12 (which would assume 30-day months).
    for m in range(12):
        hours_in_month = agg.MONTH_START_HOUR[m + 1] - agg.MONTH_START_HOUR[m]
        expected_mwh = hours_in_month * 1000.0 / 1e6
        assert totals.heating_mwh[m] == pytest.approx(expected_mwh, rel=1e-6), (
            f"month {agg.MONTH_LABELS[m]} ({hours_in_month} h) expected "
            f"{expected_mwh:.4f} MWh, got {totals.heating_mwh[m]:.4f}"
        )
    assert all(m == 0.0 for m in totals.cooling_mwh)


def test_aggregate_monthly_constant_cooling_sums_to_annual(agg):
    # 500 W × 8760 h = 4.38 MWh
    rows = _full_year_constant_power(agg, 0.0, 500.0)
    totals = agg.aggregate_monthly(rows)
    assert totals.annual_cooling_mwh == pytest.approx(4.38, rel=1e-6)
    for m in range(12):
        hours_in_month = agg.MONTH_START_HOUR[m + 1] - agg.MONTH_START_HOUR[m]
        expected_mwh = hours_in_month * 500.0 / 1e6
        assert totals.cooling_mwh[m] == pytest.approx(expected_mwh, rel=1e-6), (
            f"month {agg.MONTH_LABELS[m]} ({hours_in_month} h) expected "
            f"{expected_mwh:.4f} MWh, got {totals.cooling_mwh[m]:.4f}"
        )
    assert all(m == 0.0 for m in totals.heating_mwh)


def test_aggregate_monthly_bucket_assignment_january_only(agg):
    """1000 W for the first 744 hours (Jan) only — January bucket should
    hold all 744 Wh, every other bucket zero."""
    rows = []
    for h in range(1, 745):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=1000.0, q_cool_w=0.0))
    for h in range(745, 8761):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=0.0, q_cool_w=0.0))
    totals = agg.aggregate_monthly(rows)
    assert totals.heating_mwh[0] == pytest.approx(744.0 / 1000.0)  # 0.744 MWh
    assert all(m == 0.0 for m in totals.heating_mwh[1:])


def test_aggregate_monthly_bucket_assignment_february_only(agg):
    """Hours 745..1416 should land in February (idx 1)."""
    rows = []
    for h in range(745, 1417):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=1000.0, q_cool_w=0.0))
    for h in list(range(1, 745)) + list(range(1417, 8761)):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=0.0, q_cool_w=0.0))
    totals = agg.aggregate_monthly(rows)
    # Feb is hours 745..1416 inclusive = 672 hours
    assert totals.heating_mwh[1] == pytest.approx(672.0 / 1000.0)
    assert all(m == 0.0 for m in totals.heating_mwh[:1])
    assert all(m == 0.0 for m in totals.heating_mwh[2:])


def test_aggregate_monthly_december_includes_hour_8760(agg):
    """Hours 8017..8760 (672 hours) belong to December."""
    rows = []
    for h in range(8017, 8761):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=1000.0, q_cool_w=0.0))
    for h in range(1, 8017):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=0.0, q_cool_w=0.0))
    totals = agg.aggregate_monthly(rows)
    # 8017..8760 = 744 hours (Dec has 31 days × 24 = 744 hours)
    assert totals.heating_mwh[11] == pytest.approx(744.0 / 1000.0)


# ---------------------------------------------------------------------------
# build_csv_rows / write_monthly_csv / round-trip
# ---------------------------------------------------------------------------


def test_build_csv_rows_emits_header_and_12_months(agg):
    rows = _full_year_constant_power(agg, 0.0, 0.0)
    totals = agg.aggregate_monthly(rows)
    csv_rows = agg.build_csv_rows(totals)
    assert csv_rows[0] == [
        "month",
        "heating_mid_mwh",
        "heating_accept_min_mwh",
        "heating_accept_max_mwh",
        "cooling_mid_mwh",
        "cooling_accept_min_mwh",
        "cooling_accept_max_mwh",
    ]
    assert len(csv_rows) == 13  # header + 12 months
    assert [r[0] for r in csv_rows[1:]] == list(agg.MONTH_LABELS)


def test_write_monthly_csv_round_trip(agg, tmp_path):
    """Write a synthetic monthly CSV and re-parse it: every field must
    reproduce the in-memory totals.
    """
    rows = []
    for h in range(1, 8761):
        rows.append(agg.HourlyRow(hour=h, q_heat_w=2000.0, q_cool_w=500.0))
    totals = agg.aggregate_monthly(rows)
    out = tmp_path / "monthly.csv"
    agg.write_monthly_csv("999", totals, out)
    txt = out.read_text(encoding="utf-8")
    # Header check
    assert "month,heating_mid_mwh" in txt
    # 12 month rows
    month_lines = [
        ln for ln in txt.splitlines() if ln.startswith(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
    ]
    assert len(month_lines) == 12
    # Spot-check January: heating = 2000 W × 744 h = 1.488 MWh
    jan_line = next(ln for ln in month_lines if ln.startswith("Jan"))
    cells = jan_line.split(",")
    assert float(cells[1]) == pytest.approx(1.488, abs=1e-3)  # heating_mid
    # Tolerance window = ±10%
    assert float(cells[2]) == pytest.approx(1.488 * 0.90, abs=1e-3)
    assert float(cells[3]) == pytest.approx(1.488 * 1.10, abs=1e-3)


# ---------------------------------------------------------------------------
# End-to-end smoke against the real Case 920 hourly export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REPO_ROOT / "tests/reference_data/zone_balance/case_920_energy_hourly.csv").exists(),
    reason="Case 920 hourly CSV not present in this checkout",
)
def test_end_to_end_case_920_sums_within_annual_band(agg):
    """The aggregator must reproduce the Case 920 annual band when run
    against the real E+ hourly export: Σ(heating) ∈ [3.26, 4.30] MWh,
    Σ(cooling) ∈ [1.84, 3.31] MWh. The bands live in
    ``tests/reference_data/zone_balance/case_920_energy_reference.csv``.
    """
    src = REPO_ROOT / "tests/reference_data/zone_balance/case_920_energy_hourly.csv"
    rows = agg.parse_hourly_csv(src)
    totals = agg.aggregate_monthly(rows)
    assert 3.26 <= totals.annual_heating_mwh <= 4.30, (
        f"Case 920 Σ(heating) = {totals.annual_heating_mwh:.4f} MWh outside [3.26, 4.30]"
    )
    assert 1.84 <= totals.annual_cooling_mwh <= 3.31, (
        f"Case 920 Σ(cooling) = {totals.annual_cooling_mwh:.4f} MWh outside [1.84, 3.31]"
    )
    # 8760 rows = 1 zone
    assert len(rows) == 8760
