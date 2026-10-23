//! Addendum a daily-hourly output leg for the ASHRAE 140 Section 5.2.1 / 5.2.2
//! fabric (envelope) cases — Issue #4268.
//!
//! # Requirement (in-repo source)
//!
//! ANSI/ASHRAE Standard 140-2017, **Addendum a** (in-repo copy
//! `docs/140_2017_a_20200901.pdf`), §6.2.1.8 "Daily Hourly Output for Building
//! Thermal Envelope and Fabric Load Tests of Sections 5.2.1 and 5.2.2" and
//! **Table 6-1** "Daily Hourly Output Requirements for Building Thermal Envelope
//! and Fabric Load Tests of Sections 5.2.1 and 5.2.2" (PDF p. 65 of the addendum
//! text) require, for the 600-series (low mass) and 900-series (high mass) fabric
//! cases, hourly integrated heating (+) / sensible cooling (−) loads in kWh and
//! hourly integrated zone air temperature in °C on named representative days.
//! Table 6-1 footnote **a** fixes the reporting granularity:
//!
//! > "Hourly data to consist of 24 hourly integrated values for each day. The
//! > first hour (hour 1) is defined to run from 0000 to 0100 hours."
//!
//! §6.2.1.8 also forbids short runs: "Running the simulation for only the required
//! days specified in the output requirements shall be prohibited, because the
//! results could contain temperature history errors." The series consumed here is
//! therefore always the full 8760-hour annual run — this module only *slices* the
//! annual series down to the representative days, it never re-simulates.
//!
//! The two representative days exercised by the fabric validator are
//! **January 4** and **July 14** (both named in the Table 6-1 rows covering
//! Cases 600/660/670/680/685/695/900/980/985/995 for the load outputs and in the
//! free-float zone-air-temperature rows).
//!
//! # Bands
//!
//! Addendum a publishes **requirements**, not example-result bands: its
//! informative example results live in the accompanying electronic files
//! (`RESULTS5-2A.XLSX`; see the addendum's Notes to Section 6.2.1), which are
//! **not** in this repository. Every hourly metric is therefore recorded here as
//! an explicit **known-fail / not-yet-banded** row with its per-metric source
//! citation. Numeric bands are never invented; a row without an in-repo numeric
//! band carries `band_min/band_max = None` and `known_fail = true`, so the report
//! shows the row instead of silently dropping it. This follows the provenance rule
//! established for issue #4168 in `scripts/check_benchmark_band_provenance.py`:
//! every band row names its source document **and** table, or is explicitly
//! marked as having no published band.

use serde::{Deserialize, Serialize};

/// Source citation for the hourly-output *requirement* (document + section +
/// table). Attached to every emitted hourly report and band row.
pub const REQUIREMENT_CITATION: &str = "ANSI/ASHRAE Standard 140-2017, Addendum a \
    (docs/140_2017_a_20200901.pdf), Section 6.2.1.8 and Table 6-1 'Daily Hourly \
    Output Requirements for Building Thermal Envelope and Fabric Load Tests of \
    Sections 5.2.1 and 5.2.2' (addendum PDF p. 65)";

/// Citation for the 24-hour granularity rule (Table 6-1, footnote a).
pub const GRANULARITY_CITATION: &str = "ANSI/ASHRAE Standard 140-2017, Addendum a \
    (docs/140_2017_a_20200901.pdf), Table 6-1, footnote a (addendum PDF p. 65): \
    'Hourly data to consist of 24 hourly integrated values for each day. The first \
    hour (hour 1) is defined to run from 0000 to 0100 hours.'";

/// Provenance note explaining why every hourly row is known-fail: the addendum's
/// informative example results are not in-repo.
pub const NO_IN_REPO_BAND_NOTE: &str = "No numeric band available in-repo: Addendum a \
    (docs/140_2017_a_20200901.pdf) states the hourly OUTPUT requirement (Section \
    6.2.1.8, Table 6-1) but publishes the updated informative example results only \
    in the accompanying electronic file (RESULTS5-2A.XLSX), which is not part of \
    this repository. Row recorded as an explicit known-fail (not-yet-banded) row; no \
    numeric band is asserted. Reference requirement: ANSI/ASHRAE Standard 140-2017, \
    Addendum a, Section 6.2.1.8 / Table 6-1.";

/// Number of hours in one representative day (Table 6-1, footnote a).
pub const HOURS_PER_REPORT_DAY: usize = 24;

/// Annual steps of the validator's hourly timesteps (§5.1.7 annual run).
pub const ANNUAL_HOURS: usize = 8760;

/// Fabric family a case belongs to (Section 5.2.1 low mass vs 5.2.2 high mass).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FabricFamily {
    /// 600-series: low mass, lightweight construction (Section 5.2.1).
    LowMass600,
    /// 900-series: high mass, heavyweight construction (Section 5.2.2).
    HighMass900,
}

impl FabricFamily {
    /// Short label used in report rows.
    pub fn label(&self) -> &'static str {
        match self {
            FabricFamily::LowMass600 => "600-series (low mass)",
            FabricFamily::HighMass900 => "900-series (high mass)",
        }
    }
}

/// One representative day of the Addendum a Table 6-1 hourly output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentativeDay {
    /// Month number (1-12).
    pub month: u32,
    /// Day of month (1-31).
    pub day: u32,
}

impl RepresentativeDay {
    /// Human label, e.g. `Jan 4`.
    pub fn label(&self) -> String {
        format!("{} {}", month_abbrev(self.month), self.day)
    }

    /// 1-based day-of-year for a non-leap year (the validator's TMY3 year).
    pub fn day_of_year(&self) -> usize {
        let mut doy = 0usize;
        for m in 1..self.month {
            doy += days_in_month(m) as usize;
        }
        doy + self.day as usize
    }

    /// Index of hour 1 (0000-0100) of this day in the annual hourly series.
    pub fn first_hour_index(&self) -> usize {
        (self.day_of_year() - 1) * HOURS_PER_REPORT_DAY
    }
}

/// January 4 — named in the Table 6-1 load row for the 600/900-series fabric
/// cases and in the free-float zone-air-temperature row (600FF/680FF/900FF/980FF).
pub const JAN_4: RepresentativeDay = RepresentativeDay { month: 1, day: 4 };

/// July 14 — named in the Table 6-1 load row for the 600/900-series fabric cases
/// and in the 650FF/950FF free-float zone-air-temperature row.
pub const JUL_14: RepresentativeDay = RepresentativeDay { month: 7, day: 14 };

/// The representative days this leg reports (Addendum a Table 6-1).
pub const REPRESENTATIVE_DAYS: [RepresentativeDay; 2] = [JAN_4, JUL_14];

/// Fabric cases the hourly leg covers by default: one baseline low-mass case,
/// one setback low-mass case, the matching high-mass pair, and the baseline
/// free-float variant of each baseline.
pub const FABRIC_CASES: [&str; 6] = ["600", "640", "900", "940", "600FF", "900FF"];

/// Metrics Addendum a Table 6-1 asks for on the fabric representative days.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HourlyMetric {
    /// Hourly integrated zone air temperature (°C).
    ZoneAirTemperature,
    /// Hourly integrated heating load (kWh, reported positive).
    HeatingLoad,
    /// Hourly integrated sensible cooling load (kWh, reported positive magnitude).
    SensibleCoolingLoad,
}

impl HourlyMetric {
    /// Stable metric key used in report rows and CSV output.
    pub fn key(&self) -> &'static str {
        match self {
            HourlyMetric::ZoneAirTemperature => "hourly_integrated_zone_air_temperature_c",
            HourlyMetric::HeatingLoad => "heating_kwh",
            HourlyMetric::SensibleCoolingLoad => "sensible_cooling_kwh",
        }
    }

    /// Unit of the metric, per Table 6-1.
    pub fn unit(&self) -> &'static str {
        match self {
            HourlyMetric::ZoneAirTemperature => "degC",
            HourlyMetric::HeatingLoad => "kWh",
            HourlyMetric::SensibleCoolingLoad => "kWh",
        }
    }

    /// Human label used in the markdown leg.
    pub fn label(&self) -> &'static str {
        match self {
            HourlyMetric::ZoneAirTemperature => "Hourly integrated zone air temperature (°C)",
            HourlyMetric::HeatingLoad => "Hourly integrated heating (+) load (kWh)",
            HourlyMetric::SensibleCoolingLoad => {
                "Hourly integrated sensible cooling (−) load (kWh)"
            }
        }
    }

    /// Per-metric citation into Addendum a Table 6-1.
    pub fn citation(&self) -> &'static str {
        match self {
            HourlyMetric::ZoneAirTemperature => {
                "ANSI/ASHRAE Standard 140-2017, Addendum a (docs/140_2017_a_20200901.pdf), \
                 Table 6-1 'Hourly integrated free-float / conditioned zone air temperature \
                 (°C)' rows, Cases 600FF, 680FF, 900FF, 980FF, 650FF, 950FF (addendum PDF p. 65)"
            }
            HourlyMetric::HeatingLoad => {
                "ANSI/ASHRAE Standard 140-2017, Addendum a (docs/140_2017_a_20200901.pdf), \
                 Table 6-1 'Hourly integrated heating (+) and sensible cooling (−) loads (kWh)' \
                 row, Cases 600, 660, 670, 680, 685, 695, 900, 980, 985, 995 \
                 (addendum PDF p. 65)"
            }
            HourlyMetric::SensibleCoolingLoad => {
                "ANSI/ASHRAE Standard 140-2017, Addendum a (docs/140_2017_a_20200901.pdf), \
                 Table 6-1 'Hourly integrated heating (+) and sensible cooling (−) loads (kWh, \
                 designate cooling with a [−] sign)' row, Cases 600, 660, 670, 680, 685, 695, \
                 900, 980, 985, 995 (addendum PDF p. 65)"
            }
        }
    }

    /// Extract this metric's values from one representative day's samples.
    pub fn sample_values(&self, day: &HourlyDayOutput) -> Vec<Option<f64>> {
        day.samples
            .iter()
            .map(|s| match self {
                HourlyMetric::ZoneAirTemperature => s.zone_air_temp_c,
                HourlyMetric::HeatingLoad => s.heating_kwh(),
                HourlyMetric::SensibleCoolingLoad => s.cooling_kwh(),
            })
            .collect()
    }
}

/// The metrics Table 6-1 requires for a case on the representative days.
pub fn metrics_for_case(case_id: &str) -> Vec<HourlyMetric> {
    let _ = case_id;
    vec![
        HourlyMetric::ZoneAirTemperature,
        HourlyMetric::HeatingLoad,
        HourlyMetric::SensibleCoolingLoad,
    ]
}

/// Provenance status of a band row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandStatus {
    /// A numeric band exists in-repo and sampled values were compared to it.
    Banded,
    /// No in-repo numeric band: recorded explicitly as a known-fail row.
    KnownFailNotBanded,
}

impl BandStatus {
    /// Display token used in report rows.
    pub fn label(&self) -> &'static str {
        match self {
            BandStatus::Banded => "BANDED",
            BandStatus::KnownFailNotBanded => "KNOWN-FAIL (not-yet-banded)",
        }
    }
}

/// One hourly acceptance band row (per case, day and metric).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyBandRow {
    /// Case the row belongs to (e.g. `600`, `900FF`).
    pub case_id: String,
    /// Fabric family of the case.
    pub family: FabricFamily,
    /// Representative day the band applies to.
    pub day: RepresentativeDay,
    /// Metric the band applies to.
    pub metric: HourlyMetric,
    /// Lower band bound, or `None` when no in-repo band exists.
    pub band_min: Option<f64>,
    /// Upper band bound, or `None` when no in-repo band exists.
    pub band_max: Option<f64>,
    /// Provenance status of the row.
    pub status: BandStatus,
    /// Per-metric source citation (document + table).
    pub citation: String,
    /// Hours actually sampled for this row (0 when the series was absent).
    pub sampled_hours: usize,
    /// Observed minimum of the sampled hours (informational only).
    pub observed_min: Option<f64>,
    /// Observed maximum of the sampled hours (informational only).
    pub observed_max: Option<f64>,
    /// Free-form note; carries the not-yet-banded justification.
    pub note: String,
}

impl HourlyBandRow {
    /// Build a band row. `band` is `None` when no numeric band is published
    /// in-repo, which yields an explicit known-fail row (never an invented band).
    pub fn new(
        case_id: &str,
        family: FabricFamily,
        day: RepresentativeDay,
        metric: HourlyMetric,
        band: Option<(f64, f64)>,
        values: &[Option<f64>],
    ) -> Self {
        let observed_min = values
            .iter()
            .flatten()
            .copied()
            .fold(None, |acc: Option<f64>, v| {
                Some(match acc {
                    Some(a) if a <= v => a,
                    _ => v,
                })
            });
        let observed_max = values
            .iter()
            .flatten()
            .copied()
            .fold(None, |acc: Option<f64>, v| {
                Some(match acc {
                    Some(a) if a >= v => a,
                    _ => v,
                })
            });
        let sampled_hours = values.iter().flatten().count();
        let status = match band {
            Some(_) => BandStatus::Banded,
            None => BandStatus::KnownFailNotBanded,
        };
        let note = if band.is_some() {
            "Band transcribed from the cited source table.".to_string()
        } else if sampled_hours == 0 {
            format!(
                "{NO_IN_REPO_BAND_NOTE} No hourly values were sampled for this row, so it \
                 cannot be evaluated."
            )
        } else {
            NO_IN_REPO_BAND_NOTE.to_string()
        };
        Self {
            case_id: case_id.to_string(),
            family,
            day,
            metric,
            band_min: band.map(|b| b.0),
            band_max: band.map(|b| b.1),
            status,
            citation: metric.citation().to_string(),
            sampled_hours,
            observed_min,
            observed_max,
            note,
        }
    }

    /// `true` when the row is an explicit known-fail row (no in-repo band).
    pub fn is_known_fail(&self) -> bool {
        self.status == BandStatus::KnownFailNotBanded
    }

    /// Verdict for one sampled value against this row's band.
    ///
    /// Rows without an in-repo band can never report a pass: they stay
    /// known-fail until a band is transcribed from the source.
    pub fn verdict(&self, value: f64) -> BandStatus {
        match (self.band_min, self.band_max) {
            (Some(lo), Some(hi)) if value >= lo && value <= hi => BandStatus::Banded,
            _ => BandStatus::KnownFailNotBanded,
        }
    }

    /// Markdown table row for the report (citation carried by the caller).
    pub fn to_markdown_row(&self) -> String {
        let band = match (self.band_min, self.band_max) {
            (Some(lo), Some(hi)) => format!("{lo:.3} .. {hi:.3}"),
            _ => "not published in-repo".to_string(),
        };
        format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            self.case_id,
            self.family.label(),
            self.day.label(),
            self.metric.key(),
            self.metric.unit(),
            band,
            self.sampled_hours,
            self.status.label(),
        )
    }

    /// CSV row (citation carried in the trailing column so exports stay auditable).
    pub fn to_csv_row(&self) -> String {
        let lo = self.band_min.map(|v| format!("{v:.6}")).unwrap_or_default();
        let hi = self.band_max.map(|v| format!("{v:.6}")).unwrap_or_default();
        let obs_lo = self
            .observed_min
            .map(|v| format!("{v:.6}"))
            .unwrap_or_default();
        let obs_hi = self
            .observed_max
            .map(|v| format!("{v:.6}"))
            .unwrap_or_default();
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            self.case_id,
            csv_escape(self.family.label()),
            csv_escape(&self.day.label()),
            self.metric.key(),
            self.metric.unit(),
            lo,
            hi,
            obs_lo,
            obs_hi,
            self.sampled_hours,
            self.status.label(),
            csv_escape(&self.citation),
        )
    }
}

/// Header emitted once above the band-row CSV block.
pub fn band_csv_header() -> String {
    "Case,Family,Day,Metric,Unit,BandMin,BandMax,ObservedMin,ObservedMax,SampledHours,Status,Citation\n"
        .to_string()
}

/// One hourly sample on a representative day (Table 6-1, footnote a: 24 per day,
/// hour 1 = 0000-0100).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HourlySample {
    /// Hour of the day, 1-24 (hour 1 runs 0000-0100).
    pub hour: u32,
    /// Hourly integrated zone air temperature (°C), when collected.
    pub zone_air_temp_c: Option<f64>,
    /// Signed hourly integrated load (kWh): heating `+`, sensible cooling `−`
    /// (Table 6-1 requires cooling to carry a `[−]` sign).
    pub load_kwh_signed: Option<f64>,
}

impl HourlySample {
    /// Heating component (kWh, positive), `None` when not sampled.
    pub fn heating_kwh(&self) -> Option<f64> {
        self.load_kwh_signed.map(|v| v.max(0.0))
    }

    /// Sensible-cooling component (kWh, positive magnitude), `None` when absent.
    pub fn cooling_kwh(&self) -> Option<f64> {
        self.load_kwh_signed.map(|v| (-v).max(0.0))
    }

    /// CSV row for the hourly block (without the leading day column).
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{}\n",
            self.hour,
            fmt_opt(self.zone_air_temp_c),
            fmt_opt(self.load_kwh_signed),
            fmt_opt(self.heating_kwh()),
            fmt_opt(self.cooling_kwh()),
            if self.zone_air_temp_c.is_some() {
                "yes"
            } else {
                "no"
            },
            if self.load_kwh_signed.is_some() {
                "yes"
            } else {
                "no"
            },
        )
    }
}

/// The hourly output produced for one case on one representative day.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyDayOutput {
    /// Case the day belongs to.
    pub case_id: String,
    /// Representative day.
    pub day: RepresentativeDay,
    /// The 24 hourly integrated samples (Table 6-1, footnote a).
    pub samples: Vec<HourlySample>,
}

impl HourlyDayOutput {
    /// `true` when all 24 hourly integrated values are present.
    pub fn is_complete(&self) -> bool {
        self.samples.len() == HOURS_PER_REPORT_DAY
            && (1..=HOURS_PER_REPORT_DAY as u32).all(|h| self.samples.iter().any(|s| s.hour == h))
    }

    /// Markdown table of the 24 hourly integrated values.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "#### Case {} — {} ({})\n\n\
             | Hour | Zone air temp (°C) | Load (kWh, + heat / − cool) |\n\
             |------|--------------------|-------------------------------|\n",
            self.case_id,
            self.day.label(),
            if self.case_id.ends_with("FF") {
                "free float"
            } else {
                "conditioned"
            }
        );
        for s in &self.samples {
            out.push_str(&format!(
                "| {:02} | {} | {} |\n",
                s.hour,
                fmt_opt(s.zone_air_temp_c),
                fmt_opt(s.load_kwh_signed)
            ));
        }
        out.push('\n');
        out
    }
}

/// Per-case Addendum a hourly report: the representative-day outputs plus the
/// band rows (including explicit known-fail rows) for that case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyCaseReport {
    /// Case this report is for.
    pub case_id: String,
    /// Fabric family of the case.
    pub family: FabricFamily,
    /// Free-float vs conditioned presentation flag.
    pub free_float: bool,
    /// Hourly outputs, one entry per representative day.
    pub days: Vec<HourlyDayOutput>,
    /// Acceptance/comparison band rows (with per-metric citations).
    pub band_rows: Vec<HourlyBandRow>,
    /// Requirement citation carried with the report.
    pub citation: String,
}

impl HourlyCaseReport {
    /// `true` when every representative day has all 24 hourly values.
    pub fn is_complete(&self) -> bool {
        self.days.len() == REPRESENTATIVE_DAYS.len() && self.days.iter().all(|d| d.is_complete())
    }

    /// Band rows that are explicit known-fail rows.
    pub fn known_fail_rows(&self) -> Vec<&HourlyBandRow> {
        self.band_rows
            .iter()
            .filter(|r| r.is_known_fail())
            .collect()
    }

    /// Markdown block: the hourly outputs followed by the band rows.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "### Case {} ({})\n\nRequirement: {}  \nGranularity: {}\n\n",
            self.case_id,
            self.family.label(),
            self.citation,
            GRANULARITY_CITATION
        );
        if self.days.iter().all(|d| d.samples.is_empty()) {
            out.push_str(
                "**KNOWN-FAIL:** no hourly series was collected for this case, so no \
                 representative-day output can be produced. The band table below records the \
                 gap explicitly.\n\n",
            );
        }
        for day in &self.days {
            out.push_str(&day.to_markdown());
        }
        out.push_str("| Case | Family | Day | Metric | Unit | Band | Hours sampled | Status |\n");
        out.push_str("|------|--------|-----|--------|------|------|---------------|--------|\n");
        for row in &self.band_rows {
            out.push_str(&row.to_markdown_row());
        }
        out.push('\n');
        for row in &self.band_rows {
            out.push_str(&format!(
                "- `{}` / `{}` / {} — {} — {}\n",
                row.case_id,
                row.metric.key(),
                row.day.label(),
                row.status.label(),
                row.citation
            ));
        }
        out.push('\n');
        out
    }

    /// CSV block: one row per hourly sample, then the band's band rows.
    pub fn to_csv(&self) -> String {
        let mut out = format!("# Case: {}\n", self.case_id);
        out.push_str(&format!(
            "# Requirement: {}\n# Granularity: {}\n",
            csv_escape(&self.citation),
            csv_escape(GRANULARITY_CITATION)
        ));
        out.push_str(
            "Day,Hour,Zone_Air_Temperature_C,Load_kWh_Signed,Heating_kWh,Cooling_kWh,\
             Temp_Collected,Load_Collected\n",
        );
        for day in &self.days {
            for sample in &day.samples {
                out.push_str(&format!(
                    "{},{}",
                    csv_escape(&day.day.label()),
                    sample.to_csv_row()
                ));
            }
        }
        out.push('\n');
        out
    }
}

/// Whole-leg report: every fabric case's hourly output plus the band table.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HourlyReport {
    /// Per-case reports.
    pub cases: Vec<HourlyCaseReport>,
    /// Requirement citation for the leg as a whole.
    pub citation: String,
}

impl HourlyReport {
    /// Band rows across every case in the leg.
    pub fn band_rows(&self) -> Vec<&HourlyBandRow> {
        self.cases.iter().flat_map(|c| c.band_rows.iter()).collect()
    }

    /// Band rows that are explicit known-fail rows.
    pub fn known_fail_rows(&self) -> Vec<&HourlyBandRow> {
        self.band_rows()
            .into_iter()
            .filter(|r| r.is_known_fail())
            .collect()
    }

    /// Markdown rendering of the whole leg (hourly outputs + band table).
    pub fn to_markdown(&self) -> String {
        let mut out = String::from("# ASHRAE 140 Addendum a hourly output (Issue #4268)\n\n");
        out.push_str(&format!("Requirement: {}\n\n", self.citation));
        for case in &self.cases {
            out.push_str(&case.to_markdown());
        }
        out
    }

    /// CSV rendering of the whole leg (hourly samples + band rows).
    pub fn to_csv(&self) -> String {
        let mut out = format!("# Requirement: {}\n", csv_escape(&self.citation));
        for case in &self.cases {
            out.push_str(&case.to_csv());
        }
        out.push_str("# Acceptance bands\n");
        out.push_str(&band_csv_header());
        for row in self.band_rows() {
            out.push_str(&row.to_csv_row());
        }
        out
    }
}

/// Raw annual hourly series collected by the simulator for the fabric cases.
///
/// Two vectors are kept (rather than re-deriving them from per-zone
/// diagnostics) so the Addendum a leg costs the same ~70 KB per vector that
/// issue #827 already accepted for the free-float temperature profile.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HourlySeries {
    /// Zone-0 air temperature (°C) at the end of each hour (8760 entries).
    pub zone_air_temp_c: Vec<f64>,
    /// Signed hourly integrated load (kWh) for each hour (8760 entries):
    /// positive = heating, negative = sensible cooling.
    pub load_kwh_signed: Vec<f64>,
}

impl HourlySeries {
    /// Pre-allocate both vectors for an annual run.
    pub fn with_capacity(hours: usize) -> Self {
        Self {
            zone_air_temp_c: Vec::with_capacity(hours),
            load_kwh_signed: Vec::with_capacity(hours),
        }
    }

    /// Append one hour of output.
    pub fn push(&mut self, zone_air_temp_c: f64, load_kwh_signed: f64) {
        self.zone_air_temp_c.push(zone_air_temp_c);
        self.load_kwh_signed.push(load_kwh_signed);
    }

    /// `true` when the series holds a full annual run (Table 6-1 / §6.2.1.8:
    /// the representative days must come from a typical annual run).
    pub fn is_annual(&self) -> bool {
        self.zone_air_temp_c.len() >= ANNUAL_HOURS && self.load_kwh_signed.len() >= ANNUAL_HOURS
    }
}

/// Slice an annual series down to one representative day (24 hourly integrated
/// values, hour 1 = 0000-0100 — Table 6-1, footnote a).
///
/// Returns an *incomplete* `HourlyDayOutput` (no samples) when the series does
/// not cover the day, so callers surface a known-fail instead of padding
/// fabricated values.
pub fn slice_day(
    case_id: &str,
    day: RepresentativeDay,
    series: Option<&HourlySeries>,
) -> HourlyDayOutput {
    let start = day.first_hour_index();
    let mut samples = Vec::with_capacity(HOURS_PER_REPORT_DAY);
    if let Some(series) = series {
        for hour in 0..HOURS_PER_REPORT_DAY {
            let idx = start + hour;
            let temp = series.zone_air_temp_c.get(idx).copied();
            let load = series.load_kwh_signed.get(idx).copied();
            if temp.is_none() && load.is_none() {
                break;
            }
            samples.push(HourlySample {
                hour: hour as u32 + 1,
                zone_air_temp_c: temp,
                load_kwh_signed: load,
            });
        }
    }
    HourlyDayOutput {
        case_id: case_id.to_string(),
        day,
        samples,
    }
}

/// Fabric family of a case ID (600-series or 900-series, `FF` suffix allowed).
/// Returns `None` for non-fabric cases (e.g. 195, 800, 810, 960).
pub fn family_for_case(case_id: &str) -> Option<FabricFamily> {
    let digits: String = case_id.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.starts_with('6') {
        Some(FabricFamily::LowMass600)
    } else if digits.starts_with('9') {
        Some(FabricFamily::HighMass900)
    } else {
        None
    }
}

/// `true` when a case belongs to the Section 5.2.1 / 5.2.2 fabric families that
/// Addendum a Table 6-1 names (600-series and 900-series, `FF` variants
/// included). The sunspace (960) and night-ventilation (970) cases are NOT
/// part of §5.2.1/§5.2.2, so a bare 6xx/9xx prefix is not sufficient.
pub fn is_fabric_case(case_id: &str) -> bool {
    matches!(
        case_id,
        "600"
            | "610"
            | "620"
            | "630"
            | "640"
            | "650"
            | "600FF"
            | "650FF"
            | "680FF"
            | "900"
            | "910"
            | "920"
            | "930"
            | "940"
            | "950"
            | "900FF"
            | "950FF"
            | "980FF"
    )
}

/// Build the Addendum a band rows for one case / representative day.
///
/// Every metric Table 6-1 names for the case is emitted. No numeric band is
/// published in-repo, so each row is an explicit known-fail row carrying its
/// per-metric citation (see [`NO_IN_REPO_BAND_NOTE`]).
pub fn band_rows_for_day(
    case_id: &str,
    day: RepresentativeDay,
    samples: &[HourlySample],
) -> Vec<HourlyBandRow> {
    let family = family_for_case(case_id).unwrap_or(FabricFamily::LowMass600);
    let day_output = HourlyDayOutput {
        case_id: case_id.to_string(),
        day,
        samples: samples.to_vec(),
    };
    metrics_for_case(case_id)
        .into_iter()
        .map(|metric| {
            let values = metric.sample_values(&day_output);
            HourlyBandRow::new(case_id, family, day, metric, None, &values)
        })
        .collect()
}

/// Build the per-case Addendum a hourly report from an annual hourly series.
///
/// `series` is the full 8760-hour annual run (§6.2.1.8 forbids re-running only
/// the representative days); this function slices it, it never re-simulates.
pub fn case_report(case_id: &str, series: Option<&HourlySeries>) -> HourlyCaseReport {
    let family = family_for_case(case_id).unwrap_or(FabricFamily::LowMass600);
    let days: Vec<HourlyDayOutput> = REPRESENTATIVE_DAYS
        .iter()
        .map(|&day| slice_day(case_id, day, series))
        .collect();
    let band_rows = days
        .iter()
        .flat_map(|day| band_rows_for_day(case_id, day.day, &day.samples))
        .collect();
    HourlyCaseReport {
        case_id: case_id.to_string(),
        family,
        free_float: case_id.ends_with("FF"),
        days,
        band_rows,
        citation: REQUIREMENT_CITATION.to_string(),
    }
}

/// Assemble a whole-leg report from per-case annual series.
pub fn report_from_series<I>(cases: I) -> HourlyReport
where
    I: IntoIterator<Item = (String, Option<HourlySeries>)>,
{
    HourlyReport {
        cases: cases
            .into_iter()
            .map(|(case_id, series)| case_report(&case_id, series.as_ref()))
            .collect(),
        citation: REQUIREMENT_CITATION.to_string(),
    }
}

fn month_abbrev(month: u32) -> &'static str {
    match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => "?",
    }
}

fn days_in_month(month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => 28,
        _ => 30,
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "n/a".to_string(),
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('\n') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_series() -> HourlySeries {
        let mut s = HourlySeries::with_capacity(ANNUAL_HOURS);
        for i in 0..ANNUAL_HOURS {
            s.push(
                20.0 + (i % 24) as f64 * 0.1,
                if i % 48 == 0 { 0.5 } else { -0.1 },
            );
        }
        s
    }

    #[test]
    fn representative_day_indices_match_table_6_1_days() {
        assert_eq!(JAN_4.day_of_year(), 4);
        assert_eq!(JAN_4.first_hour_index(), 72);
        assert_eq!(JAN_4.label(), "Jan 4");
        // Jan 31 + Feb 28 + Mar 31 + Apr 30 + May 31 + Jun 30 = 181; + 14 = 195.
        assert_eq!(JUL_14.day_of_year(), 195);
        assert_eq!(JUL_14.first_hour_index(), 194 * HOURS_PER_REPORT_DAY);
        assert_eq!(JUL_14.label(), "Jul 14");
    }

    #[test]
    fn case_report_has_24_entries_per_representative_day() {
        let series = synthetic_series();
        for case_id in ["600", "900", "600FF", "900FF"] {
            let report = case_report(case_id, Some(&series));
            assert_eq!(report.days.len(), 2, "{case_id}");
            for day in &report.days {
                assert_eq!(day.samples.len(), HOURS_PER_REPORT_DAY, "{case_id}");
                assert!(day.is_complete(), "{case_id} {}", day.day.label());
                assert_eq!(day.samples[0].hour, 1);
                assert_eq!(day.samples[23].hour, 24);
            }
            assert!(report.is_complete(), "{case_id}");
        }
    }

    #[test]
    fn families_split_low_and_high_mass() {
        assert_eq!(family_for_case("600"), Some(FabricFamily::LowMass600));
        assert_eq!(family_for_case("650FF"), Some(FabricFamily::LowMass600));
        assert_eq!(family_for_case("900"), Some(FabricFamily::HighMass900));
        assert_eq!(family_for_case("900FF"), Some(FabricFamily::HighMass900));
        assert_eq!(family_for_case("195"), None);
        assert!(is_fabric_case("640"));
        assert!(!is_fabric_case("960"));
    }

    #[test]
    fn band_rows_carry_citations_and_are_known_fail() {
        let series = synthetic_series();
        let report = report_from_series([("600".to_string(), Some(series))]);
        let rows = report.band_rows();
        // 1 case x 2 days x 3 metrics
        assert_eq!(rows.len(), 6);
        for row in rows {
            assert!(
                row.citation.contains("140-2017") && row.citation.contains("Table 6-1"),
                "row {} lacks a document+table citation: {}",
                row.metric.key(),
                row.citation
            );
            assert!(row.is_known_fail());
            assert!(row.band_min.is_none() && row.band_max.is_none());
            assert_eq!(row.sampled_hours, HOURS_PER_REPORT_DAY);
            assert!(row.observed_min.is_some() && row.observed_max.is_some());
            // A row without a band can never report a pass.
            assert_eq!(row.verdict(0.0), BandStatus::KnownFailNotBanded);
        }
        assert_eq!(report.known_fail_rows().len(), 6);
    }

    #[test]
    fn missing_series_yields_known_fail_rows_not_fabricated_values() {
        let report = case_report("900", None);
        assert!(!report.is_complete());
        assert_eq!(report.days.len(), 2);
        for day in &report.days {
            assert!(day.samples.is_empty());
        }
        assert_eq!(report.known_fail_rows().len(), 6);
        for row in &report.band_rows {
            assert_eq!(row.sampled_hours, 0);
            assert!(row.note.contains("cannot be evaluated"));
        }
        let markdown = report.to_markdown();
        assert!(markdown.contains("KNOWN-FAIL"));
        assert!(markdown.contains("Table 6-1"));
    }

    #[test]
    fn markdown_and_csv_carry_the_hourly_leg() {
        let series = synthetic_series();
        let report = report_from_series([
            ("600".to_string(), Some(series.clone())),
            ("900".to_string(), Some(series)),
        ]);
        let md = report.to_markdown();
        assert!(md.contains("Jan 4") && md.contains("Jul 14"));
        assert!(md.contains("hourly_integrated_zone_air_temperature_c"));
        let csv = report.to_csv();
        assert!(csv.contains("Jan 4,1,") && csv.contains("Jul 14,24,"));
        assert!(csv.contains("KNOWN-FAIL"));
        assert!(csv.contains("Table 6-1"));
    }

    #[test]
    fn slice_day_reads_the_annual_series_indices() {
        let mut s = HourlySeries::with_capacity(ANNUAL_HOURS);
        for i in 0..ANNUAL_HOURS {
            s.push(i as f64, if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let day = slice_day("600", JAN_4, Some(&s));
        assert_eq!(day.samples.len(), 24);
        assert_eq!(day.samples[0].zone_air_temp_c, Some(72.0));
        assert_eq!(day.samples[23].zone_air_temp_c, Some(95.0));
        // Table 6-1 footnote a: the first hour runs 0000-0100.
        assert_eq!(day.samples[0].hour, 1);
        assert_eq!(day.samples[23].hour, 24);
    }
}
