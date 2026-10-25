// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Versioned standard results report (Issue #4104, closes #3988).
//!
//! This module owns the *single* constructor that every user-facing surface
//! uses to turn a finished simulation into the versioned, byte-deterministic
//! JSON standard report described by `schemas/results_v1.schema.json`:
//!
//! * REST — `POST /v1/simulate` (`options.report = true`)
//! * CLI — `fluxion --report …` (writes `{prefix}_report.json`)
//! * Python — `MultiZoneThermalModel.get_standard_report(...)`
//!
//! The report is assembled from the results produced by the earlier parts of
//! the results story:
//!
//! * the timestep-indexed end-use series (Issue #4101),
//! * the monthly end-use summaries (Issue #4102),
//! * the unmet-hours accounting, both conventions (Issue #3988/#4103).
//!
//! # Determinism
//!
//! Byte-determinism is a hard requirement of the standard, so the document is
//! emitted with:
//!
//! * **sorted object keys** — every struct declares its fields in
//!   lexicographic order (a unit test pins this),
//! * **fixed float formatting** — every float is rendered with exactly
//!   [`FLOAT_DECIMALS`] decimals (never scientific notation, never a
//!   platform-dependent shortest-round-trip form),
//! * **index-ordered arrays** — `monthly` is January…December, `by_zone` is
//!   zone index 0…n-1, and the end-use maps are keyed by end-use name.
//!
//! No wall-clock timestamps are emitted unless the caller supplies them, so
//! two runs over identical inputs produce byte-identical documents.
//!
//! # Versioning policy
//!
//! `schema_version` is semver. A **minor** bump is additive (new optional
//! fields / new sections); a **major** bump is breaking (removed or
//! re-typed fields, changed units or conventions).

use serde::{Deserialize, Serialize};

use crate::api::schema::SimulationOutput;
use crate::sim::schedule::DailySchedule;

/// Version of the standard results report schema (`schemas/results_v1.schema.json`).
pub const RESULTS_SCHEMA_VERSION: &str = "1.0.0";

/// Stable id of the hourly series referenced (not embedded) by the report.
pub const REPORT_SERIES_ID: &str = "hourly-end-use";

/// Number of decimals every float in the report is rendered with.
pub const FLOAT_DECIMALS: usize = 6;

/// End uses carried by the report, in the order used by the series
/// (heating, cooling, lighting, equipment — the Issue #4101 order).
pub const END_USES: [&str; 4] = ["heating", "cooling", "lighting", "equipment"];

/// Column names of the referenced hourly series, in series order.
pub const SERIES_COLUMNS: [&str; 4] = [
    "heating_kwh",
    "cooling_kwh",
    "lighting_kwh",
    "equipment_kwh",
];

/// Calendar month names, January first.
pub const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

// ---------------------------------------------------------------------------
// Deterministic float formatting
// ---------------------------------------------------------------------------

/// Render a float with the report's fixed precision.
///
/// The value is rounded to [`FLOAT_DECIMALS`] decimals and rendered in plain
/// (never scientific) notation, so the same number always produces the same
/// bytes. Negative zero is normalised to `0.000000`. Non-finite values are
/// rejected: a report must never contain `NaN`/`Infinity` (which are not
/// valid JSON numbers anyway).
fn format_fixed(value: f64) -> Result<String, String> {
    if !value.is_finite() {
        return Err(format!(
            "non-finite value ({value}) cannot be represented in the results report"
        ));
    }
    if value == 0.0 {
        // Normalises -0.0 to 0.000000.
        return Ok(format!("{:.*}", FLOAT_DECIMALS, 0.0f64));
    }
    Ok(format!("{:.*}", FLOAT_DECIMALS, value))
}

/// `serde` helper serializing an `f64` with the report's fixed precision.
pub fn serialize_fixed<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let text = format_fixed(*value).map_err(serde::ser::Error::custom)?;
    let raw = serde_json::value::RawValue::from_string(text).map_err(serde::ser::Error::custom)?;
    raw.serialize(serializer)
}

/// Replace non-finite values with `0.0` so the report stays schema-valid.
fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Report sections
// ---------------------------------------------------------------------------

/// End-use energy in kWh (annual or per month).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndUseEnergy {
    /// Cooling energy in kWh.
    #[serde(serialize_with = "serialize_fixed")]
    pub cooling: f64,
    /// Equipment electric energy in kWh.
    #[serde(serialize_with = "serialize_fixed")]
    pub equipment: f64,
    /// Heating energy in kWh.
    #[serde(serialize_with = "serialize_fixed")]
    pub heating: f64,
    /// Lighting electric energy in kWh.
    #[serde(serialize_with = "serialize_fixed")]
    pub lighting: f64,
}

impl EndUseEnergy {
    /// Builds an end-use energy block from the four end-use totals.
    pub fn new(heating: f64, cooling: f64, lighting: f64, equipment: f64) -> Self {
        Self {
            cooling: finite(cooling),
            equipment: finite(equipment),
            heating: finite(heating),
            lighting: finite(lighting),
        }
    }

    /// Sum of the four end uses.
    pub fn total(&self) -> f64 {
        self.heating + self.cooling + self.lighting + self.equipment
    }
}

/// Peak end-use demand in kW (timestep-average).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndUsePeak {
    /// Peak cooling demand in kW.
    #[serde(serialize_with = "serialize_fixed")]
    pub cooling: f64,
    /// Peak equipment demand in kW.
    #[serde(serialize_with = "serialize_fixed")]
    pub equipment: f64,
    /// Peak heating demand in kW.
    #[serde(serialize_with = "serialize_fixed")]
    pub heating: f64,
    /// Peak lighting demand in kW.
    #[serde(serialize_with = "serialize_fixed")]
    pub lighting: f64,
}

impl EndUsePeak {
    /// Builds a peak block from the four end-use peaks.
    pub fn new(heating: f64, cooling: f64, lighting: f64, equipment: f64) -> Self {
        Self {
            cooling: finite(cooling),
            equipment: finite(equipment),
            heating: finite(heating),
            lighting: finite(lighting),
        }
    }
}

/// The simulated period the report covers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunPeriod {
    /// Optional ISO-8601 UTC end of the run period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
    /// Human-readable period label (e.g. `annual`).
    pub label: String,
    /// Optional ISO-8601 UTC start of the run period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    /// Timestep duration in seconds.
    #[serde(serialize_with = "serialize_fixed")]
    pub timestep_seconds: f64,
    /// Number of timesteps in the referenced hourly series.
    pub timesteps: usize,
    /// Number of simulated years.
    pub years: usize,
}

/// Document metadata block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Name of the simulated model.
    pub model_name: String,
    /// Provenance of the simulated model.
    pub model_source: String,
    /// The simulated period.
    pub run_period: RunPeriod,
    /// Results-report schema version (semver).
    pub schema_version: String,
    /// Version of the emitting engine.
    pub tool_version: String,
}

/// Annual (whole run period) summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnualSection {
    /// End-use energy for the whole run period.
    pub end_use_kwh: EndUseEnergy,
    /// Energy use intensity in kWh/m².
    #[serde(serialize_with = "serialize_fixed")]
    pub eui_kwh_per_m2: f64,
    /// Conditioned floor area in m².
    #[serde(serialize_with = "serialize_fixed")]
    pub floor_area_m2: f64,
    /// Peak end-use demand for the whole run period.
    pub peaks: EndUsePeak,
    /// Total energy for the whole run period.
    #[serde(serialize_with = "serialize_fixed")]
    pub total_energy_kwh: f64,
    /// Number of zones in the model.
    pub zone_count: usize,
}

/// One calendar month of the run period.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonthlyMonth {
    /// End-use energy for the month.
    pub end_use_kwh: EndUseEnergy,
    /// Peak end-use demand for the month.
    pub end_use_peak_kw: EndUsePeak,
    /// Calendar month number, January = 1.
    pub month: u32,
    /// Calendar month name.
    pub name: String,
}

/// Unmet heating/cooling hours for one reporting convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnmetPair {
    /// Unmet cooling hours.
    #[serde(serialize_with = "serialize_fixed")]
    pub cooling: f64,
    /// Unmet heating hours.
    #[serde(serialize_with = "serialize_fixed")]
    pub heating: f64,
}

impl UnmetPair {
    /// Builds an unmet-hours pair.
    pub fn new(heating: f64, cooling: f64) -> Self {
        Self {
            cooling: finite(cooling),
            heating: finite(heating),
        }
    }
}

/// Unmet hours for a single zone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnmetZone {
    /// All-hours unmet hours for the zone.
    pub all_hours: UnmetPair,
    /// Zero-based zone index.
    pub index: usize,
    /// Zone name (`zone-<index>` when the model does not name the zone).
    pub name: String,
    /// Occupied-only unmet hours for the zone.
    pub occupied: UnmetPair,
}

/// Unmet-hours section: both reporting conventions plus the per-zone split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnmetHoursSection {
    /// All-hours convention (ASHRAE 90.1 Appendix G §G3.1.2.2).
    pub all_hours: UnmetPair,
    /// Per-zone unmet hours, ordered by zone index.
    pub by_zone: Vec<UnmetZone>,
    /// Occupied-only convention (EnergyPlus "Time Setpoint Not Met" parity).
    pub occupied: UnmetPair,
    /// Setpoint tolerance in °C used for the setpoint-deviation test.
    #[serde(serialize_with = "serialize_fixed")]
    pub tolerance_c: f64,
}

/// Reference to the timestep-indexed end-use series (never embedded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyReference {
    /// SHA-256 (hex) of the canonical series text.
    pub checksum: String,
    /// Column names of the referenced series, in series order.
    pub columns: Vec<String>,
    /// End uses carried by the referenced series, in series order.
    pub end_uses: Vec<String>,
    /// Stable identifier of the referenced series.
    pub series_id: String,
    /// Timestep duration in seconds.
    #[serde(serialize_with = "serialize_fixed")]
    pub timestep_seconds: f64,
    /// Number of timesteps in the referenced series.
    pub timesteps: usize,
}

/// The versioned standard results report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StandardReport {
    /// Annual summary.
    pub annual: AnnualSection,
    /// Reference to the hourly series.
    pub hourly: HourlyReference,
    /// Document metadata.
    pub metadata: ReportMetadata,
    /// Twelve calendar-month bins, January first.
    pub monthly: Vec<MonthlyMonth>,
    /// Unmet-hours accounting.
    pub unmet_hours: UnmetHoursSection,
}

// ---------------------------------------------------------------------------
// Construction context
// ---------------------------------------------------------------------------

/// Everything the report needs that is *not* carried by the
/// [`SimulationOutput`] itself (model identity, geometry, setpoints,
/// occupancy and the run period).
#[derive(Debug, Clone)]
pub struct StandardReportContext {
    /// Cooling setpoint in °C (used for the per-zone unmet-hours split).
    pub cooling_setpoint_c: f64,
    /// Conditioned floor area in m² (EUI denominator).
    pub floor_area_m2: f64,
    /// Model name.
    pub model_name: String,
    /// Model provenance.
    pub model_source: String,
    /// Occupancy schedule (daily profile) used for the occupied convention.
    pub occupancy: DailySchedule,
    /// Optional ISO-8601 UTC end of the run period.
    pub run_end: Option<String>,
    /// Human-readable run-period label.
    pub run_label: String,
    /// Optional ISO-8601 UTC start of the run period.
    pub run_start: Option<String>,
    /// Heating setpoint in °C (used for the per-zone unmet-hours split).
    pub heating_setpoint_c: f64,
    /// Timestep duration in seconds.
    pub timestep_seconds: f64,
    /// Setpoint tolerance in °C used for the unmet-hours test.
    pub unmet_hours_tolerance_c: f64,
    /// Number of zones in the model.
    pub zone_count: usize,
    /// Zone names in zone-index order (may be shorter than `zone_count`).
    pub zone_names: Vec<String>,
    /// Number of simulated years.
    pub years: usize,
}

impl Default for StandardReportContext {
    fn default() -> Self {
        Self {
            cooling_setpoint_c: 24.0,
            floor_area_m2: 1.0,
            model_name: "unnitled".to_string(),
            model_source: "unknown".to_string(),
            occupancy: DailySchedule::new(),
            run_end: None,
            run_label: "annual".to_string(),
            run_start: None,
            heating_setpoint_c: 20.0,
            timestep_seconds: 3600.0,
            unmet_hours_tolerance_c: crate::api::schema::UNMET_HOURS_TOLERANCE_DEFAULT_C,
            zone_count: 0,
            zone_names: Vec::new(),
            years: 1,
        }
    }
}

impl StandardReportContext {
    /// Creates a context for the given model identity.
    pub fn new(model_name: impl Into<String>, model_source: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            model_source: model_source.into(),
            ..Self::default()
        }
    }

    /// Sets the conditioned floor area (m²).
    pub fn with_floor_area(mut self, floor_area_m2: f64) -> Self {
        self.floor_area_m2 = floor_area_m2;
        self
    }

    /// Sets the number of zones and their names (index order).
    pub fn with_zones(mut self, zone_count: usize, zone_names: Vec<String>) -> Self {
        self.zone_count = zone_count;
        self.zone_names = zone_names;
        self
    }

    /// Sets the timestep duration (seconds) and the number of simulated years.
    pub fn with_timestep(mut self, timestep_seconds: f64, years: usize) -> Self {
        self.timestep_seconds = timestep_seconds;
        self.years = years;
        self
    }

    /// Sets the heating/cooling setpoints (°C).
    pub fn with_setpoints(mut self, heating_setpoint_c: f64, cooling_setpoint_c: f64) -> Self {
        self.heating_setpoint_c = heating_setpoint_c;
        self.cooling_setpoint_c = cooling_setpoint_c;
        self
    }

    /// Sets the occupancy schedule used for the occupied-only convention.
    pub fn with_occupancy(mut self, occupancy: DailySchedule) -> Self {
        self.occupancy = occupancy;
        self
    }

    /// Sets the run-period label and (optional) explicit bounds.
    pub fn with_run_period(
        mut self,
        label: impl Into<String>,
        start: Option<String>,
        end: Option<String>,
    ) -> Self {
        self.run_label = label.into();
        self.run_start = start;
        self.run_end = end;
        self
    }

    /// Sets the unmet-hours setpoint tolerance (°C).
    pub fn with_unmet_hours_tolerance(mut self, tolerance_c: f64) -> Self {
        self.unmet_hours_tolerance_c = tolerance_c;
        self
    }
}

// ---------------------------------------------------------------------------
// The constructor
// ---------------------------------------------------------------------------

/// Canonical text of the hourly series: one fixed-precision value per line,
/// end uses in heating/cooling/lighting/equipment order, timesteps in order.
///
/// This is the byte sequence the `hourly.checksum` is computed over, so a
/// consumer that fetches the series can recompute the checksum and detect a
/// mismatch with the report.
fn canonical_series_text(series: [Option<&Vec<f64>>; 4]) -> String {
    let mut text = String::new();
    for s in series {
        if let Some(values) = s {
            for v in values {
                // Fixed precision, one value per line; a non-finite value is
                // impossible here (the engine never records NaN energy), but
                // fall back to a canonical zero rather than panicking.
                let value = if v.is_finite() { *v } else { 0.0 };
                text.push_str(&format_fixed(value).unwrap_or_else(|_| "0.000000".to_string()));
                text.push('\n');
            }
        }
    }
    text
}

/// SHA-256 (hex) of the canonical series text.
fn series_checksum(series: [Option<&Vec<f64>>; 4]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(canonical_series_text(series).as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Sum of a series, or `None` when the series was not recorded.
///
/// Non-finite samples count as zero (the same convention as
/// [`canonical_series_text`]): a single NaN must not poison the total.
fn series_sum(series: Option<&Vec<f64>>) -> Option<f64> {
    series.map(|values| values.iter().map(|v| finite(*v)).sum::<f64>())
}

/// Maximum timestep-average kW of a series, or `None` when not recorded.
fn series_peak_kw(series: Option<&Vec<f64>>, timestep_seconds: f64) -> Option<f64> {
    let hours_per_step = timestep_seconds / 3600.0;
    if hours_per_step <= 0.0 {
        return None;
    }
    series.map(|values| {
        let peak = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, v| if v > acc { v } else { acc });
        if peak.is_finite() {
            peak / hours_per_step
        } else {
            0.0
        }
    })
}

/// Aggregates the monthly end-use summaries (`[year][month]`) over the years
/// of the run period: kWh is summed, peak kW is the maximum.
fn aggregate_monthly(
    kwh: Option<&Vec<Vec<f64>>>,
    peak_kw: Option<&Vec<Vec<f64>>>,
) -> ([f64; 12], [f64; 12]) {
    let mut kwh_out = [0.0f64; 12];
    let mut peak_out = [0.0f64; 12];
    if let Some(years) = kwh {
        for (y, months) in years.iter().enumerate() {
            for (m, value) in months.iter().enumerate().take(12) {
                kwh_out[m] += finite(*value);
                if let Some(peaks) = peak_kw {
                    if let Some(p) = peaks.get(y).and_then(|row| row.get(m)) {
                        if *p > peak_out[m] {
                            peak_out[m] = *p;
                        }
                    }
                }
            }
        }
    }
    (kwh_out, peak_out)
}

impl StandardReport {
    /// Builds the standard results report from a finished simulation.
    ///
    /// This is the *only* constructor: the REST, CLI and Python surfaces all
    /// call it, so the three surfaces cannot drift apart.
    ///
    /// * `output` — the simulation result (Issue #4101 hourly series, #4102
    ///   monthly summaries, #3988/#4103 unmet hours).
    /// * `ctx` — model identity, geometry, setpoints, occupancy and run
    ///   period.
    pub fn from_simulation(output: &SimulationOutput, ctx: &StandardReportContext) -> Self {
        let heating = output.hourly_heating_kwh.as_ref();
        let cooling = output.hourly_cooling_kwh.as_ref();
        let lighting = output.hourly_lighting_kwh.as_ref();
        let equipment = output.hourly_equipment_kwh.as_ref();

        let timesteps = [heating, cooling, lighting, equipment]
            .iter()
            .filter_map(|s| s.as_ref().map(|v| v.len()))
            .max()
            .unwrap_or(0);

        // ---- annual end-use energy -------------------------------------
        let heating_kwh = series_sum(heating).unwrap_or(output.heating_energy);
        let cooling_kwh = series_sum(cooling).unwrap_or(output.cooling_energy);
        let lighting_kwh = series_sum(lighting).unwrap_or(0.0);
        let equipment_kwh = series_sum(equipment).unwrap_or(0.0);
        let end_use_kwh = EndUseEnergy::new(heating_kwh, cooling_kwh, lighting_kwh, equipment_kwh);

        // ---- annual peaks (timestep-average kW) ------------------------
        let peaks = EndUsePeak::new(
            series_peak_kw(heating, ctx.timestep_seconds)
                .unwrap_or(output.peak_heating_load / 1000.0),
            series_peak_kw(cooling, ctx.timestep_seconds)
                .unwrap_or(output.peak_cooling_load / 1000.0),
            series_peak_kw(lighting, ctx.timestep_seconds).unwrap_or(0.0),
            series_peak_kw(equipment, ctx.timestep_seconds).unwrap_or(0.0),
        );

        // ---- monthly bins ----------------------------------------------
        let monthly = Self::monthly_section(output, ctx, heating, cooling, lighting, equipment);

        // ---- unmet hours -------------------------------------------------
        let unmet_hours = UnmetHoursSection {
            all_hours: UnmetPair::new(
                output.unmet_heating_hours_all_hours,
                output.unmet_cooling_hours_all_hours,
            ),
            by_zone: Self::by_zone_section(output, ctx),
            occupied: UnmetPair::new(output.unmet_heating_hours, output.unmet_cooling_hours),
            tolerance_c: finite(ctx.unmet_hours_tolerance_c),
        };

        // ---- hourly reference --------------------------------------------
        let hourly = HourlyReference {
            checksum: format!(
                "sha256:{}",
                series_checksum([heating, cooling, lighting, equipment])
            ),
            columns: SERIES_COLUMNS.iter().map(|s| s.to_string()).collect(),
            end_uses: END_USES.iter().map(|s| s.to_string()).collect(),
            series_id: REPORT_SERIES_ID.to_string(),
            timestep_seconds: finite(ctx.timestep_seconds),
            timesteps,
        };

        // ---- metadata ----------------------------------------------------
        let metadata = ReportMetadata {
            model_name: ctx.model_name.clone(),
            model_source: ctx.model_source.clone(),
            run_period: RunPeriod {
                end: ctx.run_end.clone(),
                label: ctx.run_label.clone(),
                start: ctx.run_start.clone(),
                timestep_seconds: finite(ctx.timestep_seconds),
                timesteps,
                years: ctx.years,
            },
            schema_version: RESULTS_SCHEMA_VERSION.to_string(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let floor_area_m2 = if ctx.floor_area_m2 > 0.0 {
            ctx.floor_area_m2
        } else {
            1.0
        };
        let total_energy_kwh = end_use_kwh.total();

        StandardReport {
            annual: AnnualSection {
                end_use_kwh,
                eui_kwh_per_m2: total_energy_kwh / floor_area_m2,
                floor_area_m2,
                peaks,
                total_energy_kwh,
                zone_count: ctx.zone_count,
            },
            hourly,
            metadata,
            monthly,
            unmet_hours,
        }
    }

    /// The twelve calendar-month bins.
    ///
    /// Prefers the monthly summaries produced by Issue #4102; falls back to
    /// re-deriving them from the hourly series (#4101) when the summaries are
    /// absent. Months are aggregated over the years of the run period (kWh
    /// summed, peak kW maximised) so that the twelve bins always add up to
    /// the annual totals.
    fn monthly_section(
        output: &SimulationOutput,
        ctx: &StandardReportContext,
        heating: Option<&Vec<f64>>,
        cooling: Option<&Vec<f64>>,
        lighting: Option<&Vec<f64>>,
        equipment: Option<&Vec<f64>>,
    ) -> Vec<MonthlyMonth> {
        // Try the #4102 summaries first.
        let mut kwh_by_use: [Option<[f64; 12]>; 4] = [None, None, None, None];
        let mut peak_by_use: [Option<[f64; 12]>; 4] = [None, None, None, None];
        let stored = [
            output.monthly_heating_kwh.as_ref(),
            output.monthly_cooling_kwh.as_ref(),
            output.monthly_lighting_kwh.as_ref(),
            output.monthly_equipment_kwh.as_ref(),
        ];
        let stored_peaks = [
            output.monthly_heating_peak_kw.as_ref(),
            output.monthly_cooling_peak_kw.as_ref(),
            output.monthly_lighting_peak_kw.as_ref(),
            output.monthly_equipment_peak_kw.as_ref(),
        ];
        let mut have_stored = true;
        for u in 0..4 {
            match (stored[u], stored_peaks[u]) {
                (Some(k), Some(p)) => {
                    let (kwh, peak) = aggregate_monthly(Some(k), Some(p));
                    kwh_by_use[u] = Some(kwh);
                    peak_by_use[u] = Some(peak);
                }
                _ => {
                    have_stored = false;
                    break;
                }
            }
        }

        // Fall back to deriving the bins from the hourly series.
        if !have_stored {
            if let Some(summary) = SimulationOutput::monthly_end_use_summary(
                heating.cloned(),
                cooling.cloned(),
                lighting.cloned(),
                equipment.cloned(),
                ctx.timestep_seconds,
            ) {
                for u in 0..4 {
                    let (kwh, peak) =
                        aggregate_monthly(Some(&summary.kwh[u]), Some(&summary.peak_kw[u]));
                    kwh_by_use[u] = Some(kwh);
                    peak_by_use[u] = Some(peak);
                }
            }
        }

        (0..12)
            .map(|m| MonthlyMonth {
                end_use_kwh: EndUseEnergy::new(
                    kwh_by_use[0].map(|v| v[m]).unwrap_or(0.0),
                    kwh_by_use[1].map(|v| v[m]).unwrap_or(0.0),
                    kwh_by_use[2].map(|v| v[m]).unwrap_or(0.0),
                    kwh_by_use[3].map(|v| v[m]).unwrap_or(0.0),
                ),
                end_use_peak_kw: EndUsePeak::new(
                    peak_by_use[0].map(|v| v[m]).unwrap_or(0.0),
                    peak_by_use[1].map(|v| v[m]).unwrap_or(0.0),
                    peak_by_use[2].map(|v| v[m]).unwrap_or(0.0),
                    peak_by_use[3].map(|v| v[m]).unwrap_or(0.0),
                ),
                month: (m + 1) as u32,
                name: MONTH_NAMES[m].to_string(),
            })
            .collect()
    }

    /// Per-zone unmet hours, ordered by zone index.
    ///
    /// Uses the same helper the surfaces use for the totals
    /// ([`SimulationOutput::unmet_hours`] /
    /// [`SimulationOutput::unmet_hours_all_hours`]), so the per-zone values
    /// sum to the reported totals. Empty when the run recorded no hourly zone
    /// temperature trace.
    fn by_zone_section(output: &SimulationOutput, ctx: &StandardReportContext) -> Vec<UnmetZone> {
        let Some(zone_temps) = output.hourly_zone_temperatures.as_ref() else {
            return Vec::new();
        };
        zone_temps
            .iter()
            .enumerate()
            .map(|(index, temps)| {
                let (occ_heating, occ_cooling) = SimulationOutput::unmet_hours(
                    std::slice::from_ref(temps),
                    &ctx.occupancy,
                    ctx.heating_setpoint_c,
                    ctx.cooling_setpoint_c,
                    ctx.unmet_hours_tolerance_c,
                );
                let (all_heating, all_cooling) = SimulationOutput::unmet_hours_all_hours(
                    std::slice::from_ref(temps),
                    ctx.heating_setpoint_c,
                    ctx.cooling_setpoint_c,
                    ctx.unmet_hours_tolerance_c,
                );
                UnmetZone {
                    all_hours: UnmetPair::new(all_heating, all_cooling),
                    index,
                    name: ctx
                        .zone_names
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| format!("zone-{index}")),
                    occupied: UnmetPair::new(occ_heating, occ_cooling),
                }
            })
            .collect()
    }

    /// Canonical JSON rendering of the report (sorted keys, fixed floats).
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Canonical JSON rendering of the report, newline-terminated.
    pub fn to_json_string_terminated(&self) -> serde_json::Result<String> {
        let mut text = self.to_json_string()?;
        text.push('\n');
        Ok(text)
    }
}

/// Convenience: the canonical JSON of a report built from a simulation.
pub fn standard_report_json(
    output: &SimulationOutput,
    ctx: &StandardReportContext,
) -> serde_json::Result<String> {
    StandardReport::from_simulation(output, ctx).to_json_string_terminated()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic but realistic simulation result: 12 hours of end-use
    /// energy in two zones.
    fn sample_output() -> SimulationOutput {
        let heating: Vec<f64> = (0..12).map(|h| (h as f64) * 0.5).collect();
        let cooling: Vec<f64> = (0..12).map(|h| (h as f64) * 0.25).collect();
        let lighting: Vec<f64> = vec![0.1; 12];
        let equipment: Vec<f64> = vec![0.2; 12];
        SimulationOutput {
            eui: 2.0,
            total_energy: 10.0,
            peak_heating_load: 5_000.0,
            peak_cooling_load: 2_000.0,
            heating_energy: heating.iter().sum(),
            cooling_energy: cooling.iter().sum(),
            zone_temperatures: None,
            hourly_zone_temperatures: Some(vec![vec![21.0; 12], vec![25.0; 12]]),
            effective_solver: Some("5r1c".to_string()),
            unmet_heating_hours: 4.0,
            unmet_cooling_hours: 2.0,
            unmet_heating_hours_all_hours: 8.0,
            unmet_cooling_hours_all_hours: 4.0,
            hourly_heating_kwh: Some(heating),
            hourly_cooling_kwh: Some(cooling),
            hourly_lighting_kwh: Some(lighting),
            hourly_equipment_kwh: Some(equipment),
            monthly_heating_kwh: None,
            monthly_heating_peak_kw: None,
            monthly_cooling_kwh: None,
            monthly_cooling_peak_kw: None,
            monthly_lighting_kwh: None,
            monthly_lighting_peak_kw: None,
            monthly_equipment_kwh: None,
            monthly_equipment_peak_kw: None,
        }
    }

    fn sample_context() -> StandardReportContext {
        let mut occupancy = DailySchedule::new();
        for hour in 0..24 {
            occupancy.set_hour(hour, 1.0).unwrap();
        }
        StandardReportContext::new("Test Model", "unit-test")
            .with_floor_area(50.0)
            .with_zones(2, vec!["Zone 1".to_string(), "Zone 2".to_string()])
            .with_timestep(3600.0, 1)
            .with_setpoints(20.0, 24.0)
            .with_occupancy(occupancy)
    }

    #[test]
    fn report_has_all_five_sections() {
        let report = StandardReport::from_simulation(&sample_output(), &sample_context());
        assert_eq!(report.metadata.schema_version, "1.0.0");
        assert_eq!(report.metadata.tool_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(report.metadata.model_name, "Test Model");
        assert_eq!(report.metadata.model_source, "unit-test");
        assert_eq!(report.metadata.run_period.timesteps, 12);
        assert_eq!(report.metadata.run_period.years, 1);
        assert_eq!(report.metadata.run_period.timestep_seconds, 3600.0);
        assert_eq!(report.annual.zone_count, 2);
        assert_eq!(report.monthly.len(), 12);
        assert_eq!(report.monthly[0].name, "January");
        assert_eq!(report.monthly[11].name, "December");
        assert_eq!(report.monthly[11].month, 12);
        assert_eq!(report.unmet_hours.occupied.heating, 4.0);
        assert_eq!(report.unmet_hours.occupied.cooling, 2.0);
        assert_eq!(report.unmet_hours.all_hours.heating, 8.0);
        assert_eq!(report.unmet_hours.all_hours.cooling, 4.0);
        assert_eq!(report.hourly.series_id, "hourly-end-use");
        assert_eq!(report.hourly.timesteps, 12);
        assert_eq!(report.hourly.end_uses.len(), 4);
        assert_eq!(report.hourly.columns.len(), 4);
        assert!(report.hourly.checksum.starts_with("sha256:"));
        assert_eq!(report.hourly.checksum.len(), "sha256:".len() + 64);
    }

    #[test]
    fn annual_energy_and_eui_are_consistent() {
        let report = StandardReport::from_simulation(&sample_output(), &sample_context());
        // heating 0.5+1+...+5.5 = 33.0, cooling 16.5, lighting 1.2, equipment 2.4
        assert!((report.annual.end_use_kwh.heating - 33.0).abs() < 1e-9);
        assert!((report.annual.end_use_kwh.cooling - 16.5).abs() < 1e-9);
        assert!((report.annual.end_use_kwh.lighting - 1.2).abs() < 1e-9);
        assert!((report.annual.end_use_kwh.equipment - 2.4).abs() < 1e-9);
        let total = report.annual.end_use_kwh.total();
        assert!((report.annual.total_energy_kwh - total).abs() < 1e-9);
        assert!((report.annual.eui_kwh_per_m2 - total / 50.0).abs() < 1e-9);
        // Peak heating timestep is 5.5 kWh in one hour -> 5.5 kW.
        assert!((report.annual.peaks.heating - 5.5).abs() < 1e-9);
        assert!((report.annual.peaks.cooling - 2.75).abs() < 1e-9);
    }

    #[test]
    fn monthly_bins_sum_to_the_annual_energy() {
        let report = StandardReport::from_simulation(&sample_output(), &sample_context());
        let heating: f64 = report.monthly.iter().map(|m| m.end_use_kwh.heating).sum();
        assert!(
            (heating - report.annual.end_use_kwh.heating).abs() < 1e-9,
            "monthly bins must add up to the annual total"
        );
        // The whole series is 12 hours, so it all lands in January.
        assert!((report.monthly[0].end_use_kwh.heating - 33.0).abs() < 1e-9);
        assert!((report.monthly[1].end_use_kwh.heating - 0.0).abs() < 1e-9);
    }

    #[test]
    fn per_zone_unmet_hours_are_in_index_order() {
        let report = StandardReport::from_simulation(&sample_output(), &sample_context());
        assert_eq!(report.unmet_hours.by_zone.len(), 2);
        assert_eq!(report.unmet_hours.by_zone[0].index, 0);
        assert_eq!(report.unmet_hours.by_zone[1].index, 1);
        assert_eq!(report.unmet_hours.by_zone[0].name, "Zone 1");
        assert_eq!(report.unmet_hours.by_zone[1].name, "Zone 2");
        // Zone 1 sits at 21 °C (inside the 19.8–24.2 °C band) and zone 2 at
        // 25 °C (above the cooling setpoint): zone 2 has 12 unmet cooling
        // hours and no unmet heating hours.
        assert_eq!(report.unmet_hours.by_zone[0].occupied.heating, 0.0);
        assert_eq!(report.unmet_hours.by_zone[0].occupied.cooling, 0.0);
        assert_eq!(report.unmet_hours.by_zone[1].occupied.heating, 0.0);
        assert_eq!(report.unmet_hours.by_zone[1].occupied.cooling, 12.0);
    }

    #[test]
    fn serialization_is_byte_deterministic() {
        let a = StandardReport::from_simulation(&sample_output(), &sample_context())
            .to_json_string()
            .unwrap();
        let b = StandardReport::from_simulation(&sample_output(), &sample_context())
            .to_json_string()
            .unwrap();
        assert_eq!(a, b, "identical inputs must yield byte-identical JSON");
        assert!(a.contains("\"schema_version\": \"1.0.0\""));
    }

    #[test]
    fn object_keys_are_sorted_and_floats_fixed() {
        let json = StandardReport::from_simulation(&sample_output(), &sample_context())
            .to_json_string()
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_sorted_keys(&value);
        // Fixed float formatting: every float is rendered with exactly six
        // decimals (no exponent, no trailing-zero trimming).
        assert!(json.contains("\"eui_kwh_per_m2\": 1.062000"), "{json}");
        assert!(
            !has_exponent_number(&json),
            "no scientific notation outside strings: {json}"
        );
    }

    /// True when a number token outside any string uses exponent notation
    /// (`1e-3`, `2.5E+6`, ...). String contents are skipped: the series
    /// checksum is hex and legitimately contains `e` after digits.
    fn has_exponent_number(json: &str) -> bool {
        let mut in_string = false;
        let mut escaped = false;
        let mut prev = '\0';
        for ch in json.chars() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                'e' | 'E' if prev.is_ascii_digit() || prev == '.' => return true,
                _ => {}
            }
            if !ch.is_whitespace() {
                prev = ch;
            }
        }
        false
    }

    /// Recursively asserts that every JSON object has lexicographically
    /// sorted keys.
    fn assert_sorted_keys(value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                let keys: Vec<&String> = map.keys().collect();
                let mut sorted = keys.clone();
                sorted.sort();
                assert_eq!(keys, sorted, "object keys must be sorted");
                for v in map.values() {
                    assert_sorted_keys(v);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    assert_sorted_keys(item);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn checksum_changes_when_the_series_changes() {
        let base = StandardReport::from_simulation(&sample_output(), &sample_context());
        let mut mutated = sample_output();
        let mut heating = mutated.hourly_heating_kwh.clone().unwrap();
        heating[0] += 0.001;
        mutated.hourly_heating_kwh = Some(heating);
        let changed = StandardReport::from_simulation(&mutated, &sample_context());
        assert_ne!(
            base.hourly.checksum, changed.hourly.checksum,
            "the series checksum must cover the hourly values"
        );
        assert_ne!(
            base.to_json_string().unwrap(),
            changed.to_json_string().unwrap()
        );
    }

    #[test]
    fn missing_series_still_produces_a_valid_report() {
        let mut output = sample_output();
        output.hourly_heating_kwh = None;
        output.hourly_cooling_kwh = None;
        output.hourly_lighting_kwh = None;
        output.hourly_equipment_kwh = None;
        let report = StandardReport::from_simulation(&output, &sample_context());
        assert_eq!(report.hourly.timesteps, 0);
        assert_eq!(report.monthly.len(), 12);
        // Falls back to the output's own annual totals.
        assert!((report.annual.end_use_kwh.heating - output.heating_energy).abs() < 1e-9);
        // Peaks fall back to the reported peak loads (W -> kW).
        assert!((report.annual.peaks.heating - 5.0).abs() < 1e-9);
    }

    #[test]
    fn non_finite_values_are_reported_as_zero() {
        let mut output = sample_output();
        output.hourly_heating_kwh = Some(vec![f64::NAN, 1.0]);
        let report = StandardReport::from_simulation(&output, &sample_context());
        assert!(report.annual.end_use_kwh.heating.is_finite());
        assert!((report.annual.end_use_kwh.heating - 1.0).abs() < 1e-9);
        let json = report.to_json_string().unwrap();
        assert!(!json.contains("NaN"), "{json}");
    }

    #[test]
    fn format_fixed_is_stable() {
        assert_eq!(format_fixed(0.0).unwrap(), "0.000000");
        assert_eq!(format_fixed(-0.0).unwrap(), "0.000000");
        assert_eq!(format_fixed(1.5).unwrap(), "1.500000");
        assert_eq!(format_fixed(1.0 / 3.0).unwrap(), "0.333333");
        assert!(format_fixed(f64::NAN).is_err());
        assert!(format_fixed(f64::INFINITY).is_err());
    }
}
