//! `mod.rs` — re-export shim + core report vocabulary. The Issue #3457
//! module-size ratchet pinned `src/validation/report.rs` at 4136/4136
//! lines (see [`tests/reference_data/module_size/report_ratchet.json`](../../../../tests/reference_data/module_size/report_ratchet.json)),
//! so the file was decomposed into focused child files at Issue #3788.
//! Public API is preserved unchanged; every `crate::validation::report::X`
//! path and every `fluxion::validation::report::X` import continues to
//! work because `mod.rs` re-exports the public symbols from the child
//! modules.
//!
//! - [`status`] vocabulary stays in this file: [`ReportHeader`],
//!   [`MetricType`] (+`Ord`/`Display`), [`ValidationStatus`] +
//!   [`compute_status`], [`ReferenceProgram`], [`BenchmarkData`], plus
//!   the shared result types [`ValidationResult`] and [`Interpretation`]
//!   that every child module crosses.
//! - [`benchmark`] — [`BenchmarkReport`] + both `impl BenchmarkReport`
//!   blocks (metric assembly, rendering, delta/sensitivity).
//! - [`multi_zone`] — [`MultiZoneValidationReport`], [`Case960Report`],
//!   [`Case970Report`], [`MultiZoneSummary`], [`ValidationSuite`] +
//!   interpretation-generation helpers.
//! - `tests` — the inline `#[cfg(test)]` unit-test module (extracted to
//!   `tests.rs` so the production code stays clear of the test bodies).
//!
//! # Validation report generation and analysis for ASHRAE 140.
//!
//! The parent module provides structures and methods for generating
//! comprehensive validation reports, including pass/fail determination,
//! delta analysis, and multiple export formats (Markdown, HTML, CSV).
//! The body below holds only the core report vocabulary and the public
//! re-exports; see the child modules for `BenchmarkReport`, the multi-zone
//! family, and the `ValidationSuite` collection wrapper.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fmt;

/// ASHRAE 140-2023 Section 8.1 compliance report header.
///
/// Required fields as specified by ASHRAE 140-2023 Section 8.1 for
/// compliance reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportHeader {
    /// Program name (e.g., "fluxion")
    pub program_name: String,
    /// Program version (e.g., "1.0.0")
    pub program_version: String,
    /// Developer name/organization (configurable)
    pub developer: String,
    /// Simulation run timestamp
    pub run_date: DateTime<Utc>,
    /// ASHRAE 140 standard edition (e.g., "ASHRAE 140-2023")
    pub ashrae_edition: String,
    /// Weather file identification (from EPW header)
    pub weather_file_id: String,
}

impl ReportHeader {
    /// Creates a new report header with current timestamp.
    ///
    /// Uses the crate version from Cargo.toml and the default developer "Fluxion Development Team".
    /// Weather file ID is extracted from the EPW header for ASHRAE 140 validation.
    pub fn new(weather_file_id: String) -> Self {
        Self {
            program_name: "fluxion".to_string(),
            program_version: env!("CARGO_PKG_VERSION").to_string(),
            developer: "Fluxion Development Team".to_string(),
            run_date: Utc::now(),
            ashrae_edition: "ASHRAE 140-2023".to_string(),
            weather_file_id,
        }
    }

    /// Creates a report header with custom developer name.
    pub fn with_developer(mut self, developer: &str) -> Self {
        self.developer = developer.to_string();
        self
    }
}

impl Default for ReportHeader {
    fn default() -> Self {
        Self {
            program_name: "fluxion".to_string(),
            program_version: env!("CARGO_PKG_VERSION").to_string(),
            developer: "Fluxion Development Team".to_string(),
            run_date: Utc::now(),
            ashrae_edition: "ASHRAE 140-2023".to_string(),
            weather_file_id: "USA_CO_Denver-Stapleton.Intl.AP.724690_TMY".to_string(),
        }
    }
}

/// Types of validation metrics for ASHRAE 140.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Annual heating energy consumption (kWh)
    AnnualHeating,
    /// Annual cooling energy consumption (kWh)
    AnnualCooling,
    /// Peak heating load (kW)
    PeakHeating,
    /// Peak cooling load (kW)
    PeakCooling,
    /// Minimum free-floating temperature (°C)
    MinFreeFloat,
    /// Maximum free-floating temperature (°C)
    MaxFreeFloat,
    /// Incident solar radiation per surface orientation (kWh/m²).
    /// Per ASHRAE 140-2023 Section 8.2.3, outputs annual and peak solar per orientation.
    IncidentSolar {
        /// Surface identifier (e.g., "roof", "N", "S", "E", "W")
        surface_id: String,
        /// Surface orientation
        orientation: crate::validation::ashrae_140_cases::Orientation,
    },
}

impl PartialOrd for MetricType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MetricType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Explicit ordering: base variants alphabetically, IncidentSolar last
        match (self, other) {
            (MetricType::AnnualHeating, MetricType::AnnualHeating) => std::cmp::Ordering::Equal,
            (MetricType::AnnualHeating, _) => std::cmp::Ordering::Less,
            (_, MetricType::AnnualHeating) => std::cmp::Ordering::Greater,
            (MetricType::AnnualCooling, MetricType::AnnualCooling) => std::cmp::Ordering::Equal,
            (MetricType::AnnualCooling, _) => std::cmp::Ordering::Less,
            (_, MetricType::AnnualCooling) => std::cmp::Ordering::Greater,
            (MetricType::PeakHeating, MetricType::PeakHeating) => std::cmp::Ordering::Equal,
            (MetricType::PeakHeating, _) => std::cmp::Ordering::Less,
            (_, MetricType::PeakHeating) => std::cmp::Ordering::Greater,
            (MetricType::PeakCooling, MetricType::PeakCooling) => std::cmp::Ordering::Equal,
            (MetricType::PeakCooling, _) => std::cmp::Ordering::Less,
            (_, MetricType::PeakCooling) => std::cmp::Ordering::Greater,
            (MetricType::MinFreeFloat, MetricType::MinFreeFloat) => std::cmp::Ordering::Equal,
            (MetricType::MinFreeFloat, _) => std::cmp::Ordering::Less,
            (_, MetricType::MinFreeFloat) => std::cmp::Ordering::Greater,
            (MetricType::MaxFreeFloat, MetricType::MaxFreeFloat) => std::cmp::Ordering::Equal,
            (MetricType::MaxFreeFloat, _) => std::cmp::Ordering::Less,
            (_, MetricType::MaxFreeFloat) => std::cmp::Ordering::Greater,
            (
                MetricType::IncidentSolar {
                    surface_id: a_sid,
                    orientation: a_ori,
                },
                MetricType::IncidentSolar {
                    surface_id: b_sid,
                    orientation: b_ori,
                },
            ) => a_sid.cmp(b_sid).then_with(|| a_ori.cmp(b_ori)),
        }
    }
}

impl MetricType {
    /// Returns the display name for this metric type (ASHRAE 140 compliant).
    pub fn display_name(&self) -> &str {
        match self {
            MetricType::AnnualHeating => "Annual Heating Energy (kWh)",
            MetricType::AnnualCooling => "Annual Cooling Energy (kWh)",
            MetricType::PeakHeating => "Peak Heating Load (kW)",
            MetricType::PeakCooling => "Peak Cooling Load (kW)",
            MetricType::MinFreeFloat => "Minimum Free-Floating Temperature (°C)",
            MetricType::MaxFreeFloat => "Maximum Free-Floating Temperature (°C)",
            MetricType::IncidentSolar { .. } => "Incident Solar Radiation (kWh/m²)",
        }
    }

    /// Returns the units for this metric type.
    pub fn units(&self) -> &str {
        match self {
            MetricType::AnnualHeating | MetricType::AnnualCooling => "kWh",
            MetricType::PeakHeating | MetricType::PeakCooling => "kW",
            MetricType::MinFreeFloat | MetricType::MaxFreeFloat => "°C",
            MetricType::IncidentSolar { .. } => "kWh/m²",
        }
    }

    /// Converts internal MWh storage to Section 8 compliant kWh output.
    ///
    /// Issue #749: Section 8 requires kWh but internal storage is MWh.
    /// This method multiplies annual energy values by 1000 to convert MWh→kWh.
    pub fn to_output_units(&self, value: f64) -> f64 {
        match self {
            MetricType::AnnualHeating | MetricType::AnnualCooling => value * 1000.0,
            _ => value,
        }
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Validation status for a single metric comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Value within 5% of reference range
    Pass,
    /// Value within reference range but >2% deviation
    Warning,
    /// Value outside 5% tolerance band
    Fail,
}

impl fmt::Display for ValidationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationStatus::Pass => write!(f, "PASS"),
            ValidationStatus::Warning => write!(f, "WARN"),
            ValidationStatus::Fail => write!(f, "FAIL"),
        }
    }
}

impl ValidationStatus {
    /// Returns the display name for this status.
    pub fn display_name(&self) -> &str {
        match self {
            ValidationStatus::Pass => "PASS",
            ValidationStatus::Warning => "WARN",
            ValidationStatus::Fail => "FAIL",
        }
    }

    /// Returns the emoji icon for this status (for terminal output).
    pub fn icon(&self) -> &str {
        match self {
            ValidationStatus::Pass => "✓",
            ValidationStatus::Warning => "⚠",
            ValidationStatus::Fail => "✗",
        }
    }

    /// Returns the color code for HTML output.
    pub fn color(&self) -> &str {
        match self {
            ValidationStatus::Pass => "green",
            ValidationStatus::Warning => "orange",
            ValidationStatus::Fail => "red",
        }
    }
}

/// Computes validation status for a given value against a reference range.
///
/// Status determination according to ASHRAE 140-2017:
/// - Pass: value within [min, max] with <10% deviation from midpoint
/// - Warning: within [min, max] but >=10% deviation, OR within tolerance band [min*0.95, max*1.05]
/// - Fail: outside tolerance band
pub fn compute_status(value: f64, ref_min: f64, ref_max: f64) -> ValidationStatus {
    let ref_mid = (ref_min + ref_max) / 2.0;
    let percent_error = if ref_mid != 0.0 {
        ((value - ref_mid) / ref_mid.abs()) * 100.0
    } else {
        0.0
    };

    let tolerance_min = ref_min * 0.95;
    let tolerance_max = ref_max * 1.05;

    if value >= ref_min && value <= ref_max {
        // Within reference range - check percent error
        // Use 10% threshold to match ValidationResult::new behavior
        if percent_error.abs() >= 10.0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Pass
        }
    } else if value >= tolerance_min && value <= tolerance_max {
        ValidationStatus::Warning
    } else {
        ValidationStatus::Fail
    }
}

/// Reference programs for ASHRAE 140 validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceProgram {
    /// EnergyPlus - DOE's flagship building energy simulation program
    EnergyPlus,
    /// ESP-r - Research-grade building energy simulation from University of Strathclyde
    EspR,
    /// TRNSYS - Transient System Simulation Tool
    TRNSYS,
    /// DOE2 - Legacy DOE building energy simulation program
    DOE2,
}

impl fmt::Display for ReferenceProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceProgram::EnergyPlus => write!(f, "EnergyPlus"),
            ReferenceProgram::EspR => write!(f, "ESP-r"),
            ReferenceProgram::TRNSYS => write!(f, "TRNSYS"),
            ReferenceProgram::DOE2 => write!(f, "DOE2"),
        }
    }
}

/// Benchmark data for a single ASHRAE 140 case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkData {
    /// Minimum annual heating (MWh) across reference programs
    pub annual_heating_min: f64,
    /// Maximum annual heating (MWh) across reference programs
    pub annual_heating_max: f64,
    /// Minimum annual cooling (MWh) across reference programs
    pub annual_cooling_min: f64,
    /// Maximum annual cooling (MWh) across reference programs
    pub annual_cooling_max: f64,
    /// Minimum peak heating load (kW) across reference programs
    pub peak_heating_min: f64,
    /// Maximum peak heating load (kW) across reference programs
    pub peak_heating_max: f64,
    /// Minimum peak cooling load (kW) across reference programs
    pub peak_cooling_min: f64,
    /// Maximum peak cooling load (kW) across reference programs
    pub peak_cooling_max: f64,
    /// Minimum free-floating temperature (°C) across reference programs
    pub min_free_float_min: f64,
    /// Maximum free-floating temperature (°C) across reference programs
    pub min_free_float_max: f64,
    /// Maximum free-floating temperature (°C) across reference programs
    pub max_free_float_min: f64,
    /// Maximum free-floating temperature (°C) across reference programs
    pub max_free_float_max: f64,
}

impl BenchmarkData {
    /// Creates a new BenchmarkData with all values initialized to zero.
    pub fn new() -> Self {
        Self {
            annual_heating_min: 0.0,
            annual_heating_max: 0.0,
            annual_cooling_min: 0.0,
            annual_cooling_max: 0.0,
            peak_heating_min: 0.0,
            peak_heating_max: 0.0,
            peak_cooling_min: 0.0,
            peak_cooling_max: 0.0,
            min_free_float_min: 0.0,
            min_free_float_max: 0.0,
            max_free_float_min: 0.0,
            max_free_float_max: 0.0,
        }
    }

    /// Returns the reference range for a given metric type.
    pub fn get_range(&self, metric: MetricType) -> Option<(f64, f64)> {
        match metric {
            MetricType::AnnualHeating => {
                if self.annual_heating_min > 0.0 || self.annual_heating_max > 0.0 {
                    Some((self.annual_heating_min, self.annual_heating_max))
                } else {
                    None
                }
            }
            MetricType::AnnualCooling => {
                if self.annual_cooling_min > 0.0 || self.annual_cooling_max > 0.0 {
                    Some((self.annual_cooling_min, self.annual_cooling_max))
                } else {
                    None
                }
            }
            MetricType::PeakHeating => {
                if self.peak_heating_min > 0.0 || self.peak_heating_max > 0.0 {
                    Some((self.peak_heating_min, self.peak_heating_max))
                } else {
                    None
                }
            }
            MetricType::PeakCooling => {
                if self.peak_cooling_min > 0.0 || self.peak_cooling_max > 0.0 {
                    Some((self.peak_cooling_min, self.peak_cooling_max))
                } else {
                    None
                }
            }
            MetricType::MinFreeFloat => {
                if self.min_free_float_min != 0.0 || self.min_free_float_max != 0.0 {
                    Some((self.min_free_float_min, self.min_free_float_max))
                } else {
                    None
                }
            }
            MetricType::MaxFreeFloat => {
                if self.max_free_float_min != 0.0 || self.max_free_float_max != 0.0 {
                    Some((self.max_free_float_min, self.max_free_float_max))
                } else {
                    None
                }
            }
            MetricType::IncidentSolar { .. } => None,
        }
    }

    /// Calculates the midpoint of the reference range for a given metric.
    pub fn midpoint(&self, metric: MetricType) -> Option<f64> {
        self.get_range(metric).map(|(min, max)| (min + max) / 2.0)
    }
}

impl Default for BenchmarkData {
    fn default() -> Self {
        Self::new()
    }
}

/// A single validation result for a specific case and metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Case identifier (e.g., "600", "900", "600FF")
    pub case_id: String,
    /// Metric type
    pub metric: MetricType,
    /// Fluxion simulation value
    pub fluxion_value: f64,
    /// Reference minimum value
    pub ref_min: f64,
    /// Reference maximum value
    pub ref_max: f64,
    /// Percent error from reference midpoint
    pub percent_error: f64,
    /// Validation status
    pub status: ValidationStatus,
    /// Per-program validation statuses for multi-reference comparison
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_program: Option<HashMap<String, ValidationStatus>>,
    /// Date of peak value occurrence (e.g., "Jan 15") for peak metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_date: Option<String>,
    /// Hour of peak value occurrence (0-23) for peak metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_hour: Option<u32>,
    /// Issue #761: Peak timestamp (month, day, hour) per ASHRAE 140-2023 Section 8.2.2.
    /// Only populated for PeakHeating and PeakCooling metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_timestamp: Option<(u32, u32, u32)>,
}

impl ValidationResult {
    /// Creates a new validation result and determines pass/fail status.
    pub fn new(
        case_id: &str,
        metric: MetricType,
        fluxion_value: f64,
        ref_min: f64,
        ref_max: f64,
    ) -> Self {
        // Calculate reference midpoint
        let ref_mid = (ref_min + ref_max) / 2.0;

        // Calculate percent error from reference midpoint
        let percent_error = if ref_mid != 0.0 {
            ((fluxion_value - ref_mid) / ref_mid.abs()) * 100.0
        } else {
            0.0
        };

        // Determine pass/fail status
        // Pass: Within [Ref Min, Ref Max] with <10% deviation from midpoint
        // Warning: Within [Ref Min, Ref Max] with >=10% deviation, OR within tolerance band but outside ref range
        // Fail: Outside 5% tolerance band
        let tolerance_min = ref_min * 0.95;
        let tolerance_max = ref_max * 1.05;

        let status = if fluxion_value >= ref_min && fluxion_value <= ref_max {
            // Within reference range - check percent error
            if percent_error.abs() >= 10.0 {
                ValidationStatus::Warning
            } else {
                ValidationStatus::Pass
            }
        } else if fluxion_value >= tolerance_min && fluxion_value <= tolerance_max {
            // Within tolerance band but outside reference range
            ValidationStatus::Warning
        } else {
            ValidationStatus::Fail
        };

        Self {
            case_id: case_id.to_string(),
            metric,
            fluxion_value,
            ref_min,
            ref_max,
            percent_error,
            status,
            per_program: None,
            peak_date: None,
            peak_hour: None,
            peak_timestamp: None,
        }
    }

    /// Returns true if this result passed validation (within reference range with <10% error).
    pub fn is_pass(&self) -> bool {
        matches!(self.status, ValidationStatus::Pass)
    }

    /// Returns true if this result is a warning (within reference range but >=10% error, or within tolerance band).
    pub fn is_warning(&self) -> bool {
        matches!(self.status, ValidationStatus::Warning)
    }

    /// Returns true if this result failed validation (outside tolerance band).
    pub fn is_fail(&self) -> bool {
        matches!(self.status, ValidationStatus::Fail)
    }

    /// Returns the deviation from reference range center as a string.
    pub fn deviation_string(&self) -> String {
        format!("{:+.2}%", self.percent_error)
    }

    /// Returns true if the value is within the reference range.
    pub fn is_within_range(&self) -> bool {
        self.fluxion_value >= self.ref_min && self.fluxion_value <= self.ref_max
    }

    /// Returns the deviation from the reference range center as a percentage.
    pub fn deviation_percent(&self) -> f64 {
        self.percent_error
    }

    /// Returns true if this result passed validation.
    pub fn passed(&self) -> bool {
        self.status == ValidationStatus::Pass
    }

    /// Returns true if this result is a warning.
    pub fn warning(&self) -> bool {
        self.status == ValidationStatus::Warning
    }

    /// Returns true if this result failed validation.
    pub fn failed(&self) -> bool {
        self.status == ValidationStatus::Fail
    }
}

/// Interpretation guidance for failed validation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interpretation {
    /// Root cause hypotheses explaining why the metric failed
    pub root_cause_hypotheses: Vec<String>,
    /// Parameter sensitivity analysis
    pub parameter_sensitivity: Vec<String>,
    /// Recommended next steps for investigation
    pub recommended_next_steps: Vec<String>,
    /// What-if scenarios for debugging approaches
    pub what_if_scenarios: Vec<String>,
    /// References to relevant documentation
    pub references: Vec<String>,
}

impl Default for Interpretation {
    fn default() -> Self {
        Self {
            root_cause_hypotheses: Vec::new(),
            parameter_sensitivity: Vec::new(),
            recommended_next_steps: Vec::new(),
            what_if_scenarios: Vec::new(),
            references: Vec::new(),
        }
    }
}

mod benchmark;
mod multi_zone;

pub use benchmark::{
    BenchmarkReport, DeltaReport, DeltaResult, DeltaTestResult, SensitivityResult,
};
pub use multi_zone::{
    Case960Report, Case970Report, MultiZoneSummary, MultiZoneValidationReport, ValidationSuite,
};

#[cfg(test)]
mod tests;
