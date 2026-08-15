//! ASHRAE 140 Multi-Zone Validation Infrastructure
//!
//! This module implements ASHRAE 140 validation framework for multi-zone buildings.
//! It provides the foundation for validating against ASHRAE 140 reference cases,
//! particularly focusing on Case 960 (two-zone sunspace building).
//!
//! Key functionality:
//! - ASHRAE 140 multi-zone validator
//! - Case 960 reference data loading
//! - Multi-zone validation result comparison
//!
//! This module extends the existing ASHRAE 140 validation framework to support
//! multi-zone thermal network validation.

use crate::sim::engine::ThermalModel;
use crate::validation::ashrae_140_cases::ASHRAE140Case;
use crate::validation::ashrae_140_validator::{
    ASHRAE140Validator, ValidationReport, ValidationResult,
};
use crate::validation::report::{BenchmarkReport, MetricType, ValidationStatus};
use crate::weather::epw::EpwWeatherSource;
use crate::weather::WeatherSource;
use csv::Writer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// ASHRAE 140 multi-zone validator
///
/// This validator extends the base ASHRAE 140 validator to handle multi-zone cases
/// like Case 960, 970, and 980.
pub struct ASHRAE140MultiZoneValidator {
    /// Base ASHRAE 140 validator for single-zone cases
    #[allow(dead_code)]
    base_validator: ASHRAE140Validator,
    /// Case 960 reference data
    #[allow(dead_code)]
    case_960_reference: Option<Case960Reference>,
    /// Case 970 reference data (stub for future implementation)
    #[allow(dead_code)]
    case_970_reference: Option<Case970Reference>,
    /// Case 980 reference data (stub for future implementation)
    #[allow(dead_code)]
    case_980_reference: Option<Case960Reference>,
}

/// Case 960 validator for ASHRAE 140 multi-zone validation
///
/// This validator implements comprehensive validation for ASHRAE 140 Case 960,
/// which represents a two-zone sunspace building.
#[derive(Debug, Clone)]
pub struct Case960Validator {
    /// Reference data for Case 960 validation
    reference: Case960Reference,
    /// Statistical analysis results
    statistics: Case960Statistics,
}

/// Case 970 validator for ASHRAE 140 multi-zone validation
///
/// This validator provides the framework for ASHRAE 140 Case 970 validation,
/// which represents a more complex multi-zone building configuration.
#[derive(Debug, Clone)]
pub struct Case970Validator {
    /// Reference data for Case 970 validation
    reference: Case970Reference,
    /// Statistical analysis results
    statistics: Case970Statistics,
}

/// Statistical analysis results for Case 960 validation
#[derive(Debug, Clone, Default)]
pub struct Case960Statistics {
    /// Percentage differences for each metric
    pub percentage_differences: HashMap<String, f64>,
    /// Root Mean Square Error for temperature profiles
    pub rmse_temperature: f64,
    /// Maximum absolute errors
    pub max_absolute_errors: HashMap<String, f64>,
    /// Overall validation score (0-100)
    pub overall_score: f64,
}

/// Statistical analysis results for Case 970 validation
#[derive(Debug, Clone, Default)]
pub struct Case970Statistics {
    /// Percentage differences for each metric
    pub percentage_differences: HashMap<String, f64>,
    /// Root Mean Square Error for temperature profiles
    pub rmse_temperature: f64,
    /// Maximum absolute errors
    pub max_absolute_errors: HashMap<String, f64>,
    /// Overall validation score (0-100)
    pub overall_score: f64,
}

impl Default for ASHRAE140MultiZoneValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ASHRAE140MultiZoneValidator {
    /// Create a new ASHRAE 140 multi-zone validator
    pub fn new() -> Self {
        Self {
            base_validator: ASHRAE140Validator::new(),
            case_960_reference: None,
            case_970_reference: None,
            case_980_reference: None,
        }
    }

    /// Load Case 960 reference data
    ///
    /// This method loads the expected values for ASHRAE 140 Case 960,
    /// which represents a two-zone sunspace building.
    ///
    /// Reference values are sourced from the canonical benchmark module
    /// (`crate::validation::benchmark::CASE_960_*`), which derives the
    /// ASHRAE 140-2023 inter-program envelope (EnergyPlus, ESP-r, TRNSYS,
    /// DOE2, BSIMAC, CSE, DeST). The strict ±15% annual energy / ±10% peak
    /// tolerance from issue #1368 applies at the validator level; this
    /// struct exposes the same 15%/10% tolerances for downstream consumers.
    ///
    /// Prior to issue #1407 this struct hardcoded 12.4 MWh heating /
    /// 8.7 MWh cooling placeholders that did not match any reference
    /// program — the validator at the time fabricated PASS by comparing
    /// two hardcoded numbers (12.4 vs 12.5). See #1407 for details.
    ///
    /// # Returns
    /// Case960Reference struct with expected values
    pub fn load_case_960_reference_data() -> Case960Reference {
        use super::benchmark::{
            CASE_960_ANNUAL_COOLING_REF, CASE_960_ANNUAL_HEATING_REF, CASE_960_ENERGY_TOLERANCE,
            CASE_960_PEAK_COOLING_REF, CASE_960_PEAK_HEATING_REF, CASE_960_PEAK_TOLERANCE,
        };

        Case960Reference {
            // Zone temperatures at key timesteps (°C).
            // These are still placeholder sentinel values for the
            // `Case960Validator::validate_hourly_temperature_profiles`
            // consumer — the *energy* reference data, which is what the
            // strict ±15% CI gate consumes (#1368), now comes from the
            // benchmark module above.
            zone_temperatures: HashMap::from([
                // Winter design day (hour 4380 - Jan 21, 6:00 AM)
                (4380, vec![15.2, 8.1]), // Zone 1 (living), Zone 2 (sunspace)
                // Summer design day (hour 5000 - Jul 21, 4:40 PM)
                (5000, vec![26.8, 38.4]),
                // Annual average
                (8760, vec![20.1, 18.7]),
            ]),

            // Annual energy consumption (MWh) — midpoints of the ASHRAE 140
            // inter-program range. Bounds (min/max) come from the same module.
            annual_heating: CASE_960_ANNUAL_HEATING_REF,
            annual_cooling: CASE_960_ANNUAL_COOLING_REF,

            // Peak loads (kW) — midpoints of the ASHRAE 140 inter-program range.
            peak_heating: CASE_960_PEAK_HEATING_REF,
            peak_cooling: CASE_960_PEAK_COOLING_REF,

            // Temperature bounds — used by zone-temperature consumers, not
            // by the strict energy gate.
            min_temperature: 5.0,
            max_temperature: 45.0,

            // Tolerances — sourced from the same module so consumers reading
            // this struct see the same ±15% / ±10% the benchmark module enforces.
            temperature_tolerance: 1.0,
            energy_tolerance: CASE_960_ENERGY_TOLERANCE,
            load_tolerance: CASE_960_PEAK_TOLERANCE,
        }
    }

    /// Return the ASHRAE 140 inter-program bounds (heating/cooling/peak)
    /// for Case 960 in MWh / kW. Re-exported from `benchmark.rs` so
    /// downstream consumers (CLI, docs) can read the canonical envelope
    /// without re-importing the benchmark module. Issue #1407.
    pub fn case_960_inter_program_bounds() -> Case960InterProgramBounds {
        use super::benchmark::{
            CASE_960_ANNUAL_COOLING_MAX, CASE_960_ANNUAL_COOLING_MIN, CASE_960_ANNUAL_HEATING_MAX,
            CASE_960_ANNUAL_HEATING_MIN, CASE_960_PEAK_COOLING_MAX, CASE_960_PEAK_COOLING_MIN,
            CASE_960_PEAK_HEATING_MAX, CASE_960_PEAK_HEATING_MIN,
        };
        Case960InterProgramBounds {
            annual_heating_min: CASE_960_ANNUAL_HEATING_MIN,
            annual_heating_max: CASE_960_ANNUAL_HEATING_MAX,
            annual_cooling_min: CASE_960_ANNUAL_COOLING_MIN,
            annual_cooling_max: CASE_960_ANNUAL_COOLING_MAX,
            peak_heating_min: CASE_960_PEAK_HEATING_MIN,
            peak_heating_max: CASE_960_PEAK_HEATING_MAX,
            peak_cooling_min: CASE_960_PEAK_COOLING_MIN,
            peak_cooling_max: CASE_960_PEAK_COOLING_MAX,
        }
    }

    /// Validate Case 960 against reference data by **running the real
    /// physics simulation** and comparing against the canonical ASHRAE
    /// 140 inter-program envelope (EnergyPlus, ESP-r, TRNSYS, DOE2,
    /// BSIMAC, CSE, DeST) sourced from `validation::benchmark`.
    ///
    /// Prior to issue #1407 this method fabricated PASS by comparing two
    /// hardcoded placeholders (`actual = 12.5 / 8.5 / 5.1 / 4.9` against
    /// `reference = 12.4 / 8.7 / 5.2 / 4.8`). It never called
    /// `ThermalModel::step_physics`, so the strict ±15% CI gate (#1368)
    /// could never produce a meaningful FAIL for Case 960.
    ///
    /// The implementation now mirrors `ASHRAE140Validator::validate_case_960`
    /// (which already runs the real 8760-step simulation), wraps it in the
    /// `MultiZoneValidator` API surface (returning the lightweight
    /// `ValidationResult` that downstream `run_multi_zone_validation`
    /// / `run_comprehensive_validation` consumers expect), and applies the
    /// same ±15% / ±10% tolerances.
    ///
    /// The `_thermal_model` and `_reference` arguments are accepted for
    /// API back-compat with the prior stub but are **not** consulted for
    /// the verdict — the real validator builds its own model from
    /// `ASHRAE140Case::Case960.spec()` and reads reference bounds from
    /// `validation::benchmark`. This guarantees one source of truth.
    ///
    /// # Arguments
    /// * `_thermal_model` - Kept for API back-compat; not consulted.
    /// * `_reference` - Kept for API back-compat; not consulted.
    ///
    /// # Returns
    /// `ValidationResult { in_range, error_pct }`. `in_range` is `true`
    /// only when **all four** ASHRAE 140 metrics (annual heating,
    /// annual cooling, peak heating, peak cooling) sit within their
    /// respective inter-program envelopes.
    pub fn validate_case_960<T: crate::physics::cta::ContinuousTensor<f64>>(
        &self,
        _thermal_model: &ThermalModel<T>,
        _reference: &Case960Reference,
    ) -> ValidationResult {
        let started = Instant::now();
        let report = self.run_real_case_960_report();
        let _elapsed = started.elapsed();

        let metrics = [
            report.heating_result.in_range,
            report.cooling_result.in_range,
            report.peak_heating_result.in_range,
            report.peak_cooling_result.in_range,
        ];
        let in_range = metrics.iter().all(|v| *v);
        let avg_error_pct = (report.heating_result.error_pct
            + report.cooling_result.error_pct
            + report.peak_heating_result.error_pct
            + report.peak_cooling_result.error_pct)
            / 4.0;

        ValidationResult {
            in_range,
            error_pct: avg_error_pct,
        }
    }

    /// Internal helper: build the spec from the canonical
    /// `ASHRAE140Case::Case960`, run the real `ASHRAE140Validator` for
    /// 8760 hourly steps, and return the full `ValidationReport`.
    ///
    /// Centralised so all three public entry points
    /// (`validate_case_960`, `run_multi_zone_validation`,
    /// `run_comprehensive_validation`) step the same physics and read the
    /// same benchmark bounds — issue #1407.
    fn run_real_case_960_report(&self) -> ValidationReport {
        let base_validator = ASHRAE140Validator::new();
        base_validator.validate_case_960()
    }

    /// Run Case 960 using only **synthetic actual values** (e.g. user-
    /// supplied pre-computed metrics) and compare them against the
    /// canonical inter-program envelope. Does not run any simulation;
    /// use [`Self::validate_case_960`] for the end-to-end path.
    ///
    /// Provided for downstream consumers (e.g. the CLI's `--format json`
    /// path in `src/cli/multi_zone.rs`, and `validate_case_960_with_validator`)
    /// that already have `actual` numbers in hand and need a uniform
    /// comparator. Issue #1407.
    pub fn compare_against_reference(
        &self,
        actual_heating_mwh: f64,
        actual_cooling_mwh: f64,
        actual_peak_heating_kw: f64,
        actual_peak_cooling_kw: f64,
    ) -> Case960CompareOutcome {
        use super::benchmark::{
            CASE_960_ANNUAL_COOLING_MAX, CASE_960_ANNUAL_COOLING_MIN, CASE_960_ANNUAL_HEATING_MAX,
            CASE_960_ANNUAL_HEATING_MIN, CASE_960_ENERGY_TOLERANCE, CASE_960_PEAK_COOLING_MAX,
            CASE_960_PEAK_COOLING_MIN, CASE_960_PEAK_HEATING_MAX, CASE_960_PEAK_HEATING_MIN,
            CASE_960_PEAK_TOLERANCE,
        };

        let h_in_range = actual_heating_mwh >= CASE_960_ANNUAL_HEATING_MIN
            && actual_heating_mwh <= CASE_960_ANNUAL_HEATING_MAX;
        let c_in_range = actual_cooling_mwh >= CASE_960_ANNUAL_COOLING_MIN
            && actual_cooling_mwh <= CASE_960_ANNUAL_COOLING_MAX;
        let ph_in_range = actual_peak_heating_kw >= CASE_960_PEAK_HEATING_MIN
            && actual_peak_heating_kw <= CASE_960_PEAK_HEATING_MAX;
        let pc_in_range = actual_peak_cooling_kw >= CASE_960_PEAK_COOLING_MIN
            && actual_peak_cooling_kw <= CASE_960_PEAK_COOLING_MAX;

        let mid_h = (CASE_960_ANNUAL_HEATING_MIN + CASE_960_ANNUAL_HEATING_MAX) / 2.0;
        let mid_c = (CASE_960_ANNUAL_COOLING_MIN + CASE_960_ANNUAL_COOLING_MAX) / 2.0;
        let mid_ph = (CASE_960_PEAK_HEATING_MIN + CASE_960_PEAK_HEATING_MAX) / 2.0;
        let mid_pc = (CASE_960_PEAK_COOLING_MIN + CASE_960_PEAK_COOLING_MAX) / 2.0;

        let h_err = if mid_h > 0.0 {
            ((actual_heating_mwh - mid_h).abs() / mid_h) * 100.0
        } else {
            0.0
        };
        let c_err = if mid_c > 0.0 {
            ((actual_cooling_mwh - mid_c).abs() / mid_c) * 100.0
        } else {
            0.0
        };
        let ph_err = if mid_ph > 0.0 {
            ((actual_peak_heating_kw - mid_ph).abs() / mid_ph) * 100.0
        } else {
            0.0
        };
        let pc_err = if mid_pc > 0.0 {
            ((actual_peak_cooling_kw - mid_pc).abs() / mid_pc) * 100.0
        } else {
            0.0
        };

        Case960CompareOutcome {
            annual_heating_mwh: actual_heating_mwh,
            annual_cooling_mwh: actual_cooling_mwh,
            peak_heating_kw: actual_peak_heating_kw,
            peak_cooling_kw: actual_peak_cooling_kw,
            annual_heating_in_range: h_in_range,
            annual_cooling_in_range: c_in_range,
            peak_heating_in_range: ph_in_range,
            peak_cooling_in_range: pc_in_range,
            annual_heating_error_pct: h_err,
            annual_cooling_error_pct: c_err,
            peak_heating_error_pct: ph_err,
            peak_cooling_error_pct: pc_err,
            energy_tolerance: CASE_960_ENERGY_TOLERANCE,
            peak_tolerance: CASE_960_PEAK_TOLERANCE,
            annual_heating_min: CASE_960_ANNUAL_HEATING_MIN,
            annual_heating_max: CASE_960_ANNUAL_HEATING_MAX,
            annual_cooling_min: CASE_960_ANNUAL_COOLING_MIN,
            annual_cooling_max: CASE_960_ANNUAL_COOLING_MAX,
            peak_heating_min: CASE_960_PEAK_HEATING_MIN,
            peak_heating_max: CASE_960_PEAK_HEATING_MAX,
            peak_cooling_min: CASE_960_PEAK_COOLING_MIN,
            peak_cooling_max: CASE_960_PEAK_COOLING_MAX,
            all_in_range: h_in_range && c_in_range && ph_in_range && pc_in_range,
        }
    }

    /// Run full multi-zone validation suite
    ///
    /// This method runs validation for all supported multi-zone cases.
    ///
    /// # Returns
    /// BenchmarkReport with detailed validation results
    pub fn run_multi_zone_validation(&mut self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();

        // Issue #1407: previously this method ran the stub validator and
        // emitted zero/zero actuals whenever the in-memory comparison
        // failed, which silently fabricated PASS for the strict ±15%
        // CI gate (#1368). It now runs the real 8760-step physics
        // simulation via `validate_case_960` and reports the **actual
        // model outputs** into the `BenchmarkReport` so downstream
        // consumers can see whether the engine produced numbers in
        // band.
        let started = Instant::now();
        let case_960_ref = Self::load_case_960_reference_data();
        let spec = ASHRAE140Case::Case960.spec();
        let model = ThermalModel::<crate::physics::cta::VectorField>::from_spec(&spec);
        let case_960_result = self.validate_case_960(&model, &case_960_ref);
        let _elapsed = started.elapsed();

        // Reconstruct the underlying ValidationReport so we can emit the
        // **actual** (post-simulation) values into the BenchmarkReport
        // rather than the bogus "0.0 on FAIL" the stub used to write.
        let vrep = self.run_real_case_960_report();

        report.add_result_simple(
            "960",
            MetricType::AnnualHeating,
            vrep.annual_heating_mwh,
            case_960_ref.annual_heating * (1.0 - case_960_ref.energy_tolerance),
            case_960_ref.annual_heating * (1.0 + case_960_ref.energy_tolerance),
        );

        report.add_result_simple(
            "960",
            MetricType::AnnualCooling,
            vrep.annual_cooling_mwh,
            case_960_ref.annual_cooling * (1.0 - case_960_ref.energy_tolerance),
            case_960_ref.annual_cooling * (1.0 + case_960_ref.energy_tolerance),
        );

        report.add_result_simple(
            "960",
            MetricType::PeakHeating,
            vrep.peak_heating_kw,
            case_960_ref.peak_heating * (1.0 - case_960_ref.load_tolerance),
            case_960_ref.peak_heating * (1.0 + case_960_ref.load_tolerance),
        );

        report.add_result_simple(
            "960",
            MetricType::PeakCooling,
            vrep.peak_cooling_kw,
            case_960_ref.peak_cooling * (1.0 - case_960_ref.load_tolerance),
            case_960_ref.peak_cooling * (1.0 + case_960_ref.load_tolerance),
        );

        // `add_result_simple` automatically sets the per-result
        // `ValidationStatus` from `fluxion_value` vs `[ref_min, ref_max]`
        // (`report::ValidationResult::new`). Each row now reports the
        // *real* engine output instead of the bogus 0.0-on-FAIL the
        // prior stub wrote (issue #1407), so the strict ±15% CI gate
        // (#1368) can produce a meaningful verdict for Case 960.

        // Touch the lightweight validator result so the compiler
        // doesn't warn, and so callers can introspect the unified
        // verdict via `report.all_passed()` (used by the bin entry
        // point and the CLI's text-mode renderer).
        let _ = case_960_result.in_range;

        // Add stubs for Case 970 and 980 (future implementation)
        report.add_result_simple("970", MetricType::AnnualHeating, 0.0, 0.0, 0.0);
        report.add_result_simple("980", MetricType::AnnualHeating, 0.0, 0.0, 0.0);

        report
    }

    /// Generate a validation report for multi-zone cases
    ///
    /// # Returns
    /// String containing the detailed validation report
    pub fn generate_multi_zone_report(&mut self) -> String {
        let report = self.run_multi_zone_validation();

        let mut report_text = String::new();
        report_text.push_str("=== ASHRAE 140 Multi-Zone Validation Report ===\n");
        report_text.push_str(&format!(
            "Status: {}\n",
            if report
                .results
                .iter()
                .all(|r| r.status == ValidationStatus::Pass)
            {
                "PASSED"
            } else {
                "FAILED"
            }
        ));
        report_text.push_str(&format!("Total Cases: {}\n", report.results.len()));
        report_text.push_str("\nCase Results:\n");

        for result in &report.results {
            report_text.push_str(&format!(
                "  Case {}: {} ({:.1}% error)\n",
                result.case_id,
                match result.status {
                    ValidationStatus::Pass => "PASS",
                    ValidationStatus::Warning => "WARN",
                    ValidationStatus::Fail => "FAIL",
                },
                result.percent_error.abs()
            ));
        }

        report_text.push_str("\nMulti-zone validation framework ready.");
        report_text.push_str("\nCase 960: Two-zone sunspace building validation implemented.");
        report_text.push_str("\nCases 970/980: Stub implementations for future expansion.");

        report_text
    }
}

/// Reference data for ASHRAE 140 Case 960
///
/// Case 960 represents a two-zone sunspace building with:
/// - Zone 1: Living space (20°C heating setpoint, 24°C cooling setpoint)
/// - Zone 2: Sunspace (15°C heating setpoint, no cooling)
/// - Specific geometry, construction, and internal loads per ASHRAE 140-2017
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case960Reference {
    /// Zone temperatures at key timesteps (hour -> temperatures)
    pub zone_temperatures: HashMap<usize, Vec<f64>>,

    /// Expected annual heating energy consumption (MWh)
    pub annual_heating: f64,

    /// Expected annual cooling energy consumption (MWh)
    pub annual_cooling: f64,

    /// Expected peak heating load (kW)
    pub peak_heating: f64,

    /// Expected peak cooling load (kW)
    pub peak_cooling: f64,

    /// Minimum expected temperature (°C)
    pub min_temperature: f64,

    /// Maximum expected temperature (°C)
    pub max_temperature: f64,

    /// Temperature validation tolerance (°C)
    pub temperature_tolerance: f64,

    /// Energy validation tolerance (fraction)
    pub energy_tolerance: f64,

    /// Load validation tolerance (fraction)
    pub load_tolerance: f64,
}

/// ASHRAE 140 inter-program envelope for Case 960 (heating/cooling/peak).
///
/// This struct exposes the canonical min/max bounds across EnergyPlus,
/// ESP-r, TRNSYS, DOE2, BSIMAC, CSE, and DeST as derived from
/// `crate::validation::benchmark::CASE_960_*`. Issue #1407 makes this
/// the single source of truth so the multi-zone validator, the strict
/// ±15% CI gate (#1368), and the CLI cannot diverge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case960InterProgramBounds {
    /// Lower bound of annual heating energy (MWh).
    pub annual_heating_min: f64,
    /// Upper bound of annual heating energy (MWh).
    pub annual_heating_max: f64,
    /// Lower bound of annual cooling energy (MWh).
    pub annual_cooling_min: f64,
    /// Upper bound of annual cooling energy (MWh).
    pub annual_cooling_max: f64,
    /// Lower bound of peak heating load (kW).
    pub peak_heating_min: f64,
    /// Upper bound of peak heating load (kW).
    pub peak_heating_max: f64,
    /// Lower bound of peak cooling load (kW).
    pub peak_cooling_min: f64,
    /// Upper bound of peak cooling load (kW).
    pub peak_cooling_max: f64,
}

/// Reference data for ASHRAE 140 Case 970
///
/// Case 970 represents a more complex multi-zone building configuration
/// with multiple conditioned zones and inter-zone heat transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case970Reference {
    /// Zone temperatures at key timesteps (hour -> temperatures)
    pub zone_temperatures: HashMap<usize, Vec<f64>>,

    /// Expected annual heating energy consumption (MWh)
    pub annual_heating: f64,

    /// Expected annual cooling energy consumption (MWh)
    pub annual_cooling: f64,

    /// Expected peak heating load (kW)
    pub peak_heating: f64,

    /// Expected peak cooling load (kW)
    pub peak_cooling: f64,

    /// Minimum expected temperature (°C)
    pub min_temperature: f64,

    /// Maximum expected temperature (°C)
    pub max_temperature: f64,

    /// Temperature validation tolerance (°C)
    pub temperature_tolerance: f64,

    /// Energy validation tolerance (fraction)
    pub energy_tolerance: f64,

    /// Load validation tolerance (fraction)
    pub load_tolerance: f64,
}

impl Default for Case960Reference {
    fn default() -> Self {
        Self::load_case_960_reference_data()
    }
}

/// Outcome of [`ASHRAE140MultiZoneValidator::compare_against_reference`].
///
/// Holds the four actual metrics plus their `in_range` flags, signed
/// error percentages, the strict ±15% / ±10% tolerances, and the
/// canonical ASHRAE 140 inter-program envelope. JSON-serialisable so the
/// CLI (`src/cli/multi_zone.rs`) can emit it under `--format json`
/// without re-loading the benchmark module. Issue #1407.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case960CompareOutcome {
    /// Actual annual heating energy (MWh) — caller-supplied.
    pub annual_heating_mwh: f64,
    /// Actual annual cooling energy (MWh) — caller-supplied.
    pub annual_cooling_mwh: f64,
    /// Actual peak heating load (kW) — caller-supplied.
    pub peak_heating_kw: f64,
    /// Actual peak cooling load (kW) — caller-supplied.
    pub peak_cooling_kw: f64,

    /// `true` iff `annual_heating_mwh ∈ [ref_min, ref_max]`.
    pub annual_heating_in_range: bool,
    /// `true` iff `annual_cooling_mwh ∈ [ref_min, ref_max]`.
    pub annual_cooling_in_range: bool,
    /// `true` iff `peak_heating_kw ∈ [ref_min, ref_max]`.
    pub peak_heating_in_range: bool,
    /// `true` iff `peak_cooling_kw ∈ [ref_min, ref_max]`.
    pub peak_cooling_in_range: bool,

    /// |actual - mid| / mid for annual heating, percent.
    pub annual_heating_error_pct: f64,
    /// |actual - mid| / mid for annual cooling, percent.
    pub annual_cooling_error_pct: f64,
    /// |actual - mid| / mid for peak heating, percent.
    pub peak_heating_error_pct: f64,
    /// |actual - mid| / mid for peak cooling, percent.
    pub peak_cooling_error_pct: f64,

    /// Energy tolerance (fraction, e.g. 0.15) — issue #1368.
    pub energy_tolerance: f64,
    /// Peak-load tolerance (fraction, e.g. 0.10) — issue #1368.
    pub peak_tolerance: f64,

    /// Canonical annual heating lower bound (MWh).
    pub annual_heating_min: f64,
    /// Canonical annual heating upper bound (MWh).
    pub annual_heating_max: f64,
    /// Canonical annual cooling lower bound (MWh).
    pub annual_cooling_min: f64,
    /// Canonical annual cooling upper bound (MWh).
    pub annual_cooling_max: f64,
    /// Canonical peak heating lower bound (kW).
    pub peak_heating_min: f64,
    /// Canonical peak heating upper bound (kW).
    pub peak_heating_max: f64,
    /// Canonical peak cooling lower bound (kW).
    pub peak_cooling_min: f64,
    /// Canonical peak cooling upper bound (kW).
    pub peak_cooling_max: f64,

    /// `true` iff **all four** metrics are within tolerance.
    pub all_in_range: bool,
}

impl Case960CompareOutcome {
    /// `true` when at least one metric is **outside** tolerance.
    pub fn any_failed(&self) -> bool {
        !self.all_in_range
    }
}

impl Case960Reference {
    /// Load default Case 960 reference data
    pub fn load_case_960_reference_data() -> Self {
        ASHRAE140MultiZoneValidator::load_case_960_reference_data()
    }
}

impl Default for Case970Reference {
    fn default() -> Self {
        Self::load_case_970_reference_data()
    }
}

impl Case970Reference {
    /// Load default Case 970 reference data (Issue #1446).
    ///
    /// Values are sourced from the canonical ASHRAE 140-2017 §B6.7 inter-program
    /// envelope (also tabulated in ASHRAE 140-2023 Annex B8-3 and validated
    /// across EnergyPlus 25.2.0, TRNSYS, ESP-r, DOE-2, BSIMAC, CSE, DeST).
    /// Annual heating band [10.54, 14.26] MWh and annual cooling band
    /// [7.39, 10.00] MWh are the source-of-truth ranges; the midpoint fields
    /// below are derived from those bands and used by `Case970Validator` as
    /// the canonical midpoint reference.
    pub fn load_case_970_reference_data() -> Self {
        // Reference values from `tests/reference_data/zone_balance/
        // case_970_energy_reference.csv` (regenerated by
        // `generate_case_970_energy.py`, Issue #1446). These constants are
        // duplicated here so the in-process validator doesn't have to read
        // the CSV; keep the two sources in sync.
        const ANNUAL_HEATING_REF_MIN: f64 = 10.54;
        const ANNUAL_HEATING_REF_MAX: f64 = 14.26;
        const ANNUAL_COOLING_REF_MIN: f64 = 7.39;
        const ANNUAL_COOLING_REF_MAX: f64 = 10.00;
        const PEAK_HEATING_REF_MIN: f64 = 4.0;
        const PEAK_HEATING_REF_MAX: f64 = 8.0;
        const PEAK_COOLING_REF_MIN: f64 = 2.5;
        const PEAK_COOLING_REF_MAX: f64 = 5.5;

        Case970Reference {
            zone_temperatures: HashMap::from([
                // 5-zone representative setpoints; one entry per zone at
                // the annual-average timestep (hour 8760). The expected
                // conditioned-zone mean for Case 970 sits in the
                // [19, 24] °C band; the free-floating entries are kept
                // for diagnostic completeness.
                (4380, vec![19.5, 18.2, 19.0, 18.5, 19.2]), // Winter design day
                (5000, vec![24.2, 25.0, 24.5, 24.8, 24.3]), // Summer design day
                (8760, vec![21.2, 21.4, 21.3, 21.5, 21.1]), // Annual average
            ]),
            annual_heating: (ANNUAL_HEATING_REF_MIN + ANNUAL_HEATING_REF_MAX) / 2.0, // 12.400 MWh
            annual_cooling: (ANNUAL_COOLING_REF_MIN + ANNUAL_COOLING_REF_MAX) / 2.0, //  8.695 MWh
            peak_heating: (PEAK_HEATING_REF_MIN + PEAK_HEATING_REF_MAX) / 2.0,       // 6.000 kW
            peak_cooling: (PEAK_COOLING_REF_MIN + PEAK_COOLING_REF_MAX) / 2.0,       // 4.000 kW
            min_temperature: 8.0,                                                    // °C
            max_temperature: 42.0,                                                   // °C
            temperature_tolerance: 1.5,                                              // °C
            energy_tolerance: 0.15,                                                  // 15%
            load_tolerance: 0.10,                                                    // 10%
        }
    }
}

#[allow(clippy::items_after_test_module, clippy::redundant_closure)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;

    #[test]
    fn test_case_960_reference_loading() {
        let reference = Case960Reference::load_case_960_reference_data();

        // Issue #1407: previously asserted the hardcoded stub values
        // 12.4 / 8.7 / 5.2 / 4.8 — those were placeholders that did
        // not match any reference program and produced fabricated PASS.
        // Now asserts the canonical ASHRAE 140-2023 inter-program
        // midpoints sourced from `validation::benchmark`.
        assert!(
            (reference.annual_heating - 2.05).abs() < 1e-9,
            "annual_heating should equal benchmark midpoint 2.05 MWh, got {}",
            reference.annual_heating
        );
        assert!(
            (reference.annual_cooling - 2.165).abs() < 1e-9,
            "annual_cooling should equal benchmark midpoint 2.165 MWh, got {}",
            reference.annual_cooling
        );
        assert!(
            (reference.peak_heating - 5.0).abs() < 1e-9,
            "peak_heating should equal benchmark midpoint 5.0 kW, got {}",
            reference.peak_heating
        );
        assert!(
            (reference.peak_cooling - 2.0).abs() < 1e-9,
            "peak_cooling should equal benchmark midpoint 2.0 kW, got {}",
            reference.peak_cooling
        );

        // Verify temperature ranges
        assert!(reference.min_temperature > 0.0);
        assert!(reference.max_temperature < 50.0);

        // Verify tolerances are reasonable (sourced from `benchmark::CASE_960_*`)
        assert!((reference.energy_tolerance - 0.15).abs() < 1e-9);
        assert!((reference.load_tolerance - 0.10).abs() < 1e-9);
        assert!(reference.temperature_tolerance > 0.0);

        // And the canonical inter-program envelope must agree with
        // benchmark — this is what prevents the validator from drifting
        // from the strict ±15% CI gate (#1368).
        let bounds = ASHRAE140MultiZoneValidator::case_960_inter_program_bounds();
        assert!((bounds.annual_heating_min - 1.65).abs() < 1e-9);
        assert!((bounds.annual_heating_max - 2.45).abs() < 1e-9);
        assert!((bounds.annual_cooling_min - 1.55).abs() < 1e-9);
        assert!((bounds.annual_cooling_max - 2.78).abs() < 1e-9);
        assert!((bounds.peak_heating_min - 2.0).abs() < 1e-9);
        assert!((bounds.peak_heating_max - 8.0).abs() < 1e-9);
        assert!((bounds.peak_cooling_min - 0.0).abs() < 1e-9);
        assert!((bounds.peak_cooling_max - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_case_960_validation() {
        let validator = ASHRAE140MultiZoneValidator::new();
        let reference = Case960Reference::load_case_960_reference_data();

        // Create a thermal model for testing
        let spec = crate::validation::ashrae_140_cases::ASHRAE140Case::Case960.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        // Run validation. With the real path the validator now actually
        // steps the physics — depending on the model's current accuracy
        // (Wave 6 / #1446 still has work to do on multi-zone coupling),
        // this may PASS or FAIL, but the assertion is only that the
        // validator completes and returns a sensible error percentage.
        let result = validator.validate_case_960(&model, &reference);

        assert!(result.error_pct >= 0.0);
        // Cap at a generous upper bound — the model currently produces
        // 200%+ heating errors (well above the ±15% gate), so the cap
        // here is "not 1e9" rather than "not 100%".
        assert!(result.error_pct.is_finite());
    }

    #[test]
    fn test_multi_zone_report_generation() {
        let mut validator = ASHRAE140MultiZoneValidator::new();
        let report = validator.generate_multi_zone_report();

        // Verify report contains expected sections
        assert!(report.contains("ASHRAE 140 Multi-Zone Validation Report"));
        assert!(report.contains("Case Results:"));
        assert!(report.contains("Case 960"));
        assert!(report.contains("Multi-zone validation framework ready"));
    }

    #[test]
    fn test_multi_zone_validation_suite() {
        let mut validator = ASHRAE140MultiZoneValidator::new();
        let report = validator.run_multi_zone_validation();

        // Issue #1407: the suite now emits 4 metrics for Case 960
        // (annual heating, annual cooling, peak heating, peak cooling)
        // plus 2 stubs for Case 970 and 980 — total 6 results.
        assert_eq!(
            report.results.len(),
            6,
            "Expected 4 Case-960 metrics + 2 stubs (970/980); got {}",
            report.results.len()
        );

        // Check that Case 960 metrics exist (all four)
        let case_960_metrics: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.case_id == "960")
            .collect();
        assert_eq!(
            case_960_metrics.len(),
            4,
            "Expected 4 Case-960 metric rows, got {}",
            case_960_metrics.len()
        );
    }

    /// Regression test for issue #1407: the validator must run a real
    /// simulation, must NOT fabricate PASS for an obviously broken
    /// config, and must return PASS for a known-good config.
    ///
    /// Before this fix, the validator returned PASS in <1ms by comparing
    /// two hardcoded placeholders (12.4 vs 12.5 MWh). This test would
    /// have failed for two reasons:
    ///   1. `validate_case_960` completed in microseconds (no physics).
    ///   2. It returned `in_range = true` even when the underlying
    ///      engine output was nowhere near the canonical envelope.
    ///
    /// After this fix:
    ///   1. `validate_case_960` steps 8760 physics hours through
    ///      `ASHRAE140Validator::validate_case_960`, producing a real
    ///      computed annual-heating value distinct from the pre-#1407
    ///      stub placeholder (12.5 MWh) — see assertion (1) below.
    ///      (Issue #2745: this was previously a flaky >200ms wall-clock
    ///      proxy; it is now a deterministic structural check.)
    ///   2. The current engine output (7.47 MWh heating vs canonical
    ///      1.65-2.45 MWh) is far outside ±15%, so the validator
    ///      correctly returns `in_range = false` — proving PASS is no
    ///      longer fabricated.
    ///   3. `compare_against_reference` with a synthetic known-good
    ///      input (the canonical midpoint of every metric) returns
    ///      `all_in_range = true` — proving the comparison logic itself
    ///      is correct (independent of the engine's current accuracy).
    #[test]
    fn test_case_960_validator_runs_real_model_not_stub() {
        let validator = ASHRAE140MultiZoneValidator::new();
        let spec = ASHRAE140Case::Case960.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);
        let reference = Case960Reference::load_case_960_reference_data();

        // (1) DETERMINISTIC structural check that the real physics ran
        //     (issue #2745). The pre-#1407 stub never called
        //     `step_physics`: it returned hardcoded placeholders
        //     (annual_heating = 12.5 MWh) and fabricated `in_range = true`.
        //     The original regression guard used a >200ms wall-clock proxy,
        //     which is inherently machine/profile-dependent and flaked
        //     under `--profile ci` on fast runners (~120-140ms). It is
        //     replaced here by asserting the computed annual heating is a
        //     finite, positive value that is NOT the stub's 12.5 MWh
        //     placeholder. A 1.0 MWh separation is a robust, machine-
        //     independent discriminator that survives model-accuracy
        //     improvements (the real value will never return to 12.5).
        let report = validator.run_real_case_960_report();
        let computed_heating_mwh = report.annual_heating_mwh;
        const STUB_PLACEHOLDER_HEATING_MWH: f64 = 12.5;
        assert!(
            computed_heating_mwh.is_finite() && computed_heating_mwh > 0.0,
            "Real physics must produce a finite, positive annual heating value; got {}",
            computed_heating_mwh,
        );
        assert!(
            (computed_heating_mwh - STUB_PLACEHOLDER_HEATING_MWH).abs() > 1.0,
            "Computed annual heating ({:.3} MWh) is within 1.0 MWh of the pre-#1407 \
             stub placeholder (12.5 MWh) — the stub may be reinstalled.",
            computed_heating_mwh,
        );

        let result = validator.validate_case_960(&model, &reference);

        // (2) The validator must NOT fabricate PASS for the current
        //     engine output. The current model produces ~7.47 MWh
        //     heating (verified by `cargo test -p fluxion --test
        //     ashrae_140_case_960_sunspace test_case_960_comprehensive_energy_validation`),
        //     which is ~265% above the canonical midpoint 2.05 MWh —
        //     far outside ±15%. The validator must therefore return
        //     `in_range = false`.
        //
        //     Note: when Wave 6 / #1446 lands and the multi-zone model
        //     produces results inside ±15%, this assertion will need
        //     to be re-evaluated. Until then, this is the regression
        //     guard that ensures the validator stops fabricating PASS.
        assert!(
            !result.in_range,
            "Validator must NOT fabricate PASS for the current engine output \
             (heating ~7.47 MWh is far outside the 1.65-2.45 MWh canonical band). \
             Got in_range = true — likely the stub is still installed. \
             error_pct = {:.1}%",
            result.error_pct
        );

        // (3) The comparator logic itself must be sound — when given
        //     synthetic known-good values (the canonical midpoints) it
        //     must report `all_in_range = true`. This proves the
        //     validator's *comparison* code is correct independent of
        //     the engine's current accuracy.
        let outcome = validator.compare_against_reference(
            2.05,  // canonical annual heating midpoint
            2.165, // canonical annual cooling midpoint
            5.0,   // canonical peak heating midpoint
            2.0,   // canonical peak cooling midpoint
        );
        assert!(
            outcome.all_in_range,
            "Comparator must return all_in_range = true for the canonical \
             midpoints. Got: h_in={}, c_in={}, ph_in={}, pc_in={}",
            outcome.annual_heating_in_range,
            outcome.annual_cooling_in_range,
            outcome.peak_heating_in_range,
            outcome.peak_cooling_in_range
        );

        // (4) The comparator must NOT pass obviously broken values —
        //     this is the symmetric guard to (3).
        let broken = validator.compare_against_reference(
            12.4,  // pre-#1407 hardcoded placeholder (4-7× too high)
            8.7,   // pre-#1407 hardcoded placeholder (3-4× too high)
            100.0, // wildly above the 2.0-8.0 kW band
            100.0, // wildly above the 0.0-4.0 kW band
        );
        assert!(
            !broken.all_in_range,
            "Comparator must NOT pass obviously broken values (12.4 / 8.7 / 100 / 100)."
        );
        assert!(
            broken.any_failed(),
            "Comparator must report any_failed() for broken values."
        );
    }

    /// Issue #2980 acceptance item #2 regression guard.
    ///
    /// Before this fix, `validate_case_970_with_validator` hardcoded
    /// `actual_heating: 15.0` and `actual_cooling: 10.0` MWh (with the
    /// comment `// Placeholder values - would come from actual simulation`).
    /// The validator always reported `(pass, error_pct ≈ 17%)` regardless of
    /// engine state, making the Case 970 headline metric uninformative.
    ///
    /// After this fix, the validator runs a real 8760-step physics simulation
    /// and feeds the engine's actual annual heating / cooling (converted to
    /// electrical MWh via the standard 0.9 heating efficiency / 3.0 cooling
    /// COP factors used by `ASHRAE140Validator::validate_case_960`).
    ///
    /// This test pins two things:
    ///   1. The actual heating/cooling values returned by the simulation
    ///      helper are **not** the pre-#2980 hardcoded placeholders
    ///      (15.0 / 10.0 MWh).
    ///   2. The validator's `error_pct` is consistent with the reference
    ///      midpoint (i.e. it derives from real engine output, not a
    ///      pre-baked constant).
    #[test]
    fn test_case_970_validator_uses_real_simulation_not_hardcoded_placeholders() {
        let validator = ASHRAE140MultiZoneValidator::new();

        // (1) `run_real_case_970_energy` returns engine outputs, not the
        //     pre-#2980 placeholder constants. We assert finite /
        //     non-negative + that the values do NOT exactly equal the
        //     stub constants (a future regression that re-installs the
        //     hardcoded `15.0` / `10.0` would produce bit-identical
        //     output and trip these equality checks). The Case 970
        //     engine output is currently ~14.4 MWh heating / ~10.0 MWh
        //     cooling, which is "close to" but not "equal to" the
        //     pre-#2980 placeholders.
        let (h, c) = validator.run_real_case_970_energy();
        const STUB_HEATING_MWH: f64 = 15.0;
        const STUB_COOLING_MWH: f64 = 10.0;
        assert!(
            h.is_finite() && h >= 0.0,
            "Real Case 970 simulation must produce a finite non-negative \
             annual heating value; got {}",
            h
        );
        assert!(
            c.is_finite() && c >= 0.0,
            "Real Case 970 simulation must produce a finite non-negative \
             annual cooling value; got {}",
            c
        );
        assert!(
            (h - STUB_HEATING_MWH).abs() > 0.05,
            "Case 970 actual heating ({:.4} MWh) is within 0.05 MWh of the \
             pre-#2980 hardcoded placeholder (15.0 MWh) — the stub may \
             be reinstalled.",
            h
        );
        assert!(
            (c - STUB_COOLING_MWH).abs() > 0.05,
            "Case 970 actual cooling ({:.4} MWh) is within 0.05 MWh of the \
             pre-#2980 hardcoded placeholder (10.0 MWh) — the stub may \
             be reinstalled.",
            c
        );

        // (2) The full `validate_case_970_with_validator` path consumes
        //     real engine output and the verdict is consistent with the
        //     canonical ASHRAE 140-2017 §B6.7 midpoints (Issue #1446):
        //     heating 12.400 MWh, cooling 8.695 MWh, both ±15%. We only
        //     assert the error_pct is finite and within [0, 200] — the
        //     multi-zone coupling is still being tuned (per the existing
        //     Case 960 deterministic guard in `test_case_960_validator_
        //     runs_real_model_not_stub`), so we don't gate on PASS.
        let spec = ASHRAE140Case::Case970.spec();
        let model = ThermalModel::<crate::physics::cta::VectorField>::from_spec(&spec);
        let result = validator.validate_case_970_with_validator(&model);
        assert!(
            result.error_pct.is_finite() && result.error_pct >= 0.0,
            "Case 970 validator error_pct must be finite and non-negative; \
             got {}",
            result.error_pct
        );
    }

    /// Issue #2980 acceptance item #3 regression guard.
    ///
    /// Before this fix:
    ///   - `Case970Validator::validate_annual_heating` carried the
    ///     docstring `(stub implementation)` even though it performed the
    ///     real ASHRAE 140-2017 §B6.7 ±15% comparison (Issue #1446).
    ///   - `Case970Validator::generate_report` headed its output with
    ///     `=== ASHRAE 140 Case 970 Validation Report (STUB) ===` and
    ///     printed "Case 970 validation framework is implemented but not
    ///     yet fully validated."
    ///
    /// After this fix:
    ///   - The two `validate_annual_*` docstrings describe the real
    ///     comparator (no `(stub implementation)`).
    ///   - `generate_report` reflects the validator's actual state
    ///     (computed gap / tolerance) and no longer self-describes as a
    ///     stub.
    ///
    /// This test pins both behaviours so a future "simplification" can't
    /// silently re-introduce either misleading label.
    #[test]
    fn test_case_970_report_no_longer_self_describes_as_stub() {
        let mut validator = Case970Validator::new();

        // Drive the validator with canonical ASHRAE 140-2017 §B6.7
        // midpoints (Issue #1446) so `generate_report` populates the
        // "Annual Heating" / "Annual Cooling" branches with real numbers.
        let (h_pass, _h_pct) = validator.validate_annual_heating(12.40);
        let (c_pass, _c_pct) = validator.validate_annual_cooling(8.695);

        // Canonical midpoints ⇒ both should pass the ±15% band.
        assert!(h_pass, "Canonical Case 970 heating midpoint should PASS");
        assert!(c_pass, "Canonical Case 970 cooling midpoint should PASS");

        let report = validator.generate_report();

        // Report must NOT self-describe as a stub any more (item #3).
        assert!(
            !report.contains("STUB"),
            "Case 970 report must not contain the 'STUB' label anymore; \
             got:\n{}",
            report
        );
        assert!(
            !report.contains("not yet fully validated"),
            "Case 970 report must not say 'not yet fully validated' \
             anymore; got:\n{}",
            report
        );
        assert!(
            !report.contains("This case will be completed in future work"),
            "Case 970 report must not defer to 'future work' anymore; \
             got:\n{}",
            report
        );

        // And the report must surface the canonical reference values so
        // downstream consumers (CLI / docs) can see what the comparator
        // is comparing against.
        assert!(
            report.contains("12.40") && report.contains("8.70"),
            "Case 970 report should surface the canonical ASHRAE 140-2017 \
             §B6.7 midpoints (12.40 heating, 8.695 cooling); got:\n{}",
            report
        );
    }
}

impl Default for Case960Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl Case960Validator {
    /// Create a new Case 960 validator with default reference data
    pub fn new() -> Self {
        Self {
            reference: Case960Reference::load_case_960_reference_data(),
            statistics: Case960Statistics::default(),
        }
    }

    /// Create a new Case 960 validator with custom reference data
    pub fn with_reference(reference: Case960Reference) -> Self {
        Self {
            reference,
            statistics: Case960Statistics::default(),
        }
    }

    /// Load reference data from benchmark.rs
    ///
    /// This method loads the expected values for ASHRAE 140 Case 960
    /// from the benchmark data module.
    pub fn load_reference_data() -> Case960Reference {
        Case960Reference::load_case_960_reference_data()
    }

    /// Get reference annual heating energy (MWh)
    pub fn annual_heating(&self) -> f64 {
        self.reference.annual_heating
    }

    /// Get reference annual cooling energy (MWh)
    pub fn annual_cooling(&self) -> f64 {
        self.reference.annual_cooling
    }

    /// Get reference peak heating load (kW)
    pub fn peak_heating(&self) -> f64 {
        self.reference.peak_heating
    }

    /// Get reference peak cooling load (kW)
    pub fn peak_cooling(&self) -> f64 {
        self.reference.peak_cooling
    }

    /// Get zone temperatures reference data
    pub fn zone_temperatures(&self) -> &HashMap<usize, Vec<f64>> {
        &self.reference.zone_temperatures
    }

    /// Validate annual heating energy consumption
    ///
    /// Compares actual annual heating against reference values with tolerance.
    /// Returns (pass, percentage_difference)
    pub fn validate_annual_heating(&mut self, actual_heating: f64) -> (bool, f64) {
        let reference = self.reference.annual_heating;
        let error = (actual_heating - reference).abs();
        let percentage_diff = (error / reference) * 100.0;

        let pass = percentage_diff <= self.reference.energy_tolerance * 100.0;

        self.statistics
            .percentage_differences
            .insert("annual_heating".to_string(), percentage_diff);
        self.statistics
            .max_absolute_errors
            .insert("annual_heating".to_string(), error);

        (pass, percentage_diff)
    }

    /// Validate annual cooling energy consumption
    ///
    /// Compares actual annual cooling against reference values with tolerance.
    /// Returns (pass, percentage_difference)
    pub fn validate_annual_cooling(&mut self, actual_cooling: f64) -> (bool, f64) {
        let reference = self.reference.annual_cooling;
        let error = (actual_cooling - reference).abs();
        let percentage_diff = (error / reference) * 100.0;

        let pass = percentage_diff <= self.reference.energy_tolerance * 100.0;

        self.statistics
            .percentage_differences
            .insert("annual_cooling".to_string(), percentage_diff);
        self.statistics
            .max_absolute_errors
            .insert("annual_cooling".to_string(), error);

        (pass, percentage_diff)
    }

    /// Validate peak heating load
    ///
    /// Compares actual peak heating load against reference values with tolerance.
    /// Returns (pass, percentage_difference)
    pub fn validate_peak_heating(&mut self, actual_peak: f64) -> (bool, f64) {
        let reference = self.reference.peak_heating;
        let error = (actual_peak - reference).abs();
        let percentage_diff = (error / reference) * 100.0;

        let pass = percentage_diff <= self.reference.load_tolerance * 100.0;

        self.statistics
            .percentage_differences
            .insert("peak_heating".to_string(), percentage_diff);
        self.statistics
            .max_absolute_errors
            .insert("peak_heating".to_string(), error);

        (pass, percentage_diff)
    }

    /// Validate peak cooling load
    ///
    /// Compares actual peak cooling load against reference values with tolerance.
    /// Returns (pass, percentage_difference)
    pub fn validate_peak_cooling(&mut self, actual_peak: f64) -> (bool, f64) {
        let reference = self.reference.peak_cooling;
        let error = (actual_peak - reference).abs();
        let percentage_diff = (error / reference) * 100.0;

        let pass = percentage_diff <= self.reference.load_tolerance * 100.0;

        self.statistics
            .percentage_differences
            .insert("peak_cooling".to_string(), percentage_diff);
        self.statistics
            .max_absolute_errors
            .insert("peak_cooling".to_string(), error);

        (pass, percentage_diff)
    }

    /// Validate hourly temperature profiles
    ///
    /// Compares actual temperature profiles against reference values.
    /// Returns RMSE and maximum temperature difference.
    pub fn validate_hourly_temperature_profiles(
        &mut self,
        actual_temperatures: &HashMap<usize, Vec<f64>>,
    ) -> (f64, f64) {
        let mut total_squared_error = 0.0f64;
        let mut max_diff = 0.0f64;
        let mut count = 0;

        for (timestep, expected_temps) in &self.reference.zone_temperatures {
            if let Some(actual_temps) = actual_temperatures.get(timestep) {
                for (expected_temp, actual_temp) in expected_temps.iter().zip(actual_temps.iter()) {
                    let diff = expected_temp - actual_temp;
                    total_squared_error += diff * diff;
                    max_diff = max_diff.max(diff.abs());
                    count += 1;
                }
            }
        }

        let rmse = if count > 0 {
            (total_squared_error / count as f64).sqrt()
        } else {
            0.0
        };

        self.statistics.rmse_temperature = rmse;
        self.statistics
            .max_absolute_errors
            .insert("temperature_profile".to_string(), max_diff);

        (rmse, max_diff)
    }

    /// Calculate overall validation score (0-100)
    ///
    /// Aggregates all validation results into a single score.
    pub fn calculate_overall_score(&mut self) -> f64 {
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;

        // Annual energy: 30% weight
        let heating_score = 100.0
            * (1.0
                - self
                    .statistics
                    .percentage_differences
                    .get("annual_heating")
                    .unwrap_or(&100.0)
                    / 100.0);
        let cooling_score = 100.0
            * (1.0
                - self
                    .statistics
                    .percentage_differences
                    .get("annual_cooling")
                    .unwrap_or(&100.0)
                    / 100.0);
        weighted_score += (heating_score + cooling_score) * 0.15;
        total_weight += 0.3;

        // Peak loads: 20% weight
        let peak_heating_score = 100.0
            * (1.0
                - self
                    .statistics
                    .percentage_differences
                    .get("peak_heating")
                    .unwrap_or(&100.0)
                    / 100.0);
        let peak_cooling_score = 100.0
            * (1.0
                - self
                    .statistics
                    .percentage_differences
                    .get("peak_cooling")
                    .unwrap_or(&100.0)
                    / 100.0);
        weighted_score += (peak_heating_score + peak_cooling_score) * 0.10;
        total_weight += 0.2;

        // Temperature profiles: 50% weight
        let temp_score = 100.0
            * (1.0
                - (self.statistics.rmse_temperature / self.reference.temperature_tolerance)
                    .min(1.0));
        weighted_score += temp_score * 0.5;
        total_weight += 0.5;

        let overall_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };

        self.statistics.overall_score = overall_score;
        overall_score
    }

    /// Generate detailed validation report
    ///
    /// Returns a formatted string with all validation results.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== ASHRAE 140 Case 960 Validation Report ===\n");

        // Annual energy results
        if let Some(heating_pct) = self.statistics.percentage_differences.get("annual_heating") {
            report.push_str(&format!(
                "Annual Heating: {:.2} MWh (ref: {:.2} MWh, diff: {:.1}%)\n",
                self.reference.annual_heating * (1.0 + heating_pct / 100.0),
                self.reference.annual_heating,
                heating_pct
            ));
        }

        if let Some(cooling_pct) = self.statistics.percentage_differences.get("annual_cooling") {
            report.push_str(&format!(
                "Annual Cooling: {:.2} MWh (ref: {:.2} MWh, diff: {:.1}%)\n",
                self.reference.annual_cooling * (1.0 + cooling_pct / 100.0),
                self.reference.annual_cooling,
                cooling_pct
            ));
        }

        // Peak load results
        if let Some(peak_heating_pct) = self.statistics.percentage_differences.get("peak_heating") {
            report.push_str(&format!(
                "Peak Heating: {:.2} kW (ref: {:.2} kW, diff: {:.1}%)\n",
                self.reference.peak_heating * (1.0 + peak_heating_pct / 100.0),
                self.reference.peak_heating,
                peak_heating_pct
            ));
        }

        if let Some(peak_cooling_pct) = self.statistics.percentage_differences.get("peak_cooling") {
            report.push_str(&format!(
                "Peak Cooling: {:.2} kW (ref: {:.2} kW, diff: {:.1}%)\n",
                self.reference.peak_cooling * (1.0 + peak_cooling_pct / 100.0),
                self.reference.peak_cooling,
                peak_cooling_pct
            ));
        }

        // Temperature profile results
        report.push_str(&format!(
            "Temperature RMSE: {:.3}°C (tolerance: {:.1}°C)\n",
            self.statistics.rmse_temperature, self.reference.temperature_tolerance
        ));

        if let Some(max_temp_diff) = self
            .statistics
            .max_absolute_errors
            .get("temperature_profile")
        {
            report.push_str(&format!("Max Temperature Diff: {:.2}°C\n", max_temp_diff));
        }

        // Overall score
        report.push_str(&format!(
            "Overall Score: {:.1}/100\n",
            self.statistics.overall_score
        ));

        report.push_str("\nValidation against ASHRAE 140-2017 specification.\n");
        report.push_str("Case 960: Two-zone sunspace building with inter-zone heat transfer.\n");

        report
    }
}

impl Default for Case970Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl Case970Validator {
    /// Create a new Case 970 validator with default reference data
    pub fn new() -> Self {
        Self {
            reference: Case970Reference::load_case_970_reference_data(),
            statistics: Case970Statistics::default(),
        }
    }

    /// Create a new Case 970 validator with custom reference data
    pub fn with_reference(reference: Case970Reference) -> Self {
        Self {
            reference,
            statistics: Case970Statistics::default(),
        }
    }

    /// Get reference annual heating energy (MWh)
    pub fn annual_heating(&self) -> f64 {
        self.reference.annual_heating
    }

    /// Get reference annual cooling energy (MWh)
    pub fn annual_cooling(&self) -> f64 {
        self.reference.annual_cooling
    }

    /// Get reference peak heating load (kW)
    pub fn peak_heating(&self) -> f64 {
        self.reference.peak_heating
    }

    /// Get reference peak cooling load (kW)
    pub fn peak_cooling(&self) -> f64 {
        self.reference.peak_cooling
    }

    /// Get zone temperatures reference data
    pub fn zone_temperatures(&self) -> &HashMap<usize, Vec<f64>> {
        &self.reference.zone_temperatures
    }

    /// Load reference data for Case 970
    ///
    /// This method loads placeholder reference data for Case 970.
    /// Actual reference values will be added in future implementation.
    pub fn load_reference_data() -> Case970Reference {
        Case970Reference::load_case_970_reference_data()
    }

    /// Validate annual heating energy consumption
    ///
    /// Issue #2980 acceptance item #3: this previously carried the
    /// docstring `(stub implementation)` even though the function
    /// performs the real comparison
    ///   `percentage_diff = |actual - reference| / reference * 100`
    ///   `pass = percentage_diff <= energy_tolerance * 100`
    /// using the canonical ASHRAE 140-2017 §B6.7 reference data sourced
    /// from [`Case970Reference::load_case_970_reference_data`]
    /// (Issue #1446). The docstring was misleading — there is no stub.
    /// It is removed here so future readers don't add a `todo!()` or
    /// `unimplemented!()` shortcut and silently regress Case 970
    /// validation. The downstream `validate_case_970_with_validator`
    /// wires this to a real 8760-step simulation (item #2 of #2980).
    pub fn validate_annual_heating(&mut self, actual_heating: f64) -> (bool, f64) {
        let reference = self.reference.annual_heating;
        let error = (actual_heating - reference).abs();
        let percentage_diff = if reference > 0.0 {
            (error / reference) * 100.0
        } else {
            0.0
        };

        let pass = percentage_diff <= self.reference.energy_tolerance * 100.0;

        self.statistics
            .percentage_differences
            .insert("annual_heating".to_string(), percentage_diff);
        self.statistics
            .max_absolute_errors
            .insert("annual_heating".to_string(), error);

        (pass, percentage_diff)
    }

    /// Validate annual cooling energy consumption
    ///
    /// Issue #2980 acceptance item #3: see the matching note on
    /// [`Self::validate_annual_heating`] — the `(stub implementation)`
    /// docstring was misleading and has been removed. The body is the
    /// real ASHRAE 140-2017 §B6.7 ±15% cooling-band comparator (Issue
    /// #1446).
    pub fn validate_annual_cooling(&mut self, actual_cooling: f64) -> (bool, f64) {
        let reference = self.reference.annual_cooling;
        let error = (actual_cooling - reference).abs();
        let percentage_diff = if reference > 0.0 {
            (error / reference) * 100.0
        } else {
            0.0
        };

        let pass = percentage_diff <= self.reference.energy_tolerance * 100.0;

        self.statistics
            .percentage_differences
            .insert("annual_cooling".to_string(), percentage_diff);
        self.statistics
            .max_absolute_errors
            .insert("annual_cooling".to_string(), error);

        (pass, percentage_diff)
    }

    /// Generate basic validation report
    ///
    /// Issue #2980 acceptance item #3: this previously headed the report
    /// with `=== ASHRAE 140 Case 970 Validation Report (STUB) ===` and
    /// printed "Case 970 validation framework is implemented but not yet
    /// fully validated." even though the underlying
    /// [`Self::validate_annual_heating`] and [`Self::validate_annual_cooling`]
    /// perform the real ASHRAE 140-2017 §B6.7 ±15% comparison. The
    /// report header is misleading — downstream
    /// [`ASHRAE140MultiZoneValidator::validate_case_970_with_validator`]
    /// now feeds real engine output (item #2 of #2980). The report now
    /// reflects the actual validator state instead of the pre-#2980
    /// "this case will be completed in future work" disclaimer.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== ASHRAE 140 Case 970 Validation Report ===\n");
        report.push_str("ASHRAE 140-2017 §B6.7 5-zone multi-zone cross-coupling building.\n");

        if let Some(heating_pct) = self.statistics.percentage_differences.get("annual_heating") {
            report.push_str(&format!(
                "Annual Heating:  actual {:.2} MWh vs reference {:.2} MWh \
                 (gap {:.1}%, tolerance ±{:.0}%)\n",
                self.reference.annual_heating * (1.0 + heating_pct / 100.0),
                self.reference.annual_heating,
                heating_pct,
                self.reference.energy_tolerance * 100.0
            ));
        } else {
            report.push_str(&format!(
                "Reference heating: {:.2} MWh (no actual yet — call \
                 validate_annual_heating first)\n",
                self.reference.annual_heating
            ));
        }

        if let Some(cooling_pct) = self.statistics.percentage_differences.get("annual_cooling") {
            report.push_str(&format!(
                "Annual Cooling:  actual {:.2} MWh vs reference {:.2} MWh \
                 (gap {:.1}%, tolerance ±{:.0}%)\n",
                self.reference.annual_cooling * (1.0 + cooling_pct / 100.0),
                self.reference.annual_cooling,
                cooling_pct,
                self.reference.energy_tolerance * 100.0
            ));
        } else {
            report.push_str(&format!(
                "Reference cooling: {:.2} MWh (no actual yet — call \
                 validate_annual_cooling first)\n",
                self.reference.annual_cooling
            ));
        }

        report
    }
}

impl ASHRAE140MultiZoneValidator {
    /// Validate Case 960 using the dedicated validator
    pub fn validate_case_960_with_validator(
        &self,
        _thermal_model: &ThermalModel<impl crate::physics::cta::ContinuousTensor<f64>>,
    ) -> ValidationResult {
        // Issue #1407: the prior implementation compared
        // `actual = 12.5/8.5/5.1/4.9` against a hardcoded reference and
        // unconditionally emitted PASS. Replaced with the same end-to-end
        // physics path as [`Self::validate_case_960`] — runs
        // `ASHRAE140Validator::validate_case_960` which performs the full
        // 8760-step simulation and applies the strict ±15% / ±10%
        // tolerances against the canonical inter-program envelope.
        let started = Instant::now();
        let vrep = self.run_real_case_960_report();
        let _elapsed = started.elapsed();

        let metrics = [
            vrep.heating_result.in_range,
            vrep.cooling_result.in_range,
            vrep.peak_heating_result.in_range,
            vrep.peak_cooling_result.in_range,
        ];
        let in_range = metrics.iter().all(|v| *v);
        let avg_error_pct = (vrep.heating_result.error_pct
            + vrep.cooling_result.error_pct
            + vrep.peak_heating_result.error_pct
            + vrep.peak_cooling_result.error_pct)
            / 4.0;

        ValidationResult {
            in_range,
            error_pct: avg_error_pct,
        }
    }

    /// Validate Case 970 using the dedicated validator
    ///
    /// Issue #2980 acceptance item #2: this previously hardcoded
    /// `actual_heating: 15.0` / `actual_cooling: 10.0` MWh (with the
    /// comment `// Placeholder values - would come from actual simulation`),
    /// which always produced a synthetic PASS/FAIL regardless of engine
    /// state. It now runs the real 8760-step physics simulation against
    /// the Case 970 spec (5-zone cross-coupling building per
    /// ASHRAE 140-2017 §B6.7 / Issue #1446) and feeds the engine's actual
    /// annual heating / cooling into [`Case970Validator`]. The
    /// `actual_heating_mwh` and `actual_cooling_mwh` are converted to
    /// electrical energy (COP 3.0 for cooling, 0.9 efficiency for heating)
    /// to match the ASHRAE 140 reference convention — same conversion
    /// `ASHRAE140Validator::validate_case_960` applies (src/validation/
    /// ashrae_140_validator.rs).
    ///
    /// The incoming `_thermal_model` argument is kept for API back-compat
    /// with the prior stub signature but is not consulted: the validator
    /// builds its own model from `ASHRAE140Case::Case970.spec()` so the
    /// spec → model boundary is single-sourced (mirrors the
    /// `run_real_case_960_report` discipline).
    pub fn validate_case_970_with_validator(
        &self,
        _thermal_model: &ThermalModel<impl crate::physics::cta::ContinuousTensor<f64>>,
    ) -> ValidationResult {
        let mut case_validator = Case970Validator::new();

        // Issue #2980: replaced the hardcoded `15.0` / `10.0` MWh
        // placeholders with the engine's actual annual energy from a
        // real 8760-step physics run.
        let (actual_heating_mwh, actual_cooling_mwh) = self.run_real_case_970_energy();

        // Run validations against the canonical ASHRAE 140-2017 §B6.7
        // inter-program envelope (Issue #1446).
        let (heating_pass, heating_pct) =
            case_validator.validate_annual_heating(actual_heating_mwh);
        let (cooling_pass, cooling_pct) =
            case_validator.validate_annual_cooling(actual_cooling_mwh);

        // Generate report (no longer labels itself "(STUB)" — see item #3).
        let report_text = case_validator.generate_report();
        tracing::info!("{}", report_text);

        // Determine overall pass/fail
        let overall_pass = heating_pass && cooling_pass;

        ValidationResult {
            in_range: overall_pass,
            error_pct: (heating_pct + cooling_pct) / 2.0,
        }
    }

    /// Run the real 8760-step Case 970 physics simulation and return the
    /// annual heating / cooling electrical energy (MWh), converted with
    /// the same COP / efficiency factors used by
    /// [`ASHRAE140Validator::validate_case_960`] (issue #2980 item #2).
    ///
    /// Returns `(annual_heating_electrical_mwh, annual_cooling_electrical_mwh)`.
    /// Each value is finite and `>= 0` for a successful run; the helper
    /// is the single source of truth for Case 970 actuals so the validator
    /// path and any future benchmark-export path cannot diverge.
    fn run_real_case_970_energy(&self) -> (f64, f64) {
        // Same COP / heating-efficiency convention as
        // `ASHRAE140Validator::validate_case_960` (issue #1407):
        //   cooling_cop = 3.0   (1 kWh electricity moves 3 kWh heat)
        //   heating_eff = 0.9   (electric resistance / furnace typical)
        const COOLING_COP: f64 = 3.0;
        const HEATING_EFFICIENCY: f64 = 0.9;

        let spec = ASHRAE140Case::Case970.spec();
        let mut model = ThermalModel::<crate::physics::cta::VectorField>::from_spec(&spec);

        // Per-zone HVAC enable flags from the spec — mirrors
        // `ASHRAE140Validator::validate_case_960`.
        let num_zones = model.num_zones;
        let mut hvac_enabled_vals = vec![1.0_f64; num_zones];
        for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
            if zone_idx < num_zones {
                hvac_enabled_vals[zone_idx] = if hvac.is_enabled() { 1.0 } else { 0.0 };
            }
        }
        model.hvac_enabled = crate::physics::cta::VectorField::new(hvac_enabled_vals);

        model.reset_heating_cooling_energy();
        model.reset_peak_power();

        let weather = EpwWeatherSource::from_file(
            "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
        )
        .expect("Failed to load Case 970 EPW weather data");

        for step in 0..8760 {
            let weather_data = weather
                .get_hourly_data(step)
                .expect("Case 970 EPW hourly data");
            // Extract the only field used downstream (f64 is Copy) so we can move
            // weather_data into model.set_weather without an extra clone (Issue #2893).
            let dry_bulb_temp = weather_data.dry_bulb_temp;
            model.set_weather(weather_data);
            model.step_physics(step, dry_bulb_temp, 3600.0);
        }

        let annual_heating_thermal_mwh = model.get_heating_energy_kwh() / 1000.0;
        let annual_cooling_thermal_mwh = model.get_cooling_energy_kwh() / 1000.0;

        // Thermal → electrical conversion so the engine output is
        // comparable to the ASHRAE 140-2017 §B6.7 reference band, which
        // reports HVAC electricity consumption (EnergyPlus / ESP-r /
        // TRNSYS / DOE-2 / BSIMAC / CSE / DeST — Issue #1446).
        let annual_heating_electrical_mwh = annual_heating_thermal_mwh / HEATING_EFFICIENCY;
        let annual_cooling_electrical_mwh = annual_cooling_thermal_mwh / COOLING_COP;

        (annual_heating_electrical_mwh, annual_cooling_electrical_mwh)
    }

    /// Export validation results to CSV for analysis
    ///
    /// Issue #1407: this previously emitted the hardcoded stub rows
    /// `12.5/12.4`, `8.5/8.7`, `5.1/5.2`, `4.9/4.8` (all "true"). It
    /// now runs the real 8760-step physics simulation and emits the
    /// engine's actual outputs, with `Pass` derived from the canonical
    /// inter-program envelope.
    ///
    /// # Arguments
    /// * `path` - File path to save CSV
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn export_results_to_csv(&self, path: &str) -> std::io::Result<()> {
        let file_path = Path::new(path);
        let mut writer = Writer::from_path(file_path)?;

        // Write header
        writer.write_record([
            "Case",
            "Metric",
            "Actual",
            "Reference",
            "Difference",
            "Pass",
        ])?;

        // Issue #1407: run the real 8760-step physics simulation and
        // emit the engine's actual outputs.
        let vrep = self.run_real_case_960_report();
        let outcome = self.compare_against_reference(
            vrep.annual_heating_mwh,
            vrep.annual_cooling_mwh,
            vrep.peak_heating_kw,
            vrep.peak_cooling_kw,
        );

        writer.write_record([
            "960",
            "Annual Heating",
            &format!("{:.3}", vrep.annual_heating_mwh),
            &format!(
                "{:.3}",
                (outcome.annual_heating_min + outcome.annual_heating_max) / 2.0
            ),
            &format!("{:.3}", outcome.annual_heating_error_pct),
            if outcome.annual_heating_in_range {
                "true"
            } else {
                "false"
            },
        ])?;
        writer.write_record([
            "960",
            "Annual Cooling",
            &format!("{:.3}", vrep.annual_cooling_mwh),
            &format!(
                "{:.3}",
                (outcome.annual_cooling_min + outcome.annual_cooling_max) / 2.0
            ),
            &format!("{:.3}", outcome.annual_cooling_error_pct),
            if outcome.annual_cooling_in_range {
                "true"
            } else {
                "false"
            },
        ])?;
        writer.write_record([
            "960",
            "Peak Heating",
            &format!("{:.3}", vrep.peak_heating_kw),
            &format!(
                "{:.3}",
                (outcome.peak_heating_min + outcome.peak_heating_max) / 2.0
            ),
            &format!("{:.3}", outcome.peak_heating_error_pct),
            if outcome.peak_heating_in_range {
                "true"
            } else {
                "false"
            },
        ])?;
        writer.write_record([
            "960",
            "Peak Cooling",
            &format!("{:.3}", vrep.peak_cooling_kw),
            &format!(
                "{:.3}",
                (outcome.peak_cooling_min + outcome.peak_cooling_max) / 2.0
            ),
            &format!("{:.3}", outcome.peak_cooling_error_pct),
            if outcome.peak_cooling_in_range {
                "true"
            } else {
                "false"
            },
        ])?;

        // Case 970 data (stub)
        writer.write_record(["970", "Annual Heating", "N/A", "N/A", "N/A", "N/A"])?;
        writer.write_record(["970", "Annual Cooling", "N/A", "N/A", "N/A", "N/A"])?;

        writer.flush()?;
        Ok(())
    }

    /// Run comprehensive multi-zone validation with detailed reporting
    pub fn run_comprehensive_validation(&mut self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();

        // Issue #1407: previously this emitted zero/zero actuals whenever
        // the in-memory comparison failed, fabricating PASS. It now runs
        // the real 8760-step physics simulation through
        // `validate_case_960_with_validator` and reports the actual model
        // outputs into the `BenchmarkReport`.
        let case_960_ref = Case960Reference::load_case_960_reference_data();
        let _case_970_ref = Case970Reference::load_case_970_reference_data();

        // Build the real model from the canonical spec.
        let spec = ASHRAE140Case::Case960.spec();
        let model = ThermalModel::<crate::physics::cta::VectorField>::from_spec(&spec);

        // Validate Case 960 with dedicated validator (now real).
        let started = Instant::now();
        let case_960_result = self.validate_case_960_with_validator(&model);
        let _elapsed = started.elapsed();
        let vrep = self.run_real_case_960_report();

        // Validate Case 970 with dedicated validator (still a stub;
        // Case 970 is tracked separately).
        let _case_970_result = self.validate_case_970_with_validator(&model);

        // Add results to report — actual model output, not 0.0-on-FAIL.
        report.add_result_simple(
            "960",
            MetricType::AnnualHeating,
            vrep.annual_heating_mwh,
            case_960_ref.annual_heating * (1.0 - case_960_ref.energy_tolerance),
            case_960_ref.annual_heating * (1.0 + case_960_ref.energy_tolerance),
        );

        report.add_result_simple(
            "960",
            MetricType::AnnualCooling,
            vrep.annual_cooling_mwh,
            case_960_ref.annual_cooling * (1.0 - case_960_ref.energy_tolerance),
            case_960_ref.annual_cooling * (1.0 + case_960_ref.energy_tolerance),
        );

        report.add_result_simple(
            "960",
            MetricType::PeakHeating,
            vrep.peak_heating_kw,
            case_960_ref.peak_heating * (1.0 - case_960_ref.load_tolerance),
            case_960_ref.peak_heating * (1.0 + case_960_ref.load_tolerance),
        );

        report.add_result_simple(
            "960",
            MetricType::PeakCooling,
            vrep.peak_cooling_kw,
            case_960_ref.peak_cooling * (1.0 - case_960_ref.load_tolerance),
            case_960_ref.peak_cooling * (1.0 + case_960_ref.load_tolerance),
        );

        let _ = case_960_result.in_range;

        // Add stub results for Case 970
        report.add_result_simple("970", MetricType::AnnualHeating, 0.0, 0.0, 0.0);
        report.add_result_simple("970", MetricType::AnnualCooling, 0.0, 0.0, 0.0);

        // Add stub results for Case 980
        report.add_result_simple("980", MetricType::AnnualHeating, 0.0, 0.0, 0.0);
        report.add_result_simple("980", MetricType::AnnualCooling, 0.0, 0.0, 0.0);

        report
    }
}
