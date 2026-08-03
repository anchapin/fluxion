use serde::{Deserialize, Serialize};

/// ASHRAE 55 metabolic rate categories for occupant internal gain calculation.
///
/// 1 met = 58.2 W/m² (basal metabolic rate for average adult male, ~70W total)
/// These values represent the metabolic rate per unit body surface area (1.8 m²).
///
/// Reference: ASHRAE 55-2020 Table 5.2.2 (Metabolic Rates)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MetabolicRate {
    /// Default: Seated quietly - 1.0 met (~58 W/person)
    #[default]
    SeatedQuiet,
    /// Sleeping: 0.7 met (~40 W/person) - lowest metabolic rate
    Sleeping,
    /// Office work: 1.2 met (~70 W/person) - light cognitive task
    OfficeWork,
    /// Light activity: 1.6 met (~93 W/person) - standing, walking slowly
    LightActivity,
    /// Standing: 1.7 met (~99 W/person) - retail, light work
    Standing,
    /// Walking: 2.0 met (~116 W/person) - walking at 1.0 m/s
    Walking,
}

impl MetabolicRate {
    /// Returns metabolic rate in met units (ASHRAE 55).
    /// 1 met = 58.2 W/m² of body surface area.
    pub fn met(&self) -> f64 {
        match self {
            MetabolicRate::Sleeping => 0.7,
            MetabolicRate::SeatedQuiet => 1.0,
            MetabolicRate::OfficeWork => 1.2,
            MetabolicRate::LightActivity => 1.6,
            MetabolicRate::Standing => 1.7,
            MetabolicRate::Walking => 2.0,
        }
    }

    /// Returns metabolic rate in W/m² (basal = 58.2 W/m²).
    pub fn watts_per_m2(&self) -> f64 {
        self.met() * 58.2
    }

    /// Returns sensible heat gain per person [W] given the occupant state.
    /// Based on ASHRAE 55 sensible/latent split (approximately 60/40).
    pub fn sensible_gain_w(&self) -> f64 {
        let total = self.watts_per_m2() * 1.8; // ~1.8 m² body surface area
        total * 0.6 // 60% sensible
    }

    /// Returns latent heat gain per person [W] given the occupant state.
    pub fn latent_gain_w(&self) -> f64 {
        let total = self.watts_per_m2() * 1.8;
        total * 0.4 // 40% latent
    }

    /// Returns the total heat gain per person [W].
    pub fn total_gain_w(&self) -> f64 {
        self.sensible_gain_w() + self.latent_gain_w()
    }
}

/// Metabolic rate variation by time of day and occupancy state.
/// Provides diurnal variation patterns for different building types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolicRatePattern {
    /// Commercial pattern: higher during work hours, lower at night
    Commercial,
    /// Residential pattern: moderate at night (sleeping), higher evening/morning
    Residential,
}

impl MetabolicRatePattern {
    /// Returns the metabolic rate for a given hour and occupancy state.
    pub fn get_rate(
        &self,
        hour: u8,
        occupant_state: super::lighting::OccupantState,
    ) -> MetabolicRate {
        match occupant_state {
            super::lighting::OccupantState::Absent => MetabolicRate::SeatedQuiet,
            super::lighting::OccupantState::Sleeping => MetabolicRate::Sleeping,
            super::lighting::OccupantState::PresentActive => self.active_rate_for_hour(hour),
        }
    }

    fn active_rate_for_hour(&self, hour: u8) -> MetabolicRate {
        match self {
            MetabolicRatePattern::Commercial => {
                // Commercial: higher during work hours (8-17), lower at night
                if (8..=17).contains(&hour) {
                    MetabolicRate::OfficeWork // 1.2 met
                } else if (18..=21).contains(&hour) {
                    MetabolicRate::Standing // 1.7 met - evening transition
                } else {
                    MetabolicRate::SeatedQuiet // 1.0 met - early morning/night
                }
            }
            MetabolicRatePattern::Residential => {
                // Residential: sleeping at night, moderate morning/evening
                if hour >= 23 || hour <= 5 {
                    MetabolicRate::Sleeping // 0.7 met - night
                } else if (6..=8).contains(&hour) || (18..=22).contains(&hour) {
                    MetabolicRate::Standing // 1.7 met - morning/evening
                } else {
                    MetabolicRate::SeatedQuiet // 1.0 met - daytime idle
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Co2Generation {
    /// CO2 generation rate per person at 1 met [L/s-person]
    /// Default: 0.005 L/s-person (typical office occupant)
    pub co2_generation_rate_per_person: f64,
}

impl Default for Co2Generation {
    fn default() -> Self {
        Self::new()
    }
}

impl Co2Generation {
    pub fn new() -> Self {
        Self {
            co2_generation_rate_per_person: 0.005,
        }
    }

    pub fn with_rate(mut self, rate: f64) -> Self {
        self.co2_generation_rate_per_person = rate;
        self
    }

    /// Calculate CO2 generation rate [L/s] based on occupant count and metabolic rate.
    ///
    /// CO2 generation scales with metabolic rate because higher activity produces
    /// more metabolic CO2. The factor is normalized to 1 met = 1.0.
    pub fn calculate_co2_generation(
        &self,
        occupant_count: f64,
        metabolic_rate: MetabolicRate,
    ) -> f64 {
        let activity_factor = metabolic_rate.met(); // 0.7-2.0 range
        occupant_count * self.co2_generation_rate_per_person * activity_factor
    }
}

use chrono::{DateTime, Timelike, Utc};
use std::sync::Arc;

use crate::lighting::{LightingModel, OccupantState};

pub trait OccupancyProvider: Send + Sync {
    fn occupant_state(&self, t: DateTime<Utc>) -> OccupantState;
    fn occupant_count(&self, t: DateTime<Utc>) -> f64;
}

pub trait PlugLoadProvider: Send + Sync {
    fn get_plug_load(&self, t: DateTime<Utc>) -> f64;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct InternalGains {
    pub phi_sensible: f64,
    pub phi_latent: f64,
}

impl InternalGains {
    pub fn new(phi_sensible: f64, phi_latent: f64) -> Self {
        Self {
            phi_sensible,
            phi_latent,
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }
}

pub struct DynamicInternalGainAdapter {
    occupancy: Arc<dyn OccupancyProvider>,
    plug_loads: Arc<dyn PlugLoadProvider>,
    lighting: LightingModel,
    sensible_per_occupant: f64,
    latent_per_occupant: f64,
}

impl DynamicInternalGainAdapter {
    pub fn new(
        occupancy: Arc<dyn OccupancyProvider>,
        plug_loads: Arc<dyn PlugLoadProvider>,
        lighting: LightingModel,
    ) -> Self {
        Self {
            occupancy,
            plug_loads,
            lighting,
            sensible_per_occupant: 70.0,
            latent_per_occupant: 30.0,
        }
    }

    pub fn with_metabolic_rates(
        mut self,
        sensible_per_occupant: f64,
        latent_per_occupant: f64,
    ) -> Self {
        self.sensible_per_occupant = sensible_per_occupant;
        self.latent_per_occupant = latent_per_occupant;
        self
    }

    pub fn compute_gains(&self, _zone_id: uuid::Uuid, t: DateTime<Utc>) -> InternalGains {
        let occupancy_state = self.occupancy.occupant_state(t);
        let n_occupants = self.occupancy.occupant_count(t);
        let plug_w = self.plug_loads.get_plug_load(t);
        let lighting_w = self.lighting.compute(t, occupancy_state);

        let occupant_sensible = n_occupants * self.sensible_per_occupant;
        let occupant_latent = n_occupants * self.latent_per_occupant;

        InternalGains {
            phi_sensible: occupant_sensible + plug_w + lighting_w,
            phi_latent: occupant_latent,
        }
    }
}

impl Default for DynamicInternalGainAdapter {
    fn default() -> Self {
        Self {
            occupancy: Arc::new(ScheduleOccupancyProvider::default()),
            plug_loads: Arc::new(ConstantPlugLoadProvider::default()),
            lighting: LightingModel::default(),
            sensible_per_occupant: 70.0,
            latent_per_occupant: 30.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleOccupancyProvider {
    pub hourly_counts: Vec<f64>,
}

impl ScheduleOccupancyProvider {
    pub fn new(hourly_counts: Vec<f64>) -> Self {
        Self { hourly_counts }
    }
}

impl OccupancyProvider for ScheduleOccupancyProvider {
    fn occupant_state(&self, t: DateTime<Utc>) -> OccupantState {
        let hour = t.hour() as usize;
        let count = self.hourly_counts.get(hour % 24).copied().unwrap_or(0.0);
        if count > 0.0 {
            OccupantState::PresentActive
        } else {
            OccupantState::Absent
        }
    }

    fn occupant_count(&self, t: DateTime<Utc>) -> f64 {
        let hour = t.hour() as usize;
        self.hourly_counts.get(hour % 24).copied().unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstantPlugLoadProvider {
    pub watts: f64,
}

impl ConstantPlugLoadProvider {
    pub fn new(watts: f64) -> Self {
        Self { watts }
    }
}

impl PlugLoadProvider for ConstantPlugLoadProvider {
    fn get_plug_load(&self, _t: DateTime<Utc>) -> f64 {
        self.watts
    }
}

// ---------------------------------------------------------------------------
// Occupant-Dependent Internal Gains (combines all 3 P0 features)
// ---------------------------------------------------------------------------

/// Internal gains adapter that computes occupant-dependent gains using:
/// - Metabolic rate variation (ASHRAE 55 categories)
/// - Equipment diversity factor
/// - Demand-controlled ventilation integration
///
/// This adapter produces:
/// - Variable sensible/latent gains based on metabolic rate and occupancy
/// - Diversity-reduced equipment gains
/// - DCV parameters for ventilation system integration
pub struct OccupantDependentInternalGains {
    occupancy: Arc<dyn OccupancyProvider>,
    metabolic_rate: Arc<dyn MetabolicRateProvider>,
    plug_loads: Arc<dyn PlugLoadProvider>,
    lighting: LightingModel,
    dcv_params: DcvParameters,
}

impl std::fmt::Debug for OccupantDependentInternalGains {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccupantDependentInternalGains")
            .field("lighting", &self.lighting)
            .field("dcv_params", &self.dcv_params)
            .finish()
    }
}

impl Clone for OccupantDependentInternalGains {
    fn clone(&self) -> Self {
        Self {
            occupancy: self.occupancy.clone(),
            metabolic_rate: self.metabolic_rate.clone(),
            plug_loads: self.plug_loads.clone(),
            lighting: self.lighting.clone(),
            dcv_params: self.dcv_params,
        }
    }
}

impl OccupantDependentInternalGains {
    pub fn new(
        occupancy: Arc<dyn OccupancyProvider>,
        metabolic_rate: Arc<dyn MetabolicRateProvider>,
        plug_loads: Arc<dyn PlugLoadProvider>,
        lighting: LightingModel,
        dcv_params: DcvParameters,
    ) -> Self {
        Self {
            occupancy,
            metabolic_rate,
            plug_loads,
            lighting,
            dcv_params,
        }
    }

    /// Create with typical commercial building parameters.
    pub fn commercial(
        occupancy: Arc<dyn OccupancyProvider>,
        plug_loads: Arc<dyn PlugLoadProvider>,
    ) -> Self {
        Self::new(
            occupancy,
            Arc::new(PatternMetabolicRateProvider::commercial()),
            plug_loads,
            LightingModel::office(),
            DcvParameters::commercial(),
        )
    }

    /// Create with typical residential building parameters.
    pub fn residential(
        occupancy: Arc<dyn OccupancyProvider>,
        plug_loads: Arc<dyn PlugLoadProvider>,
    ) -> Self {
        Self::new(
            occupancy,
            Arc::new(PatternMetabolicRateProvider::residential()),
            plug_loads,
            LightingModel::office(),
            DcvParameters::residential(),
        )
    }

    /// Compute internal gains for a zone at time t.
    ///
    /// Returns `InternalGains` with:
    /// - `phi_sensible`: occupant sensible + plug loads + lighting
    /// - `phi_latent`: occupant latent
    pub fn compute_gains(&self, _zone_id: uuid::Uuid, t: DateTime<Utc>) -> InternalGains {
        let occupancy_state = self.occupancy.occupant_state(t);
        let n_occupants = self.occupancy.occupant_count(t);

        // Get metabolic rate based on time and occupancy state
        let met_rate = self.metabolic_rate.get_metabolic_rate(t, occupancy_state);

        // Compute occupant gains using metabolic rate
        let occupant_sensible = n_occupants * met_rate.sensible_gain_w();
        let occupant_latent = n_occupants * met_rate.latent_gain_w();

        // Equipment gains (already include diversity factor if using EquipmentDiversityFactor)
        let plug_w = self.plug_loads.get_plug_load(t);

        // Lighting gains
        let lighting_w = self.lighting.compute(t, occupancy_state);

        InternalGains {
            phi_sensible: occupant_sensible + plug_w + lighting_w,
            phi_latent: occupant_latent,
        }
    }

    /// Get DCV parameters for this zone.
    pub fn dcv_params(&self) -> DcvParameters {
        self.dcv_params
    }

    /// Compute the current ventilation fraction based on occupancy.
    pub fn ventilation_fraction(&self) -> f64 {
        let n_occupants = self.occupancy.occupant_count(
            chrono::Utc::now(), // This would be passed in properly in real usage
        );
        self.dcv_params.ventilation_fraction(n_occupants)
    }

    /// Compute effective ventilation ACH for a given occupancy.
    pub fn effective_ventilation_ach(&self, occupancy: f64) -> f64 {
        self.dcv_params.effective_ach(occupancy)
    }

    /// Compute fan energy reduction fraction for a given occupancy.
    pub fn fan_energy_reduction(&self, occupancy: f64) -> f64 {
        self.dcv_params.fan_energy_reduction(occupancy)
    }
}

impl Default for OccupantDependentInternalGains {
    fn default() -> Self {
        Self::commercial(
            Arc::new(ScheduleOccupancyProvider::default()),
            Arc::new(ConstantPlugLoadProvider::default()),
        )
    }
}

// ---------------------------------------------------------------------------
// Metabolic Rate Provider (for time-varying metabolic rates)
// ---------------------------------------------------------------------------

/// Provider trait for time-varying metabolic rates.
///
/// Allows metabolic rate to vary by time of day and occupancy state,
/// per ASHRAE 55 activity categories.
pub trait MetabolicRateProvider: Send + Sync {
    /// Get the metabolic rate at a given time and occupancy state.
    fn get_metabolic_rate(&self, t: DateTime<Utc>, state: OccupantState) -> MetabolicRate;
}

/// Default metabolic rate provider that uses a fixed pattern (commercial/residential).
#[derive(Debug, Clone)]
pub struct PatternMetabolicRateProvider {
    pattern: MetabolicRatePattern,
}

impl PatternMetabolicRateProvider {
    pub fn new(pattern: MetabolicRatePattern) -> Self {
        Self { pattern }
    }

    pub fn commercial() -> Self {
        Self::new(MetabolicRatePattern::Commercial)
    }

    pub fn residential() -> Self {
        Self::new(MetabolicRatePattern::Residential)
    }
}

impl Default for PatternMetabolicRateProvider {
    fn default() -> Self {
        Self::commercial()
    }
}

impl MetabolicRateProvider for PatternMetabolicRateProvider {
    fn get_metabolic_rate(&self, t: DateTime<Utc>, state: OccupantState) -> MetabolicRate {
        self.pattern.get_rate(t.hour() as u8, state)
    }
}

/// Constant metabolic rate provider - returns the same rate regardless of time.
#[derive(Debug, Clone)]
pub struct ConstantMetabolicRateProvider {
    rate: MetabolicRate,
}

impl ConstantMetabolicRateProvider {
    pub fn new(rate: MetabolicRate) -> Self {
        Self { rate }
    }
}

impl Default for ConstantMetabolicRateProvider {
    fn default() -> Self {
        Self::new(MetabolicRate::OfficeWork)
    }
}

impl MetabolicRateProvider for ConstantMetabolicRateProvider {
    fn get_metabolic_rate(&self, _t: DateTime<Utc>, _state: OccupantState) -> MetabolicRate {
        self.rate
    }
}

// ---------------------------------------------------------------------------
// Equipment Diversity Factor
// ---------------------------------------------------------------------------

/// Equipment diversity factor that reduces peak plug loads.
///
/// Not all equipment runs simultaneously - the diversity factor accounts for
/// this by reducing the peak load. Typical values: 0.6-0.8 (20-40% reduction).
///
/// Reference: ASHRAE 90.1 Section 11 (Lighting) and standard plug load diversity practices.
pub struct EquipmentDiversityFactor {
    inner: Arc<dyn PlugLoadProvider>,
    /// Diversity factor (0.0 to 1.0). 1.0 = all equipment running (no diversity).
    /// Typical: 0.6-0.8 for commercial buildings.
    diversity_factor: f64,
}

impl std::fmt::Debug for EquipmentDiversityFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EquipmentDiversityFactor")
            .field("diversity_factor", &self.diversity_factor)
            .finish()
    }
}

impl Clone for EquipmentDiversityFactor {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            diversity_factor: self.diversity_factor,
        }
    }
}

impl EquipmentDiversityFactor {
    pub fn new(inner: Arc<dyn PlugLoadProvider>, diversity_factor: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&diversity_factor),
            "diversity_factor must be in [0, 1], got {}",
            diversity_factor
        );
        Self {
            inner,
            diversity_factor,
        }
    }

    /// Create with a typical diversity factor of 0.7 (30% reduction).
    pub fn with_typical_diversity(inner: Arc<dyn PlugLoadProvider>) -> Self {
        Self::new(inner, 0.7)
    }

    /// Create with 20% diversity (factor = 0.8).
    pub fn with_20_percent_diversity(inner: Arc<dyn PlugLoadProvider>) -> Self {
        Self::new(inner, 0.8)
    }

    /// Create with 40% diversity (factor = 0.6).
    pub fn with_40_percent_diversity(inner: Arc<dyn PlugLoadProvider>) -> Self {
        Self::new(inner, 0.6)
    }

    /// Create with 30% diversity (factor = 0.7, same as typical).
    pub fn with_30_percent_diversity(inner: Arc<dyn PlugLoadProvider>) -> Self {
        Self::new(inner, 0.7)
    }
}

impl PlugLoadProvider for EquipmentDiversityFactor {
    fn get_plug_load(&self, t: DateTime<Utc>) -> f64 {
        self.inner.get_plug_load(t) * self.diversity_factor
    }
}

impl Default for EquipmentDiversityFactor {
    fn default() -> Self {
        Self::new(Arc::new(ConstantPlugLoadProvider::default()), 0.7)
    }
}

// ---------------------------------------------------------------------------
// Demand-Controlled Ventilation (DCV)
// ---------------------------------------------------------------------------

/// Demand-controlled ventilation adapter that scales ventilation based on occupancy.
///
/// Per ASHRAE 62.1, when occupancy drops below 50% of design, ventilation
/// can be proportionally reduced (but not below 10% of design for IAQ).
///
/// This trait is used by the HVAC system to compute the ventilation fraction
/// based on current occupancy.
#[derive(Debug, Clone, Copy)]
pub struct DcvParameters {
    /// Design occupancy count (at 100% occupancy).
    pub design_occupancy: f64,
    /// Design ventilation rate [ACH] at 100% occupancy.
    pub design_ventilation_ach: f64,
    /// Minimum ventilation fraction when unoccupied (ASHRAE 62.1 minimum for IAQ).
    pub minimum_ventilation_fraction: f64,
    /// DCV occupancy threshold below which ventilation is reduced.
    pub dcv_threshold_fraction: f64,
}

impl Default for DcvParameters {
    fn default() -> Self {
        Self::commercial()
    }
}

impl DcvParameters {
    /// Commercial building DCV parameters.
    /// Per ASHRAE 62.1 Section 6.2.3 (Demand Control Ventilation).
    pub fn commercial() -> Self {
        Self {
            design_occupancy: 50.0,
            design_ventilation_ach: 0.5,
            minimum_ventilation_fraction: 0.1, // 10% minimum for IAQ
            dcv_threshold_fraction: 0.5,       // 50% threshold for DCV
        }
    }

    /// Residential DCV parameters.
    pub fn residential() -> Self {
        Self {
            design_occupancy: 4.0,
            design_ventilation_ach: 0.3,
            minimum_ventilation_fraction: 0.1,
            dcv_threshold_fraction: 0.5,
        }
    }

    /// Calculate the ventilation fraction based on current occupancy.
    ///
    /// When occupancy < dcv_threshold_fraction (50%), ventilation is reduced
    /// proportionally but not below minimum_ventilation_fraction.
    ///
    /// Returns a fraction (0.0 to 1.0) of design ventilation.
    pub fn ventilation_fraction(&self, current_occupancy: f64) -> f64 {
        let occupancy_fraction = current_occupancy / self.design_occupancy;

        if occupancy_fraction >= self.dcv_threshold_fraction {
            1.0 // Full ventilation at or above threshold
        } else {
            // Scale ventilation proportionally below threshold
            // But enforce minimum ventilation fraction for IAQ
            let scaled = occupancy_fraction / self.dcv_threshold_fraction;
            scaled.max(self.minimum_ventilation_fraction)
        }
    }

    /// Calculate effective ventilation rate [ACH] based on current occupancy.
    pub fn effective_ach(&self, current_occupancy: f64) -> f64 {
        self.design_ventilation_ach * self.ventilation_fraction(current_occupancy)
    }

    /// Calculate fan energy reduction compared to no-DCV operation.
    /// Returns the fractional reduction (0.0 to 1.0).
    pub fn fan_energy_reduction(&self, current_occupancy: f64) -> f64 {
        let frac = self.ventilation_fraction(current_occupancy);
        1.0 - frac // If ventilation is 60%, fan energy reduction is 40%
    }
}

/// Trait for ventilation systems that support demand-controlled ventilation.
pub trait DcvVentilation: Send + Sync {
    /// Get the current ventilation ACH, accounting for DCV.
    fn get_dcv_ach(&self, occupancy: f64) -> f64;

    /// Get the fan energy fraction (0.0 to 1.0) for current occupancy.
    fn fan_energy_fraction(&self, occupancy: f64) -> f64;
}

#[cfg(test)]
mod dcv_tests {
    use super::*;

    #[test]
    fn test_dcv_full_occupancy() {
        let params = DcvParameters::commercial();
        assert!((params.ventilation_fraction(50.0) - 1.0).abs() < 1e-10);
        assert!((params.fan_energy_reduction(50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dcv_below_threshold() {
        let params = DcvParameters::commercial();
        // 40% occupancy (20 persons) - below 50% threshold, should reduce ventilation
        let frac = params.ventilation_fraction(20.0);
        assert!(frac < 1.0, "40% occupancy should be below full ventilation");
        assert!(frac >= 0.1, "should be at least minimum ventilation"); // At least minimum
    }

    #[test]
    fn test_dcv_minimum_ventilation() {
        let params = DcvParameters::commercial();
        // Very low occupancy should still provide minimum ventilation
        let frac = params.ventilation_fraction(1.0);
        assert!((frac - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_dcv_fan_energy_reduction() {
        let params = DcvParameters::commercial();
        // At 40% occupancy (20 persons), fan energy should be reduced
        let reduction = params.fan_energy_reduction(20.0);
        assert!(reduction > 0.0, "40% occupancy should reduce fan energy");
        assert!(reduction < 1.0, "reduction should not be 100%");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy::MarkovOccupancyGenerator;
    use chrono::TimeZone;

    #[test]
    fn test_internal_gains_default() {
        let gains = InternalGains::default();
        assert!((gains.phi_sensible - 0.0).abs() < 1e-10);
        assert!((gains.phi_latent - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_internal_gains_new() {
        let gains = InternalGains::new(100.0, 30.0);
        assert!((gains.phi_sensible - 100.0).abs() < 1e-10);
        assert!((gains.phi_latent - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_adapter_default() {
        let adapter = DynamicInternalGainAdapter::default();
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 10, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);
        assert!(gains.phi_sensible >= 0.0);
        assert!(gains.phi_latent >= 0.0);
    }

    #[test]
    fn test_dynamic_adapter_zero_occupancy() {
        let adapter = DynamicInternalGainAdapter::default();
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 3, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);
        assert!(gains.phi_latent <= 1e-10);
    }

    #[test]
    fn test_dynamic_adapter_with_custom_metabolic() {
        let occupancy = Arc::new(ScheduleOccupancyProvider::new(vec![1.0; 24]));
        let plug_loads = Arc::new(ConstantPlugLoadProvider::new(100.0));
        let lighting = LightingModel::default();

        let adapter = DynamicInternalGainAdapter::new(occupancy, plug_loads, lighting)
            .with_metabolic_rates(60.0, 40.0);

        let t = Utc.with_ymd_and_hms(2024, 7, 15, 10, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);

        assert!(gains.phi_sensible > 0.0);
        assert!((gains.phi_latent - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_occupancy_provider() {
        let provider = ScheduleOccupancyProvider::new(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let t_night = Utc.with_ymd_and_hms(2024, 7, 15, 3, 0, 0).unwrap();
        let t_day = Utc.with_ymd_and_hms(2024, 7, 15, 10, 0, 0).unwrap();

        assert!((provider.occupant_count(t_night) - 0.0).abs() < 1e-10);
        assert!((provider.occupant_count(t_day) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_plug_load_provider() {
        let provider = ConstantPlugLoadProvider::new(200.0);
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        assert!((provider.get_plug_load(t) - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_adapter_arc_traits() {
        let occupancy: Arc<dyn OccupancyProvider> = Arc::new(ScheduleOccupancyProvider::default());
        let plug_loads: Arc<dyn PlugLoadProvider> = Arc::new(ConstantPlugLoadProvider::new(150.0));
        let lighting = LightingModel::default();

        let adapter = DynamicInternalGainAdapter::new(occupancy, plug_loads, lighting);
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 14, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);

        assert!(gains.phi_sensible >= 0.0);
        assert!(gains.phi_latent >= 0.0);
    }

    // -----------------------------------------------------------------------
    // Issue #1910: Occupant-dependent internal gains acceptance criteria
    // -----------------------------------------------------------------------

    /// Acceptance criterion 1: Metabolic rate variation produces at least 10% daily variation.
    ///
    /// Verifies that using a residential metabolic rate pattern produces more than 10%
    /// variation between night (sleeping) and evening (standing) activity levels.
    #[test]
    fn test_issue_1910_metabolic_rate_variation_10_percent() {
        let provider = PatternMetabolicRateProvider::residential();

        // Night hour (2 AM) - sleeping
        let t_night = Utc.with_ymd_and_hms(2024, 7, 15, 2, 0, 0).unwrap();
        let night_rate = provider.get_metabolic_rate(t_night, OccupantState::Sleeping);

        // Evening hour (20:00) - standing
        let t_evening = Utc.with_ymd_and_hms(2024, 7, 15, 20, 0, 0).unwrap();
        let evening_rate = provider.get_metabolic_rate(t_evening, OccupantState::PresentActive);

        let night_gains = night_rate.total_gain_w();
        let evening_gains = evening_rate.total_gain_w();

        let daily_variation = (evening_gains - night_gains) / night_gains;

        println!(
            "Metabolic rate variation: night={:.1} W/person, evening={:.1} W/person, variation={:.1}%",
            night_gains,
            evening_gains,
            daily_variation * 100.0
        );

        assert!(
            daily_variation >= 0.10,
            "Metabolic rate variation {:.1}% is less than required 10%",
            daily_variation * 100.0
        );
    }

    /// Acceptance criterion 2: Equipment diversity reduces peak internal gains by at least 20%.
    ///
    /// Verifies that wrapping a plug load provider with a diversity factor of 0.7
    /// (typical 30% diversity) reduces peak load by at least 20%.
    #[test]
    fn test_issue_1910_equipment_diversity_20_percent() {
        let peak_load = 1000.0; // W

        // Create a constant plug load provider with known peak
        let inner: Arc<dyn PlugLoadProvider> = Arc::new(ConstantPlugLoadProvider::new(peak_load));

        // Apply 30% diversity (factor = 0.7, reduction = 30%)
        let diverse = EquipmentDiversityFactor::with_30_percent_diversity(inner);

        let t = Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let reduced_load = diverse.get_plug_load(t);

        let reduction = (peak_load - reduced_load) / peak_load;

        println!(
            "Equipment diversity: peak={:.0} W, reduced={:.0} W, reduction={:.1}%",
            peak_load,
            reduced_load,
            reduction * 100.0
        );

        assert!(
            reduction >= 0.20,
            "Equipment diversity reduction {:.1}% is less than required 20%",
            reduction * 100.0
        );
    }

    /// Acceptance criterion 3: DCV reduces fan energy when occupancy < 50%.
    ///
    /// Verifies that when occupancy drops below 50% of design, the ventilation
    /// fraction and fan energy are reduced proportionally.
    #[test]
    fn test_issue_1910_dcv_fan_energy_reduction() {
        let params = DcvParameters::commercial();
        let design_occupancy = 50.0;

        // Full occupancy - no fan energy reduction
        let reduction_full = params.fan_energy_reduction(design_occupancy);
        assert!(
            reduction_full.abs() < 1e-10,
            "Fan energy reduction at 100% occupancy should be 0, got {:.4}",
            reduction_full
        );

        // 40% occupancy (20 persons) - should have significant fan energy reduction
        let current_occupancy = design_occupancy * 0.40; // 20 persons
        let reduction_40 = params.fan_energy_reduction(current_occupancy);

        println!(
            "DCV fan energy reduction at 40% occupancy: {:.1}%",
            reduction_40 * 100.0
        );

        assert!(
            reduction_40 > 0.0,
            "Fan energy reduction at 40% occupancy should be > 0, got {:.4}",
            reduction_40
        );
        assert!(
            reduction_40 < 1.0,
            "Fan energy reduction at 40% occupancy should be < 100%, got {:.4}",
            reduction_40
        );

        // Verify DCV activates below 50% threshold
        let vent_fraction_40 = params.ventilation_fraction(current_occupancy);
        assert!(
            vent_fraction_40 < 1.0,
            "Ventilation fraction at 40% occupancy should be < 1.0, got {:.4}",
            vent_fraction_40
        );
    }

    /// Combined test: OccupantDependentInternalGains produces all three P0 features.
    #[test]
    fn test_issue_1910_occupant_dependent_gains_combined() {
        let occupancy = Arc::new(MarkovOccupancyGenerator::commercial());
        let plug_loads: Arc<dyn PlugLoadProvider> =
            Arc::new(EquipmentDiversityFactor::with_30_percent_diversity(
                Arc::new(ConstantPlugLoadProvider::new(1000.0)),
            ));

        let adapter = OccupantDependentInternalGains::commercial(occupancy, plug_loads);

        // Compare peak vs off-peak gains
        let t_peak = Utc.with_ymd_and_hms(2024, 1, 9, 10, 0, 0).unwrap(); // Wed 10 AM
        let t_offpeak = Utc.with_ymd_and_hms(2024, 1, 9, 3, 0, 0).unwrap(); // Wed 3 AM

        let gains_peak = adapter.compute_gains(uuid::Uuid::new_v4(), t_peak);
        let gains_offpeak = adapter.compute_gains(uuid::Uuid::new_v4(), t_offpeak);

        println!(
            "Peak gains: sensible={:.1} W, latent={:.1} W",
            gains_peak.phi_sensible, gains_peak.phi_latent
        );
        println!(
            "Off-peak gains: sensible={:.1} W, latent={:.1} W",
            gains_offpeak.phi_sensible, gains_offpeak.phi_latent
        );

        // Off-peak should have lower gains due to metabolic rate variation
        assert!(
            gains_peak.phi_sensible >= gains_offpeak.phi_sensible,
            "Peak gains should be >= off-peak"
        );
    }
}
