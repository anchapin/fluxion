//! Thermal Network Diagnostic Instrumentation for Phase 3 Physics Analysis
//!
//! This module provides detailed tracking of:
//! - Conductance values per thermal path
//! - Heat flow breakdown by component
//! - Thermal mass energy changes
//! - Energy balance verification
//!
//! Purpose: Identify root causes of 4x higher HVAC energy in ASHRAE 140 validation.

use std::collections::HashMap;

/// Detailed energy flow tracking for a single timestep.
#[derive(Debug, Clone, Default)]
pub struct EnergyFlowDiagnostics {
    /// Heat flow through exterior-to-mass path: Q_em = h_tr_em × (T_ext - T_m) [W]
    pub q_em: f64,
    /// Heat flow through mass-to-surface path: Q_ms = h_tr_ms × (T_m - T_s) [W]
    pub q_ms: f64,
    /// Heat flow through surface-to-interior path: Q_is = h_tr_is × (T_s - T_i) [W]
    pub q_is: f64,
    /// Heat flow through windows: Q_w = h_tr_w × (T_ext - T_i) [W]
    pub q_w: f64,
    /// Heat flow through ventilation: Q_ve = h_ve × (T_ext - T_i) [W]
    pub q_ve: f64,
    /// Heat flow through floor: Q_floor = h_tr_floor × (T_g - T_s) [W]
    pub q_floor: f64,
    /// Radiative gains to surface node [W]
    pub phi_st: f64,
    /// Radiative gains directly to mass [W]
    pub phi_m: f64,
    /// Convective internal gains to air [W]
    pub phi_ia: f64,
    /// Inter-zone heat transfer [W]
    pub q_iz: f64,
    /// HVAC heating/cooling output [W]
    pub hvac_output: f64,
    /// Thermal mass energy change: ΔE = C_m × (Tm_new - Tm_old) [J]
    pub mass_energy_change: f64,
    /// Net energy (HVAC - mass_change) [J]
    pub net_energy: f64,
}

/// Conductance values for thermal network analysis.
#[derive(Debug, Clone, Default)]
pub struct ConductanceDiagnostics {
    /// Exterior-to-mass conductance [W/K]
    pub h_tr_em: f64,
    /// Mass-to-surface conductance [W/K]
    pub h_tr_ms: f64,
    /// Surface-to-interior conductance [W/K]
    pub h_tr_is: f64,
    /// Window conductance [W/K]
    pub h_tr_w: f64,
    /// Ventilation conductance [W/K]
    pub h_ve: f64,
    /// Floor conductance [W/K]
    pub h_tr_floor: f64,
    /// Thermal capacitance [J/K]
    pub thermal_capacitance: f64,
    /// Effective mass area A_m [m²]
    pub a_m: f64,
    /// Calculated thermal time constant τ = C_m / h_tr_ms [seconds]
    pub thermal_time_constant: f64,
}

/// Cumulative energy tracking over simulation period.
#[derive(Debug, Clone, Default)]
pub struct CumulativeEnergyDiagnostics {
    /// Total heating energy [kWh]
    pub total_heating_kwh: f64,
    /// Total cooling energy [kWh]
    pub total_cooling_kwh: f64,
    /// Cumulative heat flow through exterior-to-mass [J]
    pub cumulative_q_em: f64,
    /// Cumulative heat flow through mass-to-surface [J]
    pub cumulative_q_ms: f64,
    /// Cumulative heat flow through surface-to-interior [J]
    pub cumulative_q_is: f64,
    /// Cumulative heat flow through windows [J]
    pub cumulative_q_w: f64,
    /// Cumulative heat flow through ventilation [J]
    pub cumulative_q_ve: f64,
    /// Cumulative heat flow through floor [J]
    pub cumulative_q_floor: f64,
    /// Cumulative thermal mass energy change [J]
    pub cumulative_mass_energy_change: f64,
    /// Peak heating demand [W]
    pub peak_heating_w: f64,
    /// Peak cooling demand [W]
    pub peak_cooling_w: f64,
}

impl EnergyFlowDiagnostics {
    /// Calculate energy balance for verification.
    ///
    /// Returns true if energy in ≈ energy out within 1% tolerance.
    pub fn verify_energy_balance(&self, tolerance: f64) -> bool {
        // Energy IN: phi_st + phi_m + phi_ia
        let energy_in = self.phi_st + self.phi_m + self.phi_ia;

        // Energy OUT: q_em + q_ms + q_w + q_ve + hvac_output
        let energy_out = self.q_em.abs()
            + self.q_ms.abs()
            + self.q_w.abs()
            + self.q_ve.abs()
            + self.hvac_output.abs();

        // Check if mass energy change accounts for imbalance
        let balance = energy_in - energy_out - self.mass_energy_change;

        balance.abs() < tolerance * energy_in.max(energy_out.abs())
    }

    /// Calculate thermal mass charging/discharging rate.
    ///
    /// Returns: "charging", "discharging", or "neutral"
    pub fn mass_state(&self) -> &'static str {
        if self.mass_energy_change > 0.1 {
            "charging"
        } else if self.mass_energy_change < -0.1 {
            "discharging"
        } else {
            "neutral"
        }
    }
}

impl ConductanceDiagnostics {
    /// Calculate thermal time constant τ = C_m / h_tr_ms.
    pub fn calculate_time_constant(&mut self) {
        if self.h_tr_ms > 0.0 {
            self.thermal_time_constant = self.thermal_capacitance / self.h_tr_ms;
        } else {
            self.thermal_time_constant = f64::INFINITY;
        }
    }

    /// Get time constant in hours for easier interpretation.
    pub fn time_constant_hours(&self) -> f64 {
        self.thermal_time_constant / 3600.0
    }

    /// Check if conductances are within expected ranges for ASHRAE 140.
    ///
    /// Returns vector of warnings about out-of-range values.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // h_tr_ms should be 10-100 W/K for realistic thermal lag
        if self.h_tr_ms > 500.0 {
            warnings.push(format!(
                "h_tr_ms = {:.2} W/K is VERY HIGH (expected 10-100 W/K). τ = {:.2} min (expected 1-4 hours).",
                self.h_tr_ms,
                self.time_constant_hours() * 60.0
            ));
        } else if self.h_tr_ms < 1.0 {
            warnings.push(format!(
                "h_tr_ms = {:.4} W/K is VERY LOW (expected 10-100 W/K). τ = {:.2} hours (expected 1-4 hours).",
                self.h_tr_ms,
                self.time_constant_hours()
            ));
        }

        // h_tr_em should be similar to opaque conductance or zero
        if self.h_tr_em < 0.0 {
            warnings.push(format!(
                "h_tr_em = {:.2} W/K is NEGATIVE (invalid thermal conductance).",
                self.h_tr_em
            ));
        }

        // h_tr_is should be 50-200 W/K for typical buildings
        if self.h_tr_is > 500.0 {
            warnings.push(format!(
                "h_tr_is = {:.2} W/K is HIGH (expected 50-200 W/K).",
                self.h_tr_is
            ));
        }

        // Check thermal time constant
        let tau_hours = self.time_constant_hours();
        if tau_hours < 0.1 {
            warnings.push(format!(
                "Thermal time constant τ = {:.2} min is TOO FAST (expected 1-4 hours). Mass responds {}x too fast.",
                tau_hours * 60.0,
                60.0 / tau_hours * 60.0
            ));
        } else if tau_hours > 10.0 {
            warnings.push(format!(
                "Thermal time constant τ = {:.2} hours is TOO SLOW (expected 1-4 hours).",
                tau_hours
            ));
        }

        warnings
    }

    /// Print conductance summary for diagnostic output.
    pub fn print_summary(&self) {
        println!("=== Conductance Diagnostics ===");
        println!("h_tr_em (exterior->mass):     {:.2} W/K", self.h_tr_em);
        println!("h_tr_ms (mass->surface):      {:.2} W/K", self.h_tr_ms);
        println!("h_tr_is (surface->interior):  {:.2} W/K", self.h_tr_is);
        println!("h_tr_w  (windows):            {:.2} W/K", self.h_tr_w);
        println!("h_ve    (ventilation):          {:.2} W/K", self.h_ve);
        println!("h_tr_floor (floor):           {:.2} W/K", self.h_tr_floor);
        println!();
        println!(
            "Thermal capacitance (C_m):     {:.2} kJ/K",
            self.thermal_capacitance / 1000.0
        );
        println!("Effective mass area (A_m):      {:.2} m²", self.a_m);
        println!();
        println!(
            "Thermal time constant (τ):      {:.2} s = {:.2} hours",
            self.thermal_time_constant,
            self.time_constant_hours()
        );
        println!();

        // Print warnings
        let warnings = self.validate();
        if !warnings.is_empty() {
            println!("=== WARNINGS ===");
            for warning in &warnings {
                println!("⚠️  {}", warning);
            }
            println!();
        }
    }
}

impl CumulativeEnergyDiagnostics {
    /// Add energy flow from one timestep to cumulative totals.
    pub fn add_timestep(&mut self, flow: &EnergyFlowDiagnostics, dt: f64) {
        let hvac_j = flow.hvac_output * dt;

        if hvac_j > 0.0 {
            self.total_heating_kwh += hvac_j / 3.6e6;
            self.peak_heating_w = self.peak_heating_w.max(flow.hvac_output);
        } else {
            self.total_cooling_kwh += (-hvac_j) / 3.6e6;
            self.peak_cooling_w = self.peak_cooling_w.max(-flow.hvac_output);
        }

        self.cumulative_q_em += flow.q_em * dt;
        self.cumulative_q_ms += flow.q_ms * dt;
        self.cumulative_q_is += flow.q_is * dt;
        self.cumulative_q_w += flow.q_w * dt;
        self.cumulative_q_ve += flow.q_ve * dt;
        self.cumulative_q_floor += flow.q_floor * dt;
        self.cumulative_mass_energy_change += flow.mass_energy_change;
    }

    /// Print cumulative energy summary.
    pub fn print_summary(&self) {
        println!("=== Cumulative Energy Diagnostics ===");
        println!(
            "Total heating:              {:.2} kWh",
            self.total_heating_kwh
        );
        println!(
            "Total cooling:              {:.2} kWh",
            self.total_cooling_kwh
        );
        println!(
            "Total HVAC:                 {:.2} kWh",
            self.total_heating_kwh + self.total_cooling_kwh
        );
        println!();
        println!("Peak heating demand:         {:.2} W", self.peak_heating_w);
        println!("Peak cooling demand:         {:.2} W", self.peak_cooling_w);
        println!();
        println!("Heat flow breakdown (cumulative):");
        println!(
            "  Through exterior->mass:    {:.2} GJ",
            self.cumulative_q_em / 1e9
        );
        println!(
            "  Through mass->surface:     {:.2} GJ",
            self.cumulative_q_ms / 1e9
        );
        println!(
            "  Through surface->interior: {:.2} GJ",
            self.cumulative_q_is / 1e9
        );
        println!(
            "  Through windows:           {:.2} GJ",
            self.cumulative_q_w / 1e9
        );
        println!(
            "  Through ventilation:        {:.2} GJ",
            self.cumulative_q_ve / 1e9
        );
        println!(
            "  Through floor:            {:.2} GJ",
            self.cumulative_q_floor / 1e9
        );
        println!();
        println!(
            "Cumulative mass energy change: {:.2} GJ",
            self.cumulative_mass_energy_change / 1e9
        );
        println!();
    }
}

/// Comprehensive diagnostics for thermal network physics.
///
/// This struct collects all diagnostic information for Phase 3 analysis.
#[derive(Debug, Clone, Default)]
pub struct ThermalNetworkDiagnostics {
    /// Case identifier (e.g., "Case 600", "Case 900")
    pub case_id: String,
    /// Conductance values
    pub conductances: ConductanceDiagnostics,
    /// Current timestep energy flow
    pub current_flow: EnergyFlowDiagnostics,
    /// Cumulative energy over simulation
    pub cumulative: CumulativeEnergyDiagnostics,
    /// Hourly energy flow tracking for detailed analysis
    pub hourly_flows: Vec<EnergyFlowDiagnostics>,
    /// Maximum mass temperature observed [°C]
    pub max_mass_temp: f64,
    /// Minimum mass temperature observed [°C]
    pub min_mass_temp: f64,
    /// Maximum zone temperature observed [°C]
    pub max_zone_temp: f64,
    /// Minimum zone temperature observed [°C]
    pub min_zone_temp: f64,
}

impl ThermalNetworkDiagnostics {
    /// Create new diagnostics for a specific ASHRAE 140 case.
    pub fn new(case_id: &str) -> Self {
        Self {
            case_id: case_id.to_string(),
            hourly_flows: Vec::with_capacity(8760),
            max_mass_temp: f64::NEG_INFINITY,
            min_mass_temp: f64::INFINITY,
            max_zone_temp: f64::NEG_INFINITY,
            min_zone_temp: f64::INFINITY,
            ..Default::default()
        }
    }

    /// Update temperature extremes.
    pub fn update_temperatures(&mut self, mass_temp: f64, zone_temp: f64) {
        self.max_mass_temp = self.max_mass_temp.max(mass_temp);
        self.min_mass_temp = self.min_mass_temp.min(mass_temp);
        self.max_zone_temp = self.max_zone_temp.max(zone_temp);
        self.min_zone_temp = self.min_zone_temp.min(zone_temp);
    }

    /// Add hourly energy flow data.
    pub fn add_hourly_flow(&mut self, flow: EnergyFlowDiagnostics) {
        self.hourly_flows.push(flow);
    }

    /// Print comprehensive diagnostic report.
    pub fn print_report(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════════╗");
        println!(
            "║  Phase 3: Thermal Network Physics Analysis - {}          ║",
            self.case_id
        );
        println!("╚════════════════════════════════════════════════════════════════════╝\n");

        // Conductance summary
        self.conductances.print_summary();

        // Energy flow summary for first timestep
        println!("=== First Hour Energy Flow ===");
        if let Some(first_hour) = self.hourly_flows.first() {
            println!("Q_em (exterior->mass):        {:+.2} W", first_hour.q_em);
            println!("Q_ms (mass->surface):         {:+.2} W", first_hour.q_ms);
            println!("Q_is (surface->interior):      {:+.2} W", first_hour.q_is);
            println!("Q_w  (windows):               {:+.2} W", first_hour.q_w);
            println!("Q_ve (ventilation):            {:+.2} W", first_hour.q_ve);
            println!(
                "Q_floor (floor):               {:+.2} W",
                first_hour.q_floor
            );
            println!("Phi_st (radiative to surface): {:+.2} W", first_hour.phi_st);
            println!("Phi_m  (radiative to mass):    {:+.2} W", first_hour.phi_m);
            println!("Phi_ia (convective to air):    {:+.2} W", first_hour.phi_ia);
            println!(
                "HVAC output:                  {:+.2} W",
                first_hour.hvac_output
            );
            println!(
                "Mass ΔE (Cm×ΔT):           {:+.2} J",
                first_hour.mass_energy_change
            );
            println!("Mass state:                    {}", first_hour.mass_state());
            println!();
        }

        // Temperature extremes
        println!("=== Temperature Extremes ===");
        println!(
            "Mass temp:  min {:.2}°C, max {:.2}°C",
            self.min_mass_temp, self.max_mass_temp
        );
        println!(
            "Zone temp:  min {:.2}°C, max {:.2}°C",
            self.min_zone_temp, self.max_zone_temp
        );
        println!();

        // Cumulative energy
        self.cumulative.print_summary();

        // Energy balance check
        println!("=== Energy Balance Verification ===");
        if let Some(last_hour) = self.hourly_flows.last() {
            let balanced = last_hour.verify_energy_balance(0.01);
            println!(
                "Last timestep energy balance: {}",
                if balanced { "✅ PASS" } else { "❌ FAIL" }
            );
        }
    }

    /// Export diagnostics to CSV for analysis.
    ///
    /// Returns CSV formatted string with hourly data.
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("hour,q_em_W,q_ms_W,q_is_W,q_w_W,q_ve_W,q_floor_W,phi_st_W,phi_m_W,phi_ia_W,hvac_W,mass_delta_J\n");

        // Data rows
        for (i, flow) in self.hourly_flows.iter().enumerate() {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                i,
                flow.q_em,
                flow.q_ms,
                flow.q_is,
                flow.q_w,
                flow.q_ve,
                flow.q_floor,
                flow.phi_st,
                flow.phi_m,
                flow.phi_ia,
                flow.hvac_output,
                flow.mass_energy_change
            ));
        }

        csv
    }

    /// Calculate statistics for key thermal network parameters.
    pub fn calculate_statistics(&self) -> HashMap<&'static str, f64> {
        let mut stats = HashMap::new();

        if self.hourly_flows.is_empty() {
            return stats;
        }

        // Average heat flows
        let q_em_avg: f64 =
            self.hourly_flows.iter().map(|f| f.q_em).sum::<f64>() / self.hourly_flows.len() as f64;
        let q_ms_avg: f64 =
            self.hourly_flows.iter().map(|f| f.q_ms).sum::<f64>() / self.hourly_flows.len() as f64;
        let hvac_avg: f64 = self
            .hourly_flows
            .iter()
            .map(|f| f.hvac_output.abs())
            .sum::<f64>()
            / self.hourly_flows.len() as f64;

        stats.insert("avg_q_em_W", q_em_avg);
        stats.insert("avg_q_ms_W", q_ms_avg);
        stats.insert("avg_hvac_W", hvac_avg);
        stats.insert(
            "thermal_time_constant_s",
            self.conductances.thermal_time_constant,
        );
        stats.insert(
            "thermal_time_constant_h",
            self.conductances.time_constant_hours(),
        );

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conductance_validation() {
        let mut cond = ConductanceDiagnostics {
            h_tr_ms: 1456.0, // Too high (Case 600 current value)
            thermal_capacitance: 100_000.0,
            ..Default::default()
        };
        cond.calculate_time_constant();

        let warnings = cond.validate();
        assert!(!warnings.is_empty(), "Should warn about high h_tr_ms");
        assert!(warnings[0].contains("VERY HIGH"));
    }

    #[test]
    fn test_energy_balance_verification() {
        let flow = EnergyFlowDiagnostics {
            phi_st: 100.0,
            phi_m: 50.0,
            phi_ia: 25.0,
            q_em: 30.0,
            q_ms: 20.0,
            q_w: 10.0,
            q_ve: 5.0,
            hvac_output: 0.0,          // Free-floating
            mass_energy_change: 110.0, // Energy stored in mass
            ..Default::default()
        };

        // Energy in: 100 + 50 + 25 = 175
        // Energy out: 30 + 20 + 10 + 5 = 65
        // Balance: 175 - 65 - 110 = 0 (balanced)
        assert!(flow.verify_energy_balance(0.01));
    }

    #[test]
    fn test_time_constant_calculation() {
        let mut cond = ConductanceDiagnostics {
            h_tr_ms: 14.0, // Corrected value
            thermal_capacitance: 100_000.0,
            ..Default::default()
        };
        cond.calculate_time_constant();

        // τ = 100000 / 14 = 7142s ≈ 2 hours
        assert!((cond.thermal_time_constant - 7142.0).abs() < 100.0);
        assert!((cond.time_constant_hours() - 2.0).abs() < 0.1);
    }
}
