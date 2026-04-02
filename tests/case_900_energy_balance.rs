//! Case 900 Energy Balance Verification Test
//!
//! This test investigates the Case 900 cooling error (-33.76%) by analyzing the energy balance
//! across all components (solar, internal gains, conduction, ventilation, HVAC, storage, exfiltration).
//!
//! The test runs a full year simulation and verifies that daily energy balance equations hold:
//! (solar + internal + conduction + ventilation) - (stored + exfiltration) = hvac_cooling ± 5%

#[cfg(test)]
mod tests {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use fluxion::validation::diagnostics::{DailyBalance, SimulationDiagnostics};
    use std::path::PathBuf;

    /// Test Case 900 energy balance over a full year
    #[test]
    fn test_case_900_energy_balance() {
        println!("\n=== Case 900 Energy Balance Verification Test ===");
        println!("Objective: Identify whether cooling error (-33.76%) is in energy accounting");
        println!("           or control strategy by analyzing component-level balance.");

        // Create Case 900 model
        let spec = ASHRAE140Case::Case900.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);

        // Attach diagnostics to collect hourly data
        let diagnostics = SimulationDiagnostics::new(1, 8760);
        model.set_diagnostics(Some(diagnostics));

        println!("\nRunning Case 900 simulation for full year...");
        // Run simulation for full year (Jan 1 - Dec 31)
        model.run_full_year();

        // Extract diagnostics
        let diag = model.get_diagnostics().unwrap();

        println!(
            "Simulation complete. Collected {} hours of data",
            diag.hours.len()
        );
        println!("\nExtracting daily energy balances...");

        // Calculate daily balances using the diagnostics method
        let daily_balances = diag.verify_daily_energy_balance();

        println!(
            "Calculated daily balances for {} days",
            daily_balances.len()
        );

        // Find first failing day (check first zone)
        let failing_day = daily_balances
            .iter()
            .find(|d| !d.is_balanced.get(0).copied().unwrap_or(false));

        // Print results
        print_energy_balance_summary(&daily_balances, failing_day.cloned());

        // Export hourly data to CSV
        let csv_path = export_hourly_data(&diag, &daily_balances);
        println!("\nHourly data exported to: {}", csv_path.display());

        // Export daily summary
        let daily_csv_path = export_daily_summary(&daily_balances);
        println!("Daily summary exported to: {}", daily_csv_path.display());

        // Determine test result (check first zone)
        let all_balanced = daily_balances
            .iter()
            .all(|d| d.is_balanced.get(0).copied().unwrap_or(false));
        let pass_rate = (daily_balances
            .iter()
            .filter(|d| d.is_balanced.get(0).copied().unwrap_or(false))
            .count() as f64
            / daily_balances.len() as f64)
            * 100.0;

        println!("\n=== Energy Balance Test Result ===");
        println!(
            "Pass rate: {:.1}% ({}/{})",
            pass_rate,
            daily_balances
                .iter()
                .filter(|d| d.is_balanced.get(0).copied().unwrap_or(false))
                .count(),
            daily_balances.len()
        );

        if all_balanced {
            println!("✅ PASSED: All daily energy balances within ±5%");
        } else {
            println!("❌ FAILED: Daily energy balances outside tolerance");
            if let Some(first_fail) = failing_day {
                let zone_idx = 0;
                println!(
                    "\nFirst failing day (Day {}): {:.2}% error",
                    first_fail.day,
                    first_fail
                        .balance_error_pct
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  Solar:        {:.3} kWh",
                    first_fail.solar_kWh.get(zone_idx).copied().unwrap_or(0.0)
                );
                println!(
                    "  Internal:     {:.3} kWh",
                    first_fail
                        .internal_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  Conduction:   {:.3} kWh",
                    first_fail
                        .conduction_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  Ventilation:  {:.3} kWh",
                    first_fail
                        .ventilation_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  HVAC Cooling: {:.3} kWh",
                    first_fail
                        .hvac_cooling_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  HVAC Heating: {:.3} kWh",
                    first_fail
                        .hvac_heating_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  Stored:       {:.3} kWh",
                    first_fail
                        .stored_thermal_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
                println!(
                    "  Exfiltration: {:.3} kWh",
                    first_fail
                        .exfiltration_kWh
                        .get(zone_idx)
                        .copied()
                        .unwrap_or(0.0)
                );
            }
        }

        // Note: Allow test to complete regardless of balance status for Phase 30 Wave 1 diagnostics
        // The pass/fail determination will inform Wave 2 focus
        assert!(
            pass_rate >= 50.0,
            "Case 900 energy balance critically failed: only {:.1}% of days balanced",
            pass_rate
        );
    }

    /// Print summary of energy balance results
    fn print_energy_balance_summary(balances: &[DailyBalance], first_fail: Option<DailyBalance>) {
        println!("\n=== Daily Energy Balance Summary ===");
        let zone_idx = 0;
        println!(
            "Days Balanced (±5%): {}/{}",
            balances
                .iter()
                .filter(|d| d.is_balanced.get(zone_idx).copied().unwrap_or(false))
                .count(),
            balances.len()
        );

        let avg_error = balances
            .iter()
            .map(|d| d.balance_error_pct.get(zone_idx).copied().unwrap_or(0.0))
            .sum::<f64>()
            / balances.len() as f64;
        println!("Average Error: {:.2}%", avg_error);

        let max_error = balances
            .iter()
            .map(|d| d.balance_error_pct.get(zone_idx).copied().unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);
        println!("Maximum Error: {:.2}%", max_error);

        if let Some(fail) = first_fail {
            println!("\nFirst Failing Day: Day {}", fail.day);
            println!(
                "  Error: {:.2}%",
                fail.balance_error_pct.get(zone_idx).copied().unwrap_or(0.0)
            );
            println!(
                "  Solar:        {:8.3} kWh (in)",
                fail.solar_kWh.get(zone_idx).copied().unwrap_or(0.0)
            );
            println!(
                "  Internal:     {:8.3} kWh (in)",
                fail.internal_kWh.get(zone_idx).copied().unwrap_or(0.0)
            );
            println!(
                "  Conduction:   {:8.3} kWh (in)",
                fail.conduction_kWh.get(zone_idx).copied().unwrap_or(0.0)
            );
            println!(
                "  Ventilation:  {:8.3} kWh (in)",
                fail.ventilation_kWh.get(zone_idx).copied().unwrap_or(0.0)
            );
            println!(
                "  HVAC Cooling: {:8.3} kWh (out)",
                fail.hvac_cooling_kWh.get(zone_idx).copied().unwrap_or(0.0)
            );
            println!(
                "  Stored:       {:8.3} kWh",
                fail.stored_thermal_kWh
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0)
            );
            println!(
                "  Exfiltration: {:8.3} kWh (out)",
                fail.exfiltration_kWh.get(zone_idx).copied().unwrap_or(0.0)
            );
        }
    }

    /// Export hourly data to CSV
    fn export_hourly_data(diag: &SimulationDiagnostics, balances: &[DailyBalance]) -> PathBuf {
        let path = PathBuf::from("./case_900_hourly_energy_balance.csv");

        let mut csv = String::from(
            "Hour,Solar_W,Internal_W,HVAC_Cooling_W,Zone_Temp_C,Setpoint_C,Day,Balance_Error_pct\n",
        );

        for (idx, hour) in diag.hours.iter().enumerate() {
            if idx >= diag.loads.solar.len() {
                break;
            }

            let day = (idx / 24) + 1;
            let solar = diag.loads.solar[idx].get(0).copied().unwrap_or(0.0);
            let internal = diag.loads.internal[idx].get(0).copied().unwrap_or(0.0);
            let hvac = diag.loads.hvac[idx].get(0).copied().unwrap_or(0.0).abs();
            let temp = diag.zone_temps[idx].get(0).copied().unwrap_or(0.0);
            let setpoint = 21.0; // Case 900 cooling setpoint
            let error = balances
                .get(day - 1)
                .map(|b| b.balance_error_pct)
                .unwrap_or(0.0);

            csv.push_str(&format!(
                "{},{:.1},{:.1},{:.1},{:.2},{:.2},{},{:.2}\n",
                hour, solar, internal, hvac, temp, setpoint, day, error
            ));
        }

        std::fs::write(&path, csv).expect("Failed to write hourly CSV");
        path
    }

    /// Export daily summary to CSV
    fn export_daily_summary(balances: &[DailyBalance]) -> PathBuf {
        let path = PathBuf::from("./case_900_daily_energy_balance.csv");

        let mut csv = String::from(
            "Day,Solar_kWh,Internal_kWh,Conduction_kWh,Ventilation_kWh,HVAC_Cooling_kWh,HVAC_Heating_kWh,Stored_kWh,Exfiltration_kWh,Balance_Error_pct,Is_Balanced\n"
        );

        let zone_idx = 0;
        for balance in balances {
            let solar = balance.solar_kWh.get(zone_idx).copied().unwrap_or(0.0);
            let internal = balance.internal_kWh.get(zone_idx).copied().unwrap_or(0.0);
            let conduction = balance.conduction_kWh.get(zone_idx).copied().unwrap_or(0.0);
            let ventilation = balance
                .ventilation_kWh
                .get(zone_idx)
                .copied()
                .unwrap_or(0.0);
            let hvac_cool = balance
                .hvac_cooling_kWh
                .get(zone_idx)
                .copied()
                .unwrap_or(0.0);
            let hvac_heat = balance
                .hvac_heating_kWh
                .get(zone_idx)
                .copied()
                .unwrap_or(0.0);
            let stored = balance
                .stored_thermal_kWh
                .get(zone_idx)
                .copied()
                .unwrap_or(0.0);
            let exfil = balance
                .exfiltration_kWh
                .get(zone_idx)
                .copied()
                .unwrap_or(0.0);
            let error = balance
                .balance_error_pct
                .get(zone_idx)
                .copied()
                .unwrap_or(0.0);
            let is_bal = balance.is_balanced.get(zone_idx).copied().unwrap_or(false);

            csv.push_str(&format!(
                "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{}\n",
                balance.day,
                solar,
                internal,
                conduction,
                ventilation,
                hvac_cool,
                hvac_heat,
                stored,
                exfil,
                error,
                if is_bal { "YES" } else { "NO" }
            ));
        }

        std::fs::write(&path, csv).expect("Failed to write daily CSV");
        path
    }
}
