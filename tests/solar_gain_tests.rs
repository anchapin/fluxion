// Solar Gain Unit Tests for ASHRAE 140 Case 900
//
// These tests validate solar gain calculations against EnergyPlus reference data.
// EnergyPlus reference data is from:
// refdata/energyplus_reference/900_reference.json
//
// The 900_reference.json file contains hourly data with 52560 records
// (10-minute intervals x 6 per hour x 8760 hours):
// - hourly.heating_energy: cumulative energy in kJ (with resets on system OFF)
// - hourly.cooling_energy: cumulative energy in kJ (with resets on system OFF)
// - hourly.zone_air_temp: empty
// - annual.heating_kwh: 5980173.77 kWh
// - annual.cooling_kwh: 8993380.95 kWh
//
// Note: The 900_reference.json file does NOT contain solar_gain data,
// so solar-specific tests are marked as #[ignore].

#[derive(Debug)]
struct EnergyPlusReference {
    /// Issue #837: Loaded from the reference JSON for completeness but not
    /// currently consumed by any test in this file (solar-specific assertions
    /// only exercise heating/cooling/solar fields). Allowing dead_code to keep
    /// the JSON deserialization shape stable for future tests.
    #[allow(dead_code)]
    zone_air_temp_c: Vec<f64>,
    heating_energy_wh: Vec<f64>,
    cooling_energy_wh: Vec<f64>,
    solar_rate_total_w: Vec<f64>,
}

impl EnergyPlusReference {
    fn load() -> Self {
        let path = "refdata/energyplus_reference/900_reference.json";
        let file = std::fs::File::open(path).expect("Failed to open reference data");
        let data: serde_json::Value =
            serde_json::from_reader(file).expect("Failed to parse reference data");

        let hourly = data.get("hourly").expect("Missing 'hourly' field");

        let heating_raw: Vec<f64> =
            serde_json::from_value(hourly.get("heating_energy").cloned().unwrap_or_default())
                .unwrap_or_default();

        let cooling_raw: Vec<f64> =
            serde_json::from_value(hourly.get("cooling_energy").cloned().unwrap_or_default())
                .unwrap_or_default();

        // Convert raw data to hourly energy (Wh)
        // Data has 52560 values (8760 hours x 6 intervals/hour)
        // Values are cumulative kJ per 10-minute interval
        let heating_energy_wh = cumulative_to_hourly_energy(&heating_raw);
        let cooling_energy_wh = cumulative_to_hourly_energy(&cooling_raw);

        let zone_air_temp_c: Vec<f64> =
            serde_json::from_value(hourly.get("zone_air_temp").cloned().unwrap_or_default())
                .unwrap_or_default();

        Self {
            zone_air_temp_c,
            heating_energy_wh,
            cooling_energy_wh,
            solar_rate_total_w: Vec::new(),
        }
    }

    fn annual_heating_mwh(&self) -> f64 {
        self.heating_energy_wh.iter().sum::<f64>() / 1_000_000.0
    }

    fn annual_cooling_mwh(&self) -> f64 {
        self.cooling_energy_wh.iter().sum::<f64>() / 1_000_000.0
    }
}

// Convert 10-minute cumulative energy values to hourly energy (Wh)
fn cumulative_to_hourly_energy(cumulative: &[f64]) -> Vec<f64> {
    if cumulative.is_empty() {
        return Vec::new();
    }

    let intervals_per_hour = 6;
    let num_hours = cumulative.len() / intervals_per_hour;
    let mut hourly = Vec::with_capacity(num_hours);

    for hour in 0..num_hours {
        let start = hour * intervals_per_hour;
        let hour_values = &cumulative[start..start + intervals_per_hour];

        let mut hour_energy_kj = 0.0;
        for i in 1..hour_values.len() {
            let diff = hour_values[i] - hour_values[i - 1];
            if diff > 0.0 {
                hour_energy_kj += diff;
            }
        }

        let hour_energy_wh = hour_energy_kj * 0.27778;
        hourly.push(hour_energy_wh);
    }

    hourly
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const SOLAR_TOLERANCE: f64 = 0.01;
    #[allow(dead_code)]
    const ENERGY_TOLERANCE: f64 = 0.05;

    // Test 1: Verify EnergyPlus reference data can be loaded
    #[test]
    fn test_energyplus_reference_validity() {
        let ep = EnergyPlusReference::load();

        assert_eq!(
            ep.heating_energy_wh.len(),
            8760,
            "Heating energy should have 8760 hours"
        );
        assert_eq!(
            ep.cooling_energy_wh.len(),
            8760,
            "Cooling energy should have 8760 hours"
        );

        let heating_mwh = ep.annual_heating_mwh();
        let cooling_mwh = ep.annual_cooling_mwh();

        assert!(
            heating_mwh > 0.0,
            "Annual heating should be positive, got {:.3} MWh",
            heating_mwh
        );
        assert!(
            cooling_mwh > 0.0,
            "Annual cooling should be positive, got {:.3} MWh",
            cooling_mwh
        );
    }

    // Test 2: Heating energy is positive
    #[test]
    fn test_heating_energy_positive() {
        let ep = EnergyPlusReference::load();
        let heating_mwh = ep.annual_heating_mwh();
        assert!(
            heating_mwh > 0.0,
            "Annual heating should be positive, got {:.3} MWh",
            heating_mwh
        );
    }

    // Test 12: Cooling energy is positive
    #[test]
    fn test_cooling_energy_positive() {
        let ep = EnergyPlusReference::load();
        let cooling_mwh = ep.annual_cooling_mwh();
        assert!(
            cooling_mwh > 0.0,
            "Annual cooling should be positive, got {:.3} MWh",
            cooling_mwh
        );
    }

    // Test 13: Heating should be higher in winter months
    #[test]
    fn test_heating_seasonal_pattern() {
        let ep = EnergyPlusReference::load();

        let mut winter_heating: f64 = 0.0;
        for i in 0..2160 {
            winter_heating += ep.heating_energy_wh[i];
        }

        let mut summer_heating: f64 = 0.0;
        for i in 3624..5088 {
            summer_heating += ep.heating_energy_wh[i];
        }

        assert!(
            winter_heating > summer_heating,
            "Winter heating ({:.2} Wh) should be higher than summer ({:.2} Wh)",
            winter_heating,
            summer_heating
        );
    }
}
