use crate::validation::diagnostic::TemperatureProfile;
use csv::Writer;
use serde::Serialize;
use std::path::Path;

/// Swing metrics for a free-floating case.
#[derive(Debug, Clone, Serialize)]
pub struct SwingMetrics {
    pub case_id: String,
    pub min_temp: f64,
    pub max_temp: f64,
    pub avg_temp: f64,
    pub swing_range: f64,
    pub comfort_hours: usize,
    pub comfort_hours_pct: f64,
}

/// Calculate swing metrics from a temperature profile.
///
/// # Arguments
/// - `profile`: Temperature profile (free-floating case)
/// - `comfort_band_min`: Lower comfort bound (default 18°C)
/// - `comfort_band_max`: Upper comfort bound (default 26°C)
pub fn calculate_swing_metrics(
    profile: &TemperatureProfile,
    comfort_band_min: f64,
    comfort_band_max: f64,
) -> SwingMetrics {
    let total_hours = profile.hourly_temps.len();
    let comfort_hours = profile
        .hourly_temps
        .iter()
        .filter(|&&t| t >= comfort_band_min && t <= comfort_band_max)
        .count();
    let comfort_hours_pct = if total_hours > 0 {
        (comfort_hours as f64 / total_hours as f64) * 100.0
    } else {
        0.0
    };
    SwingMetrics {
        case_id: profile.case_id.clone(),
        min_temp: profile.min_temp,
        max_temp: profile.max_temp,
        avg_temp: profile.avg_temp,
        swing_range: profile.swing,
        comfort_hours,
        comfort_hours_pct,
    }
}

/// Export swing metrics to CSV.
pub fn export_swing_csv(
    metrics: &[SwingMetrics],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = Writer::from_path(path)?;
    wtr.write_record(&[
        "Case",
        "Min_Temp",
        "Max_Temp",
        "Avg_Temp",
        "Swing_Range",
        "Comfort_Hours",
        "Comfort_Percent",
    ])?;
    for m in metrics {
        wtr.write_record(&[
            &m.case_id,
            &format!("{:.2}", m.min_temp),
            &format!("{:.2}", m.max_temp),
            &format!("{:.2}", m.avg_temp),
            &format!("{:.2}", m.swing_range),
            &m.comfort_hours.to_string(),
            &format!("{:.1}", m.comfort_hours_pct),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

/// Interpretation of swing metrics with diagnostic insights.
#[derive(Debug, Clone, Serialize)]
pub struct SwingInterpretation {
    pub case_id: String,
    pub thermal_mass_effectiveness: String,
    pub passive_cooling_potential: String,
    pub passive_heating_potential: String,
    pub recommendations: Vec<String>,
}

/// Generate interpretation from swing metrics.
pub fn interpret_swing_metrics(metrics: &SwingMetrics) -> SwingInterpretation {
    // Thermal mass effectiveness based on swing range and comfort percentage
    let (thermal_mass_effectiveness, mut recs) = if metrics.swing_range < 5.0
        && metrics.comfort_hours_pct >= 80.0
    {
        ("High".to_string(), vec![])
    } else if metrics.swing_range < 10.0 && metrics.comfort_hours_pct >= 50.0 {
        ("Moderate".to_string(), vec![])
    } else {
        ("Low".to_string(), vec![
            "Consider increasing thermal mass or improving insulation to reduce temperature swings".to_string()
        ])
    };

    // Passive cooling potential
    let passive_cooling_potential = if metrics.comfort_hours_pct >= 70.0
        && metrics.avg_temp >= 18.0
        && metrics.avg_temp <= 26.0
    {
        "High"
    } else if metrics.comfort_hours_pct >= 40.0 && metrics.avg_temp <= 28.0 {
        "Moderate"
    } else {
        "Low"
    };

    // Passive heating potential
    let passive_heating_potential = if metrics.comfort_hours_pct >= 70.0 && metrics.avg_temp >= 18.0
    {
        "High"
    } else if metrics.comfort_hours_pct >= 40.0 && metrics.avg_temp >= 15.0 {
        "Moderate"
    } else {
        "Low"
    };

    // Additional recommendations based on potentials
    if passive_cooling_potential == "Low" && metrics.avg_temp > 28.0 {
        recs.push("Improve shading, increase ventilation, or reduce window-to-wall ratio to reduce cooling load".to_string());
    }
    if passive_heating_potential == "Low" && metrics.avg_temp < 18.0 {
        recs.push("Increase solar gain (south-facing windows), add thermal mass, or improve insulation for better heating".to_string());
    }

    SwingInterpretation {
        case_id: metrics.case_id.clone(),
        thermal_mass_effectiveness,
        passive_cooling_potential: passive_cooling_potential.to_string(),
        passive_heating_potential: passive_heating_potential.to_string(),
        recommendations: recs,
    }
}

/// Generate a Markdown report from interpretations.
pub fn generate_swing_report(interpretations: &[SwingInterpretation]) -> String {
    let mut out = String::new();
    out.push_str("# Swing Analysis Report\n\n");
    out.push_str("| Case | Thermal Mass Effectiveness | Passive Cooling Potential | Passive Heating Potential | Recommendations |\n");
    out.push_str("|------|---------------------------|---------------------------|---------------------------|-----------------|\n");
    for interp in interpretations {
        let recs = interp.recommendations.join("; ");
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            interp.case_id,
            interp.thermal_mass_effectiveness,
            interp.passive_cooling_potential,
            interp.passive_heating_potential,
            recs
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swing_metrics_basic() {
        let mut profile = TemperatureProfile::new("600FF");
        profile.update(15.0);
        profile.update(25.0);
        profile.update(18.0);
        profile.finalize();
        // Should be min=15, max=25, avg=(15+25+18)/3=19.33, swing=10
        let metrics = calculate_swing_metrics(&profile, 18.0, 26.0);
        assert_eq!(metrics.min_temp, 15.0);
        assert_eq!(metrics.max_temp, 25.0);
        assert!((metrics.avg_temp - 19.333).abs() < 0.01);
        assert_eq!(metrics.swing_range, 10.0);
        // Comfort hours: only 18 (within 18-26) maybe also? temps: 15 -> not, 25-> yes (between 18-26), 18-> yes. That's 2/3 = 66.67%
        assert_eq!(metrics.comfort_hours, 2);
        assert!((metrics.comfort_hours_pct - (2.0 / 3.0) * 100.0).abs() < 0.1);
    }

    #[test]
    fn test_interpretation_high() {
        let metrics = SwingMetrics {
            case_id: "600FF".to_string(),
            min_temp: 20.0,
            max_temp: 24.0,
            avg_temp: 22.0,
            swing_range: 4.0,
            comfort_hours: 8000,
            comfort_hours_pct: 91.0,
        };
        let interp = interpret_swing_metrics(&metrics);
        assert_eq!(interp.thermal_mass_effectiveness, "High");
        assert_eq!(interp.passive_cooling_potential, "High");
        assert_eq!(interp.passive_heating_potential, "High");
    }

    #[test]
    fn test_interpretation_low() {
        let metrics = SwingMetrics {
            case_id: "600FF".to_string(),
            min_temp: 10.0,
            max_temp: 35.0,
            avg_temp: 22.0,
            swing_range: 25.0,
            comfort_hours: 1000,
            comfort_hours_pct: 11.4,
        };
        let interp = interpret_swing_metrics(&metrics);
        assert_eq!(interp.thermal_mass_effectiveness, "Low");
        assert_eq!(interp.passive_cooling_potential, "Low");
        assert_eq!(interp.passive_heating_potential, "Low");
        assert!(interp.recommendations.len() >= 1);
    }

    #[test]
    fn test_generate_swing_report() {
        let interp = SwingInterpretation {
            case_id: "600FF".to_string(),
            thermal_mass_effectiveness: "High".to_string(),
            passive_cooling_potential: "Moderate".to_string(),
            passive_heating_potential: "High".to_string(),
            recommendations: vec![],
        };
        let report = generate_swing_report(&[interp]);
        assert!(report.contains("Swing Analysis Report"));
        assert!(report.contains("600FF"));
    }
}
