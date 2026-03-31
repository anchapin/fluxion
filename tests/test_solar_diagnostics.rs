#[cfg(test)]
mod tests {
    use fluxion::sim::solar::SolarDiagnostic;
    use fluxion::validation::solar_diagnostics::SolarGainDiagnostics;
    use tempfile::NamedTempFile;

    fn make_diagnostic(
        month: u32,
        day: u32,
        hour: u32,
        orientation: &str,
        total_gain_w: f64,
    ) -> SolarDiagnostic {
        SolarDiagnostic {
            month,
            day,
            hour: hour as f64,
            orientation: orientation.to_string(),
            dni: 500.0,
            dhi: 100.0,
            ghi: 600.0,
            beam_irradiance: 400.0,
            diffuse_irradiance: 80.0,
            ground_reflected_irradiance: 20.0,
            total_irradiance: 500.0,
            incidence_angle: 45.0,
            shgc_effective: 0.5,
            beam_gain_w: total_gain_w * 0.6,
            diffuse_gain_w: total_gain_w * 0.3,
            ground_gain_w: total_gain_w * 0.1,
            total_gain_w,
            outdoor_temp: 25.0,
        }
    }

    #[test]
    fn test_solar_diagnostics_new() {
        let diagnostics = SolarGainDiagnostics::new();
        assert_eq!(diagnostics.records.len(), 0);
    }

    #[test]
    fn test_solar_diagnostics_default() {
        let diagnostics = SolarGainDiagnostics::default();
        assert_eq!(diagnostics.records.len(), 0);
    }

    #[test]
    fn test_solar_diagnostics_record() {
        let mut diagnostics = SolarGainDiagnostics::new();
        let diag = make_diagnostic(1, 15, 12, "South", 100.0);
        diagnostics.record(diag);
        assert_eq!(diagnostics.records.len(), 1);
    }

    #[test]
    fn test_solar_diagnostics_multiple_records() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(1, 15, 12, "South", 100.0));
        diagnostics.record(make_diagnostic(6, 15, 12, "East", 200.0));
        diagnostics.record(make_diagnostic(6, 15, 18, "West", 150.0));
        assert_eq!(diagnostics.records.len(), 3);
    }

    #[test]
    fn test_solar_diagnostics_export_csv() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(1, 15, 12, "South", 100.0));
        diagnostics.record(make_diagnostic(6, 15, 12, "East", 200.0));

        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let path = file.path().to_path_buf();

        diagnostics.export_csv(&path).expect("Failed to export CSV");

        let content = std::fs::read_to_string(&path).expect("Failed to read CSV");
        assert!(content.contains("Month,Day,HourOfDay,Orientation"));
        assert!(content.contains("South"));
        assert!(content.contains("East"));
    }

    #[test]
    fn test_solar_diagnostics_csv_header() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(1, 15, 12, "South", 100.0));

        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let path = file.path().to_path_buf();

        diagnostics.export_csv(&path).expect("Failed to export CSV");

        let content = std::fs::read_to_string(&path).expect("Failed to read CSV");
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 2, "Should have header + data rows");
        assert!(lines[0].contains("DNI"));
        assert!(lines[0].contains("SHGC_Effective"));
        assert!(lines[0].contains("OutdoorTemp_C"));
    }

    #[test]
    fn test_solar_diagnostics_csv_data_values() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(6, 15, 12, "East", 500.0));

        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let path = file.path().to_path_buf();

        diagnostics.export_csv(&path).expect("Failed to export CSV");

        let content = std::fs::read_to_string(&path).expect("Failed to read CSV");
        assert!(content.contains("6,"));
        assert!(content.contains("15,"));
        assert!(content.contains("12,East"));
    }

    #[test]
    fn test_solar_diagnostics_print_summary() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(1, 15, 12, "South", 100.0));
        diagnostics.record(make_diagnostic(6, 15, 12, "East", 200.0));
        diagnostics.record(make_diagnostic(6, 15, 18, "West", 150.0));

        // Just verify it doesn't panic
        diagnostics.print_summary();
    }

    #[test]
    fn test_solar_diagnostics_print_summary_empty() {
        let diagnostics = SolarGainDiagnostics::new();
        // Should not panic with empty records
        diagnostics.print_summary();
    }

    #[test]
    fn test_solar_diagnostics_print_summary_cooling_season() {
        let mut diagnostics = SolarGainDiagnostics::new();
        // Add cooling season records (May-Sep)
        for month in 5..=9 {
            diagnostics.record(make_diagnostic(month, 15, 12, "South", 300.0));
        }
        // Should include cooling season analysis
        diagnostics.print_summary();
    }

    #[test]
    fn test_solar_diagnostics_clone() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(1, 15, 12, "South", 100.0));

        let cloned = diagnostics.clone();
        assert_eq!(cloned.records.len(), diagnostics.records.len());
    }

    #[test]
    fn test_solar_diagnostics_debug() {
        let diagnostics = SolarGainDiagnostics::new();
        let debug_str = format!("{:?}", diagnostics);
        assert!(debug_str.contains("SolarGainDiagnostics"));
    }

    #[test]
    fn test_solar_diagnostics_export_to_nonexistent_dir() {
        let mut diagnostics = SolarGainDiagnostics::new();
        diagnostics.record(make_diagnostic(1, 15, 12, "South", 100.0));

        let result = diagnostics.export_csv("/nonexistent/path/file.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_solar_diagnostics_capacity_hint() {
        // The new() method pre-allocates for ~4 orientations * 8760 hours
        let diagnostics = SolarGainDiagnostics::new();
        assert!(diagnostics.records.capacity() >= 8760);
    }
}
