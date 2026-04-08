// validation/reporting/tests.rs
#[cfg(test)]
mod tests {
    use super::super::*;
    use std::path::PathBuf;

    #[test]
    fn test_reporting_config() {
        let config = ReportingConfig::default();
        assert_eq!(config.output_dir, "validation/reports");
        assert!(matches!(config.format, ReportFormat::Markdown));
        assert!(config.include_diagnostics);
        assert!(config.comprehensive);
    }

    #[test]
    fn test_report_generator_creation() {
        let generator = ComprehensiveReportGenerator::new();
        // Just test that it can be created without panicking
        assert!(true, "ComprehensiveReportGenerator should be creatable");
    }

    #[test]
    fn test_report_generation() {
        let generator = ComprehensiveReportGenerator::new();
        let result = generator.generate_report();
        assert!(result.is_ok(), "Report generation should succeed");

        let report = result.unwrap();
        assert_eq!(report.metadata.fluxion_version, env!("CARGO_PKG_VERSION"));
        assert!(report
            .metadata
            .validation_coverage
            .contains("Comprehensive"));
    }

    #[test]
    fn test_json_export() {
        use tempfile::NamedTempFile;

        let generator = ComprehensiveReportGenerator::new();
        let report = generator.generate_report().unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        let result = generator.export_json(&report, path);
        assert!(result.is_ok(), "JSON export should succeed");

        // Verify the file was created and contains JSON
        let content = std::fs::read_to_string(path).unwrap();
        assert!(
            content.contains("Fluxion"),
            "JSON should contain Fluxion version"
        );
        assert!(
            content.contains("Comprehensive"),
            "JSON should contain coverage info"
        );
    }

    #[test]
    fn test_markdown_generation() {
        let generator = ComprehensiveReportGenerator::new();
        let report = generator.generate_report().unwrap();

        let result = generator.generate_markdown(&report);
        assert!(result.is_ok(), "Markdown generation should succeed");

        let markdown = result.unwrap();
        assert!(markdown.contains("# Comprehensive Validation Report"));
        assert!(markdown.contains("Generated:"));
    }

    #[test]
    fn test_html_generation() {
        let generator = ComprehensiveReportGenerator::new();
        let report = generator.generate_report().unwrap();

        let result = generator.generate_html(&report);
        assert!(result.is_ok(), "HTML generation should succeed");

        let html = result.unwrap();
        assert!(html.contains("<html>"));
        assert!(html.contains("<h1>"));
    }
}
