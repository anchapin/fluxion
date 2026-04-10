// validation/automation/runner.rs
use clap::{Arg, Command};
/// Test automation runner for cross-validation
///
/// This module provides a CLI-based test runner for automated cross-validation
/// testing workflows, integrating with ESP-r validation and report generation.
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use tempfile::TempDir;

/// Test runner configuration
#[derive(Debug)]
pub struct TestRunnerConfig {
    /// Test cases directory
    pub test_cases_dir: PathBuf,
    /// Output directory for reports
    pub output_dir: PathBuf,
    /// Temperature tolerance for validation
    pub tolerance: f64,
    /// Verbose output flag
    pub verbose: bool,
    /// Output format (markdown, json)
    pub format: String,
}

impl TestRunnerConfig {
    /// Create a new test runner configuration
    pub fn new(
        test_cases_dir: PathBuf,
        output_dir: PathBuf,
        tolerance: f64,
        verbose: bool,
        format: String,
    ) -> Self {
        Self {
            test_cases_dir,
            output_dir,
            tolerance,
            verbose,
            format,
        }
    }
}

/// Test runner main struct
pub struct TestRunner {
    config: TestRunnerConfig,
    temp_dir: Option<TempDir>,
}

impl TestRunner {
    /// Create a new test runner
    pub fn new(config: TestRunnerConfig) -> Self {
        Self {
            config,
            temp_dir: None,
        }
    }

    /// Initialize test runner with temporary directory
    pub fn initialize(&mut self) -> Result<(), Box<dyn Error>> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.config.output_dir)?;

        // Create temporary directory for test execution
        let temp_dir = TempDir::new()?;
        self.temp_dir = Some(temp_dir);

        if self.config.verbose {
            println!("Test runner initialized");
            println!(
                "Test cases directory: {}",
                self.config.test_cases_dir.display()
            );
            println!("Output directory: {}", self.config.output_dir.display());
            println!("Tolerance: {}°C", self.config.tolerance);
            println!("Format: {}", self.config.format);
        }

        Ok(())
    }

    /// Discover test cases in the test cases directory
    pub fn discover_test_cases(&self) -> Result<Vec<PathBuf>, Box<dyn Error>> {
        let mut test_cases = Vec::new();
        
        if !self.config.test_cases_dir.exists() {
            return Err(format!("Test cases directory not found: {}", self.config.test_cases_dir.display()).into());
        }
        
        for entry in fs::read_dir(&self.config.test_cases_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            // Look for test case directories
            if path.is_dir() {
                let test_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                // Check if this is a valid test case (has reference data)
                let reference_path = path.join("reference.csv");
                if reference_path.exists() {
                    // Validate test data structure
                    if self.validate_test_data(&path).is_ok() {
                        test_cases.push(path);
                        if self.config.verbose {
                            println!("Found test case: {}", test_name);
                        }
                    }
                }
            }
        }
        
        if test_cases.is_empty() {
            return Err("No valid test cases found in directory".into());
        }
        
        Ok(test_cases)
    }

    /// Validate test data structure
    fn validate_test_data(&self, test_case_path: &Path) -> Result<(), Box<dyn Error>> {
        // Check for required files
        let reference_path = test_case_path.join("reference.csv");
        if !reference_path.exists() {
            return Err("Missing reference.csv file".into());
        }
        
        // Check reference file is readable and not empty
        let reference_content = fs::read_to_string(&reference_path)?;
        if reference_content.trim().is_empty() {
            return Err("Reference file is empty".into());
        }
        
        // Optional: Check for configuration file
        let config_path = test_case_path.join("config.json");
        if config_path.exists() {
            let config_content = fs::read_to_string(&config_path)?;
            if config_content.trim().is_empty() {
                return Err("Config file is empty".into());
            }
        }
        
        Ok(())
    }

    /// Run a single test case
    pub fn run_test_case(&self, test_case_path: &Path) -> Result<crate::validation::reports::CrossValidationReport, Box<dyn Error>> {
        let test_name = test_case_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        
        if self.config.verbose {
            println!("\nRunning test case: {}", test_name);
        }
        
        // Set up temporary file handling
        let temp_dir = TempDir::new()?;
        
        // Copy test data to temporary location for isolation
        let temp_test_case = temp_dir.path().join(test_name);
        self.copy_test_data(test_case_path, &temp_test_case)?;
        
        // Create ESP-r validator
        let reference_path = temp_test_case.join("reference.csv");
        let validator = crate::validation::esp_r::EspRValidator::new(reference_path, self.config.tolerance);
        
        // Load Fluxion results (this would be generated by running the simulation)
        // For now, we'll create a dummy validation result for demonstration
        let fluxion_results = crate::validation::ValidationResults::default();
        
        // Run validation
        let report = validator.validate(&fluxion_results)?;
        
        // Clean up temporary files
        temp_dir.close()?;
        
        if self.config.verbose {
            println!("Test case completed: {}", test_name);
            println!(
                "Overall status: {}",
                if report.overall_pass { "PASS" } else { "FAIL" }
            );
        }
        
        Ok(report)
    }
        
        Ok(report)
    }

    /// Copy test data to temporary location
    fn copy_test_data(&self, source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
        if !source.exists() {
            return Err(format!("Source test data not found: {}", source.display()).into());
        }
        
        // Create destination directory
        fs::create_dir_all(destination)?;
        
        // Copy all files from source to destination
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let dest_path = destination.join(entry.file_name());
            
            if entry.file_type()?.is_file() {
                fs::copy(entry.path(), dest_path)?;
            }
        }
        
        Ok(())
    }

        // Create ESP-r validator
        let reference_path = test_case_path.join("reference.csv");
        let validator =
            crate::validation::esp_r::EspRValidator::new(reference_path, self.config.tolerance);

        // Load Fluxion results (this would be generated by running the simulation)
        // For now, we'll create a dummy validation result for demonstration
        let fluxion_results = crate::validation::ValidationResults::default();

        // Run validation
        let report = validator.validate(&fluxion_results)?;

        if self.config.verbose {
            println!("Test case completed: {}", test_name);
            println!(
                "Overall status: {}",
                if report.overall_pass { "PASS" } else { "FAIL" }
            );
        }

        Ok(report)
    }

    /// Run all test cases
    pub fn run_all_tests(
        &self,
    ) -> Result<Vec<crate::validation::reports::CrossValidationReport>, Box<dyn Error>> {
        let test_cases = self.discover_test_cases()?;
        let mut reports = Vec::new();

        for test_case in &test_cases {
            let report = self.run_test_case(test_case)?;
            reports.push(report);
        }

        Ok(reports)
    }

    /// Generate combined report
    pub fn generate_combined_report(
        &self,
        reports: &[crate::validation::reports::CrossValidationReport],
    ) -> Result<String, Box<dyn Error>> {
        match self.config.format.as_str() {
            "markdown" => {
                let mut combined_md = String::new();
                combined_md.push_str(&format!("# Combined Cross-Validation Report\n\n"));
                combined_md.push_str(&format!(
                    "Generated: {}\n\n",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
                ));

                let total_tests = reports.len();
                let passed_tests = reports.iter().filter(|r| r.overall_pass).count();

                combined_md.push_str(&format!(
                    "**Summary:** {} of {} test cases passed\n\n",
                    passed_tests, total_tests
                ));

                for (i, report) in reports.iter().enumerate() {
                    combined_md.push_str(&format!("---\n\n## Test Case {}\n\n", i + 1));
                    combined_md.push_str(
                        &crate::validation::reports::cross_validation::generate_markdown_report(
                            report,
                        ),
                    );
                    combined_md.push('\n');
                }

                Ok(combined_md)
            }
            "json" => {
                use serde_json::json;
                let json_reports: Vec<_> = reports.iter().map(|r| json!(r)).collect();
                Ok(serde_json::to_string_pretty(&json_reports)?)
            }
            _ => Err(format!("Unsupported output format: {}", self.config.format).into()),
        }
    }

    /// Save report to file
    pub fn save_report(&self, report_content: &str, filename: &str) -> Result<(), Box<dyn Error>> {
        let output_path = self.config.output_dir.join(filename);
        fs::write(&output_path, report_content)?;

        if self.config.verbose {
            println!("Report saved to: {}", output_path.display());
        }

        Ok(())
    }

    /// Clean up temporary files
    pub fn cleanup(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(temp_dir) = self.temp_dir.take() {
            if self.config.verbose {
                println!("Cleaning up temporary files");
            }
            temp_dir.close()?;
        }
        Ok(())
    }
}

/// Main entry point for test runner
pub fn run_test_automation() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let matches = Command::new("Fluxion Cross-Validation Test Runner")
        .version("1.0")
        .author("Fluxion Team")
        .about("Automated test runner for cross-validation workflows")
        .arg(
            Arg::new("test-cases")
                .short('t')
                .long("test-cases")
                .value_name("DIRECTORY")
                .help("Directory containing test cases")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("DIRECTORY")
                .help("Output directory for reports")
                .required(true),
        )
        .arg(
            Arg::new("tolerance")
                .short('T')
                .long("tolerance")
                .value_name("DEGREES")
                .help("Temperature tolerance for validation (default: 0.5)")
                .default_value("0.5"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output"),
        )
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .value_name("FORMAT")
                .help("Output format (markdown, json)")
                .default_value("markdown"),
        )
        .get_matches();

    // Parse arguments
    let test_cases_dir = PathBuf::from(matches.get_one::<String>("test-cases").unwrap());
    let output_dir = PathBuf::from(matches.get_one::<String>("output").unwrap());
    let tolerance = matches
        .get_one::<String>("tolerance")
        .unwrap()
        .parse::<f64>()?;
    let verbose = matches.get_flag("verbose");
    let format = matches.get_one::<String>("format").unwrap().to_string();

    // Create configuration
    let config = TestRunnerConfig::new(test_cases_dir, output_dir, tolerance, verbose, format);

    // Create and initialize test runner
    let mut runner = TestRunner::new(config);
    runner.initialize()?;

    // Run all tests
    let reports = runner.run_all_tests()?;

    // Generate combined report
    let combined_report = runner.generate_combined_report(&reports)?;

    // Save report
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let filename = format!(
        "cross_validation_report_{}.{}",
        timestamp, runner.config.format
    );
    runner.save_report(&combined_report, &filename)?;

    // Clean up
    runner.cleanup()?;

    // Determine exit code
    let all_passed = reports.iter().all(|r| r.overall_pass);
    if all_passed {
        println!("✅ All tests passed!");
        Ok(())
    } else {
        println!("❌ Some tests failed!");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_runner_initialization() {
        let config = TestRunnerConfig::new(
            PathBuf::from("tests/fixtures"),
            PathBuf::from("target/test_output"),
            0.5,
            true,
            "markdown".to_string(),
        );

        let mut runner = TestRunner::new(config);
        assert!(runner.initialize().is_ok());
    }

    #[test]
    fn test_discover_test_cases() {
        // Create a temporary test case directory
        let temp_dir = tempfile::tempdir().unwrap();
        let test_case_dir = temp_dir.path().join("test_case_1");
        fs::create_dir_all(&test_case_dir).unwrap();

        // Create a reference file
        let reference_path = test_case_dir.join("reference.csv");
        File::create(&reference_path).unwrap();

        let config = TestRunnerConfig::new(
            temp_dir.path().to_path_buf(),
            PathBuf::from("target/test_output"),
            0.5,
            false,
            "markdown".to_string(),
        );

        let runner = TestRunner::new(config);
        let test_cases = runner.discover_test_cases().unwrap();
        assert_eq!(test_cases.len(), 1);
    }
}
