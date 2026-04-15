// validation/automation/github/workflow.rs
/// GitHub workflow generation and management
///
/// This module provides workflow generation capabilities for GitHub Actions,
/// including template management, YAML generation, and workflow validation.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Workflow generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowGeneratorConfig {
    /// Template directory
    pub template_dir: PathBuf,
    /// Output directory
    pub output_dir: PathBuf,
    /// Default workflow name
    pub default_name: String,
    /// Default workflow description
    pub default_description: String,
}

impl Default for WorkflowGeneratorConfig {
    fn default() -> Self {
        Self {
            template_dir: PathBuf::from(".github/workflow_templates"),
            output_dir: PathBuf::from(".github/workflows"),
            default_name: "Cross-Validation CI".to_string(),
            default_description: "Automated cross-validation testing".to_string(),
        }
    }
}

/// Workflow generator
#[derive(Debug)]
pub struct WorkflowGenerator {
    config: WorkflowGeneratorConfig,
    templates: HashMap<String, String>,
}

impl WorkflowGenerator {
    /// Create a new workflow generator
    pub fn new(config: WorkflowGeneratorConfig) -> Result<Self, Box<dyn Error>> {
        let mut generator = Self {
            config,
            templates: HashMap::new(),
        };
        generator.load_templates()?;
        Ok(generator)
    }

    /// Load workflow templates from template directory
    fn load_templates(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.config.template_dir.exists() {
            fs::create_dir_all(&self.config.template_dir)?;
            return Ok(()); // No templates to load
        }

        for entry in fs::read_dir(&self.config.template_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("yaml") {
                let template_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let content = fs::read_to_string(&path)?;
                self.templates.insert(template_name, content);
            }
        }

        Ok(())
    }

    /// Generate a cross-validation workflow
    pub fn generate_cross_validation_workflow(
        &self,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<String, Box<dyn Error>> {
        let name = name.unwrap_or_else(|| self.config.default_name.clone());
        let description = description.unwrap_or_else(|| self.config.default_description.clone());

        let yaml = format!(
            "name: {}\n"
            + "description: {}\n"
            + "on:\n"
            + "  push:\n"
            + "    branches: [main]\n"
            + "  pull_request:\n"
            + "    branches: [main]\n"
            + "  workflow_dispatch:\n"
            + "\n"
            + "env:\n"
            + "  RUST_VERSION: '1.70.0'\n"
            + "  CARGO_TERM_COLOR: always\n"
            + "\n"
            + "jobs:\n"
            + "  cross-validation:\n"
            + "    name: Cross-Validation Tests\n"
            + "    runs-on: ubuntu-latest\n"
            + "    steps:\n"
            + "      - name: Checkout repository\n"
            + "        uses: actions/checkout@v4\n"
            + "\n"
            + "      - name: Install Rust toolchain\n"
            + "        uses: dtolnay/rust-toolchain@stable\n"
            + "        with:\n"
            + "          toolchain: ${{{{ env.RUST_VERSION }}}}\n"
            + "\n"
            + "      - name: Cache cargo dependencies\n"
            + "        uses: actions/cache@v3\n"
            + "        with:\n"
            + "          path: |\n"
            + "            ~/.cargo/registry\n"
            + "            ~/.cargo/git\n"
            + "            target\n"
            + "          key: ${{{{ runner.os }}}}-cargo-${{{{ hashFiles('**/Cargo.lock') }}}}\n"
            + "\n"
            + "      - name: Build project\n"
            + "        run: cargo build --release\n"
            + "\n"
            + "      - name: Run cross-validation tests\n"
            + "        run: cargo test --release -- --test-threads=1\n"
            + "\n"
            + "      - name: Generate validation report\n"
            + "        run: cargo run --release -- validate --format=markdown --output-file=validation_report.md\n"
            + "\n"
            + "      - name: Upload validation report\n"
            + "        uses: actions/upload-artifact@v3\n"
            + "        with:\n"
            + "          name: validation-report\n"
            + "          path: validation_report.md\n",
            name, description
        );

        Ok(yaml)
    }

    /// Generate a performance validation workflow
    pub fn generate_performance_workflow(
        &self,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<String, Box<dyn Error>> {
        let name = name.unwrap_or_else(|| "Performance Validation".to_string());
        let description =
            description.unwrap_or_else(|| "Performance validation and benchmarking".to_string());

        let yaml = format!(
            "name: {}\n"
                + "description: {}\n"
                + "on:\n"
                + "  push:\n"
                + "    branches: [main]\n"
                + "  pull_request:\n"
                + "    branches: [main]\n"
                + "  schedule:\n"
                + "    - cron: '0 0 * * 0'  # Weekly on Sunday\n"
                + "\n"
                + "env:\n"
                + "  RUST_VERSION: '1.70.0'\n"
                + "\n"
                + "jobs:\n"
                + "  performance-benchmark:\n"
                + "    name: Performance Benchmark\n"
                + "    runs-on: ubuntu-latest\n"
                + "    steps:\n"
                + "      - name: Checkout repository\n"
                + "        uses: actions/checkout@v4\n"
                + "\n"
                + "      - name: Install Rust toolchain\n"
                + "        uses: dtolnay/rust-toolchain@stable\n"
                + "        with:\n"
                + "          toolchain: ${{{{ env.RUST_VERSION }}}}\n"
                + "\n"
                + "      - name: Run performance benchmarks\n"
                + "        run: cargo bench --all\n"
                + "\n"
                + "      - name: Upload benchmark results\n"
                + "        uses: actions/upload-artifact@v3\n"
                + "        with:\n"
                + "          name: benchmark-results\n"
                + "          path: target/criterion\n",
            name, description
        );

        Ok(yaml)
    }

    /// Generate a CI/CD pipeline workflow
    pub fn generate_ci_cd_workflow(
        &self,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<String, Box<dyn Error>> {
        let name = name.unwrap_or_else(|| "CI/CD Pipeline".to_string());
        let description = description
            .unwrap_or_else(|| "Complete CI/CD pipeline with testing and deployment".to_string());

        let yaml = format!(
            "name: {}\n"
                + "description: {}\n"
                + "on:\n"
                + "  push:\n"
                + "    branches: [main]\n"
                + "  pull_request:\n"
                + "    branches: [main]\n"
                + "\n"
                + "env:\n"
                + "  RUST_VERSION: '1.70.0'\n"
                + "\n"
                + "jobs:\n"
                + "  test:\n"
                + "    name: Run Tests\n"
                + "    runs-on: ubuntu-latest\n"
                + "    steps:\n"
                + "      - name: Checkout repository\n"
                + "        uses: actions/checkout@v4\n"
                + "\n"
                + "      - name: Install Rust toolchain\n"
                + "        uses: dtolnay/rust-toolchain@stable\n"
                + "        with:\n"
                + "          toolchain: ${{{{ env.RUST_VERSION }}}}\n"
                + "\n"
                + "      - name: Run tests\n"
                + "        run: cargo test --all\n"
                + "\n"
                + "  build:\n"
                + "    name: Build Release\n"
                + "    needs: test\n"
                + "    runs-on: ubuntu-latest\n"
                + "    steps:\n"
                + "      - name: Checkout repository\n"
                + "        uses: actions/checkout@v4\n"
                + "\n"
                + "      - name: Install Rust toolchain\n"
                + "        uses: dtolnay/rust-toolchain@stable\n"
                + "        with:\n"
                + "          toolchain: ${{{{ env.RUST_VERSION }}}}\n"
                + "\n"
                + "      - name: Build release\n"
                + "        run: cargo build --release\n"
                + "\n"
                + "      - name: Upload artifacts\n"
                + "        uses: actions/upload-artifact@v3\n"
                + "        with:\n"
                + "          name: release-binaries\n"
                + "          path: target/release/fluxion\n",
            name, description
        );

        Ok(yaml)
    }

    /// Save workflow to file
    pub fn save_workflow(&self, yaml_content: &str, filename: &str) -> Result<(), Box<dyn Error>> {
        if !self.config.output_dir.exists() {
            fs::create_dir_all(&self.config.output_dir)?;
        }

        let output_path = self.config.output_dir.join(filename);
        fs::write(&output_path, yaml_content)?;

        Ok(())
    }

    /// Validate generated workflow YAML
    pub fn validate_workflow_yaml(&self, yaml_content: &str) -> Result<(), Box<dyn Error>> {
        // Basic validation - check for required sections
        let required_sections = ["name:", "on:", "jobs:"];

        for section in required_sections {
            if !yaml_content.contains(section) {
                return Err(format!("Missing required section: {}", section).into());
            }
        }

        // Check for common syntax errors
        let lines: Vec<&str> = yaml_content.split('\n').collect();
        for (i, line) in lines.iter().enumerate() {
            // Check for unbalanced quotes
            let single_quotes = line.matches('\'').count();
            let double_quotes = line.matches('"').count();

            if single_quotes % 2 != 0 || double_quotes % 2 != 0 {
                return Err(format!("Unbalanced quotes on line {}", i + 1).into());
            }
        }

        Ok(())
    }

    /// Apply template variables to workflow content
    pub fn apply_template_variables(
        &self,
        template_content: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, Box<dyn Error>> {
        let mut result = template_content.to_string();

        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }
}

/// Template management
#[derive(Debug)]
pub struct TemplateManager {
    template_dir: PathBuf,
}

impl TemplateManager {
    /// Create a new template manager
    pub fn new(template_dir: PathBuf) -> Self {
        Self { template_dir }
    }

    /// Create a new template
    pub fn create_template(&self, name: &str, content: &str) -> Result<(), Box<dyn Error>> {
        if !self.template_dir.exists() {
            fs::create_dir_all(&self.template_dir)?;
        }

        let template_path = self.template_dir.join(format!("{}.yaml", name));
        fs::write(&template_path, content)?;

        Ok(())
    }

    /// Get template content
    pub fn get_template(&self, name: &str) -> Result<String, Box<dyn Error>> {
        let template_path = self.template_dir.join(format!("{}.yaml", name));

        if !template_path.exists() {
            return Err(format!("Template not found: {}", name).into());
        }

        let content = fs::read_to_string(&template_path)?;
        Ok(content)
    }

    /// List available templates
    pub fn list_templates(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let mut templates = Vec::new();

        if !self.template_dir.exists() {
            return Ok(templates);
        }

        for entry in fs::read_dir(&self.template_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("yaml") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    templates.push(name.to_string());
                }
            }
        }

        Ok(templates)
    }

    /// Delete a template
    pub fn delete_template(&self, name: &str) -> Result<(), Box<dyn Error>> {
        let template_path = self.template_dir.join(format!("{}.yaml", name));

        if template_path.exists() {
            fs::remove_file(&template_path)?;
        }

        Ok(())
    }
}

/// Workflow generation error types
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("Template error: {0}")]
    TemplateError(String),
    #[error("Generation error: {0}")]
    GenerationError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for workflow operations
pub type WorkflowResult<T> = Result<T, WorkflowError>;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_workflow_generator_creation() {
        let config = WorkflowGeneratorConfig::default();
        let generator = WorkflowGenerator::new(config).unwrap();
        assert_eq!(generator.config.default_name, "Cross-Validation CI");
    }

    #[test]
    fn test_cross_validation_workflow_generation() {
        let config = WorkflowGeneratorConfig::default();
        let generator = WorkflowGenerator::new(config).unwrap();

        let yaml = generator
            .generate_cross_validation_workflow(None, None)
            .unwrap();
        assert!(yaml.contains("name: Cross-Validation CI"));
        assert!(yaml.contains("description: Automated cross-validation testing"));
        assert!(yaml.contains("actions/checkout@v4"));
        assert!(yaml.contains("cargo test --release"));
    }

    #[test]
    fn test_performance_workflow_generation() {
        let config = WorkflowGeneratorConfig::default();
        let generator = WorkflowGenerator::new(config).unwrap();

        let yaml = generator.generate_performance_workflow(None, None).unwrap();
        assert!(yaml.contains("name: Performance Validation"));
        assert!(yaml.contains("cargo bench --all"));
        assert!(yaml.contains("schedule:"));
    }

    #[test]
    fn test_workflow_validation() {
        let config = WorkflowGeneratorConfig::default();
        let generator = WorkflowGenerator::new(config).unwrap();

        let valid_yaml = generator
            .generate_cross_validation_workflow(None, None)
            .unwrap();
        assert!(generator.validate_workflow_yaml(&valid_yaml).is_ok());

        let invalid_yaml = "name: Test\n".to_string();
        assert!(generator.validate_workflow_yaml(&invalid_yaml).is_err());
    }

    #[test]
    fn test_template_management() {
        let temp_dir = tempdir().unwrap();
        let template_dir = temp_dir.path().to_path_buf();
        let manager = TemplateManager::new(template_dir.clone());

        // Create a template
        manager
            .create_template("test", "name: Test\non: push")
            .unwrap();

        // Get the template
        let content = manager.get_template("test").unwrap();
        assert!(content.contains("name: Test"));

        // List templates
        let templates = manager.list_templates().unwrap();
        assert!(templates.contains(&"test".to_string()));

        // Delete the template
        manager.delete_template("test").unwrap();
        let templates = manager.list_templates().unwrap();
        assert!(!templates.contains(&"test".to_string()));
    }

    #[test]
    fn test_template_variables() {
        let config = WorkflowGeneratorConfig::default();
        let generator = WorkflowGenerator::new(config).unwrap();

        let template = "name: {{{{WORKFLOW_NAME}}}}\ndescription: {{{{WORKFLOW_DESC}}}}";
        let mut variables = HashMap::new();
        variables.insert("WORKFLOW_NAME".to_string(), "Test Workflow".to_string());
        variables.insert("WORKFLOW_DESC".to_string(), "Test description".to_string());

        let result = generator
            .apply_template_variables(template, &variables)
            .unwrap();
        assert!(result.contains("name: Test Workflow"));
        assert!(result.contains("description: Test description"));
    }
}
