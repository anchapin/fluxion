//! LLM-Powered Validation Co-pilot for BEM Inputs
//!
//! This module provides an "Engineering Co-pilot" that uses local LLM inference
//! via Ollama to validate building energy model configurations before simulation.
//!
//! # Features
//!
//! - **Local LLM Integration**: Uses Ollama for privacy-preserving, offline-capable inference
//! - **BEM Input Validation**: Checks for common setup errors in building configurations
//! - **Natural Language Troubleshooting**: Provides actionable recommendations
//!
//! # Validation Categories
//!
//! - Window-to-Wall Ratio (WWR) checks
//! - Internal load schedule validation
//! - ASHRAE 90.1 baseline compliance
//! - Physical impossibility checks
//! - Material property range validation
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluxion::validation::copilot::{Copilot, CopilotConfig};
//!
//! let config = CopilotConfig::default();
//! let mut copilot = Copilot::new(config);
//!
//! // Analyze a building configuration
//! let result = copilot.analyze(&building_config).await?;
//! if !result.is_valid() {
//!     for issue in &result.issues {
//!         println!("{}: {}", issue.severity, issue.message);
//!     }
//! }
//! ```

pub mod checker;
pub mod ollama;
pub mod prompt;
pub mod types;

pub use checker::BemChecker;
pub use ollama::OllamaClient;
pub use types::{BemIssue, BemIssueSeverity, CopilotResult, ValidationChecks, OLLAMA_DEFAULT_URL};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the BEM Co-pilot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotConfig {
    /// Ollama server URL
    pub ollama_url: String,
    /// Model name to use for inference
    pub model: String,
    /// Timeout for LLM requests
    pub timeout: Duration,
    /// Enable verbose output
    pub verbose: bool,
    /// Skip LLM call and only use rule-based checks
    pub rule_based_only: bool,
}

impl Default for CopilotConfig {
    fn default() -> Self {
        Self {
            ollama_url: OLLAMA_DEFAULT_URL.to_string(),
            model: "llama3.2:latest".to_string(),
            timeout: Duration::from_secs(30),
            verbose: false,
            rule_based_only: false,
        }
    }
}

impl CopilotConfig {
    /// Create a new config with custom Ollama URL
    pub fn with_ollama_url(mut self, url: String) -> Self {
        self.ollama_url = url;
        self
    }

    /// Create a new config with custom model
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Disable LLM and use only rule-based checks
    pub fn rule_based_only(mut self) -> Self {
        self.rule_based_only = true;
        self
    }
}

/// BEM Co-pilot main entry point
pub struct Copilot {
    config: CopilotConfig,
    ollama: OllamaClient,
    checker: BemChecker,
}

impl Copilot {
    /// Create a new BEM Co-pilot instance
    pub fn new(config: CopilotConfig) -> Self {
        let ollama = OllamaClient::new(config.ollama_url.clone(), config.timeout);
        let checker = BemChecker::new();
        Self {
            config,
            ollama,
            checker,
        }
    }

    /// Analyze a building configuration for issues
    pub async fn analyze(&mut self, config_json: &str) -> Result<CopilotResult> {
        // Step 1: Run rule-based validation checks
        let rule_issues = self.checker.check(config_json);

        if self.config.verbose {
            tracing::warn!(
                "[Copilot] Rule-based checks found {} issues",
                rule_issues.len()
            );
        }

        // Step 2: If LLM is enabled, get natural language analysis
        let llm_analysis = if self.config.rule_based_only {
            None
        } else {
            match self.ollama.analyze(config_json, &rule_issues).await {
                Ok(analysis) => {
                    if self.config.verbose {
                        tracing::warn!("[Copilot] LLM analysis complete");
                    }
                    Some(analysis)
                }
                Err(e) => {
                    if self.config.verbose {
                        tracing::warn!("[Copilot] LLM analysis failed: {}", e);
                    }
                    None
                }
            }
        };

        // Step 3: Combine results
        let all_issues = if let Some(ref analysis) = llm_analysis {
            self.merge_issues(&rule_issues, analysis)
        } else {
            rule_issues
        };

        let config_valid = all_issues
            .iter()
            .all(|i| i.severity != BemIssueSeverity::Error);

        Ok(CopilotResult {
            issues: all_issues,
            llm_analysis,
            config_valid,
        })
    }

    /// Merge rule-based issues with LLM analysis
    fn merge_issues(&self, rule_issues: &[BemIssue], llm_analysis: &str) -> Vec<BemIssue> {
        let mut merged = rule_issues.to_vec();

        // Add LLM insights as informational issues
        if !llm_analysis.is_empty() {
            merged.push(BemIssue {
                severity: BemIssueSeverity::Info,
                category: "llm_insight".to_string(),
                field: "configuration".to_string(),
                message: llm_analysis.to_string(),
                suggestion: None,
            });
        }

        merged
    }

    /// Check if Ollama is available
    pub async fn is_ollama_available(&self) -> bool {
        self.ollama.is_available().await
    }

    /// Get supported validation checks
    pub fn get_validation_checks() -> ValidationChecks {
        ValidationChecks::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copilot_config_default() {
        let config = CopilotConfig::default();
        assert_eq!(config.ollama_url, OLLAMA_DEFAULT_URL);
        assert_eq!(config.model, "llama3.2:latest");
        assert!(!config.rule_based_only);
    }

    #[test]
    fn test_copilot_config_builder() {
        let config = CopilotConfig::default()
            .with_ollama_url("http://localhost:11434".to_string())
            .with_model("mistral:latest".to_string())
            .verbose()
            .rule_based_only();

        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.model, "mistral:latest");
        assert!(config.verbose);
        assert!(config.rule_based_only);
    }

    #[test]
    fn test_bem_checker_basic() {
        let checker = BemChecker::new();

        // Test with empty config
        let issues = checker.check("{}");
        assert!(!issues.is_empty()); // Should find missing required fields

        // Test with invalid WWR
        let config = r#"{"window_wall_ratio": 1.5}"#;
        let issues = checker.check(config);
        let wwr_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.category == "window_to_wall_ratio")
            .collect();
        assert!(!wwr_issues.is_empty());
    }
}
