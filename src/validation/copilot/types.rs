//! Types for the BEM Co-pilot validation system

use serde::{Deserialize, Serialize};

/// Default Ollama server URL
pub const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434";

/// Severity level for BEM issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BemIssueSeverity {
    /// Critical error that prevents simulation
    Error,
    /// Warning about potential issues
    Warning,
    /// Informational notice
    Info,
    /// Hint or suggestion
    Hint,
}

impl std::fmt::Display for BemIssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BemIssueSeverity::Error => write!(f, "ERROR"),
            BemIssueSeverity::Warning => write!(f, "WARNING"),
            BemIssueSeverity::Info => write!(f, "INFO"),
            BemIssueSeverity::Hint => write!(f, "HINT"),
        }
    }
}

/// A detected issue in a BEM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BemIssue {
    /// Severity level of the issue
    pub severity: BemIssueSeverity,
    /// Category of the issue (e.g., "window_to_wall_ratio", "internal_loads")
    pub category: String,
    /// Field in the configuration that has the issue
    pub field: String,
    /// Human-readable message describing the issue
    pub message: String,
    /// Optional suggestion for fixing the issue
    pub suggestion: Option<String>,
}

impl BemIssue {
    /// Create a new error issue
    pub fn error(category: &str, field: &str, message: &str) -> Self {
        Self {
            severity: BemIssueSeverity::Error,
            category: category.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }

    /// Create a new error issue with suggestion
    pub fn error_with_suggestion(
        category: &str,
        field: &str,
        message: &str,
        suggestion: &str,
    ) -> Self {
        Self {
            severity: BemIssueSeverity::Error,
            category: category.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            suggestion: Some(suggestion.to_string()),
        }
    }

    /// Create a new warning issue
    pub fn warning(category: &str, field: &str, message: &str) -> Self {
        Self {
            severity: BemIssueSeverity::Warning,
            category: category.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }

    /// Create a new warning issue with suggestion
    pub fn warning_with_suggestion(
        category: &str,
        field: &str,
        message: &str,
        suggestion: &str,
    ) -> Self {
        Self {
            severity: BemIssueSeverity::Warning,
            category: category.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            suggestion: Some(suggestion.to_string()),
        }
    }

    /// Create a new info issue
    pub fn info(category: &str, field: &str, message: &str) -> Self {
        Self {
            severity: BemIssueSeverity::Info,
            category: category.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }

    /// Create a new hint issue
    pub fn hint(category: &str, field: &str, message: &str) -> Self {
        Self {
            severity: BemIssueSeverity::Hint,
            category: category.to_string(),
            field: field.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }
}

/// Result of BEM configuration analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotResult {
    /// All detected issues
    pub issues: Vec<BemIssue>,
    /// LLM-generated natural language analysis (if available)
    pub llm_analysis: Option<String>,
    /// Whether the configuration is valid for simulation
    pub config_valid: bool,
}

impl CopilotResult {
    /// Create a successful result with no issues
    pub fn success() -> Self {
        Self {
            issues: Vec::new(),
            llm_analysis: None,
            config_valid: true,
        }
    }

    /// Check if the configuration is valid
    pub fn is_valid(&self) -> bool {
        self.config_valid
    }

    /// Get all error issues
    pub fn errors(&self) -> Vec<&BemIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Error)
            .collect()
    }

    /// Get all warning issues
    pub fn warnings(&self) -> Vec<&BemIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Warning)
            .collect()
    }

    /// Get all info issues
    pub fn infos(&self) -> Vec<&BemIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Info)
            .collect()
    }

    /// Get count of issues by severity
    pub fn count_by_severity(&self) -> (usize, usize, usize, usize) {
        let errors = self
            .issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Error)
            .count();
        let warnings = self
            .issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Warning)
            .count();
        let infos = self
            .issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Info)
            .count();
        let hints = self
            .issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Hint)
            .count();
        (errors, warnings, infos, hints)
    }

    /// Print issues in a human-readable format
    pub fn print_summary(&self) {
        let (errors, warnings, infos, hints) = self.count_by_severity();

        tracing::info!("\n╔══════════════════════════════════════════════════════════════╗");
        tracing::info!("║              BEM Configuration Analysis Results              ║");
        tracing::info!("╠══════════════════════════════════════════════════════════════╣");
        tracing::info!(
            "║  Issues Found: {} errors, {} warnings, {} info, {} hints     ║",
            errors,
            warnings,
            infos,
            hints
        );
        tracing::info!("╚══════════════════════════════════════════════════════════════╝");

        if self.issues.is_empty() {
            tracing::info!("\n✓ Configuration appears valid!");
            return;
        }

        if errors > 0 {
            tracing::info!("\n❌ ERRORS (must fix before simulation):");
            for issue in self
                .issues
                .iter()
                .filter(|i| i.severity == BemIssueSeverity::Error)
            {
                tracing::info!("  • [{}] {}", issue.category, issue.message);
                if let Some(ref s) = issue.suggestion {
                    tracing::info!("    → Suggestion: {}", s);
                }
            }
        }

        if warnings > 0 {
            tracing::info!("\n⚠ WARNINGS (may affect accuracy):");
            for issue in self
                .issues
                .iter()
                .filter(|i| i.severity == BemIssueSeverity::Warning)
            {
                tracing::info!("  • [{}] {}", issue.category, issue.message);
                if let Some(ref s) = issue.suggestion {
                    tracing::info!("    → Suggestion: {}", s);
                }
            }
        }

        if infos > 0 || hints > 0 {
            tracing::info!("\n💡 INFO / HINTS:");
            for issue in self.issues.iter().filter(|i| {
                i.severity == BemIssueSeverity::Info || i.severity == BemIssueSeverity::Hint
            }) {
                tracing::info!("  • [{}] {}", issue.category, issue.message);
            }
        }

        if let Some(ref analysis) = self.llm_analysis {
            tracing::info!("\n🤖 LLM Analysis:");
            tracing::info!("{}", analysis);
        }
    }
}

/// Validation checks available in the Co-pilot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationChecks {
    /// Check window-to-wall ratio
    pub wwr_check: bool,
    /// Check internal load schedules
    pub schedule_check: bool,
    /// Check material properties
    pub material_check: bool,
    /// Check ASHRAE 90.1 baseline compliance
    pub baseline_check: bool,
    /// Check physical constraints
    pub physics_check: bool,
    /// Check HVAC configuration
    pub hvac_check: bool,
}

impl ValidationChecks {
    /// Get all validation checks enabled
    pub fn all() -> Self {
        Self {
            wwr_check: true,
            schedule_check: true,
            material_check: true,
            baseline_check: true,
            physics_check: true,
            hvac_check: true,
        }
    }

    /// Get only essential validation checks
    pub fn essential() -> Self {
        Self {
            wwr_check: true,
            schedule_check: true,
            material_check: false,
            baseline_check: false,
            physics_check: true,
            hvac_check: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bem_issue_error() {
        let issue = BemIssue::error("wwr", "window_wall_ratio", "WWR too high");
        assert_eq!(issue.severity, BemIssueSeverity::Error);
        assert_eq!(issue.category, "wwr");
    }

    #[test]
    fn test_bem_issue_with_suggestion() {
        let issue = BemIssue::warning_with_suggestion(
            "schedule",
            "lighting_schedule",
            "Schedule may be incorrect",
            "Use a typical occupancy schedule from ASHRAE 90.1",
        );
        assert_eq!(issue.severity, BemIssueSeverity::Warning);
        assert!(issue.suggestion.is_some());
    }

    #[test]
    fn test_copilot_result_valid() {
        let result = CopilotResult::success();
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
    }

    #[test]
    fn test_copilot_result_count_by_severity() {
        let result = CopilotResult {
            issues: vec![
                BemIssue::error("wwr", "wwr", "error"),
                BemIssue::error("schedule", "schedule", "error2"),
                BemIssue::warning("material", "material", "warning"),
                BemIssue::info("physics", "physics", "info"),
            ],
            llm_analysis: None,
            config_valid: false,
        };

        let (errors, warnings, infos, hints) = result.count_by_severity();
        assert_eq!(errors, 2);
        assert_eq!(warnings, 1);
        assert_eq!(infos, 1);
        assert_eq!(hints, 0);
    }
}
