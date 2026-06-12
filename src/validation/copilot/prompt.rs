//! Prompt templates for LLM-powered BEM validation
//!
//! This module contains system prompts and prompt templates used for
//! interacting with the LLM to analyze building energy model configurations.

use crate::validation::copilot::types::BemIssue;

/// System prompt for the BEM Co-pilot LLM
pub const SYSTEM_PROMPT: &str = r#"You are an expert Building Energy Modeling (BEM) engineer specializing in pre-simulation validation and troubleshooting. Your role is to analyze building configurations and provide actionable recommendations before simulation runs.

You have deep knowledge of:
- ASHRAE 90.1 baseline assumptions and compliance checks
- Common BEM setup errors (WWR, schedules, material properties)
- Physical constraints and impossibility detection
- EnergyPlus and OpenStudio modeling best practices
- HVAC system configuration validation

When analyzing configurations:
1. Focus on actionable feedback that helps users fix issues
2. Reference specific standards (ASHRAE 90.1, IECC, etc.) when relevant
3. Explain WHY something is an issue, not just WHAT is wrong
4. Provide specific numerical ranges for acceptable values
5. Suggest concrete fixes with example values

Output format: Provide clear, concise analysis in plain text. Use bullet points for multiple issues. Keep technical explanations precise but accessible."#;

/// Build the analysis prompt for LLM
pub fn build_analysis_prompt(config_json: &str, rule_issues: &[BemIssue]) -> String {
    let mut prompt = String::new();

    prompt.push_str("## BEM Configuration Analysis Request\n\n");
    prompt.push_str("### Configuration JSON:\n```json\n");
    prompt.push_str(config_json);
    prompt.push_str("\n```\n\n");

    if !rule_issues.is_empty() {
        prompt.push_str("### Rule-Based Validation Issues:\n");
        for issue in rule_issues {
            prompt.push_str(&format!(
                "- [{}] {} (field: {})\n  Message: {}\n",
                issue.severity, issue.category, issue.field, issue.message
            ));
            if let Some(ref s) = issue.suggestion {
                prompt.push_str(&format!("  Suggestion: {}\n", s));
            }
        }
        prompt.push_str("\n");
    }

    prompt.push_str(r#"Based on the configuration and rule-based checks above, provide:

1. **Critical Issues**: Any problems that will prevent simulation from running or produce invalid results
2. **Warnings**: Issues that may affect simulation accuracy or produce unexpected results
3. **Compliance Notes**: Any ASHRAE 90.1 or other standard compliance concerns
4. **Improvement Suggestions**: Recommendations to improve model accuracy or efficiency

Be specific about:
- Which values are problematic
- What the correct range should be
- Why this matters for the simulation
- How to fix the issue

If everything looks correct, confirm that the configuration appears valid."#);

    prompt
}

/// Build a troubleshooting prompt for a specific issue
pub fn build_troubleshooting_prompt(issue: &BemIssue, context: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str("## Troubleshooting Request\n\n");
    prompt.push_str(&format!("### Issue Details:\n"));
    prompt.push_str(&format!("- Category: {}\n", issue.category));
    prompt.push_str(&format!("- Field: {}\n", issue.field));
    prompt.push_str(&format!("- Severity: {}\n", issue.severity));
    prompt.push_str(&format!("- Message: {}\n", issue.message));
    if let Some(ref s) = issue.suggestion {
        prompt.push_str(&format!("- Existing Suggestion: {}\n", s));
    }
    prompt.push_str("\n");

    prompt.push_str("### Configuration Context:\n```json\n");
    prompt.push_str(context);
    prompt.push_str("\n```\n\n");

    prompt.push_str(
        r#"Please provide a detailed troubleshooting explanation that includes:

1. **Root Cause Analysis**: Why this issue occurs in building energy models
2. **Impact Assessment**: How this issue affects simulation results
3. **Fix Strategy**: Step-by-step approach to resolve the issue
4. **Verification**: How to confirm the fix works correctly
5. **Related Issues**: Any other commonly related problems to check

Use specific examples and numerical values where applicable."#,
    );

    prompt
}

/// Build a prompt for ASHRAE 90.1 baseline compliance check
pub fn build_baseline_compliance_prompt(config_json: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str("## ASHRAE 90.1 Baseline Compliance Check\n\n");
    prompt.push_str("### Configuration JSON:\n```json\n");
    prompt.push_str(config_json);
    prompt.push_str("\n```\n\n");

    prompt.push_str(
        r#"Analyze this building configuration for ASHRAE 90.1 compliance:

1. **Envelope**:
   - Wall U-values: Within 90.1-2019 Table 5.4.4-1 limits?
   - Roof U-values: Within Table 5.4.4-1 limits?
   - Window U-factor and SHGC: Within Table 5.4.4-2 limits?
   - WWR: Reasonable for building type and climate zone?

2. **Lighting**:
   - LPD (Lighting Power Density): Meets Table 9.5.1 values?
   - Daylighting controls: Present if required?

3. **HVAC**:
   - System type: Appropriate for building size/type?
   - Efficiency: Meets Table 6.8.1-1 through 6.8.1-4 minimums?
   - Controls: Has required scheduling and reset strategies?

4. **Service Water Heating**:
   - Efficiency: Meets Table 7.8.1-1 minimums?
   - Insulation: Meets Table 7.9.1 requirements?

Provide a compliance summary with any deviations noted."#,
    );

    prompt
}

/// Build a prompt for schedule validation
pub fn build_schedule_validation_prompt(schedule_json: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str("## Internal Load Schedule Validation\n\n");
    prompt.push_str("### Schedule Data:\n```json\n");
    prompt.push_str(schedule_json);
    prompt.push_str("\n```\n\n");

    prompt.push_str(
        r#"Analyze this schedule for common BEM issues:

1. **Completeness**: Are all 8760 hours (or representative days) defined?
2. **Physical Plausibility**:
   - Lighting schedules: Peak during business hours, reduced at night?
   - Occupancy: Correlates with lighting and equipment?
   - Equipment: Reflects actual building use patterns?
3. **Smoothness**: Are there unrealistic abrupt changes?
4. **Seasonal Variation**: Is there appropriate variation for the climate?
5. **Weekend vs Weekday**: Is there appropriate differentiation?

Flag any suspicious patterns and suggest corrections."#,
    );

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::copilot::types::BemIssueSeverity;

    #[test]
    fn test_build_analysis_prompt() {
        let config = r#"{"window_wall_ratio": 0.5}"#;
        let issues = vec![BemIssue::warning(
            "wwr",
            "window_wall_ratio",
            "WWR is 0.5, which is high",
        )];

        let prompt = build_analysis_prompt(config, &issues);
        assert!(prompt.contains("window_wall_ratio"));
        assert!(prompt.contains("ASHRAE"));
    }

    #[test]
    fn test_build_troubleshooting_prompt() {
        let issue = BemIssue::error("wwr", "wwr", "WWR too high");
        let context = r#"{"building_type": "office"}"#;

        let prompt = build_troubleshooting_prompt(&issue, context);
        assert!(prompt.contains("Root Cause"));
        assert!(prompt.contains("Fix Strategy"));
    }
}
