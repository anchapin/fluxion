use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditFinding {
    file: String,
    line: usize,
    pattern: String,
    content: String,
    priority: String,            // "critical", "warning", "info"
    requirement: Option<String>, // e.g., "PHYS-01", "PHYS-04"
    issue_url: Option<String>,   // GitHub issue URL
}

#[derive(Debug, Serialize, Deserialize)]
struct AuditReport {
    generated: String,
    findings: Vec<AuditFinding>,
    summary: AuditSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditSummary {
    total_findings: usize,
    critical: usize,
    warning: usize,
    info: usize,
    files_scanned: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let pattern_str = match args.get(1).map(|s| s.as_str()) {
        Some("all") => r"TODO|FIXME|mock|placeholder|hardcoded",
        Some("critical") => r"TODO|FIXME|mock|placeholder",
        Some("todo") => r"TODO|FIXME",
        Some("mock") => r"mock|placeholder",
        Some("hardcoded") => r"hardcoded",
        Some("--help" | "-h") => {
            print_usage();
            return Ok(());
        }
        _ => {
            println!("Usage: audit_codebase [all|critical|todo|mock|hardcoded] [--validate]");
            println!("  all       - Scan for all patterns (default)");
            println!("  critical  - Scan for TODO/FIXME/mock/placeholder");
            println!("  todo      - Scan for TODO/FIXME");
            println!("  mock      - Scan for mock/placeholder");
            println!("  hardcoded - Scan for hardcoded values");
            println!("  --validate - Validate JSON output and exit");
            println!("  --help    - Show this help message");
            return Ok(());
        }
    };

    if args.contains(&"--validate".to_string()) {
        run_audit(pattern_str)?;
        validate_audit_report()?;
        return Ok(());
    }

    run_audit(pattern_str)?;
    Ok(())
}

fn print_usage() {
    println!("Fluxion Codebase Audit Tool");
    println!();
    println!("Scans src/ directory for TODO/FIXME/mock/placeholder/hardcoded patterns");
    println!("Generates audit_report.json with structured findings");
    println!();
    println!("Usage: cargo run --bin audit_codebase [PATTERN] [OPTIONS]");
    println!();
    println!("Patterns:");
    println!("  all       - Scan for all patterns (default)");
    println!("  critical  - Scan for TODO/FIXME/mock/placeholder");
    println!("  todo      - Scan for TODO/FIXME");
    println!("  mock      - Scan for mock/placeholder");
    println!("  hardcoded - Scan for hardcoded values");
    println!();
    println!("Options:");
    println!("  --validate - Validate JSON output and exit");
    println!("  --help    - Show this help message");
}

fn run_audit(pattern_str: &str) -> Result<(), Box<dyn std::error::Error>> {
    let pattern = Regex::new(pattern_str)?;
    let mut findings = BTreeMap::new(); // Sort by file name
    let mut files_scanned = HashSet::new();

    WalkDir::new("src")
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
        .for_each(|entry| {
            if let Ok(e) = entry {
                if e.file_type().is_file() && e.path().extension().map_or(false, |ext| ext == "rs")
                {
                    files_scanned.insert(e.path().display().to_string());
                    if let Err(err) = search_file(&e.path(), &pattern, &mut findings) {
                        eprintln!("Error searching {}: {}", e.path().display(), err);
                    }
                }
            }
        });

    let findings_vec: Vec<AuditFinding> = findings.into_values().flatten().collect();

    let summary = calculate_summary(&findings_vec, files_scanned.len());
    let critical_count = summary.critical;
    let warning_count = summary.warning;
    let info_count = summary.info;
    let files_scanned_count = summary.files_scanned;

    let report = AuditReport {
        generated: Utc::now().to_rfc3339(),
        findings: findings_vec,
        summary: summary.clone(),
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write("audit_report.json", &json)?;

    println!(
        "Audit complete: {} findings written to audit_report.json",
        report.findings.len()
    );
    println!("  Critical: {}", critical_count);
    println!("  Warning: {}", warning_count);
    println!("  Info: {}", info_count);
    println!("  Files scanned: {}", files_scanned_count);

    Ok(())
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with('.') || s == "target")
        .unwrap_or(false)
}

fn search_file(
    path: &Path,
    pattern: &Regex,
    findings: &mut BTreeMap<String, Vec<AuditFinding>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut matches = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        if pattern.is_match(line) {
            let line_content = line.trim().to_string();

            // Extract matched pattern from original line
            if let Some(mat) = pattern.find(line) {
                let matched_text = &line[mat.range()];
                let pattern_name = determine_pattern_name(matched_text);
                let priority = determine_priority(&pattern_name);

                matches.push(AuditFinding {
                    file: path.display().to_string(),
                    line: line_num + 1, // 1-based line numbers
                    pattern: pattern_name,
                    content: line_content,
                    priority,
                    requirement: None, // Filled in post-processing
                    issue_url: None,   // Filled in post-processing
                });
            }
        }
    }

    if !matches.is_empty() {
        findings.insert(path.display().to_string(), matches);
    }

    Ok(())
}

fn determine_pattern_name(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("todo") {
        "TODO".to_string()
    } else if lower.contains("fixme") {
        "FIXME".to_string()
    } else if lower.contains("mock") {
        "mock".to_string()
    } else if lower.contains("placeholder") {
        "placeholder".to_string()
    } else if lower.contains("hardcoded") {
        "hardcoded".to_string()
    } else {
        "unknown".to_string()
    }
}

fn determine_priority(pattern: &str) -> String {
    match pattern {
        "TODO" | "FIXME" => "warning".to_string(),
        "mock" | "placeholder" => "critical".to_string(),
        "hardcoded" => "warning".to_string(),
        _ => "info".to_string(),
    }
}

fn calculate_summary(findings: &[AuditFinding], files_scanned: usize) -> AuditSummary {
    let total_findings = findings.len();
    let critical = findings.iter().filter(|f| f.priority == "critical").count();
    let warning = findings.iter().filter(|f| f.priority == "warning").count();
    let info = findings.iter().filter(|f| f.priority == "info").count();

    AuditSummary {
        total_findings,
        critical,
        warning,
        info,
        files_scanned,
    }
}

fn validate_audit_report() -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string("audit_report.json")?;
    let _report: AuditReport = serde_json::from_str(&content)?;
    println!("✓ audit_report.json is valid JSON");
    Ok(())
}
