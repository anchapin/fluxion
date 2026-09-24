//! `fluxion topology` subcommand (Issue #3963) plus the backward-compatibility
//! aliases `fluxion export-topology` (Issue #3966) and `fluxion lint-topology`
//! (Issue #3964).
//!
//! `topology export` builds the in-memory simulation topology graph for a
//! model — either an ASHRAE 140 registry case (`--case N`) or a serialized
//! `CaseSpec` document (`--model <path>`) — finalizes, validates, and writes it
//! as deterministic JSON (identical inputs → byte-identical output).
//!
//! `topology lint` checks a topology — registry case (`--case`), CaseSpec
//! file (`--model`), or a previously exported topology document (`--input`,
//! fully offline) — against the #3964 structural rule taxonomy
//! (E001–E007 errors, W001–W002 warnings) and exits 0 clean / 1 findings /
//! 2 usage or internal errors.
//!
//! The handlers deliberately mirror the fail-loud CLI contract: mutually
//! exclusive source flags are enforced, unknown case ids error out non-zero,
//! and no path here silently succeeds (cf. issue #2947 stub policy).

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};

use crate::sim::topology_lint::{lint_topology, LintReport, LintSeverity};
use crate::topology::{TopologyContext, TopologyGraph};
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use crate::validation::topology_bridge;

/// Arguments for `fluxion topology export` / `fluxion export-topology`.
#[derive(Debug, Args, Clone)]
pub struct ExportArgs {
    /// ASHRAE 140 case id from the registry (e.g. 600, 610, 620, 630, 640,
    /// 650, 900, 960, 650FF, 950FF, ...). Mutually exclusive with --model.
    #[arg(long, value_name = "N", conflicts_with = "model")]
    pub case: Option<String>,

    /// Path to a serialized CaseSpec model document (JSON). Mutually
    /// exclusive with --case.
    #[arg(long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Output file path; writes to stdout when omitted.
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Optional export timestamp embedded in the metadata block. Omitted by
    /// default so repeated exports are byte-identical (determinism contract).
    #[arg(long, value_name = "ISO8601")]
    pub timestamp: Option<String>,
}

/// `fluxion topology ...` subcommands.
#[derive(Debug, Subcommand, Clone)]
pub enum TopologyCommand {
    /// Export the simulation topology graph as deterministic JSON.
    Export(ExportArgs),
    /// Lint a topology for structural defects (Issue #3964).
    Lint(LintArgs),
}

/// Dispatches `fluxion topology <command>`.
pub fn dispatch(command: TopologyCommand) -> Result<()> {
    match command {
        TopologyCommand::Export(args) => handle_export(args),
        TopologyCommand::Lint(args) => handle_lint(args),
    }
}

/// Builds and writes the topology export. Shared by `topology export` and the
/// `export-topology` alias (identical behavior by construction: both call
/// this function).
pub fn handle_export(args: ExportArgs) -> Result<()> {
    let (spec, ctx) = match (&args.case, &args.model) {
        (Some(case_id), None) => {
            let case = ASHRAE140Case::from_case_id(case_id).ok_or_else(|| {
                anyhow!(
                    "unknown ASHRAE 140 case id: {case_id:?}; expected a registry id \
                     such as 600, 610, 620, 630, 640, 650, 900, 950, 960 (see \
                     `fluxion validate --help`)"
                )
            })?;
            let spec = case.spec();
            let ctx = TopologyContext {
                model_name: format!("ASHRAE 140 Case {}", spec.case_id),
                model_source: format!("ashrae-140-registry:{}", spec.case_id),
                timestamp: args.timestamp.clone(),
            };
            (spec, ctx)
        }
        (None, Some(path)) => {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("failed to read model file {}", path.display()))?;
            let spec: CaseSpec = serde_json::from_str(&raw).with_context(|| {
                format!("failed to parse CaseSpec JSON from {}", path.display())
            })?;
            let model_name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string());
            let ctx = TopologyContext {
                model_name,
                model_source: format!("model-file:{}", path.display()),
                timestamp: args.timestamp.clone(),
            };
            (spec, ctx)
        }
        _ => {
            // clap's conflicts_with already rejects case+model; this arm only
            // fires when neither was provided.
            bail!(
                "exactly one of --case <N> or --model <PATH> is required \
                 (they are mutually exclusive)"
            );
        }
    };

    let mut graph = topology_bridge::case_topology_graph(&spec, &ctx);
    graph
        .finalize()
        .map_err(|e| anyhow::anyhow!("topology finalization failed: {e}"))?;
    graph
        .validate()
        .map_err(|e| anyhow::anyhow!("topology validation failed: {e}"))?;

    let json = graph
        .to_json_string()
        .context("failed to serialize topology graph")?;

    match &args.output {
        Some(path) => {
            fs::write(path, json.as_bytes()).with_context(|| {
                format!("failed to write topology export to {}", path.display())
            })?;
            println!(
                "wrote topology export ({} nodes, {} edges) to {}",
                graph.metadata.node_count,
                graph.metadata.edge_count,
                path.display()
            );
        }
        None => println!("{json}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neither_source_nor_model_is_rejected() {
        let args = ExportArgs {
            case: None,
            model: None,
            output: None,
            timestamp: None,
        };
        assert!(handle_export(args).is_err());
    }

    #[test]
    fn unknown_case_id_is_rejected() {
        let args = ExportArgs {
            case: Some("not-a-case".to_string()),
            model: None,
            output: None,
            timestamp: None,
        };
        let err = handle_export(args).unwrap_err().to_string();
        assert!(err.contains("unknown ASHRAE 140 case id"));
    }
}

// ---------------------------------------------------------------------------
// Issue #3964 — `fluxion topology lint` / `fluxion lint-topology`
// ---------------------------------------------------------------------------

/// Output format for the lint report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LintFormat {
    /// Human-readable summary (default).
    Text,
    /// Machine-readable JSON with stable finding codes.
    Json,
}

/// Arguments for `fluxion topology lint` / `fluxion lint-topology` (Issue
/// #3964). Exactly one of `--case`, `--model`, or `--input` is required.
#[derive(Debug, Args, Clone)]
pub struct LintArgs {
    /// ASHRAE 140 registry case id to lint (e.g. 600, 900). Mutually
    /// exclusive with --model and --input.
    #[arg(long, value_name = "N", conflicts_with_all = ["model", "input"])]
    pub case: Option<String>,

    /// Path to a serialized CaseSpec model document (JSON) to lint. Mutually
    /// exclusive with --case and --input.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["case", "input"])]
    pub model: Option<PathBuf>,

    /// Path to a previously exported topology document (`topology export`
    /// output) to lint offline — no registry access required.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["case", "model"])]
    pub input: Option<PathBuf>,

    /// Treat warnings as errors for the exit code (exit 1 when only
    /// warnings are found).
    #[arg(long)]
    pub strict: bool,

    /// Output format for the lint report.
    #[arg(long, value_enum, default_value_t = LintFormat::Text)]
    pub format: LintFormat,
}

/// Resolves the graph to lint from `--case` / `--model` / `--input`.
///
/// Returns the graph plus its provenance label (`metadata.model_source`).
/// The `--input` path deliberately skips finalize/validate: the linter must
/// be able to speak about broken documents, not refuse them.
fn lint_graph_source(args: &LintArgs) -> Result<(TopologyGraph, String)> {
    match (&args.case, &args.model, &args.input) {
        (Some(case_id), None, None) => {
            let case = ASHRAE140Case::from_case_id(case_id).ok_or_else(|| {
                anyhow!(
                    "unknown ASHRAE 140 case id: {case_id:?}; expected a registry id \
                     such as 600, 610, 620, 630, 640, 650, 900, 950, 960 (see \
                     `fluxion validate --help`)"
                )
            })?;
            let spec = case.spec();
            let ctx = TopologyContext {
                model_name: format!("ASHRAE 140 Case {}", spec.case_id),
                model_source: format!("ashrae-140-registry:{}", spec.case_id),
                timestamp: None,
            };
            let mut graph = topology_bridge::case_topology_graph(&spec, &ctx);
            graph
                .finalize()
                .map_err(|e| anyhow!("topology finalization failed: {e}"))?;
            graph
                .validate()
                .map_err(|e| anyhow!("topology validation failed: {e}"))?;
            Ok((graph, ctx.model_source))
        }
        (None, Some(path), None) => {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("failed to read model file {}", path.display()))?;
            let spec: CaseSpec = serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse CaseSpec document {}", path.display()))?;
            let ctx = TopologyContext {
                model_name: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string(),
                model_source: format!("model-file:{}", path.display()),
                timestamp: None,
            };
            let mut graph = topology_bridge::case_topology_graph(&spec, &ctx);
            graph
                .finalize()
                .map_err(|e| anyhow!("topology finalization failed: {e}"))?;
            graph
                .validate()
                .map_err(|e| anyhow!("topology validation failed: {e}"))?;
            Ok((graph, ctx.model_source))
        }
        (None, None, Some(path)) => {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("failed to read topology document {}", path.display()))?;
            let graph: TopologyGraph = serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse topology document {}", path.display()))?;
            let source = graph.metadata.model_source.clone();
            Ok((graph, source))
        }
        _ => bail!(
            "exactly one of --case <N>, --model <PATH>, or --input <PATH> is required \
             (they are mutually exclusive)"
        ),
    }
}

/// Runs the #3964 topology linter and enforces the exit-code contract:
/// 0 = clean, 1 = findings (errors always; warnings under `--strict`),
/// 2 = usage/internal errors.
pub fn handle_lint(args: LintArgs) -> Result<()> {
    let (graph, source) = match lint_graph_source(&args) {
        Ok(pair) => pair,
        Err(err) => {
            // Usage/internal errors exit 2 per the #3964 CLI contract (clap
            // itself already exits 2 for flag-conflict usage errors).
            eprintln!("error: {err:#}");
            std::io::stdout().flush().ok();
            std::process::exit(2);
        }
    };
    let report = lint_topology(&graph);
    let blocking = report.has_blocking(args.strict);
    match args.format {
        LintFormat::Json => print_lint_json(&report, &source, args.strict),
        LintFormat::Text => print_lint_text(&report, &source),
    }
    if blocking {
        // Findings: exit non-zero so CI can gate on the linter.
        std::io::stdout().flush().ok();
        std::process::exit(1);
    }
    Ok(())
}

fn severity_label(severity: LintSeverity) -> &'static str {
    match severity {
        LintSeverity::Error => "ERROR",
        LintSeverity::Warning => "WARNING",
    }
}

fn print_lint_text(report: &LintReport, source: &str) {
    println!("topology lint: {source}");
    println!(
        "  findings: {} ({} error(s), {} warning(s))",
        report.findings.len(),
        report.error_count(),
        report.warning_count(),
    );
    for finding in &report.findings {
        match &finding.node_id {
            Some(node) => println!(
                "{} [{}] node `{node}`: {}",
                finding.code,
                severity_label(finding.severity),
                finding.message
            ),
            None => println!(
                "{} [{}] {}",
                finding.code,
                severity_label(finding.severity),
                finding.message
            ),
        }
        if let Some(edge) = &finding.edge {
            println!("    edge: {} -> {}", edge.source, edge.target);
        }
    }
}

fn print_lint_json(report: &LintReport, source: &str, strict: bool) {
    let doc = serde_json::json!({
        "source": source,
        "strict": strict,
        "clean": report.is_clean(),
        "summary": {
            "total": report.findings.len(),
            "errors": report.error_count(),
            "warnings": report.warning_count(),
        },
        "findings": &report.findings,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&doc).expect("lint report serializes")
    );
}

#[cfg(test)]
mod lint_tests {
    use super::*;

    fn lint_args(case: Option<&str>) -> LintArgs {
        LintArgs {
            case: case.map(str::to_string),
            model: None,
            input: None,
            strict: false,
            format: LintFormat::Text,
        }
    }

    #[test]
    fn lint_rejects_missing_source() {
        let err = lint_graph_source(&lint_args(None)).unwrap_err().to_string();
        assert!(err.contains("exactly one of"), "got: {err}");
    }

    #[test]
    fn lint_rejects_unknown_case_id() {
        let err = lint_graph_source(&lint_args(Some("not-a-case")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown ASHRAE 140 case id"), "got: {err}");
    }

    #[test]
    fn lint_registry_case_graph_lints_clean() {
        let (graph, source) = lint_graph_source(&lint_args(Some("600"))).expect("case 600");
        assert_eq!(source, "ashrae-140-registry:600");
        let report = lint_topology(&graph);
        assert!(
            report.is_clean(),
            "case 600 should lint clean, got {:?}",
            report.findings
        );
    }
}
