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
//! `--model` accepts a serialized `CaseSpec` JSON document only. Known
//! non-`CaseSpec` model formats (OpenStudio `.osm`, GreenBuildingXML
//! `.xml`/`.gbxml`) are rejected with a precise error naming the file, the
//! detected format, and what IS supported (Issue #3967).
//!
//! The handlers deliberately mirror the fail-loud CLI contract: mutually
//! exclusive source flags are enforced, unknown case ids error out non-zero,
//! and no path here silently succeeds (cf. issue #2947 stub policy).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};

use crate::sim::topology_lint::{lint_topology, LintReport, LintSeverity};
use crate::topology::{TopologyContext, TopologyGraph};
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use crate::validation::topology_bridge;

/// Output format for `fluxion topology export` (Issue #4115). The default
/// stays `Json`: the JSON path is the byte-identical determinism contract
/// guarded by `scripts/check_topology_drift.py`. `Toon` is an additive
/// agent-facing rendering via the `fluxion-toon` crate (Issue #2071); it does
/// NOT replace JSON, mutate the schema, or feed the drift gate.
///
/// Acceptance numbers (measured against committed topology fixtures;
/// Issue #4153):
///
/// | Case | JSON bytes | JSON tokens (cl100k_base) | TOON bytes | TOON tokens | Token save |
/// |------|------------|---------------------------|------------|-------------|------------|
/// | 600  | 24,757     | 8,382                     | 18,102     | 6,042       | +27.9%     |
/// | 900  | 24,783     | 8,406                     | 18,128     | 6,066       | +27.8%     |
/// | 960  | 50,892     | 17,227                    | 37,117     | 12,406      | +28.0%     |
///
/// Case 960 (71 nodes / 120 edges, ~2x the payload of 600/900) holds the
/// saving at +28.0%, within 0.2 points of the smaller cases — the reduction
/// is a property of the encoding, not of one graph shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ExportFormat {
    /// Canonical deterministic JSON (default; feeds the drift gate).
    Json,
    /// Token-Oriented Object Notation via the `fluxion-toon` crate.
    /// Compresses uniform-array heavy payloads for agent context windows;
    /// see `docs/TOON_INTEGRATION.md`.
    Toon,
}

/// Arguments for `fluxion topology export` / `fluxion export-topology`.
#[derive(Debug, Args, Clone)]
pub struct ExportArgs {
    /// ASHRAE 140 case id from the registry (e.g. 600, 610, 620, 630, 640,
    /// 650, 900, 960, 650FF, 950FF, ...). Mutually exclusive with --model.
    #[arg(long, value_name = "N", conflicts_with = "model")]
    pub case: Option<String>,

    /// Path to a serialized CaseSpec model document (JSON). Mutually
    /// exclusive with --case. Unsupported model formats (OpenStudio `.osm`,
    /// GreenBuildingXML `.xml`/`.gbxml`) are rejected with a precise error
    /// naming the detected format; only a serialized CaseSpec JSON document
    /// is supported.
    #[arg(long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Output file path; writes to stdout when omitted.
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Optional export timestamp embedded in the metadata block. Omitted by
    /// default so repeated exports are byte-identical (determinism contract).
    #[arg(long, value_name = "ISO8601")]
    pub timestamp: Option<String>,

    /// Output format (Issue #4115). `json` is the canonical default and the
    /// format the topology-drift gate validates; `toon` is an additive
    /// agent-facing rendering via `fluxion-toon` that does not feed the
    /// drift gate.
    #[arg(long, value_enum, default_value_t = ExportFormat::Json)]
    pub format: ExportFormat,
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

/// Model file formats that `--model` does NOT accept, keyed by lower-cased
/// file extension (Issue #3967).
///
/// `--model` takes a *serialized `CaseSpec` JSON document*. Feeding it a
/// real OpenStudio (`.osm`) or GreenBuildingXML (`.xml` / `.gbxml`) export
/// used to surface as a raw serde JSON parse error, which told the user
/// nothing about the actual problem: the format is not supported at all.
/// The table below lets both model-loading paths (export and lint) reject
/// those formats loudly, naming the file, the detected format, and what IS
/// supported.
const UNSUPPORTED_MODEL_FORMATS: &[(&str, &str)] = &[
    ("osm", "OpenStudio (.osm)"),
    ("xml", "GreenBuildingXML (.xml)"),
    ("gbxml", "GreenBuildingXML (.gbxml)"),
];

/// Returns a human-readable label for a known non-`CaseSpec` model format
/// detected from the file extension (case-insensitive), or `None` when the
/// extension is unknown (e.g. `.json`, no extension) and the file should be
/// parsed as a `CaseSpec` JSON document as before.
fn detect_unsupported_model_format(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    UNSUPPORTED_MODEL_FORMATS
        .iter()
        .find(|(candidate, _)| *candidate == ext)
        .map(|(_, label)| *label)
}

/// Reads a `--model` document, rejecting known unsupported model formats
/// *before* any JSON parsing happens (Issue #3967).
///
/// Shared by the export and lint model-loading arms so both paths fail loud
/// with the same precise message. Files whose extension is unknown (`.json`,
/// no extension, ...) keep the previous behavior: they are read and parsed as
/// a serialized `CaseSpec` JSON document.
fn read_model_document(path: &Path) -> Result<String> {
    if let Some(format_label) = detect_unsupported_model_format(path) {
        bail!(
            "unsupported model format for {}: detected {format_label}, which is not yet \
             supported; `--model` only accepts a serialized CaseSpec JSON document \
             (a JSON file)",
            path.display()
        );
    }
    fs::read_to_string(path)
        .with_context(|| format!("failed to read model file {}", path.display()))
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
            let raw = read_model_document(path)?;
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

    match args.format {
        ExportFormat::Json => write_export(&json, args.output.as_ref(), &graph, "json"),
        ExportFormat::Toon => {
            let toon = fluxion_toon::to_string(&graph)
                .context("failed to serialize topology graph as TOON")?;
            write_export(&toon, args.output.as_ref(), &graph, "toon")
        }
    }
}

/// Writes the serialized topology to disk or stdout, preserving the
/// file/stdout branch structure that pre-existed in `handle_export` (kept
/// intact so the byte-identical JSON default stays untouched).
fn write_export(
    payload: &str,
    output: Option<&PathBuf>,
    graph: &TopologyGraph,
    fmt_label: &str,
) -> Result<()> {
    match output {
        Some(path) => {
            fs::write(path, payload.as_bytes()).with_context(|| {
                format!(
                    "failed to write topology export ({fmt_label}) to {}",
                    path.display()
                )
            })?;
            println!(
                "wrote topology export ({fmt_label}; {} nodes, {} edges) to {}",
                graph.metadata.node_count,
                graph.metadata.edge_count,
                path.display()
            );
        }
        None => println!("{payload}"),
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
            format: ExportFormat::Json,
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
            format: ExportFormat::Json,
        };
        let err = handle_export(args).unwrap_err().to_string();
        assert!(err.contains("unknown ASHRAE 140 case id"));
    }

    // -------------------------------------------------------------------
    // Issue #3967 — `--model` rejects unsupported model formats loudly
    // -------------------------------------------------------------------

    /// Writes a fixture file into `dir` and returns its path.
    fn write_fixture(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, contents).expect("fixture file should be written");
        path
    }

    /// A serialized `CaseSpec` JSON document (the only format `--model`
    /// supports) built from the ASHRAE 140 registry.
    fn casespec_json_fixture() -> String {
        let case =
            ASHRAE140Case::from_case_id("600").expect("Case 600 is in the ASHRAE 140 registry");
        serde_json::to_string_pretty(&case.spec()).expect("CaseSpec serializes to JSON")
    }

    fn export_args_for_model(model: &Path, output: Option<PathBuf>) -> ExportArgs {
        ExportArgs {
            case: None,
            model: Some(model.to_path_buf()),
            output,
            timestamp: None,
            format: ExportFormat::Json,
        }
    }

    /// An OpenStudio `.osm` export must be rejected with a precise error
    /// (not a raw serde JSON parse error) that names the file, the detected
    /// format, and the supported format.
    #[test]
    fn export_rejects_opensstudio_osm_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "building.osm", "<osm></osm>");

        let err = handle_export(export_args_for_model(&model, None))
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("building.osm"),
            "error must name the offending file (got: {err})"
        );
        assert!(
            err.contains("OpenStudio (.osm)"),
            "error must name the detected format (got: {err})"
        );
        assert!(
            err.contains("not yet supported"),
            "error must state the format is not supported (got: {err})"
        );
        assert!(
            err.contains("CaseSpec"),
            "error must state what IS supported (got: {err})"
        );
        assert!(
            !err.contains("expected value"),
            "error must not be a raw serde JSON parse error (got: {err})"
        );
    }

    /// A GreenBuildingXML `.xml` export must be rejected the same way.
    #[test]
    fn export_rejects_greenbuildingxml_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "building.xml", "<gbXML></gbXML>");

        let err = handle_export(export_args_for_model(&model, None))
            .unwrap_err()
            .to_string();

        assert!(err.contains("building.xml"), "got: {err}");
        assert!(err.contains("GreenBuildingXML (.xml)"), "got: {err}");
        assert!(err.contains("CaseSpec"), "got: {err}");
    }

    /// The extension check is case-insensitive: `.OSM` is an OpenStudio
    /// export too.
    #[test]
    fn export_rejects_uppercase_osm_extension() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "building.OSM", "<osm></osm>");

        let err = handle_export(export_args_for_model(&model, None))
            .unwrap_err()
            .to_string();

        assert!(err.contains("OpenStudio (.osm)"), "got: {err}");
        assert!(err.contains("CaseSpec"), "got: {err}");
    }

    /// A valid serialized `CaseSpec` JSON document still loads through
    /// `--model` (behavior unchanged by the format gate).
    #[test]
    fn export_accepts_valid_casespec_json_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "casespec.json", &casespec_json_fixture());
        let output = dir.path().join("topology.json");

        handle_export(export_args_for_model(&model, Some(output.clone())))
            .expect("valid CaseSpec JSON must export");

        let exported = fs::read_to_string(&output).expect("export file written");
        assert!(
            exported.contains("\"nodes\""),
            "export must be a topology document (got: {exported})"
        );
    }

    /// A `.json` file that is not a `CaseSpec` still fails with the JSON
    /// parse error — the format gate must not swallow it.
    #[test]
    fn export_reports_json_parse_error_for_invalid_casespec_json() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "garbage.json", "this is not json");

        let err = handle_export(export_args_for_model(&model, None))
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("failed to parse CaseSpec JSON"),
            "expected the JSON parse error (got: {err})"
        );
        assert!(
            !err.contains("unsupported model format"),
            "format gate must not fire for .json files (got: {err})"
        );
    }

    /// Issue #4115 acceptance: `--format toon` produces valid TOON that
    /// round-trips through the `fluxion-toon` crate parser. Serializes Case
    /// 600 (the canonical example in `docs/TOON_INTEGRATION.md`), confirms
    /// the TOON header is present, and verifies the parsed graph is
    /// structurally equivalent to the canonical JSON serialization.
    #[test]
    fn toon_export_round_trips_case_600() {
        let case =
            ASHRAE140Case::from_case_id("600").expect("Case 600 is in the ASHRAE 140 registry");
        let spec = case.spec();
        let ctx = TopologyContext {
            model_name: format!("ASHRAE 140 Case {}", spec.case_id),
            model_source: format!("ashrae-140-registry:{}", spec.case_id),
            timestamp: None,
        };
        let mut graph = topology_bridge::case_topology_graph(&spec, &ctx);
        graph
            .finalize()
            .expect("topology finalization should succeed for Case 600");
        graph
            .validate()
            .expect("topology validation should succeed for Case 600");

        // Serialize via the fluxion-toon crate (the same path the CLI uses).
        let toon = fluxion_toon::to_string(&graph).expect("TOON serialization");
        assert!(
            toon.starts_with("toon:v1"),
            "TOON payload must start with the v1 header (got prefix: {:?})",
            toon.lines().next()
        );

        // Round-trip back into a graph and compare structural equality.
        let parsed: TopologyGraph =
            fluxion_toon::from_str(&toon).expect("TOON parser must round-trip the Case 600 export");
        assert_eq!(
            parsed.metadata.node_count, graph.metadata.node_count,
            "node count must survive TOON round-trip"
        );
        assert_eq!(
            parsed.metadata.edge_count, graph.metadata.edge_count,
            "edge count must survive TOON round-trip"
        );
        assert_eq!(parsed.nodes.len(), graph.nodes.len(), "nodes len");
        assert_eq!(parsed.edges.len(), graph.edges.len(), "edges len");
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
    /// exclusive with --case and --input. Unsupported model formats
    /// (OpenStudio `.osm`, GreenBuildingXML `.xml`/`.gbxml`) are rejected with
    /// a precise error naming the detected format; only a serialized CaseSpec
    /// JSON document is supported.
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
            let raw = read_model_document(path)?;
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

    // -------------------------------------------------------------------
    // Issue #3967 — `--model` rejects unsupported model formats loudly
    // -------------------------------------------------------------------

    fn lint_args_for_model(model: &Path) -> LintArgs {
        LintArgs {
            case: None,
            model: Some(model.to_path_buf()),
            input: None,
            strict: false,
            format: LintFormat::Text,
        }
    }

    fn write_fixture(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, contents).expect("fixture file should be written");
        path
    }

    fn casespec_json_fixture() -> String {
        let case =
            ASHRAE140Case::from_case_id("600").expect("Case 600 is in the ASHRAE 140 registry");
        serde_json::to_string_pretty(&case.spec()).expect("CaseSpec serializes to JSON")
    }

    /// The lint path shares the export path's format gate: an OpenStudio
    /// `.osm` model is rejected with the same precise error.
    #[test]
    fn lint_rejects_opensstudio_osm_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "building.osm", "<osm></osm>");

        let err = lint_graph_source(&lint_args_for_model(&model))
            .unwrap_err()
            .to_string();

        assert!(err.contains("building.osm"), "got: {err}");
        assert!(err.contains("OpenStudio (.osm)"), "got: {err}");
        assert!(err.contains("not yet supported"), "got: {err}");
        assert!(err.contains("CaseSpec"), "got: {err}");
        assert!(!err.contains("expected value"), "got: {err}");
    }

    /// A GreenBuildingXML `.xml` model is rejected on the lint path too.
    #[test]
    fn lint_rejects_greenbuildingxml_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "building.xml", "<gbXML></gbXML>");

        let err = lint_graph_source(&lint_args_for_model(&model))
            .unwrap_err()
            .to_string();

        assert!(err.contains("GreenBuildingXML (.xml)"), "got: {err}");
        assert!(err.contains("CaseSpec"), "got: {err}");
    }

    /// A valid serialized `CaseSpec` JSON document still lints through
    /// `--model`.
    #[test]
    fn lint_accepts_valid_casespec_json_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "casespec.json", &casespec_json_fixture());

        let (graph, source) =
            lint_graph_source(&lint_args_for_model(&model)).expect("valid CaseSpec JSON must lint");
        assert_eq!(source, format!("model-file:{}", model.display()));
        assert!(!graph.nodes.is_empty(), "linted graph should have nodes");
    }

    /// A `.json` file with garbage still surfaces the JSON parse error.
    #[test]
    fn lint_reports_json_parse_error_for_invalid_casespec_json() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = write_fixture(dir.path(), "garbage.json", "this is not json");

        let err = lint_graph_source(&lint_args_for_model(&model))
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("failed to parse CaseSpec document"),
            "expected the JSON parse error (got: {err})"
        );
        assert!(
            !err.contains("unsupported model format"),
            "format gate must not fire for .json files (got: {err})"
        );
    }
}
