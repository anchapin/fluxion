//! `fluxion topology` subcommand (Issue #3963) plus the backward-compatibility
//! alias `fluxion export-topology` (Issue #3966).
//!
//! `topology export` builds the in-memory simulation topology graph for a
//! model — either an ASHRAE 140 registry case (`--case N`) or a serialized
//! `CaseSpec` document (`--model <path>`) — finalizes, validates, and writes it
//! as deterministic JSON (identical inputs → byte-identical output).
//!
//! The handler deliberately mirrors the fail-loud CLI contract: mutually
//! exclusive source flags are enforced, unknown case ids error out non-zero,
//! and no path here silently succeeds (cf. issue #2947 stub policy).

use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand};

use crate::sim::topology::{ToTopologyGraph, TopologyContext};
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};

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
}

/// Dispatches `fluxion topology <command>`.
pub fn dispatch(command: TopologyCommand) -> Result<()> {
    match command {
        TopologyCommand::Export(args) => handle_export(args),
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

    let mut graph = spec.to_topology_graph(&ctx);
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
