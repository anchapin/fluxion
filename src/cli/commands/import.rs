// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `fluxion import` CLI subcommand (issue #1900).
//!
//! Parses an EnergyPlus IDF or epJSON file and converts it to a
//! [`SimulationSchemaV1`] JSON document, written to `--output` or stdout.
//!
//! ```text
//! fluxion import model.idf
//! fluxion import model.epjson -o schema.json
//! ```

use std::convert::TryFrom;
use std::io::Write;
use std::path::PathBuf;

use clap::Args;

use crate::api::schema::SimulationSchemaV1;
use crate::io::idf::{IdfError, IdfParser};

/// Import an EnergyPlus IDF or epJSON model into Fluxion's SimulationSchemaV1 format.
#[derive(Args, Debug)]
pub struct ImportCommand {
    /// Path to the input file (.idf or .epjson / .epJSON).
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output file path. If omitted, JSON is written to stdout.
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<PathBuf>,
}

/// Errors specific to the import CLI command.
#[derive(Debug)]
pub enum ImportError {
    /// The file extension was not `.idf` or `.epjson` / `.epJSON`.
    UnsupportedFormat(String),
    /// Wraps an [`IdfError`] from the parser or converter.
    Idf(IdfError),
    /// JSON serialization failed.
    Serialize(serde_json::Error),
    /// Writing to the output file or stdout failed.
    Write(std::io::Error),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::UnsupportedFormat(ext) => {
                write!(f, "unsupported file extension '{ext}': expected .idf or .epjson/.epJSON")
            }
            ImportError::Idf(e) => write!(f, "{e}"),
            ImportError::Serialize(e) => write!(f, "JSON serialization failed: {e}"),
            ImportError::Write(e) => write!(f, "write failed: {e}"),
        }
    }
}

impl std::error::Error for ImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ImportError::Idf(e) => Some(e),
            ImportError::Serialize(e) => Some(e),
            ImportError::Write(e) => Some(e),
            ImportError::UnsupportedFormat(_) => None,
        }
    }
}

impl From<IdfError> for ImportError {
    fn from(e: IdfError) -> Self {
        ImportError::Idf(e)
    }
}

impl From<serde_json::Error> for ImportError {
    fn from(e: serde_json::Error) -> Self {
        ImportError::Serialize(e)
    }
}

impl From<std::io::Error> for ImportError {
    fn from(e: std::io::Error) -> Self {
        ImportError::Write(e)
    }
}

/// Determine the input format from the file extension.
fn detect_format(path: &std::path::Path) -> Result<InputFormat, ImportError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "idf" => Ok(InputFormat::Idf),
        "epjson" => Ok(InputFormat::Epjson),
        other => Err(ImportError::UnsupportedFormat(other.to_string())),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputFormat {
    Idf,
    Epjson,
}

/// Execute the `fluxion import` command.
pub fn execute_import(cmd: &ImportCommand) -> Result<(), ImportError> {
    let format = detect_format(&cmd.input)?;

    let idf = match format {
        InputFormat::Idf => IdfParser::from_path(&cmd.input)?,
        InputFormat::Epjson => IdfParser::from_epjson_path(&cmd.input)?,
    };

    let schema = SimulationSchemaV1::try_from(&idf)?;
    let json = serde_json::to_string_pretty(&schema)?;

    match &cmd.output {
        Some(out_path) => {
            let mut file = std::fs::File::create(out_path)?;
            file.write_all(json.as_bytes())?;
            file.write_all(b"\n")?;
            eprintln!("Wrote SimulationSchemaV1 to {}", out_path.display());
        }
        None => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(json.as_bytes())?;
            handle.write_all(b"\n")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn detect_format_idf() {
        assert_eq!(
            detect_format(Path::new("model.idf")).unwrap(),
            InputFormat::Idf
        );
    }

    #[test]
    fn detect_format_epjson() {
        assert_eq!(
            detect_format(Path::new("model.epjson")).unwrap(),
            InputFormat::Epjson
        );
    }

    #[test]
    fn detect_format_epjson_uppercase() {
        // .epJSON extension (as seen in EnergyPlus distribution)
        assert_eq!(
            detect_format(Path::new("model.epJSON")).unwrap(),
            InputFormat::Epjson
        );
    }

    #[test]
    fn detect_format_unsupported() {
        let err = detect_format(Path::new("model.txt")).unwrap_err();
        match err {
            ImportError::UnsupportedFormat(ext) => assert_eq!(ext, "txt"),
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn import_idf_produces_valid_schema() {
        let cmd = ImportCommand {
            input: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/reference_data/energyplus_models/ashrae_140_case_600.idf"),
            output: None,
        };
        // We can't capture stdout in a unit test easily, so just verify the
        // parsing + conversion path succeeds.
        let idf = IdfParser::from_path(&cmd.input).expect("IDF parses");
        let schema = SimulationSchemaV1::try_from(&idf).expect("converts");
        assert_eq!(schema.geometry.zones.len(), 1);
        assert!((schema.geometry.zones[0].floor_area - 48.0).abs() < 1e-3);
    }

    #[test]
    fn import_epjson_produces_valid_schema() {
        let cmd = ImportCommand {
            input: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/reference_data/energyplus_models/case900.epJSON"),
            output: None,
        };
        let idf = IdfParser::from_epjson_path(&cmd.input).expect("epJSON parses");
        let schema = SimulationSchemaV1::try_from(&idf).expect("converts");
        // The case900.epJSON fixture only has Version + Building (no Zone),
        // so just verify the schema was produced successfully.
        assert_eq!(schema.version, crate::api::schema::SchemaVersion::V1);
    }

    #[test]
    fn import_missing_file_returns_io_error() {
        let cmd = ImportCommand {
            input: PathBuf::from("/nonexistent/model.idf"),
            output: None,
        };
        let err = execute_import(&cmd).unwrap_err();
        assert!(matches!(err, ImportError::Idf(IdfError::Io(_))));
    }

    #[test]
    fn import_unsupported_version_returns_error() {
        // IDF with unsupported version — parser succeeds but converter rejects it.
        let tmp = tempfile::tempdir().unwrap();
        let bad_idf = tmp.path().join("bad.idf");
        std::fs::write(&bad_idf, "Version, 99.9;\n").unwrap();

        let err = execute_import(&ImportCommand {
            input: bad_idf,
            output: None,
        })
        .unwrap_err();
        assert!(
            matches!(
                err,
                ImportError::Idf(IdfError::UnsupportedVersion(_))
            ),
            "expected UnsupportedVersion, got: {err}"
        );
    }

    #[test]
    fn import_writes_to_file() {
        let tmp = tempfile::tempdir().unwrap();
        let out_path = tmp.path().join("schema.json");

        let cmd = ImportCommand {
            input: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/reference_data/energyplus_models/ashrae_140_case_600.idf"),
            output: Some(out_path.clone()),
        };
        execute_import(&cmd).expect("import succeeds");

        let content = std::fs::read_to_string(&out_path).unwrap();
        let schema: SimulationSchemaV1 = serde_json::from_str(&content).unwrap();
        assert_eq!(schema.geometry.zones.len(), 1);
    }
}
