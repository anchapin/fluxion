// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! EnergyPlus IDF (Input Data File) import — **MVP scaffold (issue #1341)**.
//!
//! This module parses the subset of IDF objects needed to drive Fluxion's
//! ASHRAE 140 validation harness and EnergyPlus model migration
//! (issue #778). The MVP covers the **10 object types** enumerated in
//! `docs/idf-import-design.md` §4.1:
//!
//! - `Version`
//! - `Timestep`
//! - `RunPeriod`
//! - `Building`
//! - `Zone`
//! - `Material`
//! - `Construction`
//! - `BuildingSurface:Detailed`
//! - `GlobalGeometryRules`
//! - `Site:GroundTemperature:BuildingSurface`
//!
//! All other object types are still **captured** into [`IdfObject`]s so
//! callers can inspect or forward them, but no typed accessor is provided
//! and they are ignored by [`IdfFile::materials`], [`IdfFile::zones`], etc.
//!
//! # Out of scope (per design §10 and issue #1341)
//!
//! - epJSON parsing (design §4.2 follow-up).
//! - HVAC, Schedule, Window/Door, `FenestrationSurface:Detailed` (design §10).
//! - IDF export (design §10).
//!
//! # Example
//!
//! ```ignore
//! use fluxion::io::idf::{IdfParser, IdfFile};
//!
//! let src = "Version, 25.2;\nTimestep, 1;\n";
//! let idf: IdfFile = IdfParser::from_str(src).expect("parses");
//! assert_eq!(idf.version.as_deref(), Some("25.2"));
//! assert_eq!(idf.objects.len(), 2);
//! ```

pub mod convert;
pub mod epjson;
pub mod error;
pub mod lexer;
pub mod parser;

pub use convert::{case_spec_from_idf, GroundTempMeta, RunPeriodMeta, SUPPORTED_VERSIONS};
pub use error::IdfError;
pub use lexer::{tokenize, RawObject};
pub use parser::{IdfFile, IdfObject, IdfParser, IdfValue};
