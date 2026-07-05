// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Generic input/output adapters — file format bridges that don't fit
//! under [`crate::interop`] (which is reserved for the named ecosystem
//! formats: OSM, gbXML, FMI, ...).
//!
//! Currently the only child is [`idf`] — the EnergyPlus IDF/epJSON import
//! scaffold from `docs/idf-import-design.md` (issue #1341).

pub mod idf;
