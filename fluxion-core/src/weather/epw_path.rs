//! EPW file path resolution via environment variable.
//!
//! ASHRAE 140 validation requires real TMY3 weather files which are not
//! committed to the repository (see `.gitignore`).  Before running the full
//! ASHRAE 140 test suite, download the required EPW files:
//!
//! ```bash
//! python3 scripts/fetch_ashrae140_epw.py
//! export FLUXION_EPW_DIR=<repo-root>/assets/weather
//! ```
//!
//! This module provides:
//! - `resolve_epw_path()` — resolve a bare filename to an absolute path using
//!   `FLUXION_EPW_DIR` (falls back to `assets/weather/` relative to the crate root)
//! - `epw_required()` — panics with a helpful message if the file is absent
//! - `epw_optional()` — returns `None` if the file is absent (for tests that
//!   should `#[ignore]` when files are missing)

use std::path::PathBuf;

/// Resolve an EPW filename to an absolute path.
///
/// Resolution order:
/// 1. `FLUXION_EPW_DIR` env var (if set) — use that directory
/// 2. `assets/weather/` relative to the crate root (default)
///
/// # Arguments
///
/// * `filename` — bare EPW filename, e.g. `"USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw"`
///
/// # Example
///
/// ```
/// use fluxion_core::weather::epw_path::resolve_epw_path;
///
/// let path = resolve_epw_path("WD600.epw");
/// if let Some(p) = path {
///     println!("EPW file: {}", p.display());
/// }
/// ```
pub fn resolve_epw_path(filename: &str) -> Option<PathBuf> {
    let epw_dir = if let Ok(dir) = std::env::var("FLUXION_EPW_DIR") {
        PathBuf::from(dir)
    } else {
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_root.join("assets").join("weather")
    };

    let path = epw_dir.join(filename);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Resolve an EPW path and panic with a helpful message if absent.
///
/// Use this in **non-test** code where an EPW file is a hard requirement.
///
/// # Arguments
///
/// * `filename` — bare EPW filename
///
/// # Panics
///
/// Panics with a message explaining how to download the missing file.
pub fn epw_required(filename: &str) -> PathBuf {
    resolve_epw_path(filename).unwrap_or_else(|| {
        panic!(
            "\
EPW file not found: {}

ASHRAE 140 validation requires real weather files.
Run the download script:

    python3 scripts/fetch_ashrae140_epw.py

Then set the environment variable:

    export FLUXION_EPW_DIR=$(pwd)/assets/weather

(Or place EPW files in assets/weather/ relative to the crate root.)",
            filename
        )
    })
}

/// Resolve an EPW path, returning `None` if absent.
///
/// Use this in test code when the test should `#[ignore]` if the file is missing.
pub fn epw_optional(filename: &str) -> Option<PathBuf> {
    resolve_epw_path(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_epw_path_returns_none_for_missing_file() {
        let result = resolve_epw_path("nonexistent_file_12345.epw");
        assert!(result.is_none());
    }

    #[test]
    fn test_epw_optional_returns_none_for_missing() {
        let result = epw_optional("nonexistent_file_12345.epw");
        assert!(result.is_none());
    }
}
