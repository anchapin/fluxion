// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Parser size/depth/repetition limits — DoS hardening (issue #2527).
//!
//! The OSM/IFC/IDF/EPW/epJSON/gbXML parsers and the fluxion-toon
//! deserializer previously accepted unbounded input: no file-size cap,
//! no line-count cap, no recursion-depth cap, no array-element cap. A
//! malicious or malformed input (Billion-Laughs style expansion,
//! zip-bomb style repetition, or a pathologically deep nesting) could
//! exhaust memory or blow the stack before any physics code ran.
//!
//! This module is an intentionally **dependency-light leaf**: it pulls
//! in only `std` and the already-crated `thiserror` macro. It must NOT
//! import `crate::sim::*` / `crate::physics::*` / `crate::ai::*` /
//! `crate::validation::*` — the cycle-regression guard
//! `scripts/check_ashrae_cases_cycle.py` enforces that.
//!
//! # Defaults
//!
//! | Limit              | Default (`default`/`http_default`) | `cli_default` |
//! |--------------------|------------------------------------|---------------|
//! | `max_file_bytes`   | 64 MiB                             | 1 GiB         |
//! | `max_lines`        | 1,000,000                          | 10,000,000    |
//! | `max_recursion_depth` | 256                             | 1,024         |
//! | `max_array_elements` | 1,000,000                        | 10,000,000    |
//!
//! The HTTP import handlers (`POST /v1/import/{osm|gbxml|idf|epjson}`)
//! are additionally bounded by the 16 MiB `DefaultBodyLimit` set in
//! #2505, which binds *before* the parser runs. The 64 MiB parser cap
//! therefore primarily protects the **in-process** `from_str` paths
//! (`BatchOracle`, `fluxion-mcp` tools), which do not pass through the
//! HTTP body limit.

use thiserror::Error;

/// 64 MiB — the default per-file byte cap (issue #2527 §acceptance a).
pub const DEFAULT_MAX_FILE_BYTES: usize = 64 * 1024 * 1024;
/// 1,000,000 — the default line-count cap (issue #2527 §acceptance b).
pub const DEFAULT_MAX_LINES: usize = 1_000_000;
/// 256 — the default recursion/nesting-depth cap (issue #2527 §acceptance c).
pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 256;
/// 1,000,000 — the default array-element cap (issue #2527 §acceptance d).
pub const DEFAULT_MAX_ARRAY_ELEMENTS: usize = 1_000_000;

/// 1 GiB — the relaxed file cap for trusted CLI/in-process ingestion.
pub const CLI_MAX_FILE_BYTES: usize = 1024 * 1024 * 1024;
/// Relaxed line-count cap for trusted CLI/in-process ingestion.
pub const CLI_MAX_LINES: usize = 10_000_000;
/// Relaxed recursion-depth cap for trusted CLI/in-process ingestion.
pub const CLI_MAX_RECURSION_DEPTH: usize = 1_024;
/// Relaxed array-element cap for trusted CLI/in-process ingestion.
pub const CLI_MAX_ARRAY_ELEMENTS: usize = 10_000_000;

/// Configuration bundle for the four parser DoS limits.
///
/// Construct via [`ParserLimits::default`] (strict, 64 MiB),
/// [`ParserLimits::http_default`] (alias for `default`), or
/// [`ParserLimits::cli_default`] (relaxed, 1 GiB). Each parser entry
/// point exposes a `_with_limits` companion that accepts a `&ParserLimits`;
/// the no-suffix entry points delegate with [`ParserLimits::default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParserLimits {
    /// Maximum accepted input size in bytes.
    pub max_file_bytes: usize,
    /// Maximum accepted number of lines (or line-equivalent records).
    pub max_lines: usize,
    /// Maximum accepted nesting/recursion depth.
    pub max_recursion_depth: usize,
    /// Maximum accepted number of elements in a single aggregate/array.
    pub max_array_elements: usize,
}

impl ParserLimits {
    /// Strict defaults (64 MiB / 1M lines / 256 depth / 1M elements).
    ///
    /// These guard the in-process `from_str` paths that do not pass
    /// through the HTTP body limit.
    pub const fn default_const() -> Self {
        ParserLimits {
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_lines: DEFAULT_MAX_LINES,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            max_array_elements: DEFAULT_MAX_ARRAY_ELEMENTS,
        }
    }

    /// Strict defaults for the HTTP import handlers.
    ///
    /// Equivalent to [`ParserLimits::default`]; the 16 MiB HTTP body
    /// limit (#2505) binds first for request bodies, while these caps
    /// still catch a pathologically line-dense input that fits in 16 MiB.
    pub const fn http_default() -> Self {
        Self::default_const()
    }

    /// Relaxed defaults for trusted CLI/in-process ingestion
    /// (1 GiB file / 10M lines / 1024 depth / 10M elements).
    ///
    /// Use this for `BatchOracle` / `fluxion-mcp` paths that read local
    /// files outside the HTTP request path.
    pub const fn cli_default() -> Self {
        ParserLimits {
            max_file_bytes: CLI_MAX_FILE_BYTES,
            max_lines: CLI_MAX_LINES,
            max_recursion_depth: CLI_MAX_RECURSION_DEPTH,
            max_array_elements: CLI_MAX_ARRAY_ELEMENTS,
        }
    }

    /// Reject if `len` exceeds [`ParserLimits::max_file_bytes`].
    pub fn check_file_bytes(&self, len: usize) -> Result<(), ParseLimitError> {
        if len > self.max_file_bytes {
            return Err(ParseLimitError::TooLarge {
                limit: LimitKind::FileBytes,
                max: self.max_file_bytes,
                actual: len,
            });
        }
        Ok(())
    }

    /// Reject if `count` exceeds [`ParserLimits::max_lines`].
    pub fn check_lines(&self, count: usize) -> Result<(), ParseLimitError> {
        if count > self.max_lines {
            return Err(ParseLimitError::TooLarge {
                limit: LimitKind::Lines,
                max: self.max_lines,
                actual: count,
            });
        }
        Ok(())
    }

    /// Reject if `depth` exceeds [`ParserLimits::max_recursion_depth`].
    pub fn check_recursion_depth(&self, depth: usize) -> Result<(), ParseLimitError> {
        if depth > self.max_recursion_depth {
            return Err(ParseLimitError::TooLarge {
                limit: LimitKind::RecursionDepth,
                max: self.max_recursion_depth,
                actual: depth,
            });
        }
        Ok(())
    }

    /// Reject if `count` exceeds [`ParserLimits::max_array_elements`].
    pub fn check_array_elements(&self, count: usize) -> Result<(), ParseLimitError> {
        if count > self.max_array_elements {
            return Err(ParseLimitError::TooLarge {
                limit: LimitKind::ArrayElements,
                max: self.max_array_elements,
                actual: count,
            });
        }
        Ok(())
    }
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self::default_const()
    }
}

/// Which configured limit was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    /// Input byte length (`max_file_bytes`).
    FileBytes,
    /// Line / record count (`max_lines`).
    Lines,
    /// Nesting / recursion depth (`max_recursion_depth`).
    RecursionDepth,
    /// Aggregate element count (`max_array_elements`).
    ArrayElements,
}

impl LimitKind {
    pub const fn label(self) -> &'static str {
        match self {
            LimitKind::FileBytes => "file size (bytes)",
            LimitKind::Lines => "line count",
            LimitKind::RecursionDepth => "recursion depth",
            LimitKind::ArrayElements => "array element count",
        }
    }
}

/// Error returned when a parser limit is exceeded.
///
/// The single [`ParseLimitError::TooLarge`] variant carries which limit
/// was hit, its configured maximum, and the observed value so callers
/// can surface a precise diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseLimitError {
    /// A configured upper bound was exceeded — the parser aborted before
    /// allocating the unbounded structure. This is the DoS guard firing.
    #[error("parser limit exceeded: {} (limit {}, got {})", .limit.label(), .max, .actual)]
    TooLarge {
        /// Which limit was exceeded.
        limit: LimitKind,
        /// The configured maximum.
        max: usize,
        /// The observed value.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_constants_match_acceptance_criteria() {
        assert_eq!(DEFAULT_MAX_FILE_BYTES, 64 * 1024 * 1024);
        assert_eq!(DEFAULT_MAX_LINES, 1_000_000);
        assert_eq!(DEFAULT_MAX_RECURSION_DEPTH, 256);
        assert_eq!(DEFAULT_MAX_ARRAY_ELEMENTS, 1_000_000);
        assert!(DEFAULT_MAX_FILE_BYTES > 16 * 1024 * 1024); // above HTTP body limit
    }

    #[test]
    fn cli_default_relaxes_every_limit() {
        let cli = ParserLimits::cli_default();
        let strict = ParserLimits::default();
        assert!(cli.max_file_bytes > strict.max_file_bytes);
        assert!(cli.max_lines > strict.max_lines);
        assert!(cli.max_recursion_depth > strict.max_recursion_depth);
        assert!(cli.max_array_elements > strict.max_array_elements);
        assert_eq!(cli.max_file_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn check_methods_pass_at_limit_and_fail_above() {
        let lim = ParserLimits::default();
        // At-limit is allowed (strictly-greater-than triggers).
        assert!(lim.check_file_bytes(DEFAULT_MAX_FILE_BYTES).is_ok());
        assert!(lim.check_lines(DEFAULT_MAX_LINES).is_ok());
        assert!(lim
            .check_recursion_depth(DEFAULT_MAX_RECURSION_DEPTH)
            .is_ok());
        assert!(lim.check_array_elements(DEFAULT_MAX_ARRAY_ELEMENTS).is_ok());
        // One over triggers TooLarge.
        assert!(lim.check_file_bytes(DEFAULT_MAX_FILE_BYTES + 1).is_err());
        assert!(lim.check_lines(DEFAULT_MAX_LINES + 1).is_err());
        assert!(lim
            .check_recursion_depth(DEFAULT_MAX_RECURSION_DEPTH + 1)
            .is_err());
        assert!(lim
            .check_array_elements(DEFAULT_MAX_ARRAY_ELEMENTS + 1)
            .is_err());
    }

    #[test]
    fn too_large_carries_limit_kind_and_values() {
        let lim = ParserLimits {
            max_file_bytes: 10,
            max_lines: 5,
            max_recursion_depth: 3,
            max_array_elements: 2,
        };
        let err = lim.check_file_bytes(42).unwrap_err();
        assert_eq!(
            err,
            ParseLimitError::TooLarge {
                limit: LimitKind::FileBytes,
                max: 10,
                actual: 42,
            }
        );
        let msg = err.to_string();
        assert!(msg.contains("file size"));
        assert!(msg.contains("42"));
    }

    #[test]
    fn small_thresholds_usable_for_tests() {
        // Downstream parser unit tests construct tiny limits to exercise
        // the rejection path without synthesising multi-MiB inputs.
        let tiny = ParserLimits {
            max_file_bytes: 100,
            max_lines: 4,
            max_recursion_depth: 2,
            max_array_elements: 3,
        };
        assert!(tiny.check_file_bytes(101).is_err());
        assert!(tiny.check_lines(5).is_err());
        assert!(tiny.check_recursion_depth(3).is_err());
        assert!(tiny.check_array_elements(4).is_err());
    }
}
