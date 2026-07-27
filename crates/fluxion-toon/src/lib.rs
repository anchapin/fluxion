//! # fluxion-toon
//!
//! **Token-Oriented Object Notation (TOON)** — a compact, tabular serialization
//! format optimized for LLM context-window efficiency in MCP tool responses.
//!
//! TOON reduces token usage by 35–50% compared to JSON by collapsing uniform
//! flat-struct arrays into CSV-style blocks with explicit count headers.
//!
//! ## When to use TOON
//!
//! TOON is ideal for:
//! - Uniform arrays of primitive values (temperatures, conductances, schedules)
//! - Flat parameter structs in MCP tool responses
//! - High-cardinality lists where field names repeat
//!
//! ## When NOT to use TOON
//!
//! - Numerical solvers (CTF coefficients, FD internal state)
//! - Multi-node thermal mass state
//! - Hand-edited configuration files
//! - Nested or recursive data structures
//! - Any context where JSON readability is preferred
//!
//! ## Format Example
//!
//! ```text
//! # 3 uniform temperature records
//! @temp_c[3]
//! 22.5, 23.1, 21.8
//!
//! # 5 conductance values in W/K
//! @conductance_WK[5]
//! 150.2, 98.7, 203.4, 175.0, 89.3
//! ```
//!
//! ## Crate Structure
//!
//! | Module   | Purpose |
//! |----------|---------|
//! | `error`  | `ToonError` type, length mismatch guardrails |
//! | `ser`    | Serde `Serializer` impl, CSV collapse logic |
//! | `de`     | `winnow`-based deserializer, length validation |
//! | `patch`  | LLM response parser, markdown codeblock stripping |
//!
//! ## References
//!
//! - Issue [#2066](https://github.com/anchapin/fluxion/issues/2066): TOON format specification
//! - Issue [#2070](https://github.com/anchapin/fluxion/issues/2070): This documentation

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod de;
pub mod error;
pub mod patch;
pub mod ser;

pub use de::deserialize_from_str as from_str;
pub use error::ToonError;
pub use ser::serialize_to_string as to_string;
