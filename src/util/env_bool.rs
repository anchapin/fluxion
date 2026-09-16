//! Unified boolean environment-variable parsing (Issue #3751).
//!
//! Before this module, boolean env vars used at least three incompatible
//! ad-hoc conventions:
//!
//! - `ASHRAE_140_DEBUG` / `ASHRAE_140_HOURLY_OUTPUT` / `ASHRAE_140_VERBOSE`
//!   were true only for exactly `"1"` or case-insensitive `"true"` — an
//!   operator setting `ASHRAE_140_DEBUG=yes` during a validation
//!   investigation silently got no diagnostics and no signal why.
//! - `FLUXION_GPU` used the opposite polarity: `0|false|<empty>` was the
//!   falsy set and everything else (including typos) was truthy.
//! - `FLUXION_REST_ALLOW_INSECURE` accepted `1|true|yes|on` (trimmed,
//!   case-sensitive) and silently ignored everything else.
//!
//! This module is the single canonical parser every boolean env read goes
//! through, so a variable's truthiness semantics are predictable from its
//! `FLUXION_*` / `ASHRAE_140_*` prefix.
//!
//! # Canonical convention
//!
//! Tokens are matched **case-insensitively** after trimming surrounding
//! whitespace:
//!
//! | Truthy (`true`)     | Falsy (`false`)      |
//! |---------------------|----------------------|
//! | `1`, `true`, `yes`, `on` | `0`, `false`, `no`, `off`, `` (empty) |
//!
//! Any other value (e.g. the typo `ture`) is *unrecognized*: the reader
//! emits a one-line `tracing::warn!` naming the variable, its raw value,
//! and the accepted token sets, then behaves as if the variable were
//! **unset** (falling back to the call site's unset default). Opt-in
//! flags therefore stay off and fail closed, while opt-out bypasses such
//! as `FLUXION_GPU` keep their documented unset behavior — loudly instead
//! of silently.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluxion::util::env_bool::env_bool;
//!
//! // Opt-in flag: disabled unless explicitly enabled.
//! let debug = env_bool("ASHRAE_140_DEBUG", false);
//!
//! // Opt-out bypass: GPU honored unless explicitly disabled.
//! let gpu_honored = env_bool("FLUXION_GPU", true);
//! ```

/// Parse a raw env-var value against the canonical truthy/falsy token sets.
///
/// Truthy (case-insensitive, whitespace-trimmed): `1` | `true` | `yes` | `on`.
/// Falsy: `0` | `false` | `no` | `off` | `` (empty string).
/// Returns `None` for unrecognized values so callers can warn (see
/// [`env_bool`]); the empty string is deliberately falsy to preserve the
/// documented `FLUXION_GPU=0|false|<empty>` bypass contract.
pub fn parse_env_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" | "" => Some(false),
        _ => None,
    }
}

/// Read the environment variable `name` as a boolean using the canonical
/// [`parse_env_bool`] token sets (Issue #3751).
///
/// - **Unset** → returns `default_when_unset`.
/// - **Recognized token** → returns the parsed value.
/// - **Unrecognized token** (e.g. a typo like `ture`) → emits a
///   `tracing::warn!` (target `fluxion::util::env_bool`) naming the
///   variable, its raw value, the accepted token sets, and the fallback
///   value actually used — then returns `default_when_unset`, i.e. the
///   variable is treated as unset *loudly*, never silently.
///
/// Call sites should pass the variable's documented unset default:
/// `false` for opt-in flags (diagnostics, experimental gates, insecure
/// escape hatches — fail closed on typos), `true` for opt-out bypasses
/// such as `FLUXION_GPU` (honored unless explicitly disabled).
pub fn env_bool(name: &str, default_when_unset: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => match parse_env_bool(&raw) {
            Some(value) => value,
            None => {
                tracing::warn!(
                    target: "fluxion::util::env_bool",
                    env_var = name,
                    value = %raw,
                    fallback = default_when_unset,
                    accepted = "1|true|yes|on (enabled) or 0|false|no|off|<empty> (disabled)",
                    "unrecognized boolean env-var value; treating it as unset",
                );
                default_when_unset
            }
        },
        Err(_) => default_when_unset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared mutex serializing env mutation in this module (same convention
    /// as `src/ai/surrogate.rs` / `src/api/schema.rs`): without it, parallel
    /// `cargo test` threads stomp on each other's env state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Scratch variable name for exercising [`env_bool`] read behavior;
    /// deliberately outside the `FLUXION_*`/`ASHRAE_140_*` namespaces so no
    /// production code path can observe it.
    const PROBE_VAR: &str = "FLUXION_TEST_ENV_BOOL_PROBE";

    /// Acceptance test (Issue #3751): pin the parser for
    /// `1/0/true/false/empty/yes` plus `no`/`off`/`on`, case-insensitivity,
    /// whitespace trimming, and the unrecognized-typo case.
    #[test]
    fn parse_env_bool_pins_canonical_token_sets() {
        // Truthy tokens.
        assert_eq!(parse_env_bool("1"), Some(true));
        assert_eq!(parse_env_bool("true"), Some(true));
        assert_eq!(parse_env_bool("yes"), Some(true));
        assert_eq!(parse_env_bool("on"), Some(true));
        // Falsy tokens — empty is falsy (FLUXION_GPU `<empty>` bypass).
        assert_eq!(parse_env_bool("0"), Some(false));
        assert_eq!(parse_env_bool("false"), Some(false));
        assert_eq!(parse_env_bool("no"), Some(false));
        assert_eq!(parse_env_bool("off"), Some(false));
        assert_eq!(parse_env_bool(""), Some(false));
        // Typo: unrecognized, never silently true or false.
        assert_eq!(parse_env_bool("ture"), None);
        // Case-insensitive matching.
        assert_eq!(parse_env_bool("TRUE"), Some(true));
        assert_eq!(parse_env_bool("Yes"), Some(true));
        assert_eq!(parse_env_bool("FALSE"), Some(false));
        assert_eq!(parse_env_bool("Off"), Some(false));
        // Surrounding whitespace is trimmed.
        assert_eq!(parse_env_bool("  1  "), Some(true));
        assert_eq!(parse_env_bool(" false "), Some(false));
        // A typo with mixed case stays unrecognized.
        assert_eq!(parse_env_bool("Ture"), None);
    }

    /// Unset variable must return the call-site default for both polarities.
    #[test]
    fn env_bool_unset_returns_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(PROBE_VAR).ok();
        std::env::remove_var(PROBE_VAR);

        assert!(!env_bool(PROBE_VAR, false));
        assert!(env_bool(PROBE_VAR, true));

        match prev {
            Some(v) => std::env::set_var(PROBE_VAR, v),
            None => std::env::remove_var(PROBE_VAR),
        }
    }

    /// Recognized tokens parse through the reader for both polarities,
    /// including the issue's headline `yes` case.
    #[test]
    fn env_bool_recognized_tokens_parse() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(PROBE_VAR).ok();

        for (raw, expected) in [
            ("yes", true), // Issue #3751 headline scenario
            ("on", true),
            ("TRUE", true),
            ("0", false),
            ("no", false), // extended falsy set (was truthy for FLUXION_GPU)
            ("", false),   // set-but-empty is falsy, not unset
        ] {
            std::env::set_var(PROBE_VAR, raw);
            assert_eq!(
                env_bool(PROBE_VAR, !expected),
                expected,
                "raw value {raw:?} must parse to {expected} regardless of default"
            );
        }

        match prev {
            Some(v) => std::env::set_var(PROBE_VAR, v),
            None => std::env::remove_var(PROBE_VAR),
        }
    }

    /// Unrecognized values fall back to the unset default (treated as unset,
    /// loudly — the warn itself is observable in tracing output).
    #[test]
    fn env_bool_unrecognized_falls_back_to_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(PROBE_VAR).ok();
        std::env::set_var(PROBE_VAR, "ture");

        // Opt-in polarity (default false): typo stays off — fail closed.
        assert!(!env_bool(PROBE_VAR, false));
        // Opt-out polarity (default true): typo keeps the documented
        // unset behavior (e.g. FLUXION_GPU stays honored) — but warned.
        assert!(env_bool(PROBE_VAR, true));

        match prev {
            Some(v) => std::env::set_var(PROBE_VAR, v),
            None => std::env::remove_var(PROBE_VAR),
        }
    }
}
