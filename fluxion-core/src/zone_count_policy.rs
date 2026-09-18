//! Zone-count envelope policy leaf module (Issues #3731, #3871).
//!
//! Hoisted out of `fluxion::physics::geometry_tensor` into the `fluxion-core`
//! dependency-light leaf crate as the **minimum fix** for Issue #3871
//! "Hoist ZoneCountPolicy into fluxion-core to remove the new sim→physics
//! edge".
//!
//! The original `#3731` change added a single new
//! `use crate::physics::geometry_tensor::ZoneCountPolicy;` import in
//! `src/sim/thermal_model_core/mod.rs:9`, which PR #3869 admitted by raising
//! `BASELINE_SIM_TO_PHYSICS` from 80 → 81. This module restores the original
//! 80-edge baseline by moving the type, its `impl` block, the
//! `MAX_ZONES = 100` constant, and the `ZoneCountTier` enum into the leaf
//! where neither `sim` nor `physics` need to import from each other.
//!
//! # Tier semantics
//!
//! `MAX_ZONES = 100` is the gauge envelope (per the Phase 1a, #1461 data-
//! structure envelope). The question "what happens when a spec calls for more
//! than 100 zones?" currently has no typed answer: a spec with `num_zones >
//! MAX_ZONES` routed through `ZoneSolverKind::Gauge` (the unconditional
//! default selector since Issue #3291 / PR #3482) would reach the gauge
//! pre-checks (which currently guard only `num_zones < 2`, and only in the
//! multi-zone path) without ever entering a typed-error branch.
//!
//! `ZoneCountPolicy` is the typed wrapper the gauge initialization entry
//! points must use to make the policy first-class. It exposes three tiers
//! (`Empty` / `Standard | AtCapacity` / `BeyondEnvelope`), classifies a
//! zone count into the appropriate tier, and emits a typed [`ZoneCountError`]
//! for the beyond-envelope case — matching the convention every other gauge
//! pre-check uses. Callers MUST go through this wrapper: a code path that
//! bypasses it (e.g. constructs a `ThermalManifold::new(num_zones)` directly)
//! will hit the `assert!(num_zones <= MAX_ZONES, ...)` in `gauge_solver.rs`
//! and panic — that path is intentionally left in place as the secondary,
//! deep-defense check, but it should be unreachable from the public gauge
//! API now that `ZoneCountPolicy::check_gauge` is wired into both
//! `enable_gauge_solver` and `enable_gauge_solver_multi_zone`.
//!
//! Tier semantics:
//!
//! * `Empty`            — `num_zones == 0`. The gauge envelope never
//!                        observes a zero-zone building; this is a
//!                        programming error from `from_spec` and a typed
//!                        rejection here surfaces the bug at the
//!                        entry point rather than mid-step.
//! * `Standard`         — `1 <= num_zones < MAX_ZONES`. The happy path;
//!                        gauge dispatch may proceed.
//! * `AtCapacity`       — `num_zones == MAX_ZONES` (== 100). Permitted but
//!                        called out so callers / dashboards can flag it
//!                        as last-slot use.
//! * `BeyondEnvelope`   — `num_zones > MAX_ZONES`. Currently no path can
//!                        resolve it: the gauge envelope's Phase-1a data
//!                        structures (CTA geometry tensors with
//!                        `(MAX_ZONES, 20)` flat layout, the spatial
//!                        adjacency matrix, the per-zone property rows)
//!                        are sized from `MAX_ZONES` and growing them
//!                        requires the Phase-1b geometry rework tracked
//!                        in §LIMIT-21 (the β-soak program, Issue #3286).
//!
//! The `BeyondEnvelope` action is **typed rejection** by default (the
//! `check_gauge` method returns `Err(ZoneCountError)`). The wrapper is the
//! canonical seam where a future policy — explicit `NineRFourC` fallback,
//! partitioned gauge solve across two ThermalManifolds, etc. — would
//! hook in: callers that want a non-reject action must call the
//! classification method (`for_count`) and switch on the tier, NOT reach
//! around the typed `check_gauge` gate.
//!
//! # Leaf-module invariant (Issue #1349, #2462, #3467)
//!
//! This module lives in `fluxion-core` so neither `fluxion::sim` nor
//! `fluxion::physics` needs to import from the other. The leaf's only
//! dependencies are `thiserror` (Display + Error derives) and `std` — no
//! reference to `crate::physics::solver_trait::PhysicsError`, because that
//! type lives in the main `fluxion` crate and pulling it into the leaf
//! would re-introduce the very edge this issue was opened to remove.
//!
//! Callers in `fluxion::sim::thermal_model_core` that need to surface a
//! [`ZoneCountError`] as the engine's `PhysicsError::initialization(...)`
//! variant use `.map_err(|e| PhysicsError::initialization(&e.to_string()))?`
//! at the `check_gauge()?` call site. The Display format of
//! [`ZoneCountError`] is byte-identical to the historic PhysicsError
//! message body, so callers see the same `format!("{err}")` output and any
//! log lines / REST envelopes that grep on the rejection text are
//! unchanged.

/// Maximum number of thermal zones supported (Issue #1461 Phase-1a envelope).
///
/// **Value:** `100`
///
/// Beyond this, the Phase-1a data structures (CTA geometry tensors with
/// `(MAX_ZONES, 20)` flat layout, the spatial adjacency matrix, and the
/// per-zone property rows) cannot be sized from a single compile-time
/// constant. Growing the envelope is owned by the Phase-1b geometry rework
/// tracked in §LIMIT-21 (Issue #3286 β-soak program).
pub const MAX_ZONES: usize = 100;

/// Classified tier for a zone count relative to the gauge envelope.
///
/// See the [module docs](self) for tier semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZoneCountTier {
    /// `num_zones == 0` — invalid (no zones to solve).
    Empty,
    /// `1 <= num_zones < MAX_ZONES` — happy path.
    Standard,
    /// `num_zones == MAX_ZONES` — last valid slot of the gauge envelope.
    AtCapacity,
    /// `num_zones > MAX_ZONES` — exceeds the Phase-1a gauge envelope.
    BeyondEnvelope,
}

impl ZoneCountTier {
    /// `true` when `num_zones` is in `1..=MAX_ZONES` (the gauge envelope).
    /// The gauge initialization entry points MUST route through this
    /// classifier instead of comparing against `MAX_ZONES` directly.
    pub fn allows_gauge(self) -> bool {
        matches!(self, Self::Standard | Self::AtCapacity)
    }
}

impl std::fmt::Display for ZoneCountTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty (0 zones)"),
            Self::Standard => write!(f, "standard (1..MAX_ZONES)"),
            Self::AtCapacity => write!(f, "at-capacity (== MAX_ZONES = {MAX_ZONES})"),
            Self::BeyondEnvelope => write!(f, "beyond envelope (> MAX_ZONES = {MAX_ZONES})"),
        }
    }
}

/// Typed wrapper that classifies a zone count against the gauge envelope
/// and produces the canonical error for `num_zones > MAX_ZONES`.
///
/// Constructed via [`ZoneCountPolicy::for_count`] (do NOT construct from a
/// literal — the tier field is private so callers cannot classify a count
/// into the wrong tier). Use:
///
/// * `ZoneCountPolicy::for_count(n).allows_gauge()` for a non-failing
///   boolean read (e.g. dashboards, decision tables).
/// * `ZoneCountPolicy::for_count(n).check_gauge()?` to surface the
///   beyond-envelope count as a typed [`ZoneCountError`] at the gauge
///   initialization entry point (single- and multi-zone). Callers that
///   need to surface the error as the engine's `PhysicsError::initialization`
///   variant use `.map_err(|e| PhysicsError::initialization(&e.to_string()))?`
///   at the call site.
///
/// Issue #3731.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneCountPolicy {
    /// Raw count from the spec / model. Preserved for diagnostics —
    /// callers may want to print `policy.num_zones` alongside the tier
    /// in a user-facing rejection message.
    num_zones: usize,
    /// Classified tier.
    tier: ZoneCountTier,
}

impl ZoneCountPolicy {
    /// Classify `num_zones` against the gauge envelope. The returned
    /// `ZoneCountPolicy` is the single typed source of truth for "is
    /// this count allowed under the gauge selector" — do NOT roll your
    /// own `num_zones <= MAX_ZONES` comparison at a callsite.
    pub fn for_count(num_zones: usize) -> Self {
        let tier = if num_zones == 0 {
            ZoneCountTier::Empty
        } else if num_zones < MAX_ZONES {
            ZoneCountTier::Standard
        } else if num_zones == MAX_ZONES {
            ZoneCountTier::AtCapacity
        } else {
            ZoneCountTier::BeyondEnvelope
        };
        Self { num_zones, tier }
    }

    /// Raw zone count the spec / model declared.
    pub fn num_zones(&self) -> usize {
        self.num_zones
    }

    /// Classified tier. Use this for dashboards / decision tables; for
    /// the gauge pre-check, prefer the fallible [`check_gauge`](Self::check_gauge).
    pub fn tier(&self) -> ZoneCountTier {
        self.tier
    }

    /// `true` for `Standard | AtCapacity` — i.e. the count fits the
    /// Phase-1a gauge envelope. `false` for `Empty` and
    /// `BeyondEnvelope`. Use this for non-failing reads; the gauge
    /// initialization entry points prefer the fallible
    /// [`check_gauge`](Self::check_gauge) so the rejection surfaces as a
    /// typed error rather than a `bool` callers may accidentally
    /// silently skip.
    pub fn allows_gauge(&self) -> bool {
        self.tier.allows_gauge()
    }

    /// Typed pre-check for the gauge initialization entry points.
    ///
    /// * `Standard | AtCapacity` → `Ok(())`.
    /// * `Empty` → `Err(ZoneCountError::Empty)`.
    /// * `BeyondEnvelope` → `Err(ZoneCountError::BeyondEnvelope { ... })`
    ///   with a message that names the spec's `num_zones`, the gauge
    ///   envelope constant `MAX_ZONES`, and the §LIMIT-21 / Issue #3731 /
    ///   Phase-1b geometry rework that owns growing the envelope.
    ///
    /// Callers in the gauge initialization entry points MUST invoke this
    /// method (or `ZoneCountPolicy::for_count` + tier switch) — the typed
    /// error is what surfaces a beyond-envelope count as a clean `Err`,
    /// instead of the deep `assert!` panic in the shadow-mode
    /// `ThermalManifold::new(num_zones)` (`src/physics/gauge_solver.rs`).
    pub fn check_gauge(&self) -> Result<(), ZoneCountError> {
        match self.tier {
            ZoneCountTier::Standard | ZoneCountTier::AtCapacity => Ok(()),
            ZoneCountTier::Empty => Err(ZoneCountError::Empty),
            ZoneCountTier::BeyondEnvelope => Err(ZoneCountError::BeyondEnvelope {
                num_zones: self.num_zones,
                max_zones: MAX_ZONES,
            }),
        }
    }
}

impl std::fmt::Display for ZoneCountPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} zones — tier: {}", self.num_zones, self.tier)
    }
}

/// Typed rejection from [`ZoneCountPolicy::check_gauge`].
///
/// Lifted out of the original `crate::physics::solver_trait::PhysicsError`
/// path so this leaf module does not depend on the main `fluxion` crate's
/// physics error type (which would re-introduce the very `sim→physics`
/// edge Issue #3871 was opened to remove). Callers that need to surface
/// a [`ZoneCountError`] as the engine's
/// `PhysicsError::initialization(...)` variant use
/// `.map_err(|e| PhysicsError::initialization(&e.to_string()))?` at the
/// `check_gauge()?` call site; the `Display` format string below is
/// byte-identical to the historic `PhysicsError::Initialization(...)`
/// message body so callers see the same `format!("{err}")` output and
/// log lines / REST envelopes that grep on the rejection text are
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ZoneCountError {
    /// `num_zones == 0` (the `Empty` tier) — the gauge selector
    /// requires at least one thermal zone. Mirrors the historic
    /// `PhysicsError::initialization("Zone-count policy rejects
    /// num_zones = 0 (Empty tier): ...")`.
    #[error(
        "Zone-count policy rejects num_zones = 0 (Empty tier): \
         the gauge selector requires at least one thermal zone. \
         See `zone_count_policy::ZoneCountPolicy` (Issue #3731)."
    )]
    Empty,

    /// `num_zones > MAX_ZONES` (the `BeyondEnvelope` tier) — the
    /// Phase-1a gauge envelope (Issue #1461) sizes the geometry
    /// tensors, adjacency matrix, and per-zone property rows from
    /// `MAX_ZONES`, so a count beyond `MAX_ZONES` cannot be routed
    /// through the default gauge selector. Resolution is owned by the
    /// Phase-1b geometry rework (§LIMIT-21 / Issue #3731). Mirrors the
    /// historic `PhysicsError::initialization(&format!("Zone-count
    /// policy rejects num_zones = {n} (> MAX_ZONES = {m}): ..."))`.
    #[error(
        "Zone-count policy rejects num_zones = {num_zones} (> MAX_ZONES = {max_zones}): \
         the Phase-1a gauge envelope (Issue #1461) sizes the geometry \
         tensors, adjacency matrix, and per-zone property rows from \
         `MAX_ZONES`, so a count beyond `MAX_ZONES` cannot be routed \
         through the default gauge selector. Resolution is owned by \
         the Phase-1b geometry rework (§LIMIT-21 / Issue #3731)."
    )]
    BeyondEnvelope { num_zones: usize, max_zones: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ZoneCountPolicy (Issue #3731) ------------------------------------

    #[test]
    fn test_zone_count_policy_classifies_empty() {
        let p = ZoneCountPolicy::for_count(0);
        assert_eq!(p.tier(), ZoneCountTier::Empty);
        assert_eq!(p.num_zones(), 0);
        assert!(!p.allows_gauge());
    }

    #[test]
    fn test_zone_count_policy_classifies_standard_low() {
        let p = ZoneCountPolicy::for_count(1);
        assert_eq!(p.tier(), ZoneCountTier::Standard);
        assert_eq!(p.num_zones(), 1);
        assert!(p.allows_gauge());
        assert!(p.check_gauge().is_ok());
    }

    #[test]
    fn test_zone_count_policy_classifies_standard_mid() {
        let p = ZoneCountPolicy::for_count(20);
        assert_eq!(p.tier(), ZoneCountTier::Standard);
        assert!(p.allows_gauge());
        assert!(p.check_gauge().is_ok());
    }

    #[test]
    fn test_zone_count_policy_classifies_standard_just_below_cap() {
        assert_eq!(
            ZoneCountPolicy::for_count(MAX_ZONES - 1).tier(),
            ZoneCountTier::Standard
        );
        assert!(ZoneCountPolicy::for_count(MAX_ZONES - 1).allows_gauge());
    }

    #[test]
    fn test_zone_count_policy_classifies_at_capacity() {
        let p = ZoneCountPolicy::for_count(MAX_ZONES);
        assert_eq!(p.tier(), ZoneCountTier::AtCapacity);
        assert_eq!(p.num_zones(), MAX_ZONES);
        assert!(p.allows_gauge());
        assert!(p.check_gauge().is_ok());
    }

    #[test]
    fn test_zone_count_policy_classifies_beyond_envelope_min() {
        let p = ZoneCountPolicy::for_count(MAX_ZONES + 1);
        assert_eq!(p.tier(), ZoneCountTier::BeyondEnvelope);
        assert_eq!(p.num_zones(), MAX_ZONES + 1);
        assert!(!p.allows_gauge());
        let err = p.check_gauge().expect_err("must reject beyond-envelope");
        let msg = format!("{err}");
        assert!(
            msg.contains(&format!("num_zones = {}", MAX_ZONES + 1)),
            "error must name the offending count, got: {msg}"
        );
        assert!(
            msg.contains(&format!("MAX_ZONES = {MAX_ZONES}")),
            "error must name MAX_ZONES, got: {msg}"
        );
        assert!(
            msg.contains("Phase-1a gauge envelope"),
            "error must reference the Phase-1a envelope, got: {msg}"
        );
    }

    #[test]
    fn test_zone_count_policy_classifies_beyond_envelope_large() {
        let p = ZoneCountPolicy::for_count(200);
        assert_eq!(p.tier(), ZoneCountTier::BeyondEnvelope);
        assert!(!p.allows_gauge());
        assert!(p.check_gauge().is_err());
    }

    #[test]
    fn test_zone_count_policy_empty_returns_typed_error() {
        let p = ZoneCountPolicy::for_count(0);
        let err = p.check_gauge().expect_err("must reject Empty tier");
        let msg = format!("{err}");
        assert!(msg.contains("num_zones = 0"), "msg: {msg}");
        assert!(msg.contains("Empty"), "msg: {msg}");
    }

    #[test]
    fn test_zone_count_tier_display() {
        assert!(format!("{}", ZoneCountTier::Empty).contains("empty"));
        assert!(format!("{}", ZoneCountTier::Standard).contains("standard"));
        assert!(
            format!("{}", ZoneCountTier::AtCapacity).contains(&format!("MAX_ZONES = {MAX_ZONES}"))
        );
        assert!(format!("{}", ZoneCountTier::BeyondEnvelope)
            .contains(&format!("MAX_ZONES = {MAX_ZONES}")));
    }

    #[test]
    fn test_zone_count_policy_display() {
        let p = ZoneCountPolicy::for_count(42);
        let s = format!("{p}");
        assert!(
            s.contains("42 zones"),
            "Display must include raw count, got: {s}"
        );
        assert!(
            s.contains("standard"),
            "Display must include tier, got: {s}"
        );
    }

    #[test]
    fn test_zone_count_policy_typed_check_is_compile_time_seam() {
        let p = ZoneCountPolicy::for_count(MAX_ZONES);
        assert!(p.allows_gauge());
    }
}
