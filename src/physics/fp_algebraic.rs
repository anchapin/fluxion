//! Algebraic floating-point helper layer (issue #3322).
//!
//! Tiny `cfg`-routed wrappers around the four basic float operations for
//! `f32` and `f64`. With default features every helper compiles to the plain
//! IEEE 754 operator, so default-feature builds stay **bit-identical** to
//! today (zero-cost passthrough). Under `--features fast-math` the same
//! helpers instead route to the Rust 1.98 std *algebraic* float methods
//! (`f32::algebraic_add` / `f64::algebraic_add`, …), which permit operand
//! reassociation, contraction, and loop vectorization comparable to
//! `-ffast-math` — enabled per call site rather than per compilation unit.
//!
//! # ⚠️ Non-determinism warning
//!
//! The algebraic methods are **non-deterministic by specification**: results
//! may differ from strict IEEE 754 evaluation at the last-ulp level, and may
//! legitimately differ between compiler versions, compilation sessions,
//! optimization levels, and targets. Never rely on bit-stability of any
//! computation that flows through these helpers while `fast-math` is
//! enabled. In particular:
//!
//! * `determinism_check.yml` enforces **bit-identical cross-platform output**
//!   with pinned `RUSTFLAGS` (issues #1297, #2549) — enabling `fast-math`
//!   anywhere in that pipeline breaks the contract, so that workflow (and
//!   every ASHRAE 140 gate) must always run default features.
//! * ASHRAE 140 validation baselines (`tests/reference_data/zone_balance/
//!   strict_energy_gate_baseline.json`, `SCORECARD.md`) are generated and
//!   judged under strict IEEE semantics (`RULES.md`); last-ulp drift from
//!   algebraic ops is a validation failure, not noise to absorb.
//!
//! # Do NOT use in energy-balance-critical paths
//!
//! `fast-math` exists so the follow-up kernel-conversion issues (#3324
//! solar/irradiance reductions, #3325 AI batch metric reductions) can opt in
//! call-site by call-site. It must **never** be applied to paths whose
//! floating-point rounding feeds an energy conservation ledger or a
//! validation baseline, including (non-exhaustive):
//!
//! * Heat-conduction solvers: [`crate::physics::ctf_solver`],
//!   [`crate::physics::ctf_solver_wrapper`],
//!   [`crate::physics::multi_node_solver`],
//!   [`crate::physics::five_r1c_solver`], [`crate::physics::fd_solver`]
//! * Zone-balance / thermal assembly: [`crate::sim::assembly`],
//!   [`crate::sim::timestep_solver`], and the thermal model solvers
//!   ([`crate::sim::thermal_model`],
//!   [`crate::sim::thermal_model_solvers`])
//!
//! Adding a call site here is a no-op until a kernel imports it; this module
//! deliberately contains no behavior of its own.
//!
//! # Usage
//!
//! ```ignore
//! use crate::physics::fp_algebraic::{algebraic_add, algebraic_mul};
//! // default build: (a + b) * c, bit-identical to writing it out
//! // fast-math build: a.algebraic_add(b).algebraic_mul(c)
//! let z = algebraic_mul(algebraic_add(a, b), c);
//! ```

// The trait is sealed so downstream crates cannot implement it with
// semantics that would silently break the bit-identical default contract.

mod sealed {
    /// Marker trait keeping [`crate::physics::fp_algebraic::AlgebraicFloat`]
    /// closed over `f32`/`f64` only.
    pub trait Sealed {}
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}

/// Primitive float types supported by the [`crate::physics::fp_algebraic`]
/// helper layer.
///
/// Each method is `#[inline(always)]` and resolves to exactly one operation
/// in either feature state — there is no runtime dispatch and no cost when
/// `fast-math` is off.
pub trait AlgebraicFloat: Copy + sealed::Sealed {
    /// Add `rhs` to `self` (IEEE `+` by default; `algebraic_add` under
    /// `fast-math`). Implementations are `#[inline(always)]`.
    fn algebraic_add(self, rhs: Self) -> Self;

    /// Subtract `rhs` from `self` (IEEE `-` by default; `algebraic_sub`
    /// under `fast-math`). Implementations are `#[inline(always)]`.
    fn algebraic_sub(self, rhs: Self) -> Self;

    /// Multiply `self` by `rhs` (IEEE `*` by default; `algebraic_mul` under
    /// `fast-math`). Implementations are `#[inline(always)]`.
    fn algebraic_mul(self, rhs: Self) -> Self;

    /// Divide `self` by `rhs` (IEEE `/` by default; `algebraic_div` under
    /// `fast-math`). Implementations are `#[inline(always)]`.
    fn algebraic_div(self, rhs: Self) -> Self;
}

impl AlgebraicFloat for f32 {
    #[inline(always)]
    fn algebraic_add(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f32::algebraic_add(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self + rhs
        }
    }

    #[inline(always)]
    fn algebraic_sub(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f32::algebraic_sub(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self - rhs
        }
    }

    #[inline(always)]
    fn algebraic_mul(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f32::algebraic_mul(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self * rhs
        }
    }

    #[inline(always)]
    fn algebraic_div(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f32::algebraic_div(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self / rhs
        }
    }
}

impl AlgebraicFloat for f64 {
    #[inline(always)]
    fn algebraic_add(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f64::algebraic_add(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self + rhs
        }
    }

    #[inline(always)]
    fn algebraic_sub(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f64::algebraic_sub(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self - rhs
        }
    }

    #[inline(always)]
    fn algebraic_mul(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f64::algebraic_mul(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self * rhs
        }
    }

    #[inline(always)]
    fn algebraic_div(self, rhs: Self) -> Self {
        #[cfg(feature = "fast-math")]
        {
            f64::algebraic_div(self, rhs)
        }
        #[cfg(not(feature = "fast-math"))]
        {
            self / rhs
        }
    }
}

/// Add two floats: plain IEEE `+` by default, `algebraic_add` under
/// `--features fast-math` (non-deterministic; see the module docs).
#[inline(always)]
pub fn algebraic_add<T: AlgebraicFloat>(a: T, b: T) -> T {
    a.algebraic_add(b)
}

/// Subtract two floats: plain IEEE `-` by default, `algebraic_sub` under
/// `--features fast-math` (non-deterministic; see the module docs).
#[inline(always)]
pub fn algebraic_sub<T: AlgebraicFloat>(a: T, b: T) -> T {
    a.algebraic_sub(b)
}

/// Multiply two floats: plain IEEE `*` by default, `algebraic_mul` under
/// `--features fast-math` (non-deterministic; see the module docs).
#[inline(always)]
pub fn algebraic_mul<T: AlgebraicFloat>(a: T, b: T) -> T {
    a.algebraic_mul(b)
}

/// Divide two floats: plain IEEE `/` by default, `algebraic_div` under
/// `--features fast-math` (non-deterministic; see the module docs).
#[inline(always)]
pub fn algebraic_div<T: AlgebraicFloat>(a: T, b: T) -> T {
    a.algebraic_div(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // black_box keeps the compiler from folding the reference expression
    // into the helper at optimization time, so the default-build bit-identity
    // assertions compare two genuinely separate computations.
    use std::hint::black_box;

    /// Finite value pairs chosen to stress IEEE rounding: exact-representable
    /// values, values that round on add/mul/div, magnitudes spanning ~1e-30
    /// to ~1e30, catastrophic-cancellation pairs, and signed zeros.
    const F64_PAIRS: &[(f64, f64)] = &[
        (0.0, 0.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (1.0, 2.0),
        (-3.5, 1.25),
        (0.1, 0.2),
        (1.0 / 3.0, 2.0 / 3.0),
        (1e300, 1e-300),
        (1e-300, 1e-300),
        (-1e308, 5e307),
        (f64::MAX, f64::MIN_POSITIVE),
        (1e16, 1.0),               // ulp > 1 at this magnitude
        (1.0000000000000002, 1.0), // adjacent-representable pair
        (9007199254740993.0, 1.0), // 2^53 + 1 rounding boundary
        (123456.789101112, 0.000000123456789),
        (0.30000000000000004, 0.3), // catastrophic cancellation pair
    ];

    const F32_PAIRS: &[(f32, f32)] = &[
        (0.0, 0.0),
        (0.0, -0.0),
        (-0.0, -0.0),
        (1.0, 2.0),
        (-3.5, 1.25),
        (0.1, 0.2),
        (1.0 / 3.0, 2.0 / 3.0),
        (1e30, 1e-30),
        (1e-30, 1e-30),
        (1e38, 5e37),
        (f32::MAX, f32::MIN_POSITIVE),
        (16777217.0, 1.0), // 2^24 + 1 rounding boundary
        (123.4567, 0.0001234),
    ];

    /// Default-feature contract: the fallback path is **bit-identical** to
    /// the plain IEEE operators across representative finite operands,
    /// including signed zeros, subnormals, overflow-to-infinity, and
    /// division-driven underflow. This test only asserts under default
    /// features because under `fast-math` last-ulp divergence from the plain
    /// operator is allowed by specification.
    #[cfg(not(feature = "fast-math"))]
    #[test]
    fn fallback_is_bit_identical_to_ieee_operators() {
        for &(a, b) in F64_PAIRS {
            let (a, b) = (black_box(a), black_box(b));
            assert_eq!(algebraic_add(a, b).to_bits(), (a + b).to_bits());
            assert_eq!(algebraic_sub(a, b).to_bits(), (a - b).to_bits());
            assert_eq!(algebraic_mul(a, b).to_bits(), (a * b).to_bits());
            assert_eq!(algebraic_div(a, b).to_bits(), (a / b).to_bits());
        }
        for &(a, b) in F32_PAIRS {
            let (a, b) = (black_box(a), black_box(b));
            assert_eq!(algebraic_add(a, b).to_bits(), (a + b).to_bits());
            assert_eq!(algebraic_sub(a, b).to_bits(), (a - b).to_bits());
            assert_eq!(algebraic_mul(a, b).to_bits(), (a * b).to_bits());
            assert_eq!(algebraic_div(a, b).to_bits(), (a / b).to_bits());
        }
    }

    /// `fast-math` contract: the helpers compile and produce numerically
    /// valid results — within a few ulp of the IEEE operator for finite
    /// operands. The algebraic methods are non-deterministic by
    /// specification, so only a relative-epsilon bound is asserted here.
    /// IEEE-defined NaN results (e.g. `0.0 / 0.0`) are skipped: NaN
    /// propagation is not guaranteed under algebraic semantics.
    #[cfg(feature = "fast-math")]
    #[test]
    fn algebraic_helpers_stay_within_a_few_ulp_of_ieee() {
        fn assert_close_f64(got: f64, want: f64, ctx: &str) {
            if got.to_bits() == want.to_bits() {
                return;
            }
            assert!(
                !got.is_nan() && !want.is_nan(),
                "{ctx}: got {got:?}, want {want:?}"
            );
            if want.is_infinite() {
                // Single operations that overflow in IEEE must still
                // overflow under algebraic evaluation.
                assert_eq!(got, want, "{ctx}: expected overflow to {want:?}");
                return;
            }
            let tol = (want.abs() * 8.0 * f64::EPSILON).max(8.0 * f64::MIN_POSITIVE);
            assert!(
                (got - want).abs() <= tol,
                "{ctx}: got {got:e}, want {want:e}, diff {:e} > tol {tol:e}",
                (got - want).abs()
            );
        }
        fn assert_close_f32(got: f32, want: f32, ctx: &str) {
            if got.to_bits() == want.to_bits() {
                return;
            }
            assert!(
                !got.is_nan() && !want.is_nan(),
                "{ctx}: got {got:?}, want {want:?}"
            );
            if want.is_infinite() {
                assert_eq!(got, want, "{ctx}: expected overflow to {want:?}");
                return;
            }
            let tol = (want.abs() * 8.0 * f32::EPSILON).max(8.0 * f32::MIN_POSITIVE);
            assert!(
                (got - want).abs() <= tol,
                "{ctx}: got {got:e}, want {want:e}, diff {:e} > tol {tol:e}",
                (got - want).abs()
            );
        }

        for &(a, b) in F64_PAIRS {
            let (a, b) = (black_box(a), black_box(b));
            let (w_add, w_sub, w_mul, w_div) = (a + b, a - b, a * b, a / b);
            if !w_add.is_nan() {
                assert_close_f64(algebraic_add(a, b), w_add, "f64 add {a:?}+{b:?}");
            }
            if !w_sub.is_nan() {
                assert_close_f64(algebraic_sub(a, b), w_sub, "f64 sub {a:?}-{b:?}");
            }
            if !w_mul.is_nan() {
                assert_close_f64(algebraic_mul(a, b), w_mul, "f64 mul {a:?}*{b:?}");
            }
            if !w_div.is_nan() {
                assert_close_f64(algebraic_div(a, b), w_div, "f64 div {a:?}/{b:?}");
            }
        }
        for &(a, b) in F32_PAIRS {
            let (a, b) = (black_box(a), black_box(b));
            let (w_add, w_sub, w_mul, w_div) = (a + b, a - b, a * b, a / b);
            if !w_add.is_nan() {
                assert_close_f32(algebraic_add(a, b), w_add, "f32 add {a:?}+{b:?}");
            }
            if !w_sub.is_nan() {
                assert_close_f32(algebraic_sub(a, b), w_sub, "f32 sub {a:?}-{b:?}");
            }
            if !w_mul.is_nan() {
                assert_close_f32(algebraic_mul(a, b), w_mul, "f32 mul {a:?}*{b:?}");
            }
            if !w_div.is_nan() {
                assert_close_f32(algebraic_div(a, b), w_div, "f32 div {a:?}/{b:?}");
            }
        }
    }

    /// Runs in BOTH feature states: whatever the routing, the helpers must
    /// agree with exact arithmetic on values that are exactly representable,
    /// so these hold bit-for-bit even under `fast-math`.
    #[test]
    fn helpers_agree_with_exact_arithmetic_on_representable_values() {
        // 0.25 / 0.5 / powers of two are exact in binary FP: add/sub/mul/div
        // results here are exactly representable, so even reassociated
        // algebraic evaluation cannot differ.
        let a64 = black_box(0.25_f64);
        let b64 = black_box(0.5_f64);
        assert_eq!(algebraic_add(a64, b64), 0.75);
        assert_eq!(algebraic_sub(a64, b64), -0.25);
        assert_eq!(algebraic_mul(a64, b64), 0.125);
        assert_eq!(algebraic_div(a64, b64), 0.5);

        let a32 = black_box(0.25_f32);
        let b32 = black_box(0.5_f32);
        assert_eq!(algebraic_add(a32, b32), 0.75);
        assert_eq!(algebraic_sub(a32, b32), -0.25);
        assert_eq!(algebraic_mul(a32, b32), 0.125);
        assert_eq!(algebraic_div(a32, b32), 0.5);
    }

    /// Runs in BOTH feature states: identity/absorbing-element behavior with
    /// ±0.0 and multiplication by 1.0 for finite operands. Exact under
    /// default features; these identities survive algebraic reassociation
    /// because each is a single operation.
    #[test]
    fn helpers_preserve_zero_and_one_identities() {
        let x = black_box(0.123456789012345_f64);
        assert_eq!(algebraic_add(x, 0.0).to_bits(), x.to_bits());
        assert_eq!(algebraic_sub(x, 0.0).to_bits(), x.to_bits());
        assert_eq!(algebraic_mul(x, 1.0).to_bits(), x.to_bits());
        assert_eq!(algebraic_div(x, 1.0).to_bits(), x.to_bits());

        let y = black_box(0.1234567_f32);
        assert_eq!(algebraic_add(y, 0.0).to_bits(), y.to_bits());
        assert_eq!(algebraic_sub(y, 0.0).to_bits(), y.to_bits());
        assert_eq!(algebraic_mul(y, 1.0).to_bits(), y.to_bits());
        assert_eq!(algebraic_div(y, 1.0).to_bits(), y.to_bits());
    }
}
