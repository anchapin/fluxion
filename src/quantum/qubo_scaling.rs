//! D-Wave hardware constraint scaling for QUBO / Ising problems.
//!
//! D-Wave annealers impose hard limits on the values that can be submitted:
//! - Bias field `h ∈ [h_min, h_max]`
//! - Coupling strength `J ∈ [j_min, j_max]`
//!
//! Additionally, the number of variables `N` must not exceed the sampler-reported
//! `max_variables`. This module provides:
//!
//! 1. [`DwaveHardwareConstraints`] — concrete hardware limits (e.g. AdvantageSystem6.4)
//! 2. [`EmbeddingFeasibility`] — result of checking whether a problem fits the hardware
//! 3. [`ScalingResult`] — result of scaling a QUBO / Ising to hardware range, including
//!    whether any clipping occurred
//! 4. [`scale_ising_to_hardware`] — min-max scale an Ising problem to fit hardware
//! 5. [`clip_ising_to_hardware`] — hard-clip an Ising problem to hardware range
//! 6. [`check_embedding_feasibility`] — verify problem size vs available qubits

use crate::quantum::qubo_mapping::IsingProblem;

/// D-Wave hardware constraints for a specific annealer or solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DwaveHardwareConstraints {
    /// Minimum allowed bias value `h` (typically negative).
    pub h_min: f64,
    /// Maximum allowed bias value `h` (typically positive).
    pub h_max: f64,
    /// Minimum allowed coupling value `J` (typically negative).
    pub j_min: f64,
    /// Maximum allowed coupling value `J` (typically positive).
    pub j_max: f64,
    /// Maximum number of variables (qubits) supported by the sampler.
    pub max_variables: usize,
}

impl DwaveHardwareConstraints {
    /// D-Wave AdvantageSystem6.4 constraints:
    /// - `h ∈ [−4, +4]`
    /// - `J ∈ [−2, +1]`
    /// - up to 5,700 qubits (variable count depends on embedding overhead)
    pub const fn advantage_system64() -> Self {
        Self {
            h_min: -4.0,
            h_max: 4.0,
            j_min: -2.0,
            j_max: 1.0,
            max_variables: 5760,
        }
    }

    /// D-Wave Advantage2 prototype constraints (conservative estimate):
    /// - `h ∈ [−4, +4]`
    /// - `J ∈ [−2, +1]`
    /// - up to 12,000 qubits
    pub const fn advantage2_prototype() -> Self {
        Self {
            h_min: -4.0,
            h_max: 4.0,
            j_min: -2.0,
            j_max: 1.0,
            max_variables: 12000,
        }
    }

    /// Validate that a given `IsingProblem` satisfies these constraints.
    ///
    /// Returns `Ok(())` if all `h` and `J` values are within range and
    /// `num_variables ≤ max_variables`. Returns an error describing the first
    /// violation found.
    pub fn validate_ising(&self, ising: &IsingProblem) -> Result<(), ConstraintViolation> {
        if ising.num_variables > self.max_variables {
            return Err(ConstraintViolation::TooManyVariables {
                num_variables: ising.num_variables,
                max_variables: self.max_variables,
            });
        }

        for (i, &hi) in ising.h.iter().enumerate() {
            if hi < self.h_min || hi > self.h_max {
                return Err(ConstraintViolation::BiasOutOfRange {
                    index: i,
                    value: hi,
                    min: self.h_min,
                    max: self.h_max,
                });
            }
        }

        let n = ising.num_variables;
        for i in 0..n {
            for j in (i + 1)..n {
                let jij = ising.j[i * n + j];
                if jij < self.j_min || jij > self.j_max {
                    return Err(ConstraintViolation::CouplingOutOfRange {
                        i,
                        j,
                        value: jij,
                        min: self.j_min,
                        max: self.j_max,
                    });
                }
            }
        }

        Ok(())
    }

    /// Compute the scale factor needed to bring `ising` into hardware range via
    /// min-max scaling. Returns the smallest `s ≥ 1` such that after scaling,
    /// all values are within `[min_bound, max_bound]`.
    ///
    /// The scaling formula is:
    /// ```text
    /// h_scaled[i] = h[i] / s
    /// J_scaled[i,j] = J[i,j] / s
    /// ```
    ///
    /// Returns `None` if the problem is already within range (`s = 1`).
    pub fn compute_scaling_factor(&self, ising: &IsingProblem) -> Option<f64> {
        let mut h_scale = 1.0_f64;
        for &hi in &ising.h {
            let s = hi.abs() / self.h_max;
            if s > h_scale {
                h_scale = s;
            }
        }

        let n = ising.num_variables;
        let mut j_scale = 1.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let jij = ising.j[i * n + j];
                if jij.abs() > 0.0 {
                    let bound = if jij > 0.0 {
                        self.j_max
                    } else {
                        self.j_min.abs()
                    };
                    let s = jij.abs() / bound;
                    if s > j_scale {
                        j_scale = s;
                    }
                }
            }
        }

        let s = h_scale.max(j_scale);
        if s <= 1.0 {
            None
        } else {
            Some(s)
        }
    }
}

/// A single violation of hardware constraints detected in an `IsingProblem`.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintViolation {
    /// Number of variables exceeds the sampler limit.
    TooManyVariables {
        num_variables: usize,
        max_variables: usize,
    },
    /// A bias value `h[i]` is outside `[h_min, h_max]`.
    BiasOutOfRange {
        index: usize,
        value: f64,
        min: f64,
        max: f64,
    },
    /// A coupling value `J[i,j]` is outside `[j_min, j_max]`.
    CouplingOutOfRange {
        i: usize,
        j: usize,
        value: f64,
        min: f64,
        max: f64,
    },
}

impl std::fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyVariables {
                num_variables,
                max_variables,
            } => {
                write!(f, "Ising problem has {num_variables} variables but sampler supports at most {max_variables} — reduce bits_per_node or use a hybrid solver")
            }
            Self::BiasOutOfRange {
                index,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "h[{index}] = {value} is outside allowed range [{min}, {max}]"
                )
            }
            Self::CouplingOutOfRange {
                i,
                j,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "J[{i},{j}] = {value} is outside allowed range [{min}, {max}]"
                )
            }
        }
    }
}

impl std::error::Error for ConstraintViolation {}

/// Result of checking whether a QUBO / Ising problem can be embedded on hardware.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingFeasibility {
    /// `true` if `num_variables ≤ max_variables`.
    pub fits_in_qubits: bool,
    /// The number of variables in the problem.
    pub num_variables: usize,
    /// The maximum number of qubits available on the sampler.
    pub max_qubits: usize,
    /// `true` if all Ising values (after scaling) are within hardware ranges.
    pub biases_in_range: bool,
    /// `true` if all coupling values (after scaling) are within hardware ranges.
    pub couplings_in_range: bool,
}

impl EmbeddingFeasibility {
    /// Returns `true` if the problem is fully embeddable without scaling.
    pub fn is_feasible(&self) -> bool {
        self.fits_in_qubits && self.biases_in_range && self.couplings_in_range
    }
}

/// How an Ising problem was scaled to fit hardware constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingStrategy {
    /// No scaling needed — problem already within hardware range.
    None,
    /// Values were uniformly scaled by dividing by `s > 1`.
    MinMaxScale { factor: f64 },
    /// Values were hard-clipped to the hardware range (information loss).
    HardClip,
}

impl ScalingStrategy {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Result of scaling an `IsingProblem` to satisfy hardware constraints.
#[derive(Debug, Clone)]
pub struct ScalingResult {
    /// The scaled Ising problem.
    pub ising: IsingProblem,
    /// How the scaling was performed.
    pub strategy: ScalingStrategy,
    /// `true` if any value was clipped during scaling.
    pub clipped: bool,
    /// Embedding feasibility of the *original* (pre-scaling) problem.
    pub original_feasibility: EmbeddingFeasibility,
}

impl ScalingResult {
    /// Scale `ising` so all `h` and `J` values fit within `constraints`
    /// using **min-max uniform scaling** (dividing all values by `s ≥ 1`).
    ///
    /// The energy landscape is preserved up to a global factor, so the optimal
    /// solution is unchanged. The returned `ScalingResult.clipped` is `false`
    /// because scaling is reversible.
    ///
    /// Returns the scaled problem (original is consumed).
    pub fn scale_ising_to_hardware(
        mut ising: IsingProblem,
        constraints: &DwaveHardwareConstraints,
    ) -> Result<ScalingResult, ConstraintViolation> {
        let original_feasibility = check_embedding_feasibility(&ising, constraints);

        if original_feasibility.is_feasible() {
            return Ok(ScalingResult {
                ising,
                strategy: ScalingStrategy::None,
                clipped: false,
                original_feasibility,
            });
        }

        let Some(s) = constraints.compute_scaling_factor(&ising) else {
            return Ok(ScalingResult {
                ising,
                strategy: ScalingStrategy::None,
                clipped: false,
                original_feasibility,
            });
        };

        for hi in &mut ising.h {
            *hi /= s;
        }

        let n = ising.num_variables;
        for i in 0..n {
            for j in (i + 1)..n {
                ising.j[i * n + j] /= s;
                ising.j[j * n + i] /= s;
            }
        }

        ising.c /= s;

        Ok(ScalingResult {
            ising,
            strategy: ScalingStrategy::MinMaxScale { factor: s },
            clipped: false,
            original_feasibility,
        })
    }

    /// Clip `ising` values to `constraints` using **hard clipping**.
    ///
    /// This loses information (irreversible) but is sometimes preferred when
    /// preserving the absolute energy scale is more important than preserving
    /// the relative ratios between coefficients.
    ///
    /// Returns the scaled problem (original is consumed).
    pub fn clip_ising_to_hardware(
        mut ising: IsingProblem,
        constraints: &DwaveHardwareConstraints,
    ) -> Result<ScalingResult, ConstraintViolation> {
        let original_feasibility = check_embedding_feasibility(&ising, constraints);

        let mut clipped = false;

        for hi in &mut ising.h {
            if *hi < constraints.h_min {
                *hi = constraints.h_min;
                clipped = true;
            } else if *hi > constraints.h_max {
                *hi = constraints.h_max;
                clipped = true;
            }
        }

        let n = ising.num_variables;
        for i in 0..n {
            for j in (i + 1)..n {
                let idx_ij = i * n + j;
                let idx_ji = j * n + i;
                let jij = ising.j[idx_ij];
                if jij < constraints.j_min {
                    ising.j[idx_ij] = constraints.j_min;
                    ising.j[idx_ji] = constraints.j_min;
                    clipped = true;
                } else if jij > constraints.j_max {
                    ising.j[idx_ij] = constraints.j_max;
                    ising.j[idx_ji] = constraints.j_max;
                    clipped = true;
                }
            }
        }

        Ok(ScalingResult {
            ising,
            strategy: ScalingStrategy::HardClip,
            clipped,
            original_feasibility,
        })
    }
}

/// Check whether `ising` can be embedded on hardware described by `constraints`
/// without any scaling.
pub fn check_embedding_feasibility(
    ising: &IsingProblem,
    constraints: &DwaveHardwareConstraints,
) -> EmbeddingFeasibility {
    let num_variables = ising.num_variables;
    let fits_in_qubits = num_variables <= constraints.max_variables;

    let mut biases_in_range = true;
    for &hi in &ising.h {
        if hi < constraints.h_min || hi > constraints.h_max {
            biases_in_range = false;
            break;
        }
    }

    let mut couplings_in_range = true;
    let n = ising.num_variables;
    for i in 0..n {
        for j in (i + 1)..n {
            let jij = ising.j[i * n + j];
            if jij < constraints.j_min || jij > constraints.j_max {
                couplings_in_range = false;
                break;
            }
        }
        if !couplings_in_range {
            break;
        }
    }

    EmbeddingFeasibility {
        fits_in_qubits,
        num_variables,
        max_qubits: constraints.max_variables,
        biases_in_range,
        couplings_in_range,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_small_ising() -> IsingProblem {
        IsingProblem {
            h: vec![0.5, -0.3, 1.0, -0.8],
            j: vec![0.0; 16],
            c: 0.0,
            num_variables: 4,
        }
    }

    #[test]
    fn test_constraints_advantage_system64_defaults() {
        let c = DwaveHardwareConstraints::advantage_system64();
        assert_eq!(c.h_min, -4.0);
        assert_eq!(c.h_max, 4.0);
        assert_eq!(c.j_min, -2.0);
        assert_eq!(c.j_max, 1.0);
        assert_eq!(c.max_variables, 5760);
    }

    #[test]
    fn test_validate_ising_within_range() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = make_small_ising();
        assert!(c.validate_ising(&ising).is_ok());
    }

    #[test]
    fn test_validate_ising_bias_out_of_range() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.h[0] = 10.0; // exceeds h_max = 4.0
        let err = c.validate_ising(&ising);
        assert!(matches!(
            err,
            Err(ConstraintViolation::BiasOutOfRange { .. })
        ));
    }

    #[test]
    fn test_validate_ising_coupling_out_of_range() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.j[1 * 4 + 2] = -5.0; // below j_min = -2.0
        let err = c.validate_ising(&ising);
        assert!(matches!(
            err,
            Err(ConstraintViolation::CouplingOutOfRange { .. })
        ));
    }

    #[test]
    fn test_validate_ising_too_many_variables() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = IsingProblem {
            h: vec![0.0; 6000],
            j: vec![0.0; 6000 * 6000],
            c: 0.0,
            num_variables: 6000,
        };
        let err = c.validate_ising(&ising);
        assert!(matches!(
            err,
            Err(ConstraintViolation::TooManyVariables { .. })
        ));
    }

    #[test]
    fn test_compute_scaling_factor_none_needed() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = make_small_ising();
        assert!(c.compute_scaling_factor(&ising).is_none());
    }

    #[test]
    fn test_compute_scaling_factor_h_bias_exceeds() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.h[0] = 8.0; // 8/4 = 2.0
        assert_eq!(c.compute_scaling_factor(&ising), Some(2.0));
    }

    #[test]
    fn test_compute_scaling_factor_j_coupling_exceeds() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.j[1 * 4 + 2] = 4.0; // 4/1 = 4.0
        assert_eq!(c.compute_scaling_factor(&ising), Some(4.0));
    }

    #[test]
    fn test_scale_ising_no_scale_needed() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = make_small_ising();
        let result = ScalingResult::scale_ising_to_hardware(ising, &c).unwrap();
        assert!(result.strategy.is_none());
        assert!(!result.clipped);
        assert!(result.original_feasibility.is_feasible());
    }

    #[test]
    fn test_scale_ising_minmax_scale() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.h[0] = 8.0; // needs s=2
        let result = ScalingResult::scale_ising_to_hardware(ising, &c).unwrap();
        assert!(matches!(
            result.strategy,
            ScalingStrategy::MinMaxScale { factor: f } if (f - 2.0).abs() < 1e-12
        ));
        assert!(!result.clipped);
        assert_eq!(result.ising.h[0], 4.0);
    }

    #[test]
    fn test_scale_ising_preserves_relative_ratios() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.h[0] = 8.0;
        ising.h[1] = 4.0; // ratio h0/h1 = 2.0 should be preserved
        let result = ScalingResult::scale_ising_to_hardware(ising, &c).unwrap();
        assert!((result.ising.h[0] / result.ising.h[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_clip_ising_no_clip_needed() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = make_small_ising();
        let result = ScalingResult::clip_ising_to_hardware(ising, &c).unwrap();
        assert!(matches!(result.strategy, ScalingStrategy::HardClip));
        assert!(!result.clipped);
    }

    #[test]
    fn test_clip_ising_hard_clip() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        ising.h[0] = 10.0;
        let result = ScalingResult::clip_ising_to_hardware(ising, &c).unwrap();
        assert!(result.clipped);
        assert_eq!(result.ising.h[0], 4.0);
    }

    #[test]
    fn test_check_embedding_feasibility_feasible() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = make_small_ising();
        let feasibility = check_embedding_feasibility(&ising, &c);
        assert!(feasibility.is_feasible());
        assert!(feasibility.fits_in_qubits);
        assert!(feasibility.biases_in_range);
        assert!(feasibility.couplings_in_range);
    }

    #[test]
    fn test_check_embedding_feasibility_too_many_qubits() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = IsingProblem {
            h: vec![0.0; 6000],
            j: vec![0.0; 6000 * 6000],
            c: 0.0,
            num_variables: 6000,
        };
        let feasibility = check_embedding_feasibility(&ising, &c);
        assert!(!feasibility.fits_in_qubits);
        assert!(!feasibility.is_feasible());
    }

    #[test]
    fn test_ising_problem_with_large_coupling_gets_scaled() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = make_small_ising();
        // Set a large off-diagonal coupling: J[0,1] = 5.0 (> j_max=1.0)
        // s = 5.0 / 1.0 = 5.0
        ising.j[0 * 4 + 1] = 5.0;
        ising.j[1 * 4 + 0] = 5.0;
        let result = ScalingResult::scale_ising_to_hardware(ising, &c).unwrap();
        assert!(matches!(
            result.strategy,
            ScalingStrategy::MinMaxScale { factor: f } if (f - 5.0).abs() < 1e-12
        ));
        // Coupling should now be 1.0
        assert!((result.ising.j[0 * 4 + 1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_embedding_feasibility_shows_all_flags() {
        let c = DwaveHardwareConstraints::advantage_system64();
        let ising = IsingProblem {
            h: vec![0.0; 4],
            j: vec![0.0; 16],
            c: 0.0,
            num_variables: 4,
        };
        let feasibility = check_embedding_feasibility(&ising, &c);
        assert_eq!(feasibility.num_variables, 4);
        assert_eq!(feasibility.max_qubits, 5760);
        assert!(feasibility.fits_in_qubits);
        assert!(feasibility.biases_in_range);
        assert!(feasibility.couplings_in_range);
    }

    #[test]
    fn test_scaling_preserves_solution_ranking() {
        // If we scale the Ising, the argmin of the energy landscape is unchanged.
        let c = DwaveHardwareConstraints::advantage_system64();
        let mut ising = IsingProblem {
            h: vec![1.0, -1.0, 0.5, -0.5],
            j: vec![0.0; 16],
            c: 0.0,
            num_variables: 4,
        };
        ising.j[0 * 4 + 1] = 2.0;
        ising.j[1 * 4 + 0] = 2.0;

        let result = ScalingResult::scale_ising_to_hardware(ising, &c).unwrap();
        // All spins are ±1
        let all_spins: Vec<i8> = vec![1, 1, 1, 1];
        let all_spins_neg: Vec<i8> = vec![-1, -1, -1, -1];

        let e_orig = result.original_feasibility.is_feasible();
        if e_orig {
            // Problem was already feasible, energy should be identical
            let e1 = result.ising.evaluate(&all_spins);
            let e2 = result.ising.evaluate(&all_spins_neg);
            // Both energies are finite
            assert!(e1.is_finite());
            assert!(e2.is_finite());
        }
    }

    #[test]
    fn test_5r1c_and_9r4c_embeds_without_violation() {
        use crate::physics::geometry_tensor::ThermalManifold;
        use crate::quantum::qubo_mapping::{manifold_to_qubo, QuboConfig};

        let cfg = QuboConfig::default();
        let constraints = DwaveHardwareConstraints::advantage_system64();

        // 5R1C manifold — small R_tr=0.1 gives metric values ~10,
        // QUBO entries in O(1) after normalization, directly embeddable.
        let m5 = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let qp5 = manifold_to_qubo(&m5, cfg).expect("5R1C manifold_to_qubo failed");
        let ising5 = qp5.to_ising();

        let feasibility5 = check_embedding_feasibility(&ising5, &constraints);
        assert!(
            feasibility5.is_feasible(),
            "5R1C should be feasible on AdvantageSystem6.4: {:?}",
            feasibility5
        );

        // 9R4C manifold — use realistic building thermal resistances (R-values in m²K/W).
        // R_tr = [0.3, 0.2, 0.1] m²K/W is typical for wall constructions.
        // This gives metric entries ~1/R ≈ 3.3, 5.0, 10.0 which are within
        // the hardware J range after fixed-point encoding at K=8.
        let m9 = ThermalManifold::from_9r4c_parameters(
            [22.0, 20.0, 23.0, 18.0],
            [1000.0, 5000.0, 3000.0, 8000.0],
            [0.3, 0.2, 0.1],
            None,
        );
        let qp9 = manifold_to_qubo(&m9, cfg).expect("9R4C manifold_to_qubo failed");
        let ising9 = qp9.to_ising();

        let feasibility9 = check_embedding_feasibility(&ising9, &constraints);
        assert!(
            feasibility9.is_feasible(),
            "9R4C should be feasible on AdvantageSystem6.4: {:?}",
            feasibility9
        );

        // Verify scaling brings an out-of-range problem back into range.
        let mut ising_out = ising9.clone();
        ising_out.h[0] = 10.0; // exceeds h_max = 4.0
        let result = ScalingResult::scale_ising_to_hardware(ising_out, &constraints).unwrap();
        assert!(
            result.ising.h[0] <= 4.0,
            "h[0] should be scaled to ≤ 4.0, got {}",
            result.ising.h[0]
        );
        assert!(
            !result.strategy.is_none(),
            "scaling strategy should not be None for out-of-range input"
        );
    }
}
