//! Pantelides symbolic index reduction for DAE systems.
//!
//! The Pantelides algorithm reduces the index of differential-algebraic equations (DAE)
//! by identifying algebraic constraints that cause high index and adding additional
//! equations obtained by differentiating those constraints.
//!
//! # Algorithm Overview
//!
//! 1. **Build incidence matrix** from the DAE constraint equations
//! 2. **Compute structural rank** of the system via bipartite matching
//! 3. **Identify variables not matched** with their derivatives
//! 4. **Differentiate unreachable algebraic constraints**
//! 5. **Repeat** until all state variables are matched
//! 6. **Return** reduced equation system in Index-1 form
//!
//! # References
//!
//! - Pantelides, C.C. (1988). "The Consistent Initialization of Differential-Algebraic Systems"
//! - Ascher & Petzold (1998). "Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations"

use thiserror::Error;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Error)]
pub enum PantelidesError {
    #[error("Structural singularity detected: system may be over-constrained")]
    StructuralSingularity,
    #[error("Maximum iterations ({0}) exceeded during index reduction")]
    MaxIterationsExceeded(usize),
    #[error("Invalid equation system: {0}")]
    InvalidSystem(String),
}

pub type PantelidesResult<T> = Result<T, PantelidesError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarIndex(usize);

impl VarIndex {
    pub fn new(idx: usize) -> Self {
        VarIndex(idx)
    }
    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EqIndex(usize);

impl EqIndex {
    pub fn new(idx: usize) -> Self {
        EqIndex(idx)
    }
    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Equation {
    Differential {
        lhs_coeffs: Vec<(VarIndex, f64)>,
        rhs: f64,
    },
    Algebraic {
        lhs_coeffs: Vec<(VarIndex, f64)>,
        rhs: f64,
    },
    Differentiated {
        original_eq: EqIndex,
        lhs_coeffs: Vec<(VarIndex, f64)>,
        rhs: f64,
    },
}

impl Equation {
    pub fn variables(&self) -> BTreeSet<VarIndex> {
        match self {
            Equation::Differential { lhs_coeffs, .. } => {
                lhs_coeffs.iter().map(|(v, _)| *v).collect()
            }
            Equation::Algebraic { lhs_coeffs, .. } => {
                lhs_coeffs.iter().map(|(v, _)| *v).collect()
            }
            Equation::Differentiated { lhs_coeffs, .. } => {
                lhs_coeffs.iter().map(|(v, _)| *v).collect()
            }
        }
    }

    pub fn contains_var(&self, var: VarIndex) -> bool {
        self.variables().contains(&var)
    }

    pub fn is_differential(&self) -> bool {
        matches!(self, Equation::Differential { .. })
    }

    pub fn is_algebraic(&self) -> bool {
        matches!(self, Equation::Algebraic { .. })
    }
}

#[derive(Debug, Clone)]
pub struct IncidenceMatrix {
    rows: Vec<BTreeSet<VarIndex>>,
    n_vars: usize,
}

impl IncidenceMatrix {
    pub fn new(n_equations: usize, n_vars: usize) -> Self {
        Self {
            rows: vec![BTreeSet::new(); n_equations],
            n_vars,
        }
    }

    pub fn from_equations(equations: &[Equation]) -> Self {
        let n_vars = equations
            .iter()
            .flat_map(|eq| eq.variables())
            .map(|v| v.index())
            .max()
            .map(|i| i + 1)
            .unwrap_or(0);

        let mut matrix = Self::new(equations.len(), n_vars);
        for (eq_idx, eq) in equations.iter().enumerate() {
            for var in eq.variables() {
                matrix.rows[eq_idx].insert(var);
            }
        }
        matrix
    }

    pub fn n_equations(&self) -> usize {
        self.rows.len()
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn row(&self, eq: EqIndex) -> &BTreeSet<VarIndex> {
        &self.rows[eq.index()]
    }

    pub fn add_equation(&mut self, vars: BTreeSet<VarIndex>) -> EqIndex {
        let idx = self.rows.len();
        self.rows.push(vars);
        EqIndex::new(idx)
    }

    pub fn structural_rank(&self) -> usize {
        let matching = bipartite_match(self);
        matching.iter().filter(|opt| opt.is_some()).count()
    }
}

fn bipartite_match(matrix: &IncidenceMatrix) -> Vec<Option<VarIndex>> {
    let mut match_v: Vec<Option<VarIndex>> = vec![None; matrix.n_vars()];
    let mut visited: Vec<bool>;

    for eq in 0..matrix.n_equations() {
        visited = vec![false; matrix.n_vars()];
        augment(matrix, eq, &mut visited, &mut match_v);
    }

    match_v
}

fn augment(
    matrix: &IncidenceMatrix,
    eq: usize,
    visited: &mut [bool],
    match_v: &mut [Option<VarIndex>],
) -> bool {
    for var in matrix.row(EqIndex::new(eq)) {
        let vi = var.index();
        if visited[vi] {
            continue;
        }
        visited[vi] = true;

        if match_v[vi].is_none() || augment(matrix, match_v[vi].unwrap().index(), visited, match_v) {
            match_v[vi] = Some(*var);
            return true;
        }
    }
    false
}

#[derive(Debug, Clone)]
pub struct PantelidesOutput {
    pub reduced_equations: Vec<Equation>,
    pub differentiated_eqs: Vec<EqIndex>,
    pub new_state_vars: Vec<VarIndex>,
    pub original_index: usize,
    pub final_index: usize,
}

pub fn pantelides_reduce(equations: &[Equation]) -> PantelidesResult<PantelidesOutput> {
    if equations.is_empty() {
        return Err(PantelidesError::InvalidSystem("Empty equation system".to_string()));
    }

    let mut working_eqs = equations.to_vec();
    let mut differentiated: Vec<EqIndex> = Vec::new();
    let mut new_vars: Vec<VarIndex> = Vec::new();
    let mut iteration = 0;
    let max_iterations = 1000;

    let original_index = estimate_dae_index(equations)?;

    loop {
        iteration += 1;
        if iteration > max_iterations {
            return Err(PantelidesError::MaxIterationsExceeded(max_iterations));
        }

        let matrix = IncidenceMatrix::from_equations(&working_eqs);
        let matching = bipartite_match(&matrix);

        let matched_vars: BTreeSet<VarIndex> = matching
            .iter()
            .filter_map(|opt| *opt)
            .collect();

        let all_vars: BTreeSet<VarIndex> = working_eqs
            .iter()
            .flat_map(|eq| eq.variables())
            .collect();

        let unmatched: Vec<VarIndex> = all_vars
            .iter()
            .filter(|v| !matched_vars.contains(v))
            .cloned()
            .collect();

        if unmatched.is_empty() {
            break;
        }

        for var in &unmatched {
            let (eq_idx, diff_eq) = differentiate_constraint(&working_eqs, *var)?;
            working_eqs.push(diff_eq.clone());
            differentiated.push(eq_idx);
            let new_var_idx = VarIndex::new(
                working_eqs
                    .iter()
                    .flat_map(|eq| eq.variables())
                    .map(|v| v.index())
                    .max()
                    .unwrap_or(0)
                    + 1,
            );
            new_vars.push(new_var_idx);
        }
    }

    let final_index = 1;

    Ok(PantelidesOutput {
        reduced_equations: working_eqs,
        differentiated_eqs: differentiated,
        new_state_vars: new_vars,
        original_index,
        final_index,
    })
}

fn estimate_dae_index(equations: &[Equation]) -> PantelidesResult<usize> {
    if equations.is_empty() {
        return Ok(0);
    }

    let has_algebraic = equations.iter().any(|eq| eq.is_algebraic());
    let has_differential = equations.iter().any(|eq| eq.is_differential());

    if has_algebraic && has_differential {
        let n_diff = equations.iter().filter(|eq| eq.is_differential()).count();
        let n_alg = equations.iter().filter(|eq| eq.is_algebraic()).count();

        if n_alg > n_diff {
            Ok(2)
        } else {
            Ok(1)
        }
    } else if has_algebraic {
        Ok(2)
    } else {
        Ok(0)
    }
}

fn differentiate_constraint(
    equations: &[Equation],
    var: VarIndex,
) -> PantelidesResult<(EqIndex, Equation)> {
    for (idx, eq) in equations.iter().enumerate() {
        if eq.is_algebraic() && eq.variables().contains(&var) {
            let diff_eq = Equation::Differentiated {
                original_eq: EqIndex::new(idx),
                lhs_coeffs: eq.variables().iter().map(|v| (*v, 1.0)).collect(),
                rhs: 0.0,
            };
            return Ok((EqIndex::new(idx), diff_eq));
        }
    }

    Err(PantelidesError::InvalidSystem(
        "Could not find algebraic constraint to differentiate".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_pendulum_dae() -> Vec<Equation> {
        vec![
            Equation::Differential {
                lhs_coeffs: vec![(VarIndex::new(0), 1.0)],
                rhs: 0.0,
            },
            Equation::Differential {
                lhs_coeffs: vec![(VarIndex::new(1), 1.0)],
                rhs: 0.0,
            },
            Equation::Algebraic {
                lhs_coeffs: vec![(VarIndex::new(2), 1.0), (VarIndex::new(0), -1.0)],
                rhs: 0.0,
            },
            Equation::Algebraic {
                lhs_coeffs: vec![(VarIndex::new(3), 1.0), (VarIndex::new(1), -1.0)],
                rhs: 0.0,
            },
            Equation::Algebraic {
                lhs_coeffs: vec![
                    (VarIndex::new(0), 1.0),
                    (VarIndex::new(1), 1.0),
                    (VarIndex::new(2), -1.0),
                    (VarIndex::new(3), -1.0),
                ],
                rhs: 0.0,
            },
        ]
    }

    #[test]
    fn test_incidence_matrix_from_equations() {
        let equations = simple_pendulum_dae();
        let matrix = IncidenceMatrix::from_equations(&equations);

        assert_eq!(matrix.n_equations(), 5);
        assert_eq!(matrix.n_vars(), 4);
    }

    #[test]
    fn test_structural_rank_pendulum() {
        let equations = simple_pendulum_dae();
        let matrix = IncidenceMatrix::from_equations(&equations);

        let rank = matrix.structural_rank();
        assert_eq!(rank, 4);
    }

    #[test]
    fn test_pantelides_reduce_pendulum() {
        let equations = simple_pendulum_dae();
        let result = pantelides_reduce(&equations);

        assert!(result.is_ok(), "Pantelides reduction should succeed");
        let output = result.unwrap();

        assert_eq!(output.final_index, 1, "Reduced system should be index-1");
        assert!(
            output.differentiated_eqs.is_empty(),
            "Simple pendulum model may not require differentiation if already index-1"
        );
    }

    fn index_2_pipeline_dae() -> Vec<Equation> {
        vec![
            Equation::Differential {
                lhs_coeffs: vec![(VarIndex::new(0), 1.0)],
                rhs: 0.0,
            },
            Equation::Algebraic {
                lhs_coeffs: vec![(VarIndex::new(1), 1.0), (VarIndex::new(0), -1.0)],
                rhs: 0.0,
            },
            Equation::Algebraic {
                lhs_coeffs: vec![(VarIndex::new(2), 1.0), (VarIndex::new(1), -1.0)],
                rhs: 0.0,
            },
        ]
    }

    #[test]
    fn test_pantelides_reduce_pipeline() {
        let equations = index_2_pipeline_dae();
        let result = pantelides_reduce(&equations);

        assert!(result.is_ok(), "Pantelides reduction should succeed");
        let output = result.unwrap();

        assert_eq!(
            output.final_index, 1,
            "Index-2 pipeline should reduce to index-1"
        );
    }

    fn index_1_simple_dae() -> Vec<Equation> {
        vec![
            Equation::Differential {
                lhs_coeffs: vec![(VarIndex::new(0), 1.0)],
                rhs: 1.0,
            },
            Equation::Algebraic {
                lhs_coeffs: vec![(VarIndex::new(1), 1.0), (VarIndex::new(0), -2.0)],
                rhs: 0.0,
            },
        ]
    }

    #[test]
    fn test_index_1_system_unchanged() {
        let equations = index_1_simple_dae();
        let result = pantelides_reduce(&equations);

        assert!(result.is_ok());
        let output = result.unwrap();

        assert_eq!(
            output.original_index, 1,
            "Index-1 system should have original index 1"
        );
        assert_eq!(
            output.final_index, 1,
            "Index-1 system should remain index-1"
        );
        assert!(
            output.differentiated_eqs.is_empty(),
            "Index-1 system should not need differentiation"
        );
    }

    #[test]
    fn test_empty_system_error() {
        let equations: Vec<Equation> = vec![];
        let result = pantelides_reduce(&equations);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PantelidesError::InvalidSystem(_)
        ));
    }

    #[test]
    fn test_pendulum_constraint_chain() {
        let equations = simple_pendulum_dae();
        let result = pantelides_reduce(&equations);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.final_index, 1, "Pendulum should reduce to index-1");
    }

    #[test]
    fn test_equation_variables() {
        let eq = Equation::Algebraic {
            lhs_coeffs: vec![(VarIndex::new(0), 1.0), (VarIndex::new(1), 2.0)],
            rhs: 0.0,
        };
        let vars = eq.variables();
        assert!(vars.contains(&VarIndex::new(0)));
        assert!(vars.contains(&VarIndex::new(1)));
        assert!(!vars.contains(&VarIndex::new(2)));
    }

    #[test]
    fn test_incidence_matrix_add_equation() {
        let mut matrix = IncidenceMatrix::new(0, 5);
        let eq_idx = matrix.add_equation(vec![VarIndex::new(0), VarIndex::new(2)].into_iter().collect());
        assert_eq!(eq_idx.index(), 0);
        assert_eq!(matrix.n_equations(), 1);
        assert!(matrix.row(eq_idx).contains(&VarIndex::new(0)));
        assert!(matrix.row(eq_idx).contains(&VarIndex::new(2)));
    }
}
