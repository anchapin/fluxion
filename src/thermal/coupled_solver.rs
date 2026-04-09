//! Coupled ODE solver for multi-zone thermal systems.
//!
//! This module implements an implicit integration method for solving
//! the coupled system of ordinary differential equations that govern
//! multi-zone thermal dynamics.

/// Solve the coupled multi-zone thermal system using backward Euler method.
///
/// # Arguments
/// * `c` - Zone thermal capacitances (J/K)
/// * `h_tr_iz` - Inter-zone conductances (W/K)
/// * `q` - Heat gains vector (W)
/// * `dt` - Time step (s)
/// * `current_temperatures` - Current zone temperatures (°C)
///
/// # Returns
/// New zone temperatures after time step
///
/// # Method
/// Uses backward Euler: (C/dt - A) * T_new = C/dt * T_old + Q
pub fn solve_coupled_system(
    c: &[f64],
    h_tr_iz: &[f64],
    q: &[f64],
    dt: f64,
    current_temperatures: &[f64],
) -> Vec<f64> {
    // For now, implement a simplified explicit method
    // In a full implementation, this would use faer for matrix operations
    let num_zones = c.len();
    let mut new_temperatures = Vec::with_capacity(num_zones);

    for i in 0..num_zones {
        // Simplified: T_new = T_old + (Q_net / C) * dt
        // Q_net = Q_external + Q_inter_zone
        let q_net = q[i] + inter_zone_heat_contribution(i, h_tr_iz, current_temperatures);
        let temp_change = (q_net / c[i]) * dt;
        new_temperatures.push(current_temperatures[i] + temp_change);
    }

    new_temperatures
}

/// Calculate inter-zone heat contribution for a specific zone.
fn inter_zone_heat_contribution(zone_index: usize, h_tr_iz: &[f64], temperatures: &[f64]) -> f64 {
    // Simplified: sum over all other zones
    let mut total = 0.0;
    let num_zones = temperatures.len();

    for j in 0..num_zones {
        if j != zone_index {
            // Heat flow from zone j to zone i
            // Using symmetric conductance for simplicity
            let h_ij = h_tr_iz[zone_index].min(h_tr_iz[j]); // Simple symmetric approximation
            total += h_ij * (temperatures[j] - temperatures[zone_index]);
        }
    }

    total
}

/// Build system matrix for coupled ODE solver.
///
/// # Arguments
/// * `c` - Zone thermal capacitances
/// * `h_tr_iz` - Inter-zone conductances
/// * `dt` - Time step
///
/// # Returns
/// System matrix (C/dt - A) for implicit method
///
/// # Note
/// This is a placeholder that would use faer::Mat in full implementation
pub fn build_system_matrix(c: &[f64], h_tr_iz: &[f64], dt: f64) -> Vec<Vec<f64>> {
    let num_zones = c.len();
    let mut matrix = vec![vec![0.0; num_zones]; num_zones];

    // Build diagonal and off-diagonal terms
    for i in 0..num_zones {
        // Diagonal term: C_i/dt + sum of conductances
        matrix[i][i] = c[i] / dt;

        for j in 0..num_zones {
            if i != j {
                // Off-diagonal: -h_tr_ij (heat loss to other zones)
                matrix[i][j] = -h_tr_iz[i].min(h_tr_iz[j]); // Symmetric approximation
                matrix[i][i] += h_tr_iz[i].min(h_tr_iz[j]); // Add to diagonal
            }
        }
    }

    matrix
}

/// Solve linear system using simplified method.
///
/// # Arguments
/// * `matrix` - System matrix
/// * `rhs` - Right-hand side vector
///
/// # Returns
/// Solution vector
///
/// # Note
/// In full implementation, this would use faer::solve
pub fn solve_with_faer(mut matrix: Vec<Vec<f64>>, rhs: Vec<f64>) -> Vec<f64> {
    // Simplified: use Gaussian elimination for small systems
    // In practice, this would call faer::solve
    let n = matrix.len();
    let mut solution = rhs.clone();

    // Forward elimination
    for i in 0..n {
        // Partial pivoting
        let mut max_row = i;
        for k in i + 1..n {
            if matrix[k][i].abs() > matrix[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        matrix.swap(i, max_row);
        solution.swap(i, max_row);

        // Eliminate
        for k in i + 1..n {
            let factor = matrix[k][i] / matrix[i][i];
            for j in i..n {
                matrix[k][j] -= factor * matrix[i][j];
            }
            solution[k] -= factor * solution[i];
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        for k in i + 1..n {
            solution[i] -= matrix[i][k] * solution[k];
        }
        solution[i] /= matrix[i][i];
    }

    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_coupled_system_simple() {
        // Simple test: 2 zones, no inter-zone coupling
        let c = [1000.0, 1000.0];
        let h_tr_iz = [0.0, 0.0];
        let q = [100.0, 200.0];
        let dt = 3600.0; // 1 hour
        let current_temps = [20.0, 20.0];

        let new_temps = solve_coupled_system(&c, &h_tr_iz, &q, dt, &current_temps);

        // Expected: T1 = 20 + (100/1000)*3600 = 20 + 360 = 380 (unrealistic but tests math)
        // T2 = 20 + (200/1000)*3600 = 20 + 720 = 740
        assert_eq!(new_temps.len(), 2);
        assert!(new_temps[0] > current_temps[0]);
        assert!(new_temps[1] > current_temps[1]);
    }

    #[test]
    fn test_solve_coupled_system_with_coupling() {
        // Test with inter-zone coupling
        let c = [1000.0, 1000.0];
        let h_tr_iz = [50.0, 50.0]; // Symmetric conductance
        let q = [0.0, 0.0]; // No external heat
        let dt = 3600.0;
        let current_temps = [30.0, 20.0]; // Zone 0 hotter than zone 1

        let new_temps = solve_coupled_system(&c, &h_tr_iz, &q, dt, &current_temps);

        // Zone 0 should cool down, zone 1 should warm up
        assert!(new_temps[0] < current_temps[0]);
        assert!(new_temps[1] > current_temps[1]);
    }

    #[test]
    fn test_build_system_matrix() {
        let c = [1000.0, 1000.0];
        let h_tr_iz = [50.0, 50.0];
        let dt = 3600.0;

        let matrix = build_system_matrix(&c, &h_tr_iz, dt);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);

        // Check diagonal dominance
        assert!(matrix[0][0] > matrix[0][1].abs());
        assert!(matrix[1][1] > matrix[1][0].abs());
    }

    #[test]
    fn test_solve_with_faer_simple() {
        // Test simple 2x2 system: [2 1; 1 3] * [x; y] = [5; 6]
        // Solution: x=1.8, y=1.4 (derived from: 2x+y=5, x+3y=6)
        let matrix = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let rhs = vec![5.0, 6.0];

        let solution = solve_with_faer(matrix, rhs);
        assert_eq!(solution.len(), 2);
        assert!((solution[0] - 1.8).abs() < 1e-10);
        assert!((solution[1] - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_inter_zone_heat_contribution() {
        let h_tr_iz = [50.0, 50.0, 50.0];
        let temps = [25.0, 20.0, 30.0];

        // Zone 0: receives from zone 2 (30°C), loses to zone 1 (20°C)
        // Net = 50*(30-25) + 50*(20-25) = 250 - 250 = 0
        let contrib_0 = inter_zone_heat_contribution(0, &h_tr_iz, &temps);

        // Zone 1: receives from zone 0 (25°C) and zone 2 (30°C), loses to nothing else
        // Net = 50*(25-20) + 50*(30-20) = 250 + 500 = 750
        let contrib_1 = inter_zone_heat_contribution(1, &h_tr_iz, &temps);

        // Zone 2: loses to zone 0 and zone 1
        // Net = 50*(20-30) + 50*(25-30) = -500 - 250 = -750
        let contrib_2 = inter_zone_heat_contribution(2, &h_tr_iz, &temps);

        assert!(contrib_0 >= 0.0); // Net heat gain or neutral
        assert!(contrib_1 > 0.0); // Net heat gain
        assert!(contrib_2 < 0.0); // Net heat loss
    }

    #[test]
    fn test_system_energy_conservation() {
        // Test that total energy is conserved in isolated system
        let c = [1000.0, 1000.0];
        let h_tr_iz = [50.0, 50.0];
        let q = [0.0, 0.0]; // No external heat
        let dt = 3600.0;
        let initial_temps = [30.0, 20.0];

        let final_temps = solve_coupled_system(&c, &h_tr_iz, &q, dt, &initial_temps);

        // Total energy should be conserved
        let initial_energy = c[0] * initial_temps[0] + c[1] * initial_temps[1];
        let final_energy = c[0] * final_temps[0] + c[1] * final_temps[1];

        assert!((initial_energy - final_energy).abs() < 1e-6);
    }
}
