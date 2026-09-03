//! Bipolar-sign regression test for parallel transport (issue #1677)
//!
//! ## Purpose
//!
//! Validates that [`ThermalManifold::compute_parallel_transport`] produces the
//! **bipolar sign** behavior required by the Phase 3 gauge validation harness
//! (issue #1465). The transport must show:
//!
//! - **Daytime**: positive flux (heat gain) when `gauge_connection[0]` (solar) > 0
//! - **Nighttime**: negative flux (heat loss) when `gauge_connection[0]` (solar) ≈ 0
//!
//! This behavior was not enforced by any existing test — the forward-Euler stub
//! in `compute_parallel_transport` was suspected to produce a monotonically damped
//! value rather than a sign-flipping transport.
//!
//! ## Test structure
//!
//! 1. **5R1C scene**: `T_air=20°C`, `T_mass=15°C`, `dt=3600s` (1 hour)
//! 2. **Day case**: `gauge_connection[0] = 800.0` (strong solar) → flux must be **positive**
//! 3. **Night case**: `gauge_connection[0] = 0.0` (no solar) → flux must be **negative**
//! 4. **Sign-flip assertion**: day transport sign ≠ night transport sign
//!
//! ## Acceptance criteria
//!
//! - `cargo test gauge_transport_bipolar_sign` passes
//! - Transport flux sign matches `gauge_connection` sign:
//!   - positive `gauge_connection[0]` → positive transport (daytime heat gain)
//!   - zero `gauge_connection[0]` → negative transport (nighttime heat loss)

use fluxion::physics::geometry_tensor::{ManifoldIndex, ThermalManifold};

/// ASHRAE 140 Case 900 envelope geometry — 200 mm heavy-weight concrete wall.
const CASE_900_HW_CONCRETE_THICKNESS_M: f64 = 0.200;
const CASE_900_HW_CONCRETE_K_W_MK: f64 = 0.51;
const CASE_900_HW_CONCRETE_RHO_KG_M3: f64 = 1400.0;
const CASE_900_HW_CONCRETE_CP_J_KGK: f64 = 840.0;

/// 5R1C equivalent resistance for a 200 mm HW concrete wall.
/// R = thickness / (k · A) = 0.2 / (0.51 · 1) ≈ 0.392 Ω·m²/K
const R_EQ_M2K_W: f64 = CASE_900_HW_CONCRETE_THICKNESS_M / (CASE_900_HW_CONCRETE_K_W_MK * 1.0);

/// Air node capacitance for a typical zone: C = ρ·V·c_p / A
/// ≈ (1.2 kg/m³ · 2.5 m ceiling · 1 m² · 1006 J/kgK) = ~3018 J/m²K
const C_AIR_J_M2K: f64 = 1.2 * 2.5 * 1006.0;

/// Wall mass capacitance: C = ρ·V·c_p / A
/// = (1400 kg/m³ · 0.2 m · 1 m² · 840 J/kgK) = 235,200 J/m²K
const C_MASS_J_M2K: f64 = CASE_900_HW_CONCRETE_RHO_KG_M3
    * CASE_900_HW_CONCRETE_THICKNESS_M
    * CASE_900_HW_CONCRETE_CP_J_KGK;

/// Timestep: 1 hour = 3600 s
const DT_SECONDS: f64 = 3600.0;

/// Build a 5R1C thermal manifold with Case 900 geometry and the given initial
/// temperatures. The air node is at `t_air`, the wall mass node at `t_mass`.
/// Roof and floor slots are parked at zero (inert).
fn make_5r1c_manifold(t_air: f64, t_mass: f64) -> ThermalManifold {
    ThermalManifold::from_5r1c_parameters(t_air, t_mass, R_EQ_M2K_W, C_AIR_J_M2K, C_MASS_J_M2K)
}

/// The transport flux is the **change** in the air-node temperature per timestep,
/// divided by dt. Units: K/s → convert to W/m² by multiplying by C_air.
/// This gives the heat flux at the air node due to gauge transport.
fn transport_flux_wm2(manifold: &ThermalManifold, dt: f64) -> f64 {
    let transported = manifold.compute_parallel_transport(dt);
    let d_t_air = transported[ManifoldIndex::Air as usize]
        - manifold.scalar_field[ManifoldIndex::Air as usize];
    // Q = C · dT/dt  →  W/m² = J/m²K · K/s
    C_AIR_J_M2K * d_t_air / dt
}

/// Regression test: parallel transport must show bipolar sign behavior.
///
/// Daytime (solar > 0): transport flux is **positive** (heat entering the zone).
/// Nighttime (solar ≈ 0): transport flux is **negative** (heat leaving the zone).
///
/// This is the core acceptance criterion for the Phase 3 gauge validation harness.
#[test]
fn test_parallel_transport_bipolar_sign() {
    // Initial conditions: T_air=20°C (slightly warm), T_mass=15°C (cooler mass).
    // At night with no solar, heat flows from air → mass (positive flux into mass).
    // During day with strong solar, heat flows from mass → air (or solar heats mass).
    let t_air = 20.0;
    let t_mass = 15.0;

    // --- DAY CASE: strong solar irradiance (800 W/m² at noon) ---
    let mut day_manifold = make_5r1c_manifold(t_air, t_mass);
    // gauge_connection[0] = solar into air node (W/m² equivalent)
    day_manifold.gauge_connection[ManifoldIndex::Air as usize] = 800.0;
    // gauge_connection[1] = solar into wall mass node (absorbed solar)
    day_manifold.gauge_connection[ManifoldIndex::Wall as usize] = 400.0;

    let day_flux = transport_flux_wm2(&day_manifold, DT_SECONDS);

    // --- NIGHT CASE: no solar (midnight) ---
    let mut night_manifold = make_5r1c_manifold(t_air, t_mass);
    // gauge_connection[0] = 0 (no solar at night)
    night_manifold.gauge_connection[ManifoldIndex::Air as usize] = 0.0;
    // No absorbed solar on wall either
    night_manifold.gauge_connection[ManifoldIndex::Wall as usize] = 0.0;

    let night_flux = transport_flux_wm2(&night_manifold, DT_SECONDS);

    // --- Bipolar sign assertions ---

    // AC1: Daytime flux must be positive (heat gain into the zone).
    assert!(
        day_flux > 0.0,
        "Daytime transport flux must be POSITIVE (heat gain), got {day_flux:.2} W/m²",
    );

    // AC2: Nighttime flux must be negative (heat loss from the zone).
    assert!(
        night_flux < 0.0,
        "Nighttime transport flux must be NEGATIVE (heat loss), got {night_flux:.2} W/m²",
    );

    // AC3: Signs must be opposite (bipolar, not unipolar or zero).
    assert!(
        day_flux.signum() != night_flux.signum(),
        "Transport must show bipolar sign flip: day={day_flux:.2} W/m², night={night_flux:.2} W/m²",
    );

    // AC4: Magnitude must be physically meaningful (> 1 W/m² to exclude rounding).
    assert!(
        day_flux.abs() > 1.0 && night_flux.abs() > 1.0,
        "Transport flux magnitude too small: day={day_flux:.2}, night={night_flux:.2} W/m²",
    );
}

/// Verify that the bipolar sign property holds across a range of solar intensities.
/// At zero solar, flux is negative. At high solar, flux is positive. The crossover
/// happens somewhere between.
#[test]
fn test_parallel_transport_bipolar_sign_transitions_correctly() {
    let t_air = 20.0;
    let t_mass = 15.0;

    // Test a sweep of solar values
    let solar_values = [0.0, 100.0, 300.0, 500.0, 800.0, 1000.0];

    for solar in solar_values {
        let mut manifold = make_5r1c_manifold(t_air, t_mass);
        manifold.gauge_connection[ManifoldIndex::Air as usize] = solar;
        manifold.gauge_connection[ManifoldIndex::Wall as usize] = solar * 0.5;

        let flux = transport_flux_wm2(&manifold, DT_SECONDS);

        if solar == 0.0 {
            assert!(
                flux < 0.0,
                "Zero solar: flux should be negative (heat loss), got {flux:.2} W/m²",
            );
        } else if solar >= 500.0 {
            assert!(
                flux > 0.0,
                "High solar ({solar} W/m²): flux should be positive (heat gain), got {flux:.2} W/m²",
            );
        }
        // Intermediate values may go either way; we only check extremes
    }
}

/// Verify that the transport is not a trivially fixed-point (zero change).
/// Non-trivial initial conditions and non-zero forcing must produce non-zero transport.
#[test]
fn test_parallel_transport_is_not_fixed_point() {
    let t_air = 20.0;
    let t_mass = 15.0;
    let solar = 800.0;

    let mut manifold = make_5r1c_manifold(t_air, t_mass);
    manifold.gauge_connection[ManifoldIndex::Air as usize] = solar;
    manifold.gauge_connection[ManifoldIndex::Wall as usize] = 400.0;

    let transported = manifold.compute_parallel_transport(DT_SECONDS);

    // At least one slot should change
    let any_change = (0..4).any(|i| (transported[i] - manifold.scalar_field[i]).abs() > 1e-10);
    assert!(
        any_change,
        "Transport with non-zero forcing should produce non-trivial field evolution",
    );

    // The air slot (which couples to the zone) should change
    let d_t_air = (transported[ManifoldIndex::Air as usize]
        - manifold.scalar_field[ManifoldIndex::Air as usize])
        .abs();
    assert!(
        d_t_air > 1e-10,
        "Air node temperature should evolve under non-zero solar forcing",
    );
}
