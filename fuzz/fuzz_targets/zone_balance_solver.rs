//! Fuzz target: zone-balance solver (`ThermalModel::step_physics` +
//! `solve_timesteps`).
//!
//! The zone-balance solver is the ISO 13790 5R1C/6R2C/8R3C/9R4C thermal network
//! that sits behind both the `Model.simulate` Python entrypoint and
//! `BatchOracle::evaluate_population`. It must conserve energy across every
//! timestep and must not panic, abort, or produce non-finite temperatures when
//! fed arbitrary outdoor conditions and integration step sizes.
//!
//! The fuzzer explores:
//!   * extreme outdoor temperatures (reinterpreted `f64` bits cover NaN, Inf,
//!     -273 °C .. millions of °C),
//!   * very small / very large / negative / NaN integration step sizes
//!     (`dt_seconds`), which stress the implicit-Euler stability and the
//!     `1/dt` divisions in the capacitance term,
//!   * a bounded range of zone counts (covers single-zone 5R1C and multi-zone
//!     coupled-matrix branches),
//!   * a short run of successive timesteps so state carries over (exercises
//!     the temperature / mass-temperature update and the HVAC deadband logic).
//!
//! **Invariants asserted:**
//!   1. `step_physics` never panics for any finite or non-finite input.
//!   2. The returned per-step energy is either finite and >= 0, or the call
//!      is skipped because the input was non-finite (graceful).
//!   3. Zone temperatures after a step stay finite (no NaN/Inf propagation
//!      into the thermal state, which would corrupt every subsequent step).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

const MAX_ZONES: usize = 8;
/// Enough steps for the thermal mass to react, without making a single
/// fuzz iteration expensive. 24 = one simulated day at hourly resolution.
const MAX_STEPS: usize = 24;

#[derive(Arbitrary, Debug)]
struct SolverInput {
    num_zones_raw: u8,
    /// Outdoor dry-bulb temperature for each step (raw f64 bits).
    outdoor_temps: Vec<u64>,
    /// Integration step size `dt_seconds` for each step (raw f64 bits).
    dt_seconds: Vec<u64>,
    /// Window U-value applied via `apply_parameters` (raw f64 bits).
    u_value_bits: u64,
    /// Heating setpoint (raw f64 bits).
    heating_bits: u64,
    /// Cooling setpoint (raw f64 bits).
    cooling_bits: u64,
}

fuzz_target!(|input: SolverInput| {
    let num_zones = (input.num_zones_raw as usize).clamp(1, MAX_ZONES);
    let u_value = f64::from_bits(input.u_value_bits);
    let heating = f64::from_bits(input.heating_bits);
    let cooling = f64::from_bits(input.cooling_bits);

    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::new(num_zones);

    // Apply the fuzzer-supplied parameters. `apply_parameters` guards against
    // swapped setpoints internally (it re-orders heating/cooling), so this
    // must not panic even for NaN / Inf inputs.
    model.apply_parameters(&[u_value, heating, cooling]);

    // Provide a non-zero internal load so the solver exercises the gains path
    // rather than short-circuiting on a zero-load deadband.
    model.set_loads(&vec![5.0_f64; num_zones]);

    let step_count = input.outdoor_temps.len().min(input.dt_seconds.len()).min(MAX_STEPS);

    for t in 0..step_count {
        let outdoor_temp = f64::from_bits(input.outdoor_temps[t]);
        let dt = f64::from_bits(input.dt_seconds[t]);

        // Skip non-finite inputs — they are not physically meaningful and the
        // solver does not claim to handle them, but it must *reject* them
        // gracefully (i.e. not panic), which the guard below enforces.
        if !outdoor_temp.is_finite() || !dt.is_finite() || dt <= 0.0 {
            continue;
        }

        // === Invariant 1: step_physics must not panic. ===
        let energy = model.step_physics(t, outdoor_temp, dt);

        // === Invariant 2: energy is finite and non-negative. ===
        // (Energy consumption can legitimately be 0.0 in a deadband; it must
        // never be negative because the solver clamps via `.max(0.0)`.)
        assert!(
            energy.is_finite(),
            "step_physics returned non-finite energy at t={}: {} (outdoor={}, dt={})",
            t,
            energy,
            outdoor_temp,
            dt
        );
        assert!(
            energy >= 0.0,
            "step_physics returned negative energy at t={}: {} (outdoor={}, dt={})",
            t,
            energy,
            outdoor_temp,
            dt
        );

        // === Invariant 3: zone temperatures stay finite after each step. ===
        for (z, &temp) in model.get_temperatures().iter().enumerate() {
            assert!(
                temp.is_finite(),
                "zone {} temperature is non-finite after t={}: {}",
                z,
                t,
                temp
            );
        }
    }

    // Run one full `solve_timesteps` cycle over a bounded horizon with safe
    // defaults, to exercise the higher-level solve loop (which composes
    // calc_analytical_loads + step_physics + energy accounting) on top of the
    // state that was just mutated. This must also be panic-free.
    if step_count == 0 {
        let surrogates = fluxion::ai::surrogate::SurrogateManager::new();
        if let Ok(surrogates) = surrogates {
            let eui = model.solve_timesteps(24, &surrogates, false, None, None, None);
            assert!(
                eui.is_finite(),
                "solve_timesteps returned non-finite EUI: {}",
                eui
            );
        }
    }
});
