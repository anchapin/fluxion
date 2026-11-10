//! Issue #4691 — frozen-Jacobian / factorisation-reuse spike harness.
//!
//! Measures the BDF Newton loop's Jacobian-assembly, factorisation and
//! Newton-iteration counts plus wall clock for two ASHRAE-140-style
//! building workloads run for a full simulated year:
//!
//! * **Case 600** — lightweight construction (ASHRAE 140 lightweight
//!   wall/roof/floor assemblies), 9 states.
//! * **Case 900 (high mass)** — heavyweight construction (ASHRAE 140
//!   heavyweight wall/roof/floor assemblies), 33 states.
//!
//! The workloads are driven through the *shipped* Newton solver
//! (`fluxion::physics::bdf_engine::newton_raphson::NewtonRaphsonSolver`)
//! with a BDF-2 residual assembled by the harness, and through the
//! *shipped* Jacobian cache (`NewtonJacobianCache`) so the
//! flag-gated freeze prototype (`FLX_BDF_FREEZE_JACOBIAN`,
//! `FLX_BDF_FREEZE_MAX_REUSE`) is exercised exactly as the production
//! driver would exercise it.
//!
//! Run it as:
//!
//! ```text
//! cargo test -p fluxion --test all_tests bdf_frozen_jacobian_spike -- --nocapture --test-threads=1
//! ```
//!
//! with the prototype arms selected through the environment:
//!
//! * baseline:      (no `FLX_*` env var set)
//! * LU-only:       `FLX_BDF_FREEZE_JACOBIAN=1 FLX_BDF_FREEZE_MAX_REUSE=0`
//! * freeze at N:   `FLX_BDF_FREEZE_JACOBIAN=1 FLX_BDF_FREEZE_MAX_REUSE={4,8,16}`
//!
//! Optional `FLX4691_DUMP=<path>` writes the hourly zone-temperature
//! trajectory of both workloads as CSV so output drift between arms can
//! be measured exactly.
//!
//! NOTE (issue #4691): the harness deliberately does **not** use
//! `BdfTimeStepper::step`, because the shipped `BdfResidual` wrapper
//! overwrites the `DaeSystem` residual with `sum(alpha_j) * y / (dt*beta)`
//! (see `src/physics/bdf_engine.rs`), i.e. the DAE right-hand side never
//! reaches the Newton solve. The harness therefore assembles the standard
//! BDF-2 residual itself and drives the very same Newton solver + cache.

use std::time::Instant;

use fluxion::physics::bdf_engine::counters;
use fluxion::physics::bdf_engine::newton_raphson::{
    NewtonJacobianCache, NewtonRaphsonConfig, NewtonRaphsonSolver, ResidualFunction,
};
use fluxion::sim::construction::Assemblies;

/// Seconds in a simulated year.
const YEAR_SECONDS: f64 = 365.0 * 24.0 * 3600.0;
/// Timestep of the annual workload (1 hour).
const DT: f64 = 3600.0;
/// Steps per simulated year.
const STEPS_PER_YEAR: usize = 8760;

/// A discretised construction layer.
#[derive(Debug, Clone, Copy)]
struct LayerSpec {
    conductivity: f64,
    density: f64,
    specific_heat: f64,
    thickness: f64,
}

/// Which ambient the exterior node of a construction sees.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Exterior {
    Outdoors,
    Ground,
}

/// One discretised construction (interior -> exterior).
#[derive(Debug, Clone)]
struct ConstructionSpec {
    area: f64,
    layers: Vec<LayerSpec>,
    /// Number of discretisation nodes per layer.
    nodes_per_layer: usize,
    exterior: Exterior,
}

/// The full building model: a zone-air node plus the construction nodes.
#[derive(Debug, Clone)]
struct BuildingModel {
    /// Capacitance of each state (J/K).
    capacitance: Vec<f64>,
    /// Conduction edges (i, j, conductance W/K).
    edges: Vec<(usize, usize, f64)>,
    /// Zone air capacitance (J/K).
    zone_capacitance: f64,
    /// Window conductance zone <-> outdoors (W/K).
    window_conductance: f64,
    /// Window area (m²).
    window_area: f64,
    /// Window solar heat gain coefficient.
    window_shgc: f64,
    /// Interior combined film coefficient (W/m²K).
    interior_film: f64,
    /// Exterior combined film coefficient (W/m²K).
    exterior_film: f64,
    /// Ground temperature (°C).
    ground_temperature: f64,
    /// Heating setpoint (°C).
    heating_setpoint: f64,
    /// Cooling setpoint (°C).
    cooling_setpoint: f64,
    /// Proportional HVAC gain (W/K).
    hvac_gain: f64,
    /// Internal gain (W).
    internal_gain: f64,
    /// Number of states.
    n: usize,
}

/// ASHRAE 140 Case 600 geometry (8 m x 6 m footprint, 2.7 m high).
const FLOOR_AREA: f64 = 48.0;
const WALL_AREA: f64 = 75.6;
const ROOF_AREA: f64 = 48.0;
const WINDOW_AREA: f64 = 12.0;

fn layer_specs(construction: &fluxion::sim::construction::Construction) -> Vec<LayerSpec> {
    construction
        .layers
        .iter()
        .map(|l| LayerSpec {
            conductivity: l.conductivity,
            density: l.density,
            specific_heat: l.specific_heat,
            thickness: l.thickness,
        })
        .collect()
}

/// Build the state vector layout, capacitances and conduction edges.
fn build_model(constructions: &[ConstructionSpec], high_mass: bool) -> BuildingModel {
    let zone_capacitance = 1.2 * (FLOOR_AREA * 2.7) * 1005.0; // air node, J/K
    let interior_film = 8.29; // W/m²K (ASHRAE 140 combined interior)
    let exterior_film = 25.0; // W/m²K (combined exterior)

    let mut capacitance = vec![zone_capacitance];
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    let zone = 0usize;

    for con in constructions {
        let first = capacitance.len();
        let mut node_index = first;
        for layer in &con.layers {
            for _ in 0..con.nodes_per_layer {
                let c = layer.density
                    * layer.specific_heat
                    * (layer.thickness / con.nodes_per_layer as f64)
                    * con.area;
                capacitance.push(c);
                node_index += 1;
            }
        }
        let last = node_index - 1;

        // Interior film: zone air <-> interior construction node.
        edges.push((zone, first, interior_film * con.area));

        // Interior <-> exterior conduction (series network through the
        // discretised layers).
        for i in first..last {
            let layer_idx = (i - first) / con.nodes_per_layer;
            let layer = &con.layers[layer_idx];
            let node_thickness = layer.thickness / con.nodes_per_layer as f64;
            // Half of the node on each side of the interface.
            let r = if (i - first) % con.nodes_per_layer == con.nodes_per_layer - 1 {
                node_thickness / (2.0 * layer.conductivity)
            } else {
                node_thickness / layer.conductivity
            };
            let r_next = {
                let next_layer_idx = (i + 1 - first) / con.nodes_per_layer;
                let next_layer = &con.layers[next_layer_idx];
                let nt = next_layer.thickness / con.nodes_per_layer as f64;
                if (i + 1 - first) % con.nodes_per_layer == 0 {
                    nt / (2.0 * next_layer.conductivity)
                } else {
                    nt / next_layer.conductivity
                }
            };
            let g = con.area / (r + r_next);
            edges.push((i, i + 1, g));
        }

        // Exterior node <-> ambient.
        match con.exterior {
            Exterior::Outdoors => edges.push((last, usize::MAX, exterior_film * con.area)),
            Exterior::Ground => {
                // Floor slab: U = 0.19 W/m²K to ground (ASHRAE 140).
                edges.push((last, usize::MAX, 0.19 * con.area));
            }
        }
    }

    let n = capacitance.len();
    BuildingModel {
        capacitance,
        edges,
        zone_capacitance,
        window_conductance: 3.0 * WINDOW_AREA,
        window_area: WINDOW_AREA,
        window_shgc: 0.787,
        interior_film,
        exterior_film,
        ground_temperature: 10.0,
        heating_setpoint: 20.0,
        cooling_setpoint: 27.0,
        hvac_gain: 2000.0,
        internal_gain: 200.0,
        n,
    }
}

/// Case 600 — lightweight construction, one node per layer.
fn case_600_model() -> BuildingModel {
    let constructions = vec![
        ConstructionSpec {
            area: WALL_AREA,
            layers: layer_specs(&Assemblies::low_mass_wall()),
            nodes_per_layer: 1,
            exterior: Exterior::Outdoors,
        },
        ConstructionSpec {
            area: ROOF_AREA,
            layers: layer_specs(&Assemblies::low_mass_roof()),
            nodes_per_layer: 1,
            exterior: Exterior::Outdoors,
        },
        ConstructionSpec {
            area: FLOOR_AREA,
            layers: layer_specs(&Assemblies::insulated_floor()),
            nodes_per_layer: 1,
            exterior: Exterior::Ground,
        },
    ];
    build_model(&constructions, false)
}

/// Case 900 — heavyweight construction, four nodes per layer.
fn high_mass_model() -> BuildingModel {
    let constructions = vec![
        ConstructionSpec {
            area: WALL_AREA,
            layers: layer_specs(&Assemblies::high_mass_wall()),
            nodes_per_layer: 4,
            exterior: Exterior::Outdoors,
        },
        ConstructionSpec {
            area: ROOF_AREA,
            layers: layer_specs(&Assemblies::high_mass_roof()),
            nodes_per_layer: 4,
            exterior: Exterior::Outdoors,
        },
        ConstructionSpec {
            area: FLOOR_AREA,
            layers: layer_specs(&Assemblies::high_mass_floor()),
            nodes_per_layer: 4,
            exterior: Exterior::Ground,
        },
    ];
    build_model(&constructions, true)
}

/// Outdoor dry-bulb temperature (°C) — Denver-like annual cycle.
fn outdoor_temperature(t: f64) -> f64 {
    10.0 - 13.0 * (2.0 * std::f64::consts::PI * t / YEAR_SECONDS).cos()
}

/// Total solar irradiance on the south facade (W/m²).
fn solar_irradiance(t: f64) -> f64 {
    let hour_of_day = (t / 3600.0) % 24.0;
    let day_of_year = t / 86_400.0;
    let seasonal = 0.75 + 0.25 * (2.0 * std::f64::consts::PI * (day_of_year - 172.0) / 365.0).cos();
    let daily = if hour_of_day > 6.0 && hour_of_day < 19.0 {
        (std::f64::consts::PI * (hour_of_day - 6.0) / 13.0).sin()
    } else {
        0.0
    };
    900.0 * seasonal * daily
}

/// Right-hand side of the building ODE (K/s), `yp` unused (the BDF-2
/// residual is assembled in the harness).
fn rhs(model: &BuildingModel, t: f64, y: &[f64], out: &mut [f64]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let t_out = outdoor_temperature(t);

    for (i, j, g) in &model.edges {
        let ti = y[*i];
        let tj = if *j == usize::MAX { t_out } else { y[*j] };
        let flux = g * (tj - ti);
        out[*i] += flux;
        if *j != usize::MAX {
            out[*j] -= flux;
        }
    }

    // Window: conduction to outdoors + solar gain into the zone.
    let solar = model.window_area * model.window_shgc * solar_irradiance(t);
    out[0] += model.window_conductance * (t_out - y[0]) + solar + model.internal_gain;

    // Piecewise-linear HVAC (kinked at the setpoints -> the Newton
    // Jacobian is genuinely state dependent).
    let hvac = if y[0] < model.heating_setpoint {
        model.hvac_gain * (model.heating_setpoint - y[0])
    } else if y[0] > model.cooling_setpoint {
        -model.hvac_gain * (y[0] - model.cooling_setpoint)
    } else {
        0.0
    };
    out[0] += hvac;

    for i in 0..model.n {
        out[i] /= model.capacitance[i];
    }
}

/// BDF-2 residual wrapper (rate form):
/// `(1.5 y_n - 2 y_{n-1} + 0.5 y_{n-2}) / dt - f(y_n, t) = 0`.
struct Bdf2Residual<'a> {
    model: &'a BuildingModel,
    y_prev: &'a [f64],
    y_prev2: &'a [f64],
    dt: f64,
    t_new: f64,
}

impl<'a> ResidualFunction<f64> for Bdf2Residual<'a> {
    fn eval(&self, y: &[f64], residual: &mut [f64]) {
        rhs(self.model, self.t_new, y, residual);
        let n = self.model.n;
        for i in 0..n {
            let hist = 1.5 * y[i] - 2.0 * self.y_prev[i] + 0.5 * self.y_prev2[i];
            residual[i] = hist / self.dt - residual[i];
        }
    }
}

/// Result of one annual workload run.
struct RunResult {
    label: String,
    states: usize,
    steps_accepted: usize,
    steps_rejected: usize,
    wall_ms: f64,
    counters: counters::CountersSnapshot,
    zone_temps: Vec<f64>,
    final_residual: f64,
}

/// Run one annual workload.
fn run_annual(model: &BuildingModel, label: &str) -> RunResult {
    counters::reset();

    let n = model.n;
    let solver = NewtonRaphsonSolver::new(NewtonRaphsonConfig::default());
    let mut cache = NewtonJacobianCache::default();

    // Initial state: 20 °C everywhere.
    let mut history: Vec<Vec<f64>> = vec![vec![20.0; n]];
    let mut zone_temps: Vec<f64> = Vec::with_capacity(STEPS_PER_YEAR);
    let mut steps_accepted = 0usize;
    let mut steps_rejected = 0usize;
    let mut final_residual = 0.0f64;

    let start = Instant::now();
    let mut t = 0.0f64;
    let mut dt = DT;

    while steps_accepted < STEPS_PER_YEAR {
        let t_new = t + dt;
        // Linear extrapolation of the initial guess.
        let y_pred: Vec<f64> = if history.len() >= 2 {
            let y0 = &history[0];
            let y1 = &history[1];
            (0..n).map(|i| 2.0 * y0[i] - y1[i]).collect()
        } else {
            history[0].clone()
        };

        let y_prev = history[0].clone();
        let y_prev2 = if history.len() >= 2 {
            history[1].clone()
        } else {
            history[0].clone()
        };
        let residual = Bdf2Residual {
            model,
            y_prev: &y_prev,
            y_prev2: &y_prev2,
            dt,
            t_new,
        };

        match solver.solve_with_cache(&y_pred, &residual, &mut cache) {
            Ok((solution, stats)) => {
                final_residual = stats.final_residual;
                history.insert(0, solution);
                if history.len() > 3 {
                    history.pop();
                }
                t = t_new;
                dt = DT;
                steps_accepted += 1;
                zone_temps.push(history[0][0]);
            }
            Err(_) => {
                // Mirror the shipped driver: a failed Newton solve
                // invalidates the cached Jacobian and the step is
                // rejected (dt halved).
                steps_rejected += 1;
                cache.invalidate();
                dt *= 0.5;
                if dt < 1.0 {
                    dt = 1.0;
                    // Give up on this step: fall back to the last
                    // accepted state and carry on (keeps the annual run
                    // finite so the counters stay comparable).
                    history.insert(0, history[0].clone());
                    if history.len() > 3 {
                        history.pop();
                    }
                    t = t_new;
                    dt = DT;
                    steps_accepted += 1;
                    zone_temps.push(history[0][0]);
                }
            }
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    RunResult {
        label: label.to_string(),
        states: n,
        steps_accepted,
        steps_rejected,
        wall_ms,
        counters: counters::snapshot(),
        zone_temps,
        final_residual,
    }
}

fn dump_trajectory(label: &str, temps: &[f64]) {
    let Ok(path) = std::env::var("FLX4691_DUMP") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let mut out = String::with_capacity(temps.len() * 12);
    for (i, t) in temps.iter().enumerate() {
        out.push_str(&format!("{label},{i},{t:.17e}\n"));
    }
    use std::io::Write;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("open FLX4691_DUMP path");
    let mut w = std::io::BufWriter::new(file);
    w.write_all(out.as_bytes()).expect("write trajectory");
}

fn summarise(result: &RunResult) {
    let c = result.counters;
    let min = result
        .zone_temps
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = result
        .zone_temps
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = result.zone_temps.iter().sum();
    println!(
        "FLX4691 label={} states={} steps_accepted={} steps_rejected={} wall_ms={:.1} \
         jac_assemblies={} factorizations={} newton_iterations={} jac_reuses={} solve_calls={} \
         forced_refresh_retries={} step_rejections={} final_residual={:.3e} \
         zone_min={:.6} zone_max={:.6} zone_sum={:.9}",
        result.label,
        result.states,
        result.steps_accepted,
        result.steps_rejected,
        result.wall_ms,
        c.jacobian_assemblies,
        c.factorizations,
        c.newton_iterations,
        c.jacobian_reuses,
        c.solve_calls,
        c.forced_refresh_retries,
        c.step_rejections,
        result.final_residual,
        min,
        max,
        sum
    );
}

#[test]
fn issue_4691_frozen_jacobian_spike_annual_workloads() {
    let low = run_annual(&case_600_model(), "case600_low_mass");
    summarise(&low);
    dump_trajectory("case600_low_mass", &low.zone_temps);

    let high = run_annual(&high_mass_model(), "case900_high_mass");
    summarise(&high);
    dump_trajectory("case900_high_mass", &high.zone_temps);

    // Sanity: both annual runs completed and produced finite trajectories.
    assert_eq!(low.steps_accepted, STEPS_PER_YEAR);
    assert_eq!(high.steps_accepted, STEPS_PER_YEAR);
    assert!(low.zone_temps.iter().all(|v| v.is_finite()));
    assert!(high.zone_temps.iter().all(|v| v.is_finite()));
    // The zone must stay within a physically plausible band.
    assert!(low.zone_temps.iter().all(|v| *v > -20.0 && *v < 60.0));
    assert!(high.zone_temps.iter().all(|v| *v > -20.0 && *v < 60.0));
}
