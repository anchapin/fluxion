//! Issue #3339 — BDF DAE strategy evaluator binary.
//!
//! Reads a JSON `DampingPolicy` (the candidate) from `--strategy-file`
//! (or stdin), runs all 5 benchmark circuits through [`BdfDriver`],
//! and emits a Schema-v1 Summary JSON on stdout. The Python seed
//! controller under `tools/evolution/seeds/dae/` drives this binary
//! per candidate; the OpenEvolve adapter (`tools/evolution/openevolve_*`)
//! reads the JSON score and feeds OpenEvolve's population DB.
//!
//! Usage:
//!
//! ```text
//! $ cat strategy.json | bdf_evaluator --candidate-id seed-0001 --generation 0
//! { "schema_version":1, "fitness": 0.42, ... }
//! ```
//!
//! Exit codes match the harness contract:
//! - 0 — evaluated (consult `fitness`)
//! - 2 — strategy parse / IO failure
//! - 3 — invariant hard-fail (fitness forced to 0.0)
//! - 4 — timeout / resource cap (currently unreachable; the
//!   per-circuit `BdfDriver` caps steps at 50_000 which keeps wall
//!   time well below 60 s on every benchmark).

use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use fluxion::physics::bdf_benchmarks::{
    ConservationProbe, CoolingCoilWetSurface, DecouplingLoopDemandStep, HeatPumpEnteringFluidStep,
    MixingValveClosure, PumpFrequencyRamp,
};
use fluxion::physics::bdf_engine::{
    AdaptiveStepConfig, BdfDriver, DampingPolicy, DriverStats, NewtonRaphsonConfig,
    TimeSteppingConfig,
};
use sha2::{Digest, Sha256};

/// Per-circuit evaluation result.
#[derive(Debug, Clone)]
struct CircuitResult {
    name: String,
    driver_stats: DriverStats,
    probe: ConservationProbe,
    wall_ms: u128,
    status: &'static str, // "ok" | "invariant_fail" | "aborted"
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StrategySpec {
    #[serde(default)]
    mode: Option<u8>,
    baseline_factor: f64,
    floor: f64,
    loose_threshold: f64,
    tight_threshold: f64,
    aggressiveness: f64,
    history_window: usize,
    /// Optional explicit driver-level `max_steps` (default 50_000).
    #[serde(default)]
    max_steps: Option<usize>,
}

impl StrategySpec {
    fn into_policy(self) -> Result<(DampingPolicy, Option<usize>), String> {
        if !self.baseline_factor.is_finite()
            || !self.floor.is_finite()
            || !self.loose_threshold.is_finite()
            || !self.tight_threshold.is_finite()
            || !self.aggressiveness.is_finite()
        {
            return Err("non-finite strategy field".into());
        }
        if !(0.0 < self.floor && self.floor <= self.baseline_factor) {
            return Err(format!(
                "require 0 < floor={} <= baseline_factor={}",
                self.floor, self.baseline_factor
            ));
        }
        if !(self.baseline_factor > 0.0 && self.baseline_factor <= 2.0) {
            return Err(format!(
                "baseline_factor {} out of (0,2]",
                self.baseline_factor
            ));
        }
        if !(0.0 < self.aggressiveness && self.aggressiveness <= 4.0) {
            return Err(format!(
                "aggressiveness {} out of (0,4]",
                self.aggressiveness
            ));
        }
        if self.loose_threshold >= self.tight_threshold {
            return Err(format!(
                "require loose_threshold={} < tight_threshold={}",
                self.loose_threshold, self.tight_threshold
            ));
        }
        let mode = self.mode.unwrap_or(0);
        if mode > 1 {
            return Err(format!("mode {} must be 0 or 1", mode));
        }
        Ok((
            DampingPolicy {
                mode,
                baseline_factor: self.baseline_factor,
                floor: self.floor,
                loose_threshold: self.loose_threshold,
                tight_threshold: self.tight_threshold,
                aggressiveness: self.aggressiveness,
                history_window: self.history_window,
            },
            self.max_steps,
        ))
    }
}

/// Argument-parsing (deliberately hand-rolled; see `fluxion-evaluator`
/// precedent — the binary is invoked per-candidate and a CLI
/// framework would be overkill).
struct Args {
    candidate_id: String,
    strategy_file: Option<PathBuf>,
    generation: Option<u32>,
    output: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut candidate_id: Option<String> = None;
    let mut strategy_file: Option<PathBuf> = None;
    let mut generation: Option<u32> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--candidate-id" => {
                candidate_id = argv.get(i + 1).cloned();
                i += 2;
            }
            "--strategy-file" => {
                strategy_file = argv.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--generation" => {
                generation = argv.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--output" => {
                output = argv.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{}`", other)),
        }
    }
    Ok(Args {
        candidate_id: candidate_id.ok_or_else(|| "--candidate-id required".to_string())?,
        strategy_file,
        generation,
        output,
    })
}

fn print_help() {
    eprintln!(
        "bdf_evaluator — Issue #3339 bounded-campaign fitness oracle\n\n\
         USAGE:\n  \
         bdf_evaluator --candidate-id <ID> [--strategy-file <PATH>]\n\
                       [--generation <N>] [--output <SUMMARY_PATH>]\n\n\
         OUTPUT:\n  \
         Schema-v1 Summary JSON on stdout (or to --output). Exit codes: \
         0 ok, 2 strategy, 3 invariant, 4 cap."
    );
}

fn read_strategy(strategy_file: Option<PathBuf>) -> Result<String, String> {
    match strategy_file {
        Some(p) => std::fs::read_to_string(&p)
            .map_err(|e| format!("read strategy file `{}` failed: {}", p.display(), e)),
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("stdin read failed: {}", e))?;
            Ok(buf)
        }
    }
}

/// Drive a single circuit through the BDF driver with the candidate's
/// policy. The five circuits have heterogeneous dimensions; we
/// instantiate the concrete type and call `DaeSystem::residual`
/// through it (no trait-object gymnastics).
fn run_circuit<S>(circuit: &S, bdf_config: NewtonRaphsonConfig, max_steps: usize) -> CircuitResult
where
    S: fluxion::physics::bdf_engine::DaeSystem<f64> + fluxion::physics::bdf_benchmarks::Circuit,
{
    let name = circuit.name().to_string();
    let step_cfg = AdaptiveStepConfig {
        initial_dt: circuit.dt_init(),
        ..AdaptiveStepConfig::default()
    };
    let ts_cfg = TimeSteppingConfig {
        bdf_config,
        step_config: step_cfg,
        max_steps,
        tolerance: 1e-6,
    };
    let mut driver = BdfDriver::new(ts_cfg);

    let circuit_t_end = circuit.t_end();
    let initial_state = circuit.initial_state();

    if driver.initialize(0.0, &initial_state).is_err() {
        return CircuitResult {
            name,
            driver_stats: DriverStats::default(),
            probe: ConservationProbe::default(),
            wall_ms: 0,
            status: "aborted",
        };
    }

    let started = std::time::Instant::now();
    let driver_outcome = driver.run(circuit, circuit_t_end, circuit.dt_init());
    let wall_ms = started.elapsed().as_millis();

    if wall_ms > Duration::from_secs(60).as_millis() {
        return CircuitResult {
            name,
            driver_stats: DriverStats::default(),
            probe: ConservationProbe::default(),
            wall_ms,
            status: "aborted",
        };
    }

    let driver_stats = match driver_outcome {
        Ok(s) => s,
        Err(_) => DriverStats {
            converged: false,
            final_time: circuit_t_end,
            ..DriverStats::default()
        },
    };

    let final_state = driver.last_state();
    let probe = circuit.finalize(&final_state, driver_stats.final_residual);

    let status = if probe.junction_violates() {
        "invariant_fail"
    } else if !driver_stats.converged {
        "aborted"
    } else {
        "ok"
    };

    CircuitResult {
        name,
        driver_stats,
        probe,
        wall_ms,
        status,
    }
}

fn aggregate(results: &[CircuitResult]) -> (DriverStats, ConservationProbe, bool) {
    let mut agg = DriverStats::default();
    let mut probe = ConservationProbe::default();
    let mut all_ok = true;
    for r in results {
        agg.newton_iterations += r.driver_stats.newton_iterations;
        agg.steps_accepted += r.driver_stats.steps_accepted;
        agg.steps_rejected += r.driver_stats.steps_rejected;
        agg.max_residual = agg.max_residual.max(r.driver_stats.max_residual);
        probe.max_mass_relative_error = probe
            .max_mass_relative_error
            .max(r.probe.max_mass_relative_error);
        probe.max_enthalpy_relative_error = probe
            .max_enthalpy_relative_error
            .max(r.probe.max_enthalpy_relative_error);
        probe.nan_or_inf_count += r.probe.nan_or_inf_count;
        probe.conservation_violations += r.probe.conservation_violations;
        if r.status != "ok" {
            all_ok = false;
        }
    }
    agg.converged = all_ok;
    (agg, probe, all_ok)
}

/// Fitness function for the campaign: minimise total Newton
/// iterations + accepted steps, subject to all hard invariants being
/// clean. Higher score = better (so we invert the cost).
fn fitness(agg: &DriverStats, probe: &ConservationProbe, all_ok: bool) -> f64 {
    if !all_ok || probe.junction_violates() || !agg.converged {
        return 0.0;
    }
    let cost = (agg.newton_iterations as f64 + agg.steps_accepted as f64).max(1.0);
    (1.0 / (1.0 + cost * 1e-4)).max(1e-3)
}

/// Build the Schema-v1 Summary JSON. The shape mirrors
/// `fluxion-evaluator`'s contract; we don't link that crate here
/// because the binary is workspace-anchored to the root fluxion
/// crate.
#[derive(Debug, serde::Serialize)]
struct Summary {
    schema_version: u32,
    candidate_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation: Option<u32>,
    fitness: f64,
    compiled: bool,
    invariants_passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_latency_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_latency_spread_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    determinism_digest: Option<String>,
    outcome: &'static str,
    invariant_violations: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    min_invariant_margin: f64,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    bdf_per_circuit: serde_json::Map<String, serde_json::Value>,
}

fn min_margin(agg: &DriverStats, probe: &ConservationProbe) -> f64 {
    let m_mass = (1.0 - probe.max_mass_relative_error).clamp(0.0, 1.0);
    let m_enth = (1.0 - probe.max_enthalpy_relative_error).clamp(0.0, 1.0);
    let m_res = (1.0 - agg.max_residual.min(1.0)).max(0.0);
    m_mass.min(m_enth).min(m_res)
}

fn canonical_input_bytes(candidate_id: &str, strategy_json: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(candidate_id.len() + strategy_json.len() + 32);
    out.extend_from_slice(candidate_id.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(strategy_json.as_bytes());
    out
}

fn digest(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn summary_failure(args: &Args, outcome: &'static str, error: String) -> Summary {
    Summary {
        schema_version: 1,
        candidate_id: args.candidate_id.clone(),
        generation: args.generation,
        fitness: 0.0,
        compiled: false,
        invariants_passed: false,
        max_error: None,
        eval_latency_ns: None,
        eval_latency_spread_ns: None,
        determinism_digest: None,
        outcome,
        invariant_violations: Vec::new(),
        error: Some(error),
        min_invariant_margin: 0.0,
        bdf_per_circuit: Default::default(),
    }
}

fn emit_and_exit(summary: Summary, code: i32, output: Option<PathBuf>) -> ! {
    let json = match serde_json::to_string(&summary) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("bdf_evaluator: failed to serialize Summary: {}", e);
            std::process::exit(1);
        }
    };
    match output {
        Some(p) => {
            if let Err(e) = std::fs::write(&p, &json) {
                eprintln!(
                    "bdf_evaluator: write Summary to {} failed: {}",
                    p.display(),
                    e
                );
                std::process::exit(1);
            }
        }
        None => println!("{}", json),
    }
    std::process::exit(code);
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bdf_evaluator: {}", e);
            print_help();
            std::process::exit(1);
        }
    };

    // Take ownership of the fields we'll move into helper calls so we
    // can freely borrow `args` afterwards (avoids partial-move errors).
    let strategy_file = args.strategy_file.clone();
    let strategy_text = match read_strategy(strategy_file) {
        Ok(s) => s,
        Err(e) => {
            emit_and_exit(summary_failure(&args, "compile_failure", e), 2, args.output);
        }
    };

    let spec: StrategySpec = match serde_json::from_str(&strategy_text) {
        Ok(s) => s,
        Err(e) => {
            emit_and_exit(
                summary_failure(
                    &args,
                    "compile_failure",
                    format!("strategy parse failed: {}", e),
                ),
                2,
                args.output,
            );
        }
    };

    let (policy, max_steps_override) = match spec.into_policy() {
        Ok(p) => p,
        Err(e) => {
            emit_and_exit(
                summary_failure(&args, "compile_failure", format!("strategy invalid: {}", e)),
                2,
                args.output,
            );
        }
    };

    let bdf_config = NewtonRaphsonConfig {
        damping: policy,
        ..NewtonRaphsonConfig::default()
    };
    let max_steps = max_steps_override.unwrap_or(50_000);

    // Run the 5 circuits with concrete types. No dyn-trait gymnastics
    // needed — `DaeSystem<f64>` is implemented for each.
    let per_circuit: Vec<CircuitResult> = vec![
        run_circuit(&MixingValveClosure::default(), bdf_config, max_steps),
        run_circuit(&PumpFrequencyRamp::default(), bdf_config, max_steps),
        run_circuit(&CoolingCoilWetSurface::default(), bdf_config, max_steps),
        run_circuit(&DecouplingLoopDemandStep::default(), bdf_config, max_steps),
        run_circuit(&HeatPumpEnteringFluidStep::default(), bdf_config, max_steps),
    ];

    let (agg, probe, all_ok) = aggregate(&per_circuit);
    let inviolations: Vec<String> = per_circuit
        .iter()
        .filter(|r| r.status != "ok")
        .map(|r| format!("{}: {}", r.name, r.status))
        .collect();
    let invariants_passed = all_ok && !probe.junction_violates();
    let mm = min_margin(&agg, &probe);
    let fitness = fitness(&agg, &probe, invariants_passed);
    let total_wall_ms: u128 = per_circuit.iter().map(|r| r.wall_ms).sum();

    let mut bdf_per_circuit = serde_json::Map::new();
    for r in &per_circuit {
        let entry = serde_json::json!({
            "newton_iterations":     r.driver_stats.newton_iterations,
            "steps_accepted":        r.driver_stats.steps_accepted,
            "steps_rejected":        r.driver_stats.steps_rejected,
            "final_time":            r.driver_stats.final_time,
            "converged":             r.driver_stats.converged,
            "max_residual":          r.driver_stats.max_residual,
            "wall_ms":               r.wall_ms,
            "status":                r.status,
            "max_mass_rel_err":      r.probe.max_mass_relative_error,
            "max_enthalpy_rel_err":  r.probe.max_enthalpy_relative_error,
            "nan_or_inf_count":      r.probe.nan_or_inf_count,
        });
        bdf_per_circuit.insert(r.name.clone(), entry);
    }

    let summary = Summary {
        schema_version: 1,
        candidate_id: args.candidate_id.clone(),
        generation: args.generation,
        fitness,
        compiled: true,
        invariants_passed,
        max_error: Some(agg.max_residual),
        eval_latency_ns: Some((total_wall_ms * 1_000_000) as u64),
        eval_latency_spread_ns: Some(0),
        determinism_digest: Some(format!(
            "sha256:{}",
            digest(&canonical_input_bytes(&args.candidate_id, &strategy_text))
        )),
        outcome: if invariants_passed {
            "evaluated"
        } else {
            "invariant_hard_fail"
        },
        invariant_violations: inviolations,
        error: None,
        min_invariant_margin: mm,
        bdf_per_circuit,
    };
    emit_and_exit(summary, if invariants_passed { 0 } else { 3 }, args.output);
}
