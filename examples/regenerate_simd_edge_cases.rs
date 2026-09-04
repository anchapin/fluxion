//! Regenerate the per-edge reference output values for the
//! `tools/evolution/edge_cases/solar_simd.json` harness fixture (issue
//! #3338).
//!
//! IMPORTANT: this is the **only** sanctioned path to update the
//! fixture file. Hand-editing the JSON is forbidden by RULES.md and
//! the issue's acceptance ("regenerate via script, never hand-edit;
//! preserve `#[rustfmt::skip]`").
//!
//! # What it does
//!
//! Runs the canonical (scalar) implementation of each seed kernel
//! against the documented input set, captures the result, and writes
//! the `reference` block back into the JSON fixture. The candidate
//! harness then asserts that the candidate's output is within
//! `tolerance.default` (default-feature build) or
//! `tolerance.simd_kernels` (under `--features simd-kernels`).
//!
//! # How
//!
//! ```text
//! $ cargo run --release --example regenerate_simd_edge_cases
//! ```
//!
//! The script is **idempotent**: running it twice in a row produces
//! identical output (modulo the embedded timestamp). This makes it
//! safe to call from CI to enforce the determinism contract.

use std::fs;
use std::path::PathBuf;

use fluxion::sim::interzone_radiation::surface_radiative_exchange;
use fluxion::solar::surface_irradiance::PerezSkyModel;

/// Path of the JSON fixture, relative to the repo root.
const FIXTURE_PATH: &str = "tools/evolution/edge_cases/solar_simd.json";

fn main() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // When invoked through `cargo run --example`, `CARGO_MANIFEST_DIR`
    // points at the example crate's manifest. We accept either the
    // examples/ crate (this one) or the workspace root by trying both.
    let workspace_root_guess = path.ancestors().filter(|p| p.is_dir()).find_map(|p| {
        if p.join("tools/evolution/edge_cases/solar_simd.json")
            .exists()
        {
            Some(p.to_path_buf())
        } else {
            None
        }
    });
    let workspace_root = workspace_root_guess.unwrap_or(path.clone());

    let fixture_path = workspace_root.join(FIXTURE_PATH);
    if !fixture_path.exists() {
        eprintln!(
            "regenerate_simd_edge_cases: fixture not found at {}",
            fixture_path.display()
        );
        std::process::exit(2);
    }

    let body = fs::read_to_string(&fixture_path).expect("read fixture");
    let mut root: serde_json::Value = serde_json::from_str(&body).expect("parse fixture");

    let n_cases = {
        let cases = root
            .get_mut("case_set")
            .and_then(|v| v.as_array_mut())
            .expect("case_set array");
        // Snapshot the inputs first so we don't borrow `cases` while
        // mutably borrowing it for the write-back below.
        let snapshots: Vec<(String, String, serde_json::Value)> = cases
            .iter()
            .map(|case| {
                let case_name = case
                    .get("case_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>")
                    .to_string();
                let focus = case
                    .get("kernel_focus")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let params = case
                    .get("input")
                    .and_then(|i| i.get("params"))
                    .cloned()
                    .expect("input.params");
                (case_name, focus, params)
            })
            .collect();

        for ((case_name, focus, params), case) in snapshots.into_iter().zip(cases.iter_mut()) {
            let (ref_key, ref_value) = match focus.as_str() {
                "perez_diffuse_tilted" => {
                    let r = PerezSkyModel::calculate_diffuse_tilted(
                        num(&params, "dhi"),
                        num(&params, "dni"),
                        num(&params, "dni_extra"),
                        num(&params, "airmass"),
                        num(&params, "zenith_deg"),
                        num(&params, "tilt_deg"),
                        num(&params, "surface_azimuth_deg"),
                        num(&params, "solar_azimuth_deg"),
                    );
                    ("diffuse_tilted_wm2".to_string(), r)
                }
                "stefan_boltzmann_pair" => {
                    let r = surface_radiative_exchange(
                        num(&params, "t_a_c"),
                        num(&params, "t_b_c"),
                        num(&params, "emissivity_a"),
                        num(&params, "emissivity_b"),
                        num(&params, "view_factor"),
                        num(&params, "area"),
                    );
                    ("q_w".to_string(), r)
                }
                other => {
                    eprintln!("regenerate_simd_edge_cases: unknown kernel_focus `{other}`");
                    std::process::exit(3);
                }
            };

            {
                let reference = case
                    .get_mut("reference")
                    .and_then(|v| v.as_object_mut())
                    .expect("reference object");
                reference.insert(ref_key.clone(), serde_json::json!(ref_value));
            }

            eprintln!("  {case_name:42} focus={focus:22} {ref_key}={ref_value:.16e}",);
        }
        cases.len()
    };

    // Stamp the regenerated JSON. Preserve all other fields verbatim.
    let body = serde_json::to_string_pretty(&root).expect("re-serialize");
    fs::write(&fixture_path, &body).expect("write fixture");
    eprintln!(
        "regenerate_simd_edge_cases: wrote {} ({} cases)",
        fixture_path.display(),
        n_cases
    );
}

fn num(params: &serde_json::Value, key: &str) -> f64 {
    params
        .get(key)
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| panic!("missing `{}`", key))
}
