//! Issues #4270 + #4274: produce
//! `data/reference/ashrae140/analytical_results.json` — the release gate's
//! analytical anchors, §8 (GC10a) and §11 (QASv2 air-side).
//!
//! Run: `cargo run --release --example section8_gc10a_report`
//!
//! Measures Case GC10a with the cell-centered finite-volume solver at the
//! default (validated) mesh, checks it against the closed-form Delsante
//! solution, solves all six §11 QAS cases and checks case 1 against the
//! published QASv2.xlsm anchors, then prints the combined gate-facing JSON.
//! The committed copy in data/reference/ashrae140/analytical_results.json
//! must match this output byte-for-byte; the release gate fails on any drift
//! and the example itself exits 1 rather than emitting a passing artifact
//! beyond tolerance.

use fluxion::ground_coupling::{gc10a_closed_form_watts, gc10a_fd_flux_watts, Gc10aConfig};
use fluxion::qas_airside;

fn main() {
    let cfg = Gc10aConfig::default();
    let measured = gc10a_fd_flux_watts(&cfg).expect("GC10a FD solve");
    let closed = gc10a_closed_form_watts(&cfg);
    let deviation_pct = (measured.floor_conduction_w - closed).abs() / closed * 100.0;
    let s8 = format!(
        "{{\n  \"section_8_ground_coupled\": {{\n    \"gc10a_steady_state_floor_conduction_w\": {:.1},\n    \"method\": \"cell-centered finite volume, quarter domain, Jacobi-PCG (Issue #4270)\",\n    \"closed_form_check_w\": {:.2},\n    \"deviation_pct\": {:.3},\n    \"unknowns\": {},\n    \"cg_iterations\": {},\n    \"relative_residual\": {:.3e},\n    \"mesh\": {{\"fine_cell_m\": {}, \"far_field_m\": {}, \"ground_depth_m\": {}}},\n    \"source\": \"Addendum a to ANSI/ASHRAE Standard 140-2011, Table 5-29 and Annex B11-1/B11-2 (Delsante, Stokes and Walsh 1983)\"\n  }},\n",
        measured.floor_conduction_w,
        closed,
        deviation_pct,
        measured.unknowns,
        measured.iterations,
        measured.relative_residual,
        cfg.fine_cell_m,
        cfg.far_field_m,
        cfg.ground_depth_m,
    );
    let s11 = qas_airside::results_section_json().expect("§11 QAS case-1 anchor check");

    let mut json = s8;
    // results_section_json opens with ` "section_11_airside": {` and closes
    // with the source note and a newline; assemble the full document.
    json.push_str(&s11);
    json.push_str("}\n");
    print!("{json}");

    if deviation_pct > 0.5 {
        eprintln!(
            "GC10a measurement {0:.1} W deviates {1:.3}% from the closed form — beyond the ±0.5% gate; refusing to emit a passing result",
            measured.floor_conduction_w, deviation_pct
        );
        std::process::exit(1);
    }
}
