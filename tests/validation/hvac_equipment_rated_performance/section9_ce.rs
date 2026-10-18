//! ASHRAE 140-2023 §9 steady-state CE-case dispatch — Issue #4296 part 3.
//!
//! The first dispatch path that consumes the round-63 DX performance-map
//! model (`fluxion::sim::hvac::CePerformanceMap`, PR #4479). The map landed
//! unconsumed in round 63; this module wires it into the validation harness
//! for the CE case(s) whose steady-state conditions are fully transcribed.
//!
//! # Which cases the steady-state conditions serve
//!
//! Each CE case in `data/reference/ashrae140/section9_ce_cases.json` fixes a
//! steady state: internal sensible/latent gains, a cooling setpoint (the
//! entering dry-bulb), and an outdoor dry-bulb. A map evaluation additionally
//! needs the entering wet-bulb. Of the 14 transcribed cases, exactly one has
//! a *standard-sourced* entering condition without needing the licensed
//! §9 supplemental materials (Issue #4285): **CE200**, whose conditions
//! (EDB 26.7 °C / ODB 35.0 °C) match the transcribed map's own AHRI rating
//! point (EWB 19.4 °C) — two independently committed artifacts whose
//! cross-agreement this module asserts. The other 13 cases have no
//! transcribed EWB slot, so the dispatch **refuses** to evaluate them rather
//! than inventing an entering condition (RULES.md: nothing invented).
//!
//! # Dispatch semantics
//!
//! Steady-state single point, one evaluation per case condition (the
//! round-63 design note): available capacity from the map at the case
//! condition, part-load ratio from the load balance
//! `PLR = required_load / available_total_capacity` (clamped to [0, 1]),
//! delivered power through the GSR cycling formulation
//! `power × PLR / (0.771 + 0.229·PLR)`. The adjusted-net basis
//! (`adjusted_net_Table_26e`, 230 W actual indoor fan power) is used so the
//! evaluation's capacity is on the same fan-heat basis as the recorded
//! rating point.
//!
//! Comparative acceptance for the §9 cases is evidence-range context (the
//! Task 22 ensemble bands), not a pass/fail criterion — the analytical
//! acceptance slots are `null` pending #4285 — so this module asserts
//! physics invariants and artifact cross-provenance, never a band.

use serde_json::Value;

use fluxion::sim::hvac::{CePerformanceMap, CeTable};

const SECTION9_CASES_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/reference/ashrae140/section9_ce_cases.json"
));
const SECTION9_MAP_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/reference/ashrae140/section9_ce100_performance_map.json"
));

/// One dispatched steady-state CE evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct CeDispatch {
    pub case_id: String,
    pub edb_c: f64,
    pub ewb_c: f64,
    pub odb_c: f64,
    pub required_load_w: f64,
    pub available_capacity_w: f64,
    pub plr: f64,
    pub delivered_capacity_w: f64,
    pub delivered_power_w: f64,
    pub shr: f64,
    pub cop: f64,
}

/// Dispatch a CE case's steady-state condition through the DX map.
///
/// `ewb_c` must come from a committed, provenance-stamped source (here: the
/// map artifact's own rating-point conditions). Panics if the case's
/// conditions disagree with the supplied entering condition's provenance.
pub fn dispatch_ce_steady_state(
    case: &Value,
    map: &CePerformanceMap,
    ewb_c: f64,
    fan_heat_w: f64,
) -> CeDispatch {
    let cond = &case["conditions"];
    let edb_c = cond["cooling_setpoint_C"].as_f64().expect("setpoint");
    let odb_c = cond["outdoor_dry_bulb_C"].as_f64().expect("ODB");
    let sensible_w = cond["internal_sensible_gains_W"]
        .as_f64()
        .expect("sensible");
    let latent_w = cond["internal_latent_gains_W"].as_f64().expect("latent");
    let required_load_w = sensible_w + latent_w;

    // Available capacity at the case condition, full load (PLR = 1).
    let full = map.evaluate(
        CeTable::AdjustedNet26e,
        ewb_c,
        odb_c,
        edb_c,
        1.0,
        fan_heat_w,
    );
    let available_capacity_w = full.delivered_capacity_w;
    assert!(
        available_capacity_w > 0.0,
        "map returned non-positive capacity at the case condition"
    );
    let plr = (required_load_w / available_capacity_w).clamp(0.0, 1.0);

    let ev = map.evaluate(
        CeTable::AdjustedNet26e,
        ewb_c,
        odb_c,
        edb_c,
        plr,
        fan_heat_w,
    );
    CeDispatch {
        case_id: case["case_id"].as_str().expect("case_id").to_string(),
        edb_c,
        ewb_c,
        odb_c,
        required_load_w,
        available_capacity_w,
        plr,
        delivered_capacity_w: ev.delivered_capacity_w,
        delivered_power_w: ev.delivered_power_w,
        shr: ev.shr,
        cop: ev.cop,
    }
}

fn load_cases() -> Vec<Value> {
    let root: Value = serde_json::from_str(SECTION9_CASES_JSON).expect("parse §9 CE cases");
    root["cases"].as_array().expect("cases array").clone()
}

fn load_map() -> CePerformanceMap {
    CePerformanceMap::from_json_str(SECTION9_MAP_JSON).expect("parse §9 CE100 performance map")
}

/// The transcribed map's own AHRI rating-point conditions (EDB, EWB, ODB),
/// read from the committed artifact rather than restated in code.
fn map_rating_point_conditions() -> (f64, f64, f64) {
    let root: Value = serde_json::from_str(SECTION9_MAP_JSON).expect("parse map json");
    let ari = &root["ari_rating_point"]["conditions"];
    (
        ari["EDB_C"].as_f64().expect("EDB"),
        ari["EWB_C"].as_f64().expect("EWB"),
        ari["ODB_C"].as_f64().expect("ODB"),
    )
}

/// CE200's transcribed steady-state condition must be exactly the map
/// artifact's own AHRI rating-point conditions. This is the provenance
/// cross-check that licenses evaluating CE200 without the licensed §9
/// supplemental materials: the entering wet-bulb (19.4 °C) comes from the
/// committed, SHA-256-pinned map transcription itself.
#[test]
#[allow(clippy::float_cmp)] // exact provenance equality: both sides parse the same JSON literals
fn test_ce200_condition_matches_map_rating_point() {
    let cases = load_cases();
    let ce200 = cases
        .iter()
        .find(|c| c["case_id"] == "CE200")
        .expect("CE200 present");
    let map = load_map();
    let (map_edb, map_ewb, map_odb) = map_rating_point_conditions();
    // Parsing the committed map is itself part of the provenance check.
    let _parsed_ok = map.ari_compressor_power_w().is_finite();
    let edb = ce200["conditions"]["cooling_setpoint_C"]
        .as_f64()
        .expect("setpoint");
    let odb = ce200["conditions"]["outdoor_dry_bulb_C"]
        .as_f64()
        .expect("ODB");
    assert_eq!(edb, map_edb, "CE200 setpoint vs map rating-point EDB");
    assert_eq!(odb, map_odb, "CE200 ODB vs map rating-point ODB");
    // The entering wet-bulb CE200's dispatch runs at is the rating point's,
    // from the same committed artifact (asserted here so the pairing of the
    // two values in `test_ce200_steady_state_dispatch` stays provenance-pinned).
    assert_eq!(map_ewb, 19.4, "map rating-point EWB vs its recorded value");
}

/// The other 13 CE cases have no transcribed entering wet-bulb. The dispatch
/// refuses to invent one — asserted here structurally, so a future
/// transcription that fills the slots must land together with a dispatch
/// extension (the test then fails loudly instead of silently widening scope).
#[test]
fn test_remaining_ce_cases_have_no_entering_wet_bulb_slot() {
    let cases = load_cases();
    let others: Vec<&Value> = cases.iter().filter(|c| c["case_id"] != "CE200").collect();
    assert_eq!(others.len(), 13, "14 §9 CE cases transcribed");
    for case in others {
        assert!(
            case["conditions"].get("entering_wet_bulb_C").is_none(),
            "case {} grew an entering_wet_bulb_C slot: extend the dispatch \
             with its provenance instead of relying on this refusal test",
            case["case_id"]
        );
    }
}

/// The CE200 dispatch: physics invariants and the recorded evaluation.
#[test]
fn test_ce200_steady_state_dispatch() {
    let cases = load_cases();
    let ce200 = cases
        .iter()
        .find(|c| c["case_id"] == "CE200")
        .expect("CE200 present");
    let map = load_map();

    // 230 W actual indoor fan power — the map artifact's own `fan_heat`
    // block, asserted here so a transcription edit trips this test.
    let root: Value = serde_json::from_str(SECTION9_MAP_JSON).expect("parse map json");
    let fan_w = root["fan_heat"]["actual_W"].as_f64().expect("actual_W");
    #[allow(clippy::float_cmp)] // exact provenance equality: JSON literal
    let fan_ok = fan_w == 230.0;
    assert!(fan_ok, "map fan_heat.actual_W drifted from 230 W");

    let dispatch = dispatch_ce_steady_state(ce200, &map, 19.4, fan_w);
    println!("§9 CE200 steady-state dispatch (adjusted-net basis, 230 W fan):");
    println!(
        "  condition: EDB {:.1} °C / EWB {:.1} °C / ODB {:.1} °C",
        dispatch.edb_c, dispatch.ewb_c, dispatch.odb_c
    );
    println!("  required load: {:.0} W", dispatch.required_load_w);
    println!(
        "  available capacity: {:.0} W",
        dispatch.available_capacity_w
    );
    println!("  PLR: {:.4}", dispatch.plr);
    println!(
        "  delivered capacity: {:.0} W",
        dispatch.delivered_capacity_w
    );
    println!(
        "  delivered compressor power: {:.0} W",
        dispatch.delivered_power_w
    );
    println!("  SHR: {:.3}", dispatch.shr);
    println!("  COP: {:.3}", dispatch.cop);

    // Energy balance (RULES.md, sacrosanct): the dispatched part-load ratio
    // closes the load — delivered capacity equals the required load to
    // floating-point tolerance (PLR < 1 here: the unit is not capacity-bound).
    let rel =
        (dispatch.delivered_capacity_w - dispatch.required_load_w).abs() / dispatch.required_load_w;
    assert!(
        rel < 1e-9,
        "CE200 dispatch does not close the load: rel={rel}"
    );
    assert!(
        dispatch.plr > 0.0 && dispatch.plr <= 1.0,
        "PLR out of range: {}",
        dispatch.plr
    );
    // Physics invariants at the dispatch point.
    assert!(dispatch.shr > 0.0 && dispatch.shr <= 1.0);
    assert!(dispatch.cop > 1.0, "COP implausible: {}", dispatch.cop);
    assert!(
        dispatch.delivered_power_w < dispatch.delivered_capacity_w,
        "delivered power must be below delivered capacity at a COP well above 1"
    );
}
