//! HVAC equipment rated-performance harness (renamed from `hvac_bestest`,
//! Issue #4294).
//!
//! **Not ASHRAE 140 §9.** This target validates the `fluxion::sim::hvac`
//! equipment models against ASHRAE 90.1-2019 rated efficiencies over an
//! 11-bin synthetic outdoor-temperature distribution labelled "TMY
//! mid-latitude climate" (UA = 337.0 W/K) — an equipment self-consistency
//! check with real bounds and provenance in `data/manifest.json`. It reads
//! no weather file, drives no CE100–CE200 case, and is not validated
//! against the §9 quasi-analytical solutions; §9 coverage today is zero
//! cases (recorded in `docs/KNOWN_ISSUES.md`).
//!
//! The old name read as §9 coverage that does not exist; renamed per #4294
//! so no CI target implies §9 work. `data/` is kept intact under the new
//! name: the bounds and manifest are real and the eventual CE cases need
//! them.
//!
//! This integration-test target is intentionally empty for issue #1754. Follow-on
//! issues will add analytical and comparative cases without coupling reference-data
//! parsing to the simulation runners.

mod analytical;
mod comparative;
mod ha00x;
mod reference_data;
mod runner;
mod section9_ce;
