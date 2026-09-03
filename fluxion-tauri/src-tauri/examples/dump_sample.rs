//! Prints the sample building geometry as JSON to stdout.
//!
//! Used to regenerate the frontend contract fixture:
//!
//! ```sh
//! cargo run --example dump_sample -p fluxion-tauri \
//!   > ../frontend/tests/fixtures/rust-sample-geometry.json
//! ```
//!
//! The frontend test suite asserts that this output and the embedded web
//! fallback (`src/lib/sampleGeometry.ts`) stay in sync with the Rust types.

// `src/geometry.rs` is standalone (serde + serde_json only), so examples can
// include it directly; bin-target modules are not importable otherwise.
mod geometry {
    include!("../src/geometry.rs");
}

fn main() {
    let json = serde_json::to_string_pretty(&geometry::BuildingGeometry::sample())
        .expect("serialize sample");
    println!("{json}");
}
