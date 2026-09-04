// Shim that wraps the seed module under a `seed::*` namespace.
//
// The seed file at `tools/evolution/seeds/ctf/seed.rs` declares
// `pub struct Candidate` and `impl Kernel for Candidate` at the
// top level. This shim re-exports `Candidate` (and only the items
// the OpenEvolve harness needs) under `seed::Candidate` so callers
// can `use seed::Candidate;` and the rest of the implementation
// (private helpers, internal types) is hidden by the namespace.
//
// We deliberately do NOT use `pub use seed_inner::*` — the seed
// declares `pub struct CTFMaterial`, which collides with the
// production `fluxion::physics::ctf_coefficients::CTFMaterial`
// used elsewhere in this test. Only `Candidate` is re-exported.
//
// The seed uses many `for i in 0..n` patterns over `Vec<Vec<f64>>`
// matrices — clippy::needless_range_loop fires, but the loops are
// actually required because Vec<Vec<f64>> indexing requires integer
// indices in two dimensions. We allow the lint at the module level
// (an `include!`'d file can't carry its own outer attrs reliably).
#[allow(clippy::needless_range_loop)]
pub mod seed {
    include!("../tools/evolution/seeds/ctf/seed.rs");
}
