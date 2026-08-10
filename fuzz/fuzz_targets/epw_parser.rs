//! Fuzz target: `EpwWeatherSource::from_file` (EPW weather-file parser).
//!
//! The EPW parser is one of the public API surfaces called out in issue #2537.
//! It ingests untrusted `*.epw` files (EnergyPlus Weather format) which are
//! frequently downloaded from third-party repositories, so it must be robust
//! against arbitrary byte streams without panicking, indexing out of bounds,
//! or integer-overflowing on malformed numeric fields.
//!
//! The fuzzer writes the raw bytes to a temporary file and hands it to
//! `from_file`, which is the only public entry point. Malformed input must
//! surface as `Err(WeatherError)` rather than a panic.
//!
//! **Invariant:** the parser never panics on arbitrary input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // `from_file` is the public API; writing to a temp file is the only way
    // to reach it without depending on the private `parse<R: BufRead>` helper.
    let mut tmp = match tempfile::Builder::new()
        .prefix("fluxion-fuzz-epw-")
        .suffix(".epw")
        .tempfile()
    {
        Ok(f) => f,
        Err(_) => return, // ephemeral FS pressure — not a parser bug.
    };
    if tmp.write_all(data).is_err() {
        return;
    }
    if tmp.flush().is_err() {
        return;
    }

    // `from_file` opens the path itself, so the NamedTempFile just needs to
    // stay alive (holding the on-disk inode) until the parse returns. It is
    // auto-deleted when `tmp` drops at end of scope.
    let path = tmp.path().to_owned();

    // === Invariant: parser never panics on arbitrary bytes. ===
    let _ = fluxion::weather::epw::EpwWeatherSource::from_file(&path);

    // `tmp` drops here, removing the temp file. No manual cleanup needed.
});
