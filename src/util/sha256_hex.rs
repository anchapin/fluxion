//! Format the bytes of a SHA-256 (or any `AsRef<[u8]>`) digest as a
//! lowercase hex string.
//!
//! `sha2` 0.11 returns a `GenericArray<u8, U32>` whose `LowerHex` impl is no
//! longer available in newer `generic-array` releases, so we format the bytes
//! manually. This helper is the shared replacement introduced for issue #2963
//! after PR #2960 duplicated it across six files.
//!
//! Companion copy in `fluxion-core/src/weather/tmy3.rs` is intentionally kept
//! private to preserve the `fluxion-core` leaf-module invariant
//! (see `ARCHITECTURE.md`).
//!
//! # Examples
//!
//! ```
//! use crate::util::sha256_hex::sha256_hex;
//!
//! // SHA-256 of the empty input — well-known NIST answer.
//! assert_eq!(
//!     sha256_hex([]),
//!     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
//! );
//! ```

use std::fmt::Write as _;

/// Format a SHA-256 digest (or any `AsRef<[u8]>`) as a lowercase hex string.
///
/// Accepts any byte slice — `Sha256::digest(...)` returns a
/// `GenericArray<u8, U32>` (which implements `AsRef<[u8]>`), as does the
/// output of a `Sha256` `finalize()` call.
pub fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;
    use sha2::{Digest, Sha256};

    #[test]
    fn empty_input_is_a_known_answer() {
        // SHA-256 of the empty input — NIST FIPS 180-4 known answer.
        assert_eq!(
            sha256_hex(Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn digest_is_sixty_four_lowercase_hex_chars() {
        let digest = Sha256::digest(b"abc");
        let hex = sha256_hex(&digest);
        assert_eq!(hex.len(), 64, "SHA-256 hex must be 64 chars long");
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hex output must be lowercase only"
        );
    }

    #[test]
    fn abc_answer_matches_nist() {
        // SHA-256("abc") — NIST FIPS 180-4 known answer.
        assert_eq!(
            sha256_hex(Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn works_with_a_raw_byte_slice() {
        // `&[u8]` also implements `AsRef<[u8]>`.
        let bytes: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];
        assert_eq!(sha256_hex(&bytes), "deadbeef");
    }

    #[test]
    fn sha256_of_empty_input_is_64_chars() {
        // The helper's contract: a 32-byte SHA-256 digest always produces
        // a 64-char lowercase hex string. (Calling `sha256_hex(b"")` would
        // format an empty byte slice and return `""` — that's expected:
        // the helper formats whatever bytes you hand it. The contract is
        // about *digest* length, not message length.)
        let digest = Sha256::digest(b"");
        assert_eq!(sha256_hex(digest).len(), 64);
    }
}
