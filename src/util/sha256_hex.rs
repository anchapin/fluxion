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

/// Decode a 64-character hex SHA-256 string into the raw 32-byte digest.
///
/// Returns `None` if the input is not exactly 64 ASCII hex characters
/// (either lowercase or uppercase). Used by `crate::ai::surrogate::validate_hash`
/// (Issue #3655, CWE-208) to convert both sides of a tag comparison into
/// bytes before invoking `subtle::ConstantTimeEq`. Accepting uppercase keeps
/// the previous `eq_ignore_ascii_case` contract for callers that hand in a
/// registry entry stored in mixed case.
///
/// `u8::from_str_radix(_, 16)` is constant in its return value across all
/// valid inputs, but we deliberately do not include any short-circuit on
/// the first invalid character — every position is visited so an attacker
/// cannot infer digest bytes from a timing differential on length-or-format
/// checks. (The byte-level equality itself is enforced by `subtle`.)
pub fn decode_sha256_hex(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let (pairs, _tail) = hex.as_bytes().as_chunks::<2>();
    debug_assert_eq!(_tail.len(), 0, "hex.len()==64 implies zero remainder");
    let mut out = [0u8; 32];
    for (i, pair) in pairs.iter().enumerate() {
        let hi = decode_nibble(pair[0])?;
        let lo = decode_nibble(pair[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

#[inline]
fn decode_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_sha256_hex, sha256_hex};
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
        let hex = sha256_hex(digest);
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
        assert_eq!(sha256_hex(bytes), "deadbeef");
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

    #[test]
    fn decode_roundtrips_sha256_hex_for_lowercase_and_uppercase() {
        let hex_lower = sha256_hex(Sha256::digest(b"abc"));
        let hex_upper = hex_lower.to_uppercase();
        let bytes_lower = decode_sha256_hex(&hex_lower).expect("lower hex must decode");
        let bytes_upper = decode_sha256_hex(&hex_upper).expect("upper hex must decode");
        assert_eq!(bytes_lower, bytes_upper);
        // Independent ground truth for SHA-256("abc").
        assert_eq!(
            bytes_lower,
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }

    #[test]
    fn decode_rejects_wrong_length_and_non_hex() {
        assert_eq!(decode_sha256_hex(""), None);
        assert_eq!(decode_sha256_hex(&"a".repeat(63)), None);
        assert_eq!(decode_sha256_hex(&"a".repeat(65)), None);
        assert_eq!(decode_sha256_hex(&"z".repeat(64)), None);
        // Non-ASCII must not panic.
        assert_eq!(decode_sha256_hex(&"ñ".repeat(64)), None);
    }
}
