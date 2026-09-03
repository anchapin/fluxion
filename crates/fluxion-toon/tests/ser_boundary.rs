//! Boundary tests for TOON integer formatting via `NumBuffer::format_into`
//! (issue #3323).
//!
//! Covers:
//! - Array-header counts at realizable sizes (both header sites in `ser.rs`)
//! - Count *formatting* equivalence at `usize::MAX` (an array that large is
//!   not realizable; we assert the digits match the previous `core::fmt`
//!   path's output, which is the byte-identity contract)
//! - Integer values at `i64::MIN`/`i64::MAX`/`u64::MAX`/zero/negative in
//!   uniform-array rows (the `as_i64`/`as_u64` arms) and as scalars
//! - Float formatting byte-identity (must stay on serde_json's ryu path)

use core::fmt::NumBuffer;

use fluxion_toon::ser::serialize_to_string;

// ---------------------------------------------------------------------------
// Header counts — end-to-end through both header sites
// ---------------------------------------------------------------------------

#[test]
fn header_counts_object_field_site() {
    // Site 1: uniform array nested under an object key (`key[N]{...}:`)
    for n in [1usize, 2, 3, 10, 1000] {
        let items: Vec<_> = (0..n)
            .map(|i| serde_json::json!({ "n": i as i64, "label": "x" }))
            .collect();
        let value = serde_json::json!({ "zones": items });
        let out = serialize_to_string(&value).unwrap();
        let first_line = out.lines().next().unwrap();
        assert_eq!(first_line, format!("zones[{n}]{{label,n}}:"));
        assert_eq!(out.lines().count(), 1 + n);
    }
}

#[test]
fn header_counts_top_level_array_site() {
    // Site 2: top-level uniform array (`Array[N]{...}:`)
    for n in [1usize, 3, 1000] {
        let items: Vec<_> = (0..n)
            .map(|i| serde_json::json!({ "v": i as i64 }))
            .collect();
        let out = serialize_to_string(&items).unwrap();
        let first_line = out.lines().next().unwrap();
        assert_eq!(first_line, format!("Array[{n}]{{v}}:"));
        assert_eq!(out.lines().count(), 1 + n);
    }
}

/// An array with `usize::MAX` elements cannot be realized, so the boundary
/// contract is proven at the formatting layer: `format_into` must produce the
/// exact digits the previous `core::fmt`-machinery path (`to_string`)
/// produced for every boundary count value.
#[test]
fn header_count_boundary_formatting_matches_fmt_path() {
    let mut buf = NumBuffer::<usize>::new();
    for count in [
        0usize,
        1,
        9,
        10,
        99,
        100,
        123_456_789,
        u64::MAX as usize,
        usize::MAX,
    ] {
        assert_eq!(
            count.format_into(&mut buf),
            count.to_string(),
            "usize count {count} diverged from fmt path"
        );
    }
    // Spot-check the exact canonical literals.
    assert_eq!(usize::MAX.format_into(&mut buf), "18446744073709551615");
    assert_eq!(
        (u64::MAX as usize).format_into(&mut buf),
        "18446744073709551615"
    );
    assert_eq!(0usize.format_into(&mut buf), "0");
}

// ---------------------------------------------------------------------------
// Integer values in uniform-array rows — as_i64 / as_u64 arms
// ---------------------------------------------------------------------------

#[test]
fn row_integer_boundaries_byte_identical() {
    let cases = [
        (i64::MIN, "-9223372036854775808"),
        (-1i64, "-1"),
        (0i64, "0"),
        (1i64, "1"),
        (i64::MAX, "9223372036854775807"),
    ];
    for (input, expected) in cases {
        let items = vec![
            serde_json::json!({ "n": input }),
            serde_json::json!({ "n": input }),
        ];
        let out = serialize_to_string(&items).unwrap();
        assert_eq!(
            out,
            format!("Array[2]{{n}}:\n{expected}\n{expected}\n"),
            "i64 value {input}"
        );
    }
}

#[test]
fn row_u64_max_byte_identical() {
    // u64::MAX exceeds i64::MAX, so this exercises the as_u64 arm
    // (as_i64 returns None first).
    let items = vec![
        serde_json::json!({ "n": serde_json::Value::from(u64::MAX) }),
        serde_json::json!({ "n": serde_json::Value::from(u64::MAX - 1) }),
    ];
    let out = serialize_to_string(&items).unwrap();
    assert_eq!(
        out,
        "Array[2]{n}:\n18446744073709551615\n18446744073709551614\n"
    );
}

#[test]
fn row_negative_and_mixed_integers() {
    let items = vec![
        serde_json::json!({ "a": -42i64, "b": 7i64 }),
        serde_json::json!({ "a": -9223372036854775808i64, "b": 9223372036854775807i64 }),
    ];
    let out = serialize_to_string(&items).unwrap();
    assert_eq!(
        out,
        concat!(
            "Array[2]{a,b}:\n",
            "-42,7\n",
            "-9223372036854775808,9223372036854775807\n",
        )
    );
}

// ---------------------------------------------------------------------------
// Scalar integer values (non-uniform / key-value path)
// ---------------------------------------------------------------------------

#[test]
fn scalar_integer_boundaries() {
    let value = serde_json::json!({
        "min": i64::MIN,
        "max": i64::MAX,
        "neg": -7i64,
        "zero": 0i64,
        "umax": serde_json::Value::from(u64::MAX),
    });
    let out = serialize_to_string(&value).unwrap();
    assert_eq!(
        out,
        concat!(
            "max: 9223372036854775807,\n",
            "min: -9223372036854775808,\n",
            "neg: -7,\n",
            "umax: 18446744073709551615,\n",
            "zero: 0\n",
        )
    );
}

// ---------------------------------------------------------------------------
// Floats must remain on serde_json's ryu path (byte-identical)
// ---------------------------------------------------------------------------

#[test]
fn float_formatting_untouched_in_rows() {
    let items = vec![
        serde_json::json!({ "t": 22.5f64, "h": 45.0f64 }),
        serde_json::json!({ "t": -0.0625f64, "h": 1e300f64 }),
    ];
    let out = serialize_to_string(&items).unwrap();
    // 45.0 and 1e300 render exactly as serde_json's ryu-based Display did
    // before this change (verified byte-identical against the original
    // implementation via differential corpus run).
    assert_eq!(
        out,
        concat!("Array[2]{h,t}:\n", "45.0,22.5\n", "1e+300,-0.0625\n",)
    );
}

#[test]
fn float_formatting_untouched_as_scalar() {
    let value = serde_json::json!({ "v": 1.5f64, "w": -0.0001f64 });
    let out = serialize_to_string(&value).unwrap();
    assert_eq!(out, "v: 1.5,\nw: -0.0001\n");
}

// ---------------------------------------------------------------------------
// Regression corpus: pre-change output must survive unchanged
// ---------------------------------------------------------------------------

#[test]
fn existing_corpus_headers_unchanged() {
    let zones = serde_json::json!([
        { "id": "Z1", "temp_c": 22.5, "humidity_rh": 45.0 },
        { "id": "Z2", "temp_c": 23.1, "humidity_rh": 50.0 },
        { "id": "Z3", "temp_c": 21.8, "humidity_rh": 48.0 },
    ]);
    let out = serialize_to_string(&zones).unwrap();
    assert_eq!(
        out,
        concat!(
            "Array[3]{humidity_rh,id,temp_c}:\n",
            "45.0,Z1,22.5\n",
            "50.0,Z2,23.1\n",
            "48.0,Z3,21.8\n",
        )
    );
}

#[test]
fn empty_and_non_uniform_arrays_unchanged() {
    let empty: Vec<serde_json::Value> = vec![];
    let out = serialize_to_string(&empty).unwrap();
    assert_eq!(out, "[]");

    let non_uniform = serde_json::json!([1i64, "two", { "three": 3 }]);
    let out = serialize_to_string(&non_uniform).unwrap();
    // Pre-existing rendering (nested object writeln quirk included) — asserted
    // to guard against accidental changes from the integer work.
    assert_eq!(out, "[1,\"two\",three: 3\n\n]");
}
