# fluxion-toon

**Token-Oriented Object Notation (TOON)** — a compact serialization format that
shrinks LLM context-window usage by 35–50% versus JSON for the uniform,
flat-struct arrays that dominate building-energy-model state vectors (zone
temperatures, surface fluxes, HVAC energy readings).

In 30 seconds: TOON adds a `toon:v1` header to a JSON body and collapses arrays
of *identical* flat structs into a CSV-style block with an explicit count and
field list (`zone_temps[3]{id,temp_c}:\n  z0, 21.4\n  …`). The count is a
hallucination guardrail — the parser rejects any document whose row count
differs from the declared `[N]`. Any value that isn't a uniform struct array
falls back to ordinary JSON, so TOON is a strict superset for round-tripping.

This crate provides drop-in `to_string` / `from_str` functions over any
`serde::Serialize` / `Deserialize` type, plus the lower-level document model.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Why TOON](#why-toon)
- [API](#api)
- [Format at a glance](#format-at-a-glance)
- [Limitations](#limitations)
- [Specification](#specification)

## Install

```bash
cargo add fluxion-toon
```

or manually:

```toml
[dependencies]
fluxion-toon = "0.1"
```

The crate re-exports everything from the root: `to_string`, `from_str`,
`token_savings_pct`, `ToonError`, and `Result`.

## Quick start

Serialize a value and deserialize it back — the round-trip is lossless for any
`serde` type:

```rust
use fluxion_toon::{to_string, from_str};

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct Zone {
    name: String,
    temperature: f64,
}

let zone = Zone {
    name: "Zone1".to_string(),
    temperature: 22.5,
};

let toon = to_string(&zone)?;
let round_trip: Zone = from_str(&toon)?;
assert_eq!(zone, round_trip);
# Ok::<(), fluxion_toon::ToonError>(())
```

The power of TOON shows with **arrays of uniform flat structs**, which collapse
into a compact CSV block. The explicit `[N]` count makes LLM omissions or
inventions fail loudly with `ToonError::LengthMismatch`:

```rust
use fluxion_toon::{to_string, from_str};

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct ZoneTemp {
    id: String,
    temp_c: f64,
    humidity_rh: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct Snapshot {
    zone_temps: Vec<ZoneTemp>,
}

let snap = Snapshot {
    zone_temps: vec![
        ZoneTemp { id: "z0".into(), temp_c: 21.4, humidity_rh: 45.0 },
        ZoneTemp { id: "z1".into(), temp_c: 22.1, humidity_rh: 44.2 },
        ZoneTemp { id: "z2".into(), temp_c: 20.8, humidity_rh: 46.1 },
    ],
};

let toon = to_string(&snap)?;
// The uniform array is emitted as a CSV-style block:
//   zone_temps[3]{id,temp_c,humidity_rh}:
//     z0, 21.4, 45.0
//     z1, 22.1, 44.2
//     z2, 20.8, 46.1

let back: Snapshot = from_str(&toon)?;
assert_eq!(snap, back);
# Ok::<(), fluxion_toon::ToonError>(())
```

Measure the savings against the JSON baseline with `token_savings_pct`:

```rust
use fluxion_toon::{to_string, token_savings_pct};

# #[derive(serde::Serialize)]
# struct Demo;
# let value = Demo;
let json = serde_json::to_string(&value).unwrap_or_default();
let toon = to_string(&value).unwrap_or_default();
let saved = token_savings_pct(json.len(), toon.len());
```

## Why TOON

Feeding building-energy state to an LLM (for surrogate selection, anomaly
diagnostics, or natural-language reporting) means serializing model snapshots to
text. Repeated JSON keys (`"temp_c":`, `"humidity_rh":`, …) blow up token counts
on every row. TOON declares the schema once in the header and writes only the
values, cutting tokens by a third to a half on realistic zone arrays.

The declared `[N]` length is also a structural guardrail: the parser verifies
the row count, so a model that drops or invents elements fails fast instead of
silently corrupting state.

## API

| Item | Signature | Purpose |
|------|-----------|---------|
| `to_string` | `fn to_string<T: Serialize>(&T) -> Result<String>` | Serialize any `serde` value to TOON. |
| `from_str` | `fn from_str<T: DeserializeOwned>(&str) -> Result<T>` | Deserialize a TOON document back into `T`. |
| `token_savings_pct` | `fn token_savings_pct(json_len: usize, toon_len: usize) -> f64` | Percentage token savings of TOON vs JSON (negative = TOON larger). |
| `ToonError` | enum | `LengthMismatch`, `InvalidSyntax`, `MalformedRow`, `InvalidHeader`, `TooLarge` (DoS guard), `Json`, … |
| `parse` module | `ToonDocument`, `ParsedScalar`, `parse_line`, … | Lower-level document model + line/array parsers for custom workflows. |
| `patch` module | `ModelPatch`, `parse_toon_patch` | Parse a TOON-encoded model patch. |

`to_string` and `from_str` are the public entry points for nearly all use cases;
the `parse` and `patch` modules are exposed for callers that need to inspect the
document structure directly.

## Format at a glance

A TOON document is a `toon:v1` header line followed by a body:

```
toon:v1
zone_temps[3]{id,temp_c,humidity_rh}:
  z0, 21.4, 45.0
  z1, 22.1, 44.2
  z2, 20.8, 46.1
```

| Construct | Example | Notes |
|-----------|---------|-------|
| Header | `toon:v1` | Required first line. |
| Scalar | `setpoint: 22.0` | |
| Uniform array | `name[N]{f1,f2}:` + N CSV rows | Only when all elements share identical flat primitive fields. |
| Primitive `f64` | `22.0`, `-1.5e-3` | Decimal or scientific. |
| Primitive `i64` | `42`, `-7` | No decimal point. |
| Primitive `bool` | `true`, `false` | Lowercase. |
| Primitive `string` | `zone_a`, `"has, comma"` | Unquoted unless it contains `:`, `,`, or `\n`. |

Rows are indented two spaces; blank lines are **not** allowed inside a uniform
array block. Anything that doesn't satisfy the uniform-collapse rules
(non-uniform fields, nested structures, mixed types) is emitted as ordinary
JSON, so round-tripping arbitrary `serde` values always works.

## Limitations

TOON is **not** a general-purpose replacement for JSON/YAML. It is unsuitable
for:

- Internal numerical solver state (CTF/FD thermal networks).
- Multi-node thermal-mass configurations with deep nesting.
- Deeply nested or non-uniform data structures (these fall back to JSON, losing
  the savings).
- Hand-edited configuration files (use JSON/YAML instead).

It is purpose-built for compactly shipping uniform state vectors to and from
LLMs.

## Specification

The complete, normative grammar, collapse rules, and error contract live in
**[SPEC.md](./SPEC.md)**. Refer there for:

- The full `zone_temps[N]{fields}:` uniform-array-collapse rules.
- Exact newline/indentation semantics.
- The full `ToonError` variant → condition table.
- The DoS-hardening array-size cap (`parse::MAX_ARRAY_ELEMENTS`, Issue #2527).

References: Issue #2066 (define the TOON spec + scaffold crate), Issue #2071
(implement the serializer).
