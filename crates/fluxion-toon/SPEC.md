# TOON Format Specification

**Token-Oriented Object Notation (TOON)** — Version 1.0

## Overview

TOON is a compact, tabular serialization format that reduces LLM context-window usage by 35–50% compared to JSON for uniform building-energy model state vectors (zone temperatures, surface fluxes, HVAC energy arrays).

## Syntax Summary

### Scalar
```
setpoint: 22.0
```

### Uniform Object Array
Header declares count + field list, then CSV-style rows follow:
```
zone_temps[3]{id,temp_c,humidity_rh}:
  z0, 21.4, 45.0
  z1, 22.1, 44.2
  z2, 20.8, 46.1
```

### Nested Block
```
model{model_id,num_zones,window_u_value}:
  mid0, 3, 1.5
```

## Primitive Type Encodings

| Type     | Syntax Example         | Notes                              |
|----------|------------------------|------------------------------------|
| `f64`    | `22.0`, `-1.5e-3`      | Decimal or scientific notation      |
| `i64`    | `42`, `-7`             | No decimal point                   |
| `bool`   | `true`, `false`        | Lowercase                          |
| `string` | `zone_a`, `"Hello World"` | Unquoted unless contains `:`,`,`,`\n` |

## Uniform Array Collapse Rules

The `[N]{fields}:` syntax applies **only** when:
1. All elements are flat objects (no nested structures)
2. All objects share the **exact same set of fields** (same names, same types)
3. The array length is explicit (`[N]`)

If any element is non-uniform or nested, fall back to per-element JSON representation.

### Examples

**Collapse (uniform):**
```
zone_temps[3]{id,temp_c}:
  z0, 21.4
  z1, 22.1
  z2, 20.8
```

**No collapse (mixed types in field):**
```
readings[2]:
  {"id": "r0", "value": 22.4}
  {"id": "r1", "value": "error"}
```

## Newline / Indentation Semantics

- Indentation uses **2 spaces** for nested blocks
- Rows under a uniform array header are indented by 2 spaces
- Blank lines are **not allowed** within uniform array blocks
- Trailing whitespace is ignored

## Error Handling Contract

A parse error occurs when:
- Header count (`[N]`) does not match the number of following rows
- Field count in header does not match comma-separated values in a row
- Unrecognized or malformed type literal
- Missing required rows for a declared array

### Error Variants

| Variant           | Condition                                           |
|-------------------|-----------------------------------------------------|
| `LengthMismatch`  | Row count differs from declared `[N]`               |
| `InvalidSyntax`   | Malformed header or type literal                    |
| `MalformedRow`    | Comma-separated values don't match field count      |

## Explicit Length Header Semantics

The `zone_temps[3]` syntax encodes the array length as a hallucination guardrail:
- Parser **must** verify the number of rows equals `3`
- Mismatch raises `LengthMismatch`
- This prevents LLM models from omitting or inventing array elements

## What Constitutes a "Uniform" Struct

A struct is uniform for TOON collapse if and only if:
1. It is **flat** — no nested objects or arrays as field values
2. All fields have **primitive types** (`f64`, `i64`, `bool`, `string`)
3. All instances in the array have **identical field names** in the same order

Example **uniform** struct:
```rust
struct ZoneTemp {
    id: String,
    temp_c: f64,
    humidity_rh: f64,
}
```

Example **non-uniform** (cannot collapse):
```rust
struct Reading {
    id: String,
    value: serde_json::Value,  // mixed types
}
```

## Limitations

TOON is **NOT** suitable for:
- Internal numerical solver state (CTF/FD thermal networks)
- Multi-node thermal mass configurations with deep nesting
- Non-uniform or deeply nested data structures
- Hand-edited configuration files (use JSON/YAML instead)

## Example Transformations

### JSON → TOON

**JSON input:**
```json
{
  "zone_temps": [
    {"id": "z0", "temp_c": 21.4, "humidity_rh": 45.0},
    {"id": "z1", "temp_c": 22.1, "humidity_rh": 44.2},
    {"id": "z2", "temp_c": 20.8, "humidity_rh": 46.1}
  ]
}
```

**TOON output:**
```
zone_temps[3]{id,temp_c,humidity_rh}:
  z0, 21.4, 45.0
  z1, 22.1, 44.2
  z2, 20.8, 46.1
```

## Implementation

See `fluxion-toon` crate documentation for serializer/deserializer implementation details.

## References

- Issue #2066: Define TOON Format Specification and Create Scaffold Crate
- Issue #2071: Implement TOON serializer (follow-up)