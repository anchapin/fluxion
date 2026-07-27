# TOON Format Specification

**Token-Oriented Object Notation (TOON)**

Version 1.0

## Overview

TOON is a compact, tabular serialization format designed to reduce LLM context-window usage by collapsing uniform flat-struct arrays into CSV-style blocks with explicit count headers.

## Format Structure

### Header
```
toon:v1
count:<N>
```

### Scalar Fields
```
key=value
```

### Array Fields (Uniform Flat-Structs)
```
array:<FIELD_NAME>:<COUNT>
<FIELD_NAME>_0,<FIELD_NAME>_1,...<FIELD_NAME>_<COUNT-1>
```

### Example

**JSON Input:**
```json
{
  "zone_count": 3,
  "zones": [
    {"name": "Zone1", "temperature": 22.5},
    {"name": "Zone2", "temperature": 23.0},
    {"name": "Zone3", "temperature": 21.8}
  ]
}
```

**TOON Output:**
```
toon:v1
count=3
zones:3
name_0,name_1,name_2
Zone1,Zone2,Zone3
temperature_0,temperature_1,temperature_2
22.5,23.0,21.8
```

## Design Principles

1. **Explicit Count Headers**: Every array field is preceded by its count, enabling length validation.
2. **CSV-Style Arrays**: Uniform flat-struct arrays are collapsed into rows for token efficiency.
3. **Dot-Separated Field Names**: Array elements use underscore indexing (`temperature_0`) for unambiguous field binding.
4. **No Nested Structures**: TOON is designed for flat, uniform data structures only.

## Token Reduction

Compared to JSON, TOON typically achieves 35-50% token reduction for uniform array data by:
- Eliminating repetitive field names within arrays
- Using compact CSV rows instead of repeated object literals
- Reducing syntactic overhead (brackets, quotes, commas)

## Limitations

TOON is NOT suitable for:
- Numerical solvers (internal CTF/FD state)
- Multi-node thermal mass configurations
- Non-uniform or deeply nested data structures
- Hand-edited configuration files

## Implementation

See `fluxion-toon` crate documentation for serializer/deserializer implementation details.