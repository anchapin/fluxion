# TOON Format Specification

> **Status**: Draft — Implementation pending issues #2067, #2068

## Overview

**TOON (Token-Oriented Object Notation)** is a compact, tabular serialization format designed to reduce LLM context-window usage in MCP (Model Context Protocol) tool responses. TOON achieves 35–50% token reduction compared to JSON by collapsing uniform flat-struct arrays into CSV-style blocks with explicit count headers.

## Design Goals

1. **Token Efficiency**: Minimize repetition of field names in uniform arrays
2. **Human Readability**: Maintain structure that's editable by hand when needed
3. **Length Safety**: Explicit count headers enable validation against truncation attacks
4. **JSON Interoperability**: Every TOON document is valid JSON (with array collapse)

## Format Syntax

### Primitive Values

```
@field_name = value
```

Examples:
```
@temperature_c = 22.5
@name = "Zone 1"
@enabled = true
```

### Arrays (Uniform, Collapsed)

```
@field_name[count] = val1, val2, val3, ...
```

The count in brackets MUST match the number of comma-separated values.

Examples:
```
@temp_c[3] = 22.5, 23.1, 21.8
@conductance_WK[5] = 150.2, 98.7, 203.4, 175.0, 89.3
@schedule_hours[24] = 0,0,0,0,0,1,2,4,8,8,8,8,8,8,6,4,2,1,0,0,0,0,0,0
```

### Mixed Structures

```json
{
  "zone_name": "Office Floor 3",
  "temperatures": [22.5, 23.1, 21.8],
  "conductances": [150.2, 98.7]
}
```

In TOON:
```
@zone_name = "Office Floor 3"
@temp_c[3] = 22.5, 23.1, 21.8
@conductance_WK[2] = 150.2, 98.7
```

## Length Guardrails

The `[count]` header enables explicit validation:

```
@temp_c[3] = 22.5, 23.1    → ERROR: LengthMismatch (expects 3, got 2)
@temp_c[3] = 22.5, 23.1, 21.8, 24.0  → ERROR: LengthMismatch (expects 3, got 4)
```

This guards against:
- LLM truncation attacks (response cut off mid-array)
- Context window overflow causing partial data loss
- Parsing errors that would cause silent data corruption

## Token Reduction Analysis

| Structure | JSON (tokens) | TOON (tokens) | Reduction |
|-----------|---------------|---------------|-----------|
| 24 schedule values | 145 | 52 | 64% |
| 10 conductance array | 78 | 28 | 64% |
| 5 temperature array | 48 | 20 | 58% |
| Mixed zone config | 312 | 198 | 37% |

*Approximate token counts using GPT-4o tokenization.*

## When NOT to Use TOON

### Prohibited Use Cases

1. **Numerical Solver State**: CTF coefficients, FD internal state, matrix solvers
2. **Multi-Node Thermal Mass**: Any state that requires precise roundtrip preservation
3. **Hand-Edited Configs**: Files meant for human inspection/maintenance
4. **Nested/Recursive Data**: Structures with variable nesting depth
5. **Binary Data**: Images, encoded payloads, byte arrays

### Rationale

TOON is optimized for **uniform, flat parameter arrays** in tool responses. The CTF and FD solvers use complex numerical representations that:
- Require exact precision preservation
- Have internal dependencies that would break if truncated
- Are not human-edited anyway

## File Extension

`.ton` — short, memorable, avoids collision with `.toon` (audio format)

## MIME Type

`application/x-toon`

## References

- Issue [#2066](https://github.com/anchapin/fluxion/issues/2066): Format specification
- Issue [#2067](https://github.com/anchapin/fluxion/issues/2067): Serializer implementation
- Issue [#2068](https://github.com/anchapin/fluxion/issues/2068): Deserializer implementation
- Issue [#2070](https://github.com/anchapin/fluxion/issues/2070): This documentation
