# Weather File Provenance

**Status:** 🔴 BLOCKED by #732 (synthetic weather data issue)  
**Last updated:** 2026-05-12

---

## ASHRAE 140 Requirement

ASHRAE 140-2023 Annex C specifies the weather file to be used for all test cases:
- Location: Denver, Colorado, USA (cold climate)
- File format: [document format — TMY2, EPW, or ASHRAE 140-specific binary]
- Source: Provided with the standard; exact file distribution via ASHRAE

A compliance submission must document:
1. The exact file used (filename, source, version)
2. SHA256 hash of the file (for reproducibility)
3. Any preprocessing or modifications applied
4. Verification that the data matches Annex C specification

---

## Current Status

⚠️ **Issue #732** documents that fluxion currently uses synthetic/generated weather data rather than the normative ASHRAE 140 Annex C weather file. This is a compliance blocker.

Until #732 is resolved, this document cannot be completed.

---

## Template (fill in when #732 is resolved)

| Field | Value |
|-------|-------|
| Filename | [STUB] |
| Source | ASHRAE 140-2023 Annex C |
| Source URL | [STUB] |
| SHA256 hash | [STUB — compute with `sha256sum <filename>`] |
| Format | [STUB — EPW / TMY2 / other] |
| Preprocessing | [STUB — none / describe any modifications] |
| Verified against Annex C | [STUB — Yes/No + method] |

---

## Verification Procedure (once file is available)

```bash
# 1. Download from ASHRAE (or extract from standard distribution)
# 2. Compute hash
sha256sum weather/ASHRAE140_Denver.epw

# 3. Spot-check against Annex C Table values:
#    - January 1, 01:00 dry-bulb temperature
#    - July peak dry-bulb temperature
#    - Annual global horizontal radiation sum
```

---

## Related Issues

- [#732](https://github.com/anchapin/fluxion/issues/732) — Synthetic weather data (primary blocker)
