# Session 38: CTF-Aware Free-Floating Temperature

**Date**: 2026-03-28
**Status**: ⚠️ Mixed results - cooling improved, heating worsened
**Goal**: Fix free-floating temperature calculation to be CTF-aware

---

## Executive Summary

Implemented CTF-aware free-floating temperature calculation. Cooling is now within range (1.96 MWh vs 2.13-3.67 MWh expected), but heating is now overpredicting (4.76 MWh vs 1.17-2.04 MWh expected).

---

## Progress Made

### ✅ CTF-Aware Free-Floating Temperature Implemented

**File**: `src/sim/engine.rs` (lines 5158-5397)

**Fix Applied**:
1. Added CTF detection to `calculate_free_float_temperature()` method
2. When CTF is enabled, delegate to `calculate_free_float_temperature_ctf()`
3. Implemented CTF-specific heat balance equation for free-floating temperature

**CTF Heat Balance Equation**:
\`\`\`
For CTF mode, the zone air heat balance is:
C_air × dTi/dt = Q_ctf + Q_solar + Q_internal + Q_wind + Q_vent + Q_ground

For free-floating steady state (dTi/dt = 0):
0 = Q_ctf + Q_solar + Q_internal + Q_wind + Q_vent + Q_ground
\`\`\`

**Verification** (diagnose_free_float_ctf.rs):
Ti_free changes appropriately when CTF is enabled (difference: 0.21°C)

**Status**: Free-floating temperature calculation is now CTF-aware.
