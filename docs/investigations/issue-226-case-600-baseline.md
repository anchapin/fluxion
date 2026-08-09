# Issue #226: ASHRAE 140 Case 600 Baseline Validation Failure - Root Cause Analysis

## Problem
Case 600 test was failing with incorrect cooling energy (0.00 MWh instead of 6.14-8.45 MWh).

## Root Cause
The cooling energy sign convention and HVAC capacity limits were not properly implemented.

## Solution
Fixed by increasing HVAC capacity limits from 5 kW to 100 kW and correcting the cooling energy sign handling in the thermal solver.

## Technical Details
- Location: `src/sim/engine.rs` and related thermal solver
- Changes: HVAC capacity calculation and energy sign conventions
- Testing: Full ASHRAE 140 validation suite

## Status
Fixed in main branch (commit fb7b418). This fix unblocks all other ASHRAE 140 validation work.

## Impact
- Case 600 now correctly reports both heating and cooling energy
- Thermal balance calculations are correct
- Foundation for all other case validations
