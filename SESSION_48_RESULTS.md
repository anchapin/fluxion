# Session 48: CTF Solver Results - Case 900

**Date**: 2026-03-27
**Status**: ✅ CTF ENABLED - Results Available
**Test Case**: ASHRAE 140 Case 900 (High Mass Baseline)

## Experimental Setup

### Configuration
- **Solver**: CTF (Conduction Transfer Function)
- **Timestep**: 3600s (1 hour)
- **History Size**: 50 timesteps
- **Warmup Period**: 7 days
- **Wall Construction**:
  - Concrete block (0.100m, k=0.51 W/m·K)
  - Foam insulation (0.0615m, k=0.04 W/m·K)
  - Wood siding (0.009m, k=0.16 W/m·K)

### Integration Method
- **Boundary Condition**: Mass temperature (T_mass)
- **Flux Integration**: Mass energy balance (phi_m)
- **Net Correction**: Q_ctf - Q_5r1c added to phi_m

## Results: Case 900

### Annual Energy Consumption

| Metric | 5R1C Result | CTF Result | Reference | 5R1C Error | CTF Error |
|--------|-------------|------------|-----------|------------|-----------|
| Annual Heating | 1.71 MWh | 1.73 MWh | 1.17-2.04 MWh | Within range | Within range ✅ |
| Annual Cooling | 2.28 MWh | 2.53 MWh | 2.13-3.67 MWh | Within range | Within range ✅ |

**Analysis**: Annual energies still pass validation. CTF shows:
- Heating: +1.2% increase (minimal change)
- Cooling: +11% increase (moderate change)

### Peak Loads

| Metric | 5R1C Result | CTF Result | Reference | 5R1C Error | CTF Error |
|--------|-------------|------------|-----------|------------|-----------|
| Peak Heating | 1.26 kW | 3.23 kW | 1.80-2.40 kW | -30% (below) | +35% (above) ❌ |
| Peak Cooling | 2.35 kW | 2.89 kW | 1.60-2.10 kW | +12% (above) | +38% (above) ❌ |

**Analysis**: Peak loads got significantly worse:
- Peak heating: 156% increase (now 35% ABOVE reference)
- Peak cooling: 23% increase (now 38% ABOVE reference)

### CTF Flux Analysis

**Debug Output (First Timestep)**:
```
🔍 SESSION 48: CTF solver step 0:
  T_mass = 20.00°C
  T_ext = -9.95°C (sol-air temperature)
  Q_CTF = -5.80 W/m²

🔧 SESSION 48: CTF flux to mass:
  Q_CTF = -278.59 W
  Q_5R1C = -3270.51 W
  Q_net = +2991.92 W (added to phi_m)
```

**Key Observations**:
1. **Magnitude Mismatch**: CTF flux is 12x smaller than 5R1C
2. **Direction**: Both negative (heat leaving zone) ✅
3. **Net Correction**: Large positive value added to mass balance

### Comparison with Reference Tools

**EnergyPlus (Reference)**:
- Annual Heating: 1.17-2.04 MWh
- Annual Cooling: 2.13-3.67 MWh
- Peak Heating: 1.80-2.40 kW
- Peak Cooling: 1.60-2.10 kW

**Fluxion 5R1C**:
- Annual Heating: 1.71 MWh (MID-RANGE) ✅
- Annual Cooling: 2.28 MWh (MID-RANGE) ✅
- Peak Heating: 1.26 kW (BELOW range) ❌
- Peak Cooling: 2.35 kW (ABOVE range) ❌

**Fluxion CTF**:
- Annual Heating: 1.73 MWh (MID-RANGE) ✅
- Annual Cooling: 2.53 MWh (MID-RANGE) ✅
- Peak Heating: 3.23 kW (ABOVE range) ❌
- Peak Cooling: 2.89 kW (ABOVE range) ❌

## Diagnosis

### What Went Wrong

**Issue 1: Flux Magnitude Mismatch**
- CTF flux (-278 W) is 12x smaller than 5R1C (-3270 W)
- Expected: Similar magnitudes for same temperature difference
- Possible causes:
  - Coefficient calculation error
  - Timestep too large (3600s)
  - Boundary condition wrong
  - Wall area mismatch

**Issue 2: Peak Load Degradation**
- Peak heating increased from 30% below to 35% above reference
- Peak cooling increased from 12% above to 38% above reference
- Suggests CTF is over-responding to transients

**Issue 3: Integration Point**
- Adding net correction (CTF - 5R1C) to mass balance
- May be causing double-counting or wrong sign
- Need to verify energy conservation

### What Went Right

**✅ CTF Infrastructure**:
- All components working correctly
- Solver stable and running
- No crashes or numerical explosions

**✅ Annual Energies**:
- Still within reference range
- Shows CTF is not catastrophically wrong

**✅ Integration Framework**:
- Enablement working
- Debug output comprehensive
- Easy to modify and test

## Recommendations

### Immediate Actions

1. **Verify CTF Coefficients**:
   ```bash
   cargo test test_case_900_coefficients -- --nocapture
   ```
   Check coefficient values against ASHRAE 140 reference

2. **Check Energy Conservation**:
   - Add energy balance check: Q_in = Q_out + dE_storage
   - Verify <1% imbalance over simulation

3. **Test Boundary Conditions**:
   - Try zone air temperature instead of mass temperature
   - Try surface temperature (between mass and air)
   - Compare flux directions

4. **Reduce Timestep**:
   - Test with 600s (10 minutes) instead of 3600s
   - Check if timestep stability is issue

### Alternative Approaches

**Option A: Fix CTF Integration**
- Debug flux magnitude mismatch
- Verify sign convention
- Test different boundary conditions
- **Timeline**: 2-3 days

**Option B: Use FD Solver**
- Finite Difference solver more intuitive
- Direct spatial discretization
- Easier to debug
- **Timeline**: 1-2 days to enable

**Option C: Accept 5R1C Limitations**
- Document peak load limitation
- Use CTF for annual energy only
- Focus on other improvements
- **Timeline**: 0 days (proceed to Session 49)

## Performance Impact

### Computational Cost
- **5R1C Baseline**: ~100ms for 8760 timesteps
- **With CTF**: ~120ms for 8760 timesteps (estimated)
- **Overhead**: ~20% increase

### Memory Usage
- **CTF Coefficients**: ~50 doubles × 4 coefficients = 1.6 KB
- **CTF Solvers**: 50 timesteps × 4 variables × 8 bytes = 1.6 KB per zone
- **Total Overhead**: ~3.2 KB (negligible)

## Conclusion

CTF solver has been successfully enabled for Case 900 and is producing different results than the 5R1C baseline. However, peak loads have degraded significantly, indicating integration issues that need to be resolved before proceeding to Session 49.

**Status**: ⚠️ CONDITIONAL SUCCESS
- ✅ CTF enabled and active
- ✅ Annual energies passing
- ❌ Peak loads failing
- ❌ Integration needs debugging

**Recommendation**: Debug flux integration (2-3 days) before enabling CTF for all 900-series cases in Session 49.

---

**Results Generated**: 2026-03-27
**Session**: 48 (CTF Solver Audit and Enablement)
**Next**: Session 49 (Enable CTF for all 900-series) - PENDING DEBUGGING
