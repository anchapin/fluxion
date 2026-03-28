# Physics-Based Refactoring - Session 5 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 4 Recap
- Created `src/sim/hvac/ideal_loads.rs` with IdealLoads system (Session 3)
- Added `IdealLoadsSystem` to ThermalModel for electrical energy tracking (Session 3)
- Added `annual_electrical_mwh` field to CaseResults (Session 4)
- Removed thermal-to-electrical conversion from validator (Session 4)
- Model now tracks electrical consumption via IdealLoadsSystem with COP=3.0/efficiency=0.9

---

## Session 5 Task: Research CTF Implementation Requirements

### Objective
Research Conduction Transfer Functions (CTF) implementation requirements to prepare for replacing RC networks with proper transient heat conduction solving.

### Background
The current 5R1C/6R2C thermal network models use lumped capacitance which doesn't properly model transient heat conduction through building envelopes. CTF is the industry-standard method (used by EnergyPlus) for calculating time-dependent heat transfer through walls, roofs, and floors.

### Steps

#### Part A: Study EnergyPlus CTF Methodology

1. **Understand the CTF Concept**:
   - CTF coefficients relate current heat flux to past temperatures
   - Formula: `Q(t) = Σ(CTF_coefficients[i] * T(t-i))` for i = 0 to τ
   - τ is the "time constant" or number of terms needed for convergence

2. **Research CTF Calculation**:
   - How CTF coefficients are derived from wall construction properties
   - Layer properties needed: thickness, conductivity, density, specific heat
   - How to handle multi-layer constructions
   - Temperature sweep method vs analytical solutions

3. **Document Key Parameters**:
   - Convergence criteria (number of terms)
   - Time step considerations (typically hourly for ASHRAE 140)
   - Interior/exterior boundary conditions

#### Part B: Review Current Implementation

1. **Examine Existing CTF Code**:
```bash
ls -la src/sim/ctf*.rs
grep -r "CTF\|ConductionTransferFunction" src/ --include="*.rs" | head -30
```
**Note**: Always use `ls -la` to see all files including hidden ones with details.

2. **Identify Gaps**:
   - Does current implementation calculate coefficients from material properties?
   - Does it handle multi-layer walls correctly?
   - Is it integrated with the thermal model solver?

#### Part C: Document Requirements for Session 6+ Implementation

Create `docs/ctf_requirements.md` with:
1. **Technical Specification**: How CTF should be implemented
2. **Material Properties Needed**: Conductivity, density, specific heat, thickness
3. **Algorithm**: Step-by-step coefficient calculation
4. **Integration Points**: Where to connect with thermal model
5. **Validation Strategy**: How to verify against ASHRAE 140

### Expected Architecture After CTF Implementation

```
Thermal Model
├── CTF Solver (replaces RC network conduction)
│   ├── Calculate coefficients from wall layers
│   ├── Apply time-series temperature history
│   └── Compute heat flux based on CTF
├── Zone Heat Balance (unchanged)
└── HVAC (IdealLoads - from Session 3)
```

### Deliverable
- `docs/ctf_requirements.md` - Technical specification for CTF implementation
- Summary of current implementation gaps
- Recommendations for Session 6 implementation approach

### Success Criteria
- [ ] EnergyPlus CTF methodology documented
- [ ] Current implementation gaps identified
- [ ] Requirements for full implementation specified
- [ ] Integration points identified for thermal model

### Important Notes
- Focus on the "universal" CTF approach - should work for all building types (low-mass and high-mass)
- Consider efficiency - CTF should be faster than RC networks once implemented
- Document any ASHRAE 140 specific requirements