# HVAC BESTEST validation scaffold

This directory is the integration-test home for Fluxion's HVAC BESTEST work. Issue
#1754 provides module and CI wiring only: no case inputs, reference bounds,
acceptance tolerances, or equipment calculations are implemented here.

## Taxonomy

HVAC BESTEST distinguishes validation methods by the source of truth:

- **Analytical verification** compares a program with a known analytical or
  generally accepted numerical solution under tightly controlled boundary
  conditions. For HVAC BESTEST, these are not free-floating envelope tests: the
  HVAC equipment or air-distribution system is active and the solution checks
  mass and energy balances, coil loads, fan heat, and related state points.
- **Comparative testing** compares a program with itself or with qualified,
  independently implemented reference programs when no analytical truth standard
  is available. Comparative ranges are evidence, not physical constants, and must
  retain source/version provenance.

Envelope-only and free-floating cases remain in the ASHRAE Standard 140 harness;
they validate the building loads presented to HVAC models but are not RP-865 HVAC
case identifiers.

## Case families

| Family | Method | Scope | Planned module |
|---|---|---|---|
| `AE101`–`AE445` | Analytical / quasi-analytical | RP-865-derived airside HVAC equipment and distribution-system configurations | `analytical.rs` |
| `E100`–`E200` | Analytical | Companion HVAC BESTEST unitary space-cooling equipment cases; dry-coil cases `E100`–`E140` and wet-coil cases `E150`–`E200` | `analytical.rs` |
| Future dynamic cases | Comparative | Cases whose dynamic boundary conditions prevent a closed-form truth solution | `comparative.rs` |

ASHRAE Standard 140 building-fabric identifiers such as Cases 600 and 900 are
intentionally excluded from this taxonomy; they remain under the existing envelope
validation suite.

## Module responsibilities

- `analytical.rs`: analytical and quasi-analytical case runners and assertions.
- `comparative.rs`: cross-program comparative cases and qualified comparison bands.
- `reference_data.rs`: typed loaders with publication, version, units, and checksum
  provenance for RP-865/HVAC BESTEST reference data.
- `mod.rs`: integration-test target root only.

Follow-on issues must add tests before implementation, preserve moist-air and dry-air
mass balances, close the sensible/latent energy balance, and avoid invented bounds
or placeholder correlations.

## Sources

- Neymark et al., *Airside HVAC BESTEST: Adaptation of ASHRAE RP 865 Airside HVAC
  Equipment Modeling Test Cases for ASHRAE Standard 140, Volume 1: Cases
  AE101–AE445*, NREL/TP-5500-66000 (2016), DOI 10.2172/1244668.
- IEA SHC Task 22, *HVAC BESTEST, Volume 1: Cases E100–E200*, Section 1.3
  (fundamental unitary space-cooling equipment tests and case summary tables).

Run the scaffold with:

```bash
cargo test --test hvac_bestest
```
