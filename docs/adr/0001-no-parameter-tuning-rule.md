# ADR-0001: No-Parameter-Tuning Rule

- **Status:** Accepted
- **Date:** 2026-08-14 (record created)
- **Deciders:** Fluxion maintainers
- **Supersedes:** None
- **Depends on:** None
- **Source:** Originally codified in `RULES.md`; promoted to ADR for cross-reference clarity.

---

## Executive Summary

We adopt a strict **no-parameter-tuning rule** for ASHRAE 140 validation and all
related gate tests. When a test fails, we fix the underlying physics (or the
test setup), never the test outcome. This ADR captures the rationale and
binding scope of the rule so that future contributors and audit reviewers can
trace the principle to a single canonical document.

## Context

Fluxion's primary validation gate is the ASHRAE 140 suite (`docs/ASHRAE140_RESULTS.md`,
`release_gates.yaml → validation.min_pass_rate: 0.60`). The suite's 18 cases
include baseline, free-floating, high-mass, and special configurations with
specific energy and peak-load tolerances.

A natural temptation when a test fails is to **adjust internal coefficients,
timestep handling, or solver tolerances** to bring the result within tolerance.
This practice — sometimes called "test-result tuning" — produces code that
**passes the test** but is **less physically accurate** in the failure mode
that the test was designed to catch. It accumulates over time: each tuned
parameter subtly distorts the model, and a project that has been tuned for
years bears little resemblance to the physics it claims to implement.

## Decision

We adopt a hard rule:

> When a validation test fails, we fix the **physics** (governing equations,
> boundary conditions, network topology, time discretization, or material
> properties) — never the test outcome.

Specifically:

1. **No parameter is tuned to make a test pass.** If a coefficient appears to
   need adjustment to satisfy a tolerance, the underlying model or boundary
   condition is the fix target, not the coefficient.
2. **No tolerance is widened to make a test pass.** A test that fails
   repeatedly indicates either a model deficiency (fix the model) or a
   tolerance that was too tight given ASHRAE 140's reference ranges (in which
   case the test must cite the new evidence and be re-approved).
3. **No baseline file is raised to hide a regression.** Strict-energy-gate
   baselines (`tests/reference_data/zone_balance/strict_energy_gate_baseline.json`)
   and drift baselines (`validation_baseline.json` once #2856 is resolved) are
   ratcheted down only, never relaxed.
4. **Structural failures are documented, not patched.** Cases 600 and 900 are
   currently excluded from the strict ±15% annual-energy gate (per
   `release_gates.yaml → validation.individual.known_failures`) precisely
   because they are **structural** failures — patching them by parameter
   adjustment would be exactly the practice this rule prohibits.

The rule applies symmetrically:

- Test outcomes cannot be tuned up to pass a gate.
- Test outcomes cannot be tuned down to hide a regression.
- The reference data cannot be edited to favor a passing answer.

## Consequences

### Positive

- The codebase retains a 1-to-1 mapping between implemented physics and
  ASHRAE 140 §5 / §6 model descriptions. Audit reviewers can trace a
  coefficient back to a published reference.
- Regressions are loud, not silent: when a coefficient's effect becomes
  unmodeled, the failure surfaces as a failed case, not as a passing case
  with degraded accuracy.
- The drift gate (Issue #1333) can meaningfully compare today's metrics
  against a historical baseline because that baseline is honest.
- Documentation, citations, and ASHRAE 140 references remain authoritative.
  Contributors do not need to remember which constant is "really" the
  reference value versus a tuned value.

### Negative

- Failures can be slower to close. Some failures (Cases 600/900, certain
  free-floating cases, some multi-zone coupling) require deep physics work
  before the test passes — work that cannot be substituted by a quick
  coefficient adjustment.
- Temptation persists. Reviewers must actively guard against tuning. The
  `RULES.md` "must-never hardcode results" clause is the primary enforcement;
  this ADR documents the principle but does not by itself prevent violation.

### Neutral

- The rule is enforced socially and by code review, not by automated tooling.
  CI gates (Issue #1333 strict-energy gate, drift gate, scorecard drift gate)
  are *detection* mechanisms — they surface regressions. They do not by
  themselves prevent parameter tuning.

## Alternatives Considered

- **Permit bounded parameter tuning within tolerance bands:** rejected. The
  tolerance bands in ASHRAE 140 are themselves the validation metric; widening
  the band for any specific case conflates the validation objective with the
  test outcome.
- **Allow tuning for "structural" failures only:** rejected. The structural
  failures are documented in `release_gates.yaml → validation.individual.known_failures`
  precisely so they are tracked separately from the 60% pass-rate gate. A
  tuning exception for them defeats the documentation.
- **No rule (status quo):** rejected. The 5R1C / 9R4C thermal-mass decisions
  in ADR-0002 and the documented 5R1C limitations in ADR-0003 both rely on
  this rule to remain meaningful. Without the rule, the model accuracy
  claims in those ADRs are not auditable.

## References

- `RULES.md` §"Mathematical Reasoning" and "Hard Constraints"
- `release_gates.yaml → validation.min_pass_rate: 0.60`
- `release_gates.yaml → validation.individual.known_failures: ["600","900"]`
- `docs/KNOWN_ISSUES.md` — documented structural failures
- `docs/ASHRAE140_RESULTS.md` — current pass rates per case
- ADR-0002 — 9R4C promotion to default (depends on this rule)
- ADR-0003 — 5R1C high-mass limitations (depends on this rule)