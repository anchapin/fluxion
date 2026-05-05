---
phase: E-sustained-validation
plan: E-01
type: execute
wave: 1
depends_on: ["D-01"]
files_modified:
  - .github/workflows/ashrae_validation.yml
  - tests/ashrae_140_blind_validation.rs
autonomous: true
requirements:
  - SUSTAIN-01
  - SUSTAIN-02

must_haves:
  truths:
    - "CI prevents merges when blind validation pass rate < 80%"
    - "Blind validation runs as part of every PR check"
  artifacts:
    - path: ".github/workflows/ashrae_validation.yml"
      provides: "CI workflow for blind validation gate"
---

<objective>
Establish CI gate that prevents blind validation pass rate from dropping below 80%. Also set up annual re-validation schedule against latest ASHRAE reference data.
</objective>

<context>
D-01 achieved ≥80% pass rate. Phase E ensures this is maintained as code evolves.

CI gate requirements:
- Runs on every PR
- Fails if blind validation pass rate < 80%
- Provides detailed failure report for debugging
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create CI validation workflow</name>
 files>.github/workflows/ashrae_validation.yml</files>
  <action>
Create .github/workflows/ashrae_validation.yml:

```yaml
name: ASHRAE 140 Blind Validation

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  blind-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-action@stable
        with:
          targets: x86_64-unknown-linux-gnu

      - name: Build
        run: cargo build --release

      - name: Run Blind Validation
        run: cargo test --test ashrae_140_blind_validation -- --nocapture
        id: validation

      - name: Check Pass Rate
        run: |
          # Parse test output for pass rate
          # Fail if pass rate < 80%
          # Upload detailed report as artifact
```

The workflow must:
1. Run on every PR to main
2. Execute full blind validation suite
3. Parse results and compute pass rate
4. FAIL the PR if pass rate < 80%
5. Upload detailed results as CI artifact
6. Comment on PR with pass/fail summary
  </action>
  <verify>
    cat .github/workflows/ashrae_validation.yml
    # Verify: cargo test --test ashrae_140_blind_validation passes locally
  </verify>
  <done>CI workflow created and verified to run</done>
</task>

<task type="auto">
  <name>Task 2: Add to PR merge blocking checks</name>
  <files>.github/workflows/</files>
  <action>
If there's an existing CI configuration (e.g., .github/workflows/ci.yml), add the ashrae_validation job to required checks for PR merging. Ensure that:

1. ashrae_validation must pass before merge is allowed
2. Failure provides actionable feedback (which cases failed, pass rate)
3. CI artifacts are retained for debugging

If no existing CI config exists, create .github/workflows/ci.yml that includes all required checks (including ashrae_validation).
  </action>
  <verify>
    # Verify: Check that ashrae_validation is a required check
    # This may require updating branch protection rules
    echo "CI configuration updated - verify in GitHub Settings"
  </verify>
  <done>Ashrae validation is required for PR merge</done>
</task>

<task type="auto">
  <name>Task 3: Document annual re-validation process</name>
  <files>docs/ASHRAE_REVALIDATION_SCHEDULE.md</files>
  <action>
Create docs/ASHRAE_REVALIDATION_SCHEDULE.md:

```markdown
# Annual ASHRAE 140 Re-validation

## Schedule
Run full blind validation suite against latest ASHRAE 140 reference data every year (January).

## Process
1. Source latest ASHRAE 140 reference data from official channels
2. Update data/ashrae_140_true_reference/ with new data
3. Run full blind validation suite
4. Document any changes in reference values
5. If pass rate drops, investigate if it's reference data change or model regression

## Reference Data Sources
- EnergyPlus: Download from NREL
- ESP-r: Download from University of Strathclyde
- TRNSYS: Download from Thermal Energy Systems Specialists

## Sign-off
Milestone v1.X validation completed: [date] [sign-off]
```

Also create scripts/annual_ashrae_revalidation.sh that automates the process.
  </action>
  <verify>
    cat docs/ASHRAE_REVALIDATION_SCHEDULE.md
    ls scripts/annual_ashrae_revalidation.sh
  </verify>
  <done>Annual re-validation process documented and automated</done>
</task>

</tasks>

<verification>
Run: cargo test --test ashrae_140_blind_validation

Verify:
- All tests pass locally (before CI)
- CI workflow file is valid YAML
</verification>

<success_criteria>
- CI workflow exists and runs on PR
- PR merge blocked if validation fails
- Annual re-validation documented and script exists
</success_criteria>

<output>
After completion, create `.planning/phases/E-sustained-validation/E-01-SUMMARY.md`
</output>