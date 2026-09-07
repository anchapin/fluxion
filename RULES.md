# Rules

Coding rules, hard constraints, and must-always/must-never boundaries for all contributors.
Read this before modifying any physics code or validation logic.
Covers: numerical reasoning via code, energy balance, ASHRAE 140 compliance, code quality.
Companion to CONTRIBUTING.md (workflow) and ARCHITECTURE.md (module contracts).
Status: Active — all rules are enforced by CI and code review.
Action: Run `cargo clippy --workspace --exclude fluxion-tauri --lib -- -D warnings` and ASHRAE validation before committing physics changes. (The `--exclude fluxion-tauri` is required because `fluxion-tauri`'s proc-macro build needs `npm run build` in `fluxion-tauri/frontend/` to materialise `../frontend/dist`; see Issue #3126 / `rust-tests.yml:428`.)

## Hard Constraints

### Must-Always

0. **Numerical Reasoning via Code**
   - All calculations, formula verification, and numerical reasoning MUST be done by writing and executing Python code
   - Never perform mental arithmetic or "reason through" math in your head
   - Use `ctx_execute(language: "python", code: "...")` to run computations — only the printed result enters context
   - LLMs are probabilistic text predictors, not calculators — mental math produces confidently wrong answers
   - Reference: [LLMs & Math: Problem Solved by Coding](https://gregrobison.medium.com/llms-math-problem-solved-by-coding-a5a5b5c4453a)

1. **Energy Balance Conservation**
   - All thermal calculations must maintain energy balance
   - Total heat transfer (conduction + convection + radiation + solar + HVAC) must sum to zero for any zone
   - Before committing any physics change, verify energy balance doesn't break

2. **ASHRAE 140 Compliance**
   - Changes must target validation within ASHRAE 140 tolerance ranges
   - Report pass/fail rates with specific case numbers and reference values
   - Never ignore failing cases — document and track them

3. **Code Quality**
   - All Rust code must pass `cargo fmt` and `cargo clippy`
   - All Python code must pass `ruff check` and type hints
   - Include unit tests for new functionality

4. **Version Control**
   - Commit changes with descriptive messages
   - Test locally before pushing to remote branches
   - Create PRs against the `develop` branch, not `main`

5. **Documentation**
   - Document any workaround or calibration constant with rationale
   - Keep validation reports (ASHRAE140_RESULTS.md) current

### Must-Never

1. **Skip Validation**
   - Never commit changes without running the validation test suite
   - Never assume a fix works without checking all related cases

2. **Break Physics**
   - Never modify thermal parameters without understanding the physics impact
   - Never hardcode results to match reference values — fix the root cause

3. **Lose History**
   - Never modify historical validation results

4. **Skip Reviews**
   - Never push directly to main or develop without PR review
   - Never ignore code review feedback

## Safety Boundaries

### Boundary: Code Changes

- **Before modifying core physics** (thermal solver, HVAC control): Create backup branch
- **For large refactors**: Split into reviewable commits with working intermediate states
- **When adding calibration constants**: Document why, expected range, and how to validate

### Boundary: Validation

- **Minimum validation**: Run affected cases before and after any change
- **Full validation**: Run entire test suite for any physics-related change
- **Performance check**: Verify throughput remains >10,000 configs/sec after optimization

### Boundary: Debugging

- **Add diagnostic output** to verify assumptions before implementing fix
- **Compare against EnergyPlus** when model behavior is unclear
- **Use existing debug tools**: diagnostic CSV generation, logging levels

## Operational Guidelines

### Session Workflow

1. Start session by checking current validation status
2. State the specific problem being addressed
3. Implement fix with incremental testing
4. Document findings and results before concluding

### Communication Protocol

- When proposing a fix: Show expected impact on validation metrics
- When encountering unknown: State what needs investigation
- When blocked: Request specific input or clarification needed

### Performance Constraints

- Annual simulation target: <100ms per simulation
- Batch evaluation: 10,000+ configurations per second
- Memory budget: Profile before and after major changes

### Error Handling

- If test fails: Analyze before modifying code
- If reference unclear: Check ASHRAE 140 specification
- If performance degrades: Profile and identify bottleneck first
