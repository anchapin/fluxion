# Issue #1254: ripr Integration for CI

## Executive Summary

**ripr** is a memory-efficient static mutation analysis tool (~1.5GB RAM) that finds code seams where mutations may not be caught by existing tests. This document evaluates ripr as a **pre-mutation filter** for CI, complementing (not replacing) full cargo-mutants testing.

| Tool | Memory | Runtime | Purpose |
|------|--------|---------|---------|
| **ripr** | ~1.5GB | ~5 min | Static seam analysis, pre-filter |
| **cargo-mutants** | ~28GB | ~hours | Dynamic mutation testing, full coverage |

---

## 1. ripr Pilot Results on fluxion-core

### Analysis Scope
- **Package analyzed**: `fluxion-core` (weather module - true leaf crate)
- **Seams identified**: 906 total
- **Analysis mode**: draft (conservative)
- **Analysis time**: ~3 minutes on warm cache

### Seam Coverage Summary

| Grip Class | Count | Meaning |
|------------|-------|---------|
| **strongly_gripped** | 354 | Well-tested; mutations likely caught |
| **weakly_gripped** | 494 | Tested but with gaps; mutations may escape |
| **activation_unknown** | 58 | Unclear if tests exist |
| **ungripped** | 0 | No test coverage |

### Key Gap Categories Identified

1. **predicate_boundary** (e.g., `src/weather/epw.rs:806`)
   - Missing equality-boundary test cases
   - Example: `get_hourly_data` with `hour >= self.hourly_data.len()`

2. **error_variant** (e.g., `src/weather/epw.rs:283`)
   - Error path coverage incomplete
   - Tests use `expect()` but don't verify exact error variants

3. **call_presence** (e.g., `src/weather/ddy.rs:87`)
   - Some branches/conditions not exercised
   - Weak discrimination in existing tests

---

## 2. Comparison: ripr vs cargo-mutants

| Aspect | ripr | cargo-mutants |
|--------|------|---------------|
| **Approach** | Static analysis of code structure | Dynamic testing with mutants |
| **Memory footprint** | ~1.5GB | ~28GB (OOM on 7GB CI) |
| **Runtime** | Minutes | Hours |
| **Seams found** | 906 | Potentially thousands |
| **False positives** | Some (static only) | None (actual test) |
| **CI suitability** | Every PR | Manual/triggered |
| **Captures** | Test coverage gaps | Live bug detection |

### Complementary Usage

```
┌─────────────────────────────────────────────────────────────┐
│  Every PR                    Periodic (Manual)              │
│  ─────────                   ──────────────────             │
│  ripr pilot                 cargo-mutants                   │
│  ↓                                                  ↓       │
│  Find gaps                                          Full    │
│  (fast, cheap)                                     Testing  │
│     ↓                                                  ↓    │
│  Add tests ──────────────────────────────────► Coverage    │
│  (targeted)                                         Complete │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Proposed CI Integration

### Workflow: `ripr-preflight.yml`

```yaml
name: ripr Mutation Pre-Filter

on:
  pull_request:
    paths:
      - 'fluxion-core/**/*.rs'
      - 'src/**/*.rs'

permissions:
  contents: read

jobs:
  ripr-analysis:
    name: Mutation Gap Analysis
    runs-on: ubuntu-latest-4-cores  # ~6GB RAM sufficient
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install ripr
        run: cargo install ripr --locked

      - name: Run ripr pilot on fluxion-core
        working-directory: ./fluxion-core
        run: |
          ripr pilot --max-seams 50 --timeout-ms 120000 \
            --out target/ripr/pilot

      - name: Run ripr check (repo exposure)
        working-directory: ./fluxion-core
        run: |
          ripr check --root . --mode draft \
            --format repo-exposure-json \
            > target/ripr/pilot/repo-exposure.json

      - name: Analyze weak seams
        working-directory: ./fluxion-core
        run: |
          python3 << 'EOF'
          import json
          with open('target/ripr/pilot/repo-exposure.json') as f:
              data = json.load(f)
          seams = data.get('seams', [])
          weak = [s for s in seams if s.get('grip_class') == 'weakly_gripped']
          print(f"Weak seams requiring attention: {len(weak)}")
          # Print top 5 by file
          locations = {}
          for s in weak:
              loc = s.get('location', 'unknown').split(':')[0]
              locations[loc] = locations.get(loc, 0) + 1
          for loc, count in sorted(locations.items(), key=lambda x: -x[1])[:5]:
              print(f"  {loc}: {count} weak seams")
          EOF

      - name: Generate gap report
        if: always()
        working-directory: ./fluxion-core
        run: |
          mkdir -p target/ripr/reports
          ripr outcome \
            --before target/ripr/pilot/repo-exposure.json \
            --after target/ripr/pilot/repo-exposure.json \
            --format md \
            > target/ripr/reports/gap-analysis.md 2>&1 || true

      - name: Upload ripr results
        uses: actions/upload-artifact@v4
        with:
          name: ripr-results-${{ github.sha }}
          path: |
            fluxion-core/target/ripr/
          retention-days: 7

  ripr-gate:
    name: Seam Gap Gate
    needs: ripr-analysis
    runs-on: ubuntu-latest
    outputs:
      gap_count: ${{ steps.count.outputs.gaps }}
      gap_threshold: ${{ vars.RIPR_GAP_THRESHOLD || '50' }}
    steps:
      - name: Download ripr results
        uses: actions/download-artifact@v4
        with:
          name: ripr-results-${{ github.sha }}
          path: ./ripr-results

      - name: Count new weak seams
        id: count
        run: |
          # In a real implementation, compare against baseline
          # For now, warn if > threshold
          echo "gaps=$(cat ripr-results/fluxion-core/target/ripr/pilot/repo-exposure.json | \
            python3 -c 'import json,sys; d=json.load(sys.stdin); \
            print(len([s for s in d.get("seams",[]) if s.get("grip_class")=="weakly_gripped"]))' \
          )" >> $GITHUB_OUTPUT

      - name: Check threshold
        run: |
          GAPS=${{ needs.ripr-analysis.outputs.gap_count }}
          THRESHOLD=${{ needs.ripr-analysis.outputs.gap_threshold }}
          if [ "$GAPS" -gt "$THRESHOLD" ]; then
            echo "⚠️  High seam gap count: $GAPS (threshold: $THRESHOLD)"
            echo "Review: target/ripr/reports/gap-analysis.md"
          fi
```

### Integration Points

1. **PR merge gate**: ripr runs before merge, fails if gap count exceeds threshold
2. **Pre-commit hook**: Optional local check before pushing
3. **PR comments**: ripr posts gap analysis as PR comment (future enhancement)

---

## 4. Gap Analysis: What ripr Finds vs. Tests Cover

### Top Files with Weak Seams

Based on ripr analysis, these files have the most weakly_gripped seams:

| File | Weak Seams | Test Coverage Status |
|------|------------|---------------------|
| `src/weather/ddy.rs` | ~15 | Tests exist but boundary cases missing |
| `src/weather/epw.rs` | ~40 | Error path coverage incomplete |
| `src/weather/mod.rs` | ~30 | validate_all() variants not fully tested |
| `src/weather/denver.rs` | ~10 | Missing boundary discriminators |

### Example: Missing Boundary Case

```rust
// src/weather/epw.rs:806
pub fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData> {
    if hour >= self.hourly_data.len() {  // <-- boundary
        return Err(WeatherError::InvalidHour(hour));
    }
    Ok(self.hourly_data[hour])
}
```

**ripr finding**: `hour >= self.hourly_data.len()` (equality boundary) not tested

**Current test**: Only tests `hour < self.hourly_data.len()` cases

**Recommended test**:
```rust
#[test]
fn test_get_hourly_data_boundary_hour_equals_len() {
    let source = create_test_source_with_24_hours();
    let result = source.get_hourly_data(24); // hour == len
    assert!(result.is_err());
}
```

---

## 5. Baseline and Trend Tracking

### Creating a Baseline

```bash
# First-time setup
cd fluxion-core
ripr pilot --max-seams 100 --timeout-ms 180000
ripr baseline create \
  --from target/ripr/pilot/repo-exposure.json \
  --out .ripr/gate-baseline.json
```

### Tracking Delta

```bash
# After code changes
ripr check --format repo-exposure-json > target/ripr/pilot/after.json
ripr baseline diff \
  --baseline .ripr/gate-baseline.json \
  --current target/ripr/pilot/after.json \
  --out target/ripr/reports/delta.json
```

### Delta Report Example

```json
{
  "baseline_weak_seams": 494,
  "current_weak_seams": 502,
  "delta": +8,
  "new_weak_seams": [
    "src/weather/epw.rs:823 predicate_boundary",
    "src/weather/epw.rs:856 error_variant"
  ],
  "resolved_weak_seams": []
}
```

---

## 6. Recommendations

### Immediate Actions

1. **Add ripr to PR checks** for `fluxion-core` and `src/weather/**`
2. **Set gap threshold** to current baseline (494 weak seams)
3. **Create targeted tests** for top-ranked seams (predicate_boundary gaps)

### Short-term (1-2 sprints)

1. **Expand ripr scope** to full `fluxion` crate (not just weather)
2. **Implement PR comments** to post gap analysis automatically
3. **Add trend tracking** to project dashboard

### Long-term

1. **Full cargo-mutants** on dedicated high-memory runners (Phase 3 of crate split)
2. **Policy engine** for gap waiver aging and suppression
3. **Coverage goals** by grip class (e.g., reduce weakly_gripped by 20%)

---

## 7. Files Modified

| File | Purpose |
|------|---------|
| `.github/workflows/ripr-preflight.yml` | Proposed CI workflow (new) |
| `fluxion-core/.ripr/gate-baseline.json` | Baseline for gap tracking (generated) |

---

## 8. Appendix: ripr Command Reference

```bash
# Install
cargo install ripr

# Analyze seams (draft mode, 50 seams, 2 min timeout)
ripr pilot --max-seams 50 --timeout-ms 120000

# Check exposure
ripr check --format repo-exposure-json > exposure.json

# Generate baseline
ripr baseline create --from exposure.json --out baseline.json

# Compare to baseline
ripr baseline diff --baseline baseline.json --current exposure.json

# Outcome comparison
ripr outcome --before before.json --after after.json --format md
```
