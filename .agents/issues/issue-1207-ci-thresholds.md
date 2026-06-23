## Issue Description

`release_gates.yaml` has validation thresholds that are far too permissive:

```yaml
validation:
  min_pass_rate: 4.0    # Only 4% pass rate required!
  max_mae: 300.0        # 300% MAE allowed!
  extreme_deviation_limit: 15  # 15 cases can exceed 100% deviation
```

A release could have:
- 96% failure rate → PASS
- 300% mean absolute error → PASS
- 15 cases with >100% deviation → still within limits

## Impact

- Releases can ship with fundamentally broken physics
- CI gate provides false confidence
- No incentive to fix failing tests (they don't block releases)

## Recommended Thresholds

```yaml
validation:
  min_pass_rate: 60.0    # ASHRAE 140 minimum
  max_mae: 50.0          # Reasonable for physics validation
  extreme_deviation_limit: 2  # Max 2 cases >50% deviation
```

Or remove the validation gates entirely if they're not actionable.

## Files Affected

- `release_gates.yaml`

## Acceptance Criteria

- [ ] Release gates have meaningful thresholds that catch physics errors
- [ ] Thresholds are documented with rationale
- [ ] Failing gate blocks release