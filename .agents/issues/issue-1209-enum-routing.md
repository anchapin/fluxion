## Issue Description

In `thermal_model_core.rs`, routing between 5R1C and 9R4C models is done via string pattern matching:

```rust
if (spec.case_id.starts_with("9") && spec.case_id != "960")
    || (spec.case_id.starts_with("6") && !["600FF", "650FF"].contains(&spec.case_id.as_str()))
{
    model.enable_9r4c_model();
}
```

This is brittle and opaque — routing is gated by string patterns rather than type-level selection.

## Fix

Use `ThermalModelType` enum throughout:

```rust
pub enum ThermalModelType {
    LowMass5R1C,
    HighMass9R4C,
}

impl From<&SimulationSpec> for ThermalModelType {
    fn from(spec: &SimulationSpec) -> Self {
        match spec.case_id.as_str() {
            "600" | "600FF" | "650" | "650FF" => ThermalModelType::LowMass5R1C,
            _ if spec.case_id.starts_with("9") && spec.case_id != "960" => ThermalModelType::HighMass9R4C,
            _ => ThermalModelType::LowMass5R1C,
        }
    }
}
```

Then use `self.thermal_model_type` for all routing decisions.

## Files Affected

- `src/sim/thermal_model_core.rs`
- `src/sim/thermal_model.rs` (add `ThermalModelType`)

## Acceptance Criteria

- [ ] `case_id` string matching replaced with `ThermalModelType` enum
- [ ] Routing logic is type-driven, not string-driven
- [ ] Test for Case 600 vs 900 routing verifies enum value, not string pattern