#!/usr/bin/env python3
"""Update Phase 8 plan with EnergyPlus comparison."""


def main():
    # Read current plan
    with open(".planning/phase_8_solar_refinement_plan.md", "r") as f:
        content = f.read()

    # Find insertion point after line 157
    insert_pos = content.find(
        "   156→**This looks correct per ASHRAE 140 spec**\n   157→\n"
    )

    new_section = """   156→**This looks correct per ASHRAE 140 spec**
   157→

## EnergyPlus Reference Data Comparison

Successfully extracted EnergyPlus hourly data from existing SQL files in benchmarks/outputs/bestest_gsr/.

**Critical Finding:**
Fluxion is systematically under-predicting energy consumption by 60-90% compared to EnergyPlus reference values. This is a fundamental issue that extends beyond case-specific sensitivity tuning.

**Comparison Summary (EnergyPlus kWh):**

| Case | Fluxion Heating | EP Heating | Error | Fluxion Cooling | EP Cooling | Error |
|-------|----------------|------------|--------|----------------|------------|--------|
| 600 | 6.10 | 15.57 | 61% under | 6.55 | 21.76 | 70% under |
| 610 | 6.19 | 15.75 | 61% under | 4.91 | 15.65 | 69% under |
| 620 | 5.25 | 16.14 | 67% under | 2.80 | 14.67 | 81% under |
| 630 | 5.53 | 17.21 | 68% under | 1.56 | 10.25 | 85% under |
| 640 | 4.28 | 9.61 | 55% under | 6.47 | 20.81 | 68% under |
| 650 | 0.00 | 0.00 | - | 5.68 | 17.44 | 67% under |
| 900 | 1.74 | 5.98 | 71% under | 3.77 | 8.99 | 58% under |
| 910 | 1.91 | 7.03 | 73% under | 2.75 | 5.00 | 45% under |
| 920 | 1.38 | 11.99 | 88% under | 1.59 | 9.87 | 84% under |
| 930 | 1.85 | 14.36 | 87% under | 0.86 | 6.94 | 87% under |
| 940 | 1.44 | 3.84 | 62% under | 3.77 | 8.76 | 57% under |
| 950 | 0.00 | 25.05 | 100% under | 1.18 | 0.92 | 28% under |

**Analysis:**
1. **Systematic Under-prediction**: All low-mass cases (600 series) show 60-70% lower heating energy
2. **High-mass Pattern**: Similar but slightly less severe under-prediction (60-90%)
3. **Cooling Energy**: Most cases show 60-85% lower cooling energy
4. **Root Cause Hypothesis**: This is likely a fundamental issue with:
   - HVAC demand calculation magnitude (still too low despite sensitivity fix)
   - Heat transfer coefficient magnitudes in the thermal network
   - Missing heat transfer paths (e.g., direct ground coupling)
   - Solar gain timing/distribution mismatch

---

### 3. Night Ventilation Analysis"""

    # Insert new section
    updated_content = content[:insert_pos] + new_section + content[insert_pos:]

    # Write back
    f.seek(0)
    f.write(updated_content)

    print("Updated Phase 8 plan successfully")


if __name__ == "__main__":
    main()
