# Known Issues Taxonomy and Schema

## Issue ID Format

Issues are categorized by domain with sequential numbering:

- `BASE-01` through `BASE-04`: Foundation (conductance, HVAC, weather)
- `SOLAR-01` through `SOLAR-04`: Solar radiation and shading
- `FREE-01` through `FREE-03`: Free-floating temperature validation
- `TEMP-01` through `TEMP-03`: Temperature swing and thermal lag
- `MULTI-01`: Multi-zone inter-zone transfer
- `GROUND-01`: Ground boundary conditions
- `REPORT-01` through `REPORT-04`: Reporting and diagnostics (meta)

## Issue Schema

Each issue entry in `KNOWN_ISSUES.md` contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `SOLAR-01`) |
| `title` | string | Brief descriptive title |
| `description` | string | Detailed explanation of the issue |
| `affected_cases` | list of strings | ASHRAE 140 case IDs affected (e.g., `["900", "910"]`) |
| `affected_metrics` | list of strings | Metrics impacted (e.g., `["Annual Cooling", "Peak Cooling"]`) |
| `severity` | enum: `Critical`, `High`, `Medium`, `Low` | Impact level (see criteria below) |
| `github_issue` | string or null | Link to GitHub issue (e.g., `#273`) or `null` |
| `status` | enum: `open`, `investigating`, `fixed`, `wontfix` | Current resolution status |
| `phase_addressed` | string or null | Phase where addressed (e.g., `Phase 3`) or `null` |
| `resolution_notes` | string or null | Details of fix or why won't fix |

## Severity Criteria

- **Critical**: Causes >50% failure rate across multiple cases, or blocks core functionality
- **High**: Causes 10-50% failure rate, or single high-impact case failure (major metric off by >50%)
- **Medium**: Causes 1-10% failures, or affects non-primary metrics, or deviation 15-50%
- **Low**: Cosmetic issues, edge cases, or deviations <15%

## Example Entry

```markdown
### SOLAR-01: Peak Cooling Under-Prediction

- **Description:** Peak cooling loads are systematically under-predicted by 40-80% across all cases, indicating insufficient solar gain absorption or incorrect solar distribution to thermal zones.
- **Affected Cases:** 600, 610, 620, 630, 640, 650, 900, 910, 920, 930, 940, 950, 960
- **Affected Metrics:** Peak Cooling (kW)
- **Severity:** Critical
- **GitHub Issue:** #274
- **Status:** open
- **Phase Addressed:** Phase 3 (target)
- **Resolution Notes:** Under investigation - likely related to solar beam-to-mass fraction and/or shading coefficient calculation.
```

## Notes

- Issues should have concrete numerical evidence from validation runs
- When an issue is fixed, update `status` to `fixed` and add resolution notes
- If multiple root causes are discovered, split into separate issue entries
- Link to GitHub issues for traceability when available
