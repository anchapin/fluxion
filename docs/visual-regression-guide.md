# Visual Regression Testing Guide

> **TL;DR**: How to run and maintain visual regression tests for Fluxion physics outputs.
> **Key decisions**: Golden images stored in tests/visual/golden/ | pytest-based | matplotlib for rendering.
> **Owned by**: QA team
> **Reviewed**: 2026-07-13

## Overview

Visual regression tests validate that physics engine outputs (charts, reports, diagnostic plots) render correctly and haven't regressed. This is distinct from UI screenshot testing — these tests focus on scientific/engineering visualization outputs.

## Test Structure

```
tests/visual/
├── golden/           # Reference images for comparison
│   ├── validation_header.png
│   ├── benchmark_comparison.png
│   └── ...
├── output/           # Test-generated images (git-ignored)
└── test_*.py         # Test modules
```

## Running Tests

```bash
# Run all visual tests
pytest tests/visual/

# Run specific test
pytest tests/visual/test_validation_report_render.py::TestValidationReportRender::test_benchmark_comparison_chart

# Update golden images (when intentionally changing output)
pytest tests/visual/ --generate-golden
```

## Adding New Visual Tests

1. Create test in `tests/visual/test_<module>.py`
2. Use matplotlib to generate the visualization
3. Save output to `output/` directory
4. If new golden image needed, run with `--generate-golden` and commit the resulting image to `golden/`

## Interpreting Failures

When a test fails:
- **Low MSE (<100)**: Minor rendering differences — acceptable, regenerate golden if intentional
- **High MSE (>100)**: Significant regression — investigate the physics code or rendering logic
- **File not found**: Golden image missing — generate and commit

## Integration with CI

Visual tests run in CI on every PR. Failures block merge. To update golden images:
```bash
git checkout HEAD~1 -- tests/visual/golden/
pytest tests/visual/ --generate-golden
git add tests/visual/golden/
git commit -m "chore: update golden images for visual regression"
```
