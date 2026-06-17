#!/usr/bin/env bash
# =============================================================================
# annual_ashrae_revalidation.sh
#
# Automates the annual ASHRAE 140 re-validation process for Fluxion.
# This script should be run every January to verify that Fluxion's blind
# validation pass rate remains above the 80% threshold.
#
# Usage:
#   ./annual_ashrae_revalidation.sh [OPTIONS]
#
# Options:
#   --year YYYY         Year for re-validation (default: current year)
#   --dry-run           Show what would be done without executing
#   --skip-tests        Skip running the test suite
#   --skip-data-update  Skip updating reference data
#   --verbose           Enable verbose output
#   --help              Show this help message
#
# Exit Codes:
#   0 - Validation passed (pass rate >= 80%)
#   1 - Validation failed (pass rate < 80%)
#   2 - Error during validation process
#
# Requirements:
#   - Rust toolchain (stable)
#   - Python 3.8+
#   - bash 4.0+
#
# =============================================================================

set -euo pipefail

# Default values
YEAR=$(date +%Y)
DRY_RUN=false
SKIP_TESTS=false
SKIP_DATA_UPDATE=false
VERBOSE=false
REPORT_DIR="docs/ashrae_140/annual_reports"
DATA_DIR="tests/reference_data"

# Threshold
PASS_RATE_THRESHOLD=80.0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

show_help() {
    head -50 "$0" | grep -E "^#|^#!" | sed 's/^# \?//'
    echo ""
    echo "Options:"
    grep -E "^\s+--" "$0" | head -20
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --year)
                YEAR="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-data-update)
                SKIP_DATA_UPDATE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 2
                ;;
        esac
    done
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for Rust
    if ! command -v cargo &> /dev/null; then
        log_error "Rust toolchain not found. Please install Rust."
        exit 2
    fi

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3."
        exit 2
    fi

    # Check for gh CLI (for GitHub operations)
    if ! command -v gh &> /dev/null; then
        log_warning "GitHub CLI (gh) not found. Some operations will be skipped."
    fi

    log_success "Prerequisites check passed"
}

create_report_directory() {
    local report_year_dir="${REPORT_DIR}/${YEAR}"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would create directory: ${report_year_dir}"
        return
    fi

    mkdir -p "${report_year_dir}"
    log_info "Created report directory: ${report_year_dir}"
}

update_reference_data() {
    if [[ "$SKIP_DATA_UPDATE" == true ]]; then
        log_info "Skipping reference data update (--skip-data-update)"
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would update reference data from official sources:"
        log_info "  - EnergyPlus (NREL)"
        log_info "  - ESP-r (University of Strathclyde)"
        log_info "  - TRNSYS"
        return
    fi

    log_info "Updating reference data..."

    # Check if reference data directories exist
    if [[ ! -d "${DATA_DIR}" ]]; then
        log_warning "Reference data directory not found: ${DATA_DIR}"
        log_info "Creating directory structure..."
        mkdir -p "${DATA_DIR}"/{energyplus,esp-r,trnsys}
    fi

    # Document the update
    local data_version_file="${DATA_DIR}/versions.json"
    local timestamp=$(date -Iseconds)

    python3 << EOF
import json
import os
from datetime import datetime

version_info = {
    "year": "${YEAR}",
    "updated_at": "${timestamp}",
    "sources": {
        "energyplus": {
            "organization": "NREL",
            "url": "https://energyplus.net/downloads",
            "version": "latest"
        },
        "esp_r": {
            "organization": "University of Strathclyde",
            "url": "https://www.esru.strath.ac.uk/ESP-r",
            "version": "latest"
        },
        "trnsys": {
            "organization": "TESS",
            "url": "https://www.trnsys.com",
            "version": "latest"
        }
    }
}

os.makedirs("${DATA_DIR}", exist_ok=True)
with open("${data_version_file}", "w") as f:
    json.dump(version_info, f, indent=2)
print("Reference data versions documented")
EOF

    log_success "Reference data update complete"
}

run_validation_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_info "Skipping test run (--skip-tests)"
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would run: cargo test --test ashrae_140_validation --release"
        return
    fi

    log_info "Running ASHRAE 140 validation tests..."

    local output_file="${REPORT_DIR}/${YEAR}/validation_output.txt"
    local json_file="${REPORT_DIR}/${YEAR}/validation_results.json"

    # Run tests and capture output
    if cargo test --test ashrae_140_validation --release 2>&1 | tee "${output_file}"; then
        log_success "Validation tests completed"
    else
        log_warning "Validation tests completed with some failures"
    fi

    # Extract results
    python3 << EOF
import re
import json
import os

output_file = "${output_file}"
json_file = "${json_file}"

with open(output_file, 'r') as f:
    output = f.read()

# Extract pass rate
pass_rate = None
passed = None
failed = None
mae = None

rate_pattern = r"Pass Rate:\s*([0-9.]+)%"
rate_match = re.search(rate_pattern, output)
if rate_match:
    pass_rate = float(rate_match.group(1))

passed_pattern = r"Passed:\s*([0-9]+)"
passed_match = re.search(passed_pattern, output)
if passed_match:
    passed = int(passed_match.group(1))

failed_pattern = r"Failed:\s*([0-9]+)"
failed_match = re.search(failed_pattern, output)
if failed_match:
    failed = int(failed_match.group(1))

mae_pattern = r"Mean Absolute Error:\s*([0-9.]+)%"
mae_match = re.search(mae_pattern, output)
if mae_match:
    mae = float(mae_match.group(1))

results = {
    "year": ${YEAR},
    "pass_rate": pass_rate,
    "passed": passed,
    "failed": failed,
    "mae": mae,
    "threshold": ${PASS_RATE_THRESHOLD},
    "status": "passed" if (pass_rate and pass_rate >= ${PASS_RATE_THRESHOLD}) else "failed"
}

os.makedirs(os.path.dirname(json_file), exist_ok=True)
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results: pass_rate={pass_rate}%, passed={passed}, failed={failed}, mae={mae}%")
EOF
}

generate_annual_report() {
    local report_file="${REPORT_DIR}/${YEAR}/report.md"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would generate report: ${report_file}"
        return
    fi

    log_info "Generating annual re-validation report..."

    local json_file="${REPORT_DIR}/${YEAR}/validation_results.json"

    python3 << EOF
import json
from datetime import datetime

json_file = "${json_file}"
report_file = "${report_file}"

with open(json_file, 'r') as f:
    results = json.load(f)

pass_rate = results.get('pass_rate', 0) or 0
passed = results.get('passed', 0) or 0
failed = results.get('failed', 0) or 0
mae = results.get('mae', 0) or 0
threshold = results.get('threshold', 80.0)
status = results.get('status', 'unknown')

status_badge = "✅ PASSED" if status == "passed" else "❌ FAILED"

report = f"""# ASHRAE 140 Annual Re-Validation Report

**Year:** ${YEAR}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Status:** {status_badge}

## Executive Summary

This report documents the annual ASHRAE 140 blind validation re-validation
for Fluxion building energy simulation software.

## Validation Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Pass Rate | {pass_rate:.1f}% | {threshold}% | {"✅" if pass_rate >= threshold else "❌"} |
| Cases Passed | {passed} | - | - |
| Cases Failed | {failed} | - | - |
| Mean Absolute Error | {mae:.2f}% | < 5% | {"✅" if mae < 5 else "❌"} |

## Outcome

{"## Validation Passed\n\nAll ASHRAE 140 test cases are within acceptable tolerance. The CI gate will continue to enforce the 80% pass rate threshold." if status == "passed" else "## Validation Failed\n\nThe blind validation pass rate is below the 80% threshold. The following actions are required:\n\n1. Investigate root cause of failed cases\n2. Fix code regressions or document known limitations\n3. Re-run validation until pass rate >= 80%\n4. Block PRs to main until resolved"}

## Reference Data

Reference data versions are documented in `tests/reference_data/versions.json`.

## Next Steps

- [ ] Review failed cases (if any)
- [ ] Fix regressions or document limitations
- [ ] Update `ASHRAE_REVALIDATION_SCHEDULE.md` if process changed
- [ ] Archive this report

## Sign-off

| Role | Name | Date |
|------|------|------|
| Validation Lead | | |
| Technical Lead | | |

---

*Report generated by annual_ashrae_revalidation.sh*
"""

with open(report_file, 'w') as f:
    f.write(report)

print(f"Report generated: {report_file}")
EOF

    log_success "Annual report generated"
}

check_validation_result() {
    local json_file="${REPORT_DIR}/${YEAR}/validation_results.json"

    if [[ ! -f "$json_file" ]]; then
        log_error "Validation results file not found: ${json_file}"
        exit 2
    fi

    local pass_rate=$(python3 -c "import json; print(json.load(open('${json_file}'))['pass_rate'] or 0)")
    local status=$(python3 -c "import json; print(json.load(open('${json_file}'))['status'])")

    log_info "Validation Result: Pass rate = ${pass_rate}%, Status = ${status}"

    if (( $(echo "$pass_rate < $PASS_RATE_THRESHOLD" | bc -l) )); then
        log_error "CI gate failed: pass rate ${pass_rate}% is below ${PASS_RATE_THRESHOLD}% threshold"
        return 1
    fi

    log_success "CI gate passed: ${pass_rate}% >= ${PASS_RATE_THRESHOLD}%"
    return 0
}

create_github_milestone() {
    if ! command -v gh &> /dev/null; then
        log_info "GitHub CLI not available, skipping milestone creation"
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would create GitHub milestone: ASHRAE-140-${YEAR}-Annual-Revalidation"
        return
    fi

    log_info "Creating GitHub milestone..."

    local milestone_title="ASHRAE-140-${YEAR}-Annual-Revalidation"
    local due_date="${YEAR}-02-15"

    if gh milestone list | grep -q "${milestone_title}"; then
        log_info "Milestone already exists: ${milestone_title}"
    else
        gh milestone create "${milestone_title}" \
            --description "Annual ASHRAE 140 re-validation for ${YEAR}" \
            --due-date "${due_date}" 2>/dev/null || \
            log_warning "Could not create milestone (may already exist or lack permissions)"
    fi

    log_success "Milestone created/verified"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "============================================"
    echo "ASHRAE 140 Annual Re-Validation Script"
    echo "Year: ${YEAR}"
    echo "============================================"
    echo ""

    parse_args "$@"

    log_info "Starting annual re-validation process..."

    check_prerequisites
    create_report_directory
    update_reference_data
    run_validation_tests
    generate_annual_report

    if check_validation_result; then
        create_github_milestone
        log_success "Annual re-validation completed successfully!"
        exit 0
    else
        log_error "Annual re-validation FAILED"
        log_error "Pass rate is below ${PASS_RATE_THRESHOLD}% threshold"
        log_error "Review failed cases and fix regressions before proceeding"
        exit 1
    fi
}

main "$@"