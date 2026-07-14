#!/bin/bash
# Fluxion Pre-commit Lint Hook
# Runs fluxion-specific linting rules before commit
# Install: ln -s ../../scripts/precommit-lint.sh .git/hooks/pre-commit

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo "Running Fluxion pre-commit lint..."

# Check for floating point equality (FLX-CODE-001)
echo "Checking for floating point equality..."
FPEQ_COUNT=$(grep -rn --include="*.rs" -E '==\s*(f64|f32|inf|nan)' "$REPO_ROOT/src" "$REPO_ROOT/tests" 2>/dev/null | grep -v '//.*==' | grep -v 'f64::(INFINITY|NAN)' | wc -l || true)
if [ "$FPEQ_COUNT" -gt 0 ]; then
    echo -e "${RED}FLX-CODE-001: Found $FPEQ_COUNT potential floating point equality checks${NC}"
    grep -rn --include="*.rs" -E '==\s*(f64|f32|inf|nan)' "$REPO_ROOT/src" "$REPO_ROOT/tests" 2>/dev/null | grep -v '//.*==' | grep -v 'f64::(INFINITY|NAN)' | head -10
    ERRORS=$((ERRORS + FPEQ_COUNT))
fi

# Check for TODO/FIXME in physics code (should use proper error handling)
echo "Checking for TODOs in physics modules..."
TODO_COUNT=$(grep -rn --include="*.rs" -E '(TODO|FIXME|HACK|XXX)' "$REPO_ROOT/src/physics" "$REPO_ROOT/src/sim" 2>/dev/null | wc -l || true)
if [ "$TODO_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Warning: Found $TODO_COUNT TODOs/FIXMEs in physics code${NC}"
    grep -rn --include="*.rs" -E '(TODO|FIXME|HACK|XXX)' "$REPO_ROOT/src/physics" "$REPO_ROOT/src/sim" 2>/dev/null | head -5
    WARNINGS=$((WARNINGS + TODO_COUNT))
fi

# Check for missing bounds checks in physics equations
echo "Checking for missing bounds in physics code..."
if [ -f "scripts/check_bounds.py" ]; then
    python3 scripts/check_bounds.py 2>/dev/null || true
fi

# Check for hardcoded tolerances
echo "Checking for hardcoded tolerances..."
TOL_COUNT=$(grep -rn --include="*.rs" -E 'assert!.*1e-\d+' "$REPO_ROOT/src" "$REPO_ROOT/tests" 2>/dev/null | grep -v '//.*assert' | wc -l || true)
if [ "$TOL_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}FLX-PHYSICS-002: Found $TOL_COUNT hardcoded tolerances - consider using named constants${NC}"
    WARNINGS=$((WARNINGS + TOL_COUNT))
fi

# Check test names follow FLX-TEST-001
echo "Checking test naming conventions..."
BAD_TEST_NAMES=$(grep -rn --include="*.rs" -E '#\[test\]' "$REPO_ROOT/src" "$REPO_ROOT/tests" 2>/dev/null | grep -E 'fn\s+(test|tests|tested)' | wc -l || true)
if [ "$BAD_TEST_NAMES" -gt 0 ]; then
    echo -e "${YELLOW}FLX-TEST-001: Found $BAD_TEST_NAMES tests with opaque names${NC}"
    WARNINGS=$((WARNINGS + BAD_TEST_NAMES))
fi

# Run cargo clippy
echo "Running clippy..."
if cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tee /tmp/clippy.log; then
    echo -e "${GREEN}Clippy passed${NC}"
else
    CLIPPY_ERRORS=$(grep -c "error:" /tmp/clippy.log || true)
    echo -e "${RED}Clippy found $CLIPPY_ERRORS errors${NC}"
    ERRORS=$((ERRORS + CLIPPY_ERRORS))
fi

# Run rustfmt check
echo "Checking formatting..."
if cargo fmt --check; then
    echo -e "${GREEN}Formatting OK${NC}"
else
    echo -e "${RED}Formatting issues found - run 'cargo fmt'${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=========================================="
echo -e "Lint Summary: ${ERRORS} errors, ${WARNINGS} warnings"
echo "=========================================="

if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}Commit blocked - fix errors before committing${NC}"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}Commit allowed with warnings - review recommended${NC}"
    exit 0
else
    echo -e "${GREEN}Commit allowed${NC}"
    exit 0
fi
