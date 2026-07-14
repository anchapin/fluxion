#!/bin/bash
# Fluxion Agent-Assisted Linting Script
# Uses LLM to analyze code for physics correctness and common bugs
# Usage: ./scripts/agent-lint.sh [file(s)]

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FILES="${1:-}"
OUTPUT_FILE="/tmp/fluxion_lint_report_$(date +%s).md"

echo -e "${BLUE}Fluxion Agent Lint Report${NC}"
echo -e "${BLUE}Generated: $(date)${NC}"
echo ""

# If no files specified, lint all Rust files
if [ -z "$FILES" ]; then
    FILES=$(find "$REPO_ROOT/src" "$REPO_ROOT/tests" -name "*.rs" -type f 2>/dev/null)
fi

echo "Analyzing files..."
echo ""

# Run static analysis checks
echo "## Static Analysis Results" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Check 1: Division by potential zero
echo "### FLX-PHYSICS-001: Division Checks" >> "$OUTPUT_FILE"
DIV_ZERO=$(grep -rn --include="*.rs" -E '/\s*\(?[a-z_]+\)?' "$REPO_ROOT/src" 2>/dev/null | grep -v '//.*/' | grep -v 'std::' | head -20 || true)
if [ -n "$DIV_ZERO" ]; then
    echo "Potential division operations found (review for zero-denominator risk):" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$DIV_ZERO" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
else
    echo "No obvious division operations found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Check 2: Floating point comparisons
echo "### FLX-CODE-001: Floating Point Comparisons" >> "$OUTPUT_FILE"
FP_COMP=$(grep -rn --include="*.rs" -E '(==|!=)\s*(f64|f32)' "$REPO_ROOT/src" 2>/dev/null | grep -v '//.*==' | grep -v 'f64::(INFINITY|NAN|EPSILON)' | head -20 || true)
if [ -n "$FP_COMP" ]; then
    echo "Potential floating point equality found:" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$FP_COMP" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "Recommendation: Use \`approx::relative_eq!\` or \`f64::abs(a - b) < tolerance\`" >> "$OUTPUT_FILE"
else
    echo "No obvious floating point equality found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Check 3: Missing Result types
echo "### FLX-CODE-003: Error Handling" >> "$OUTPUT_FILE"
MISSING_ERR=$(grep -rn --include="*.rs" -E '(fn\s+\w+[^)]*)\s*->\s*(?!Result)' "$REPO_ROOT/src" 2>/dev/null | grep -v '//.*fn' | grep -E '(load|parse|read|open|fetch)' | head -10 || true)
if [ -n "$MISSING_ERR" ]; then
    echo "Functions loading data without Result return type:" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$MISSING_ERR" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
else
    echo "All data-loading functions appear to use proper error handling." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Check 4: Physics bounds
echo "### FLX-PHYSICS-003: State Variable Bounds" >> "$OUTPUT_FILE"
echo "Checking for temperature/humidity bounds validation..." >> "$OUTPUT_FILE"
BOUNDS_CHECKS=$(grep -rn --include="*.rs" -E '(clamp|max|min).*(-50|0\.|50|80|200)' "$REPO_ROOT/src" 2>/dev/null | head -10 || true)
if [ -n "$BOUNDS_CHECKS" ]; then
    echo "Found bounds checks:" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$BOUNDS_CHECKS" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
else
    echo "Warning: No explicit bounds checks found for temperature/humidity values." >> "$OUTPUT_FILE"
    echo "Consider adding \`.clamp(MIN_TEMP, MAX_TEMP)\` to prevent unphysical values." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Check 5: Test quality
echo "### FLX-TEST-001: Test Naming" >> "$OUTPUT_FILE"
OPAQUE_TESTS=$(grep -rn --include="*.rs" -E '#\[test\]' "$REPO_ROOT/src" "$REPO_ROOT/tests" 2>/dev/null | grep -E 'fn\s+(test|tests|tested|basic|simple)' | head -10 || true)
if [ -n "$OPAQUE_TESTS" ]; then
    echo "Tests with opaque names (consider renaming for clarity):" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$OPAQUE_TESTS" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
else
    echo "All test names appear descriptive." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Check 6: Energy balance
echo "### FLX-PHYSICS-002: Energy Balance" >> "$OUTPUT_FILE"
BALANCE=$(grep -rn --include="*.rs" -E '(energy|heat|balance|imbalance)' "$REPO_ROOT/src" 2>/dev/null | grep -E '(assert|if|warn|error)' | head -10 || true)
if [ -n "$BALANCE" ]; then
    echo "Found energy balance checks:" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$BALANCE" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
else
    echo "No explicit energy balance checks found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# LLM Analysis Section
echo "## LLM Analysis Recommendations" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "The following analysis requires review by an agent with physics knowledge:" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Generate code snippets for LLM review
echo "### Code Snippets for Review" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find physics-related functions
PHYSICS_FNS=$(grep -rn --include="*.rs" -E '(pub fn|solar|thermal|conduction|ventilation|heat)' "$REPO_ROOT/src" 2>/dev/null | head -20 || true)
if [ -n "$PHYSICS_FNS" ]; then
    echo "Physics-related functions found in:" >> "$OUTPUT_FILE"
    echo "\`\`\`rust" >> "$OUTPUT_FILE"
    echo "$PHYSICS_FNS" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Print report
cat "$OUTPUT_FILE"
echo ""

# Cleanup old reports
find /tmp -name "fluxion_lint_report_*.md" -mtime +1 -delete 2>/dev/null || true

echo ""
echo -e "${GREEN}Lint report saved to: $OUTPUT_FILE${NC}"
echo ""
echo "Common fixes:"
echo "  1. Floating point equality: Use \`approx::relative_eq!(a, b, max_relative = 1e-6)\`"
echo "  2. Division by zero: Add \`.max(MIN_VALUE)\` to denominators"
echo "  3. Missing bounds: Add \`.clamp(MIN, MAX)\` to temperature/humidity calculations"
echo "  4. Test names: Use pattern \`{unit}_{method}_{scenario}_{expected}\`"
echo ""
