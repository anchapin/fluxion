#!/usr/bin/env bash
# Physics Coverage Pre-Commit Hook
#
# Checks that physics module coverage meets 90% threshold
# Run manually with: pre-commit run physics-coverage --all-files

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Threshold configuration
PHYSICS_COVERAGE_THRESHOLD=90
COVERAGE_REPORT_DIR="coverage/physics"
TEMP_REPORT="$COVERAGE_REPORT_DIR/pre-commit-cobertura.xml"

echo -e "${YELLOW}Running physics coverage check...${NC}"
echo "Target coverage: ${PHYSICS_COVERAGE_THRESHOLD}%"
echo ""

# Check if tarpaulin is installed
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo -e "${RED}Error: cargo-tarpaulin not found${NC}"
    echo "Install with: cargo install cargo-tarpaulin"
    exit 1
fi

# Clean previous coverage data
echo "Cleaning previous coverage data..."
cargo clean

# Run tarpaulin with physics-only filtering
echo "Generating coverage report (this may take a minute)..."
mkdir -p "$COVERAGE_REPORT_DIR"

# Run tarpaulin with timeout and output to cobertura format
timeout 600 cargo tarpaulin \
    --timeout 600 \
    --out Xml \
    --output-dir "$COVERAGE_REPORT_DIR" \
    --ignore-panics \
    --exclude-files '*/tests/*' \
    --exclude-files '*/benches/*' \
    --exclude-files '*/examples/*' \
    2>&1 | tee "$COVERAGE_REPORT_DIR/tarpaulin.log" || {
        echo -e "${RED}Error: tarpaulin execution failed${NC}"
        echo "Check log at: $COVERAGE_REPORT_DIR/tarpaulin.log"
        exit 1
    }

# Verify coverage XML was generated
if [ ! -f "$TEMP_REPORT" ]; then
    echo -e "${RED}Error: Coverage XML not generated at $TEMP_REPORT${NC}"
    echo "Available files in $COVERAGE_REPORT_DIR:"
    ls -la "$COVERAGE_REPORT_DIR"
    exit 1
fi

# Extract physics coverage using Python
echo ""
echo "Analyzing physics coverage..."
COVERAGE=$(python3 << 'EOF'
import xml.etree.ElementTree as ET
import sys

try:
    tree = ET.parse('$TEMP_REPORT')
    root = tree.getroot()

    physics_files = []
    total_lines = 0
    covered_lines = 0

    # Find all classes in coverage report
    for cls in root.findall('.//class'):
        filename = cls.get('filename', '')
        if 'src/physics/' in filename:
            physics_files.append(filename)
            line_rate = float(cls.get('line-rate', '0.0'))
            lines_valid = int(cls.get('lines-valid', '0'))

            total_lines += lines_valid
            covered_lines += lines_valid * line_rate

    if total_lines > 0:
        physics_coverage = (covered_lines / total_lines) * 100
        print(f"{physics_coverage:.2f}")
        print(f"Files: {len(physics_files)}")
        print(f"Total lines: {total_lines}")
        print(f"Covered lines: {covered_lines:.0f}")

        # Show detailed file coverage
        for cls in root.findall('.//class'):
            filename = cls.get('filename', '')
            if 'src/physics/' in filename:
                line_rate = float(cls.get('line-rate', '0.0')) * 100
                file_lines = int(cls.get('lines-valid', '0'))
                print(f"  {filename}: {line_rate:.2f}% ({file_lines} lines)")
    else:
        print("0.00")
        print("Error: No physics files found in coverage report", file=sys.stderr)
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

# Capture the output
PHYSICS_COVERAGE=$(echo "$COVERAGE" | head -n 1)
FILE_COUNT=$(echo "$COVERAGE" | tail -n +2 | head -n 1)
TOTAL_LINES=$(echo "$COVERAGE" | tail -n +3 | head -n 1)
COVERED_LINES=$(echo "$COVERAGE" | tail -n +4 | head -n 1)

echo ""
echo "======================================"
echo "Physics Coverage Report"
echo "======================================"
echo "Coverage: ${PHYSICS_COVERAGE}%"
echo "Target:   ${PHYSICS_COVERAGE_THRESHOLD}%"
echo "Files:    ${FILE_COUNT}"
echo "Lines:    ${COVERED_LINES}/${TOTAL_LINES}"
echo "======================================"

# Check threshold
COVERAGE_INT=${PHYSICS_COVERAGE%.*}

if (( COVERAGE_INT < PHYSICS_COVERAGE_THRESHOLD )); then
    GAP=$((PHYSICS_COVERAGE_THRESHOLD - COVERAGE_INT))
    echo ""
    echo -e "${RED}❌ FAIL: Physics coverage ${PHYSICS_COVERAGE}% is below ${PHYSICS_COVERAGE_THRESHOLD}% threshold${NC}"
    echo -e "${RED}Gap: ${GAP}%${NC}"
    echo ""
    echo "To fix this:"
    echo "  1. Add tests to uncovered physics modules"
    echo "  2. Run: cargo llvm-cov test --lib physics:: -- --nocapture"
    echo "  3. Generate detailed report: cargo llvm-cov report --html"
    echo ""
    echo "Missing coverage by module (from tarpaulin log):"
    grep -E "(src/physics/|Coverage:)" "$COVERAGE_REPORT_DIR/tarpaulin.log" | tail -20 || true
    exit 1
else
    echo ""
    echo -e "${GREEN}✅ PASS: Physics coverage ${PHYSICS_COVERAGE}% meets ${PHYSICS_COVERAGE_THRESHOLD}% threshold${NC}"
    exit 0
fi
