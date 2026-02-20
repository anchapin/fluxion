#!/bin/bash
# Fluxion Parallel Worktree Setup Script
# Creates git worktrees for parallel development based on issues analysis

set -e

echo "================================================================================"
echo "FLUXION PARALLEL WORKTREE SETUP"
echo "================================================================================"
echo ""

# Check if we're in the fluxion repo
if [ ! -f "Cargo.toml" ] || [ ! -d ".git" ]; then
    echo "Error: Please run this script from the fluxion repository root"
    exit 1
fi

# Function to create a worktree
create_worktree() {
    local issue_num=$1
    local issue_title=$2

    # Sanitize title for directory name
    local safe_title=$(echo "$issue_title" | tr '[:upper:]' '[:lower:]' | \
        sed 's/[:\/()]/-/g' | \
        sed 's/ /-/g' | \
        sed 's/--*/-/g' | \
        cut -c1-50)

    local worktree_path="../feature-issue-${issue_num}-${safe_title}"
    local branch_name="feature/issue-${issue_num}"

    # Check if worktree already exists
    if [ -d "$worktree_path" ]; then
        echo "  ⚠️  Worktree already exists: $worktree_path"
        return 0
    fi

    # Create the worktree
    echo "  Creating: $worktree_path"
    git worktree add "$worktree_path" -b "$branch_name"
}

echo "Creating worktrees for 18 issues..."
echo ""

# Track 1: Solar & HVAC
echo "Track 1: Solar & HVAC"
create_worktree 303 "Detailed Internal Radiation Network"
create_worktree 299 "Refine Window Angular Dependence Model"
create_worktree 281 "Investigation: Construction U-values"
create_worktree 278 "Investigation: Solar gain calculation"
create_worktree 276 "Enhancement: Implement ideal HVAC control"
echo ""

# Track 2: Physics Core & Construction
echo "Track 2: Physics Core & Construction"
create_worktree 273 "Investigation: Case 960 multi-zone"
create_worktree 294 "Implement Rigorous ISO 13790 Annex C"
create_worktree 280 "Investigation: Internal heat gains"
create_worktree 272 "Investigation: Peak load values"
echo ""

# Track 3: Zone Physics & Radiation
echo "Track 3: Zone Physics & Radiation"
create_worktree 302 "Refine Inter-Zone Longwave Radiation"
create_worktree 295 "Implement Multiple Surface Conductances"
create_worktree 279 "Investigation: Infiltration modeling"
create_worktree 274 "Investigation: Thermal mass modeling"
echo ""

# Track 4: Ventilation & Tools
echo "Track 4: Ventilation & Tools"
create_worktree 301 "Dynamic Sensitivity Tensors"
create_worktree 297 "Geometric Solar Distribution"
create_worktree 304 "Automated Hourly Delta Analysis"
create_worktree 275 "Investigation: Free-floating temperatures"
create_worktree 277 "Roadmap: ASHRAE 140 CI pass rate"
echo ""

echo "================================================================================"
echo "SETUP COMPLETE"
echo "================================================================================"
echo ""
echo "List all worktrees:"
git worktree list
echo ""
echo "Next steps:"
echo "  1. Navigate to a worktree: cd ../feature-issue-XXX"
echo "  2. Work on the issue following the plan in docs/PARALLEL_EXECUTION_PLAN.md"
echo "  3. Create PR when done: git push && gh pr create"
echo "  4. Remove worktree: git worktree remove ../feature-issue-XXX"
echo ""
