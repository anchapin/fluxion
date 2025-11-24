#!/usr/bin/env python3
"""
Hook: Validate Rust doc comments on public API
Purpose: Ensure all public functions/structs have /// documentation
This prevents undocumented surrogates, physics changes, and parameter mappings
"""

import re
import sys

exit_code = 0

for filepath in sys.argv[1:]:
    if not filepath.endswith(".rs"):
        continue

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠ Error reading {filepath}: {e}")
        continue

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for public function/struct/impl without preceding doc comment
        if re.match(r"^\s*pub\s+(fn|struct|impl|async fn|enum|trait)", line):
            # Skip test functions
            if "test" in line or "cfg(test)" in lines[i - 1] if i > 0 else False:
                i += 1
                continue

            # Check if there is a doc comment before this pub item
            # Walk backwards skipping attributes like #[derive(...)]
            has_doc = False
            j = i - 1
            while j >= 0:
                check_line = lines[j]
                # Stop if we hit a doc comment
                if re.match(r"^\s*///", check_line):
                    has_doc = True
                    break
                # Skip attributes and empty lines
                if re.match(r"^\s*#\[", check_line) or re.match(r"^\s*$", check_line):
                    j -= 1
                    continue
                # Stop if we hit anything else (previous item, impl block, etc.)
                break

            if not has_doc:
                print(f"❌ {filepath}:{i + 1}: Public item missing doc comment")
                print(f"   {line.strip()}")
                print("   ✓ Add doc comment: /// Your documentation here")
                exit_code = 1

        i += 1

sys.exit(exit_code)
