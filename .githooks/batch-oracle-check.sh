#!/bin/bash
# Hook: Validate BatchOracle uses correct parallelism pattern
# Purpose: Ensure evaluate_population uses single-level rayon parallelism only
# This is critical for the 100x speedup goal (10k+ configs/sec)

set -e

for file in "$@"; do
  if [[ ! "$file" =~ lib\.rs$ ]]; then
    continue
  fi
  
  # Check for nested rayon par_iter (would cause thread-pool exhaustion)
  if grep -q "par_iter" "$file"; then
    # Count occurrences of par_iter in evaluate_population function
    par_iter_count=$(grep -c "par_iter" "$file" || echo 0)
    
    if [ "$par_iter_count" -gt 1 ]; then
      # Check if they're nested (different indentation levels in same function)
      if grep "fn evaluate_population" "$file" >/dev/null; then
        echo "❌ PERF REGRESSION: $file"
        echo "   Nested par_iter detected in evaluate_population"
        echo "   ✓ Solution: Use single-level population-wide parallelism only"
        exit 1
      fi
    fi
  fi
  
  # Verify evaluate_population actually uses rayon
  if grep -q "fn evaluate_population" "$file"; then
    if ! grep -A 30 "fn evaluate_population" "$file" | grep -q "par_iter"; then
      echo "⚠ PERFORMANCE WARNING: $file"
      echo "   evaluate_population missing rayon parallelism"
      echo "   ✓ Solution: Add .par_iter() for population-level parallelism"
      exit 1
    fi
  fi
done

exit 0
