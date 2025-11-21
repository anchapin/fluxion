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
    # Extract lines within evaluate_population function and count par_iter only there
    eval_pop_start=$(grep -n "fn evaluate_population" "$file" | cut -d: -f1)
    if [ -n "$eval_pop_start" ]; then
      # Find the end of the function by searching for the next unindented closing brace
      eval_pop_end=$(tail -n +"$eval_pop_start" "$file" | grep -n -m1 "^}" | head -n1 | cut -d: -f1)
      if [ -n "$eval_pop_end" ]; then
        eval_pop_end=$((eval_pop_start + eval_pop_end - 1))
        # Extract the function body
        eval_pop_body=$(sed -n "${eval_pop_start},${eval_pop_end}p" "$file")
        par_iter_count=$(echo "$eval_pop_body" | grep -c "par_iter" || echo 0)
        if [ "$par_iter_count" -gt 1 ]; then
          echo "❌ PERF REGRESSION: $file"
          echo "   Nested par_iter detected in evaluate_population"
          echo "   ✓ Solution: Use single-level population-wide parallelism only"
          exit 1
        fi
      fi
    fi
  fi
  
  # Verify evaluate_population actually uses rayon
  if grep -q "fn evaluate_population" "$file"; then
    # Increase the search range to 300 lines after the function definition
    if ! grep -A 300 "fn evaluate_population" "$file" | grep -q "par_iter"; then
      echo "⚠ PERFORMANCE WARNING: $file"
      echo "   evaluate_population missing rayon parallelism"
      echo "   ✓ Solution: Add .par_iter() for population-level parallelism"
      exit 1
    fi
  fi
done

exit 0
