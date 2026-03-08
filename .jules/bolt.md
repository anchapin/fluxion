## 2024-03-XX - [Initial Setup]
**Learning:** Just starting to log learnings for performance optimization.
**Action:** Always look for intermediate vector allocations to eliminate in hot loops.
## 2024-06-25 - [O(N^2) to O(N) via algebraic grouping]
**Learning:** Nested loops checking N zones (O(N^2)) when transferring heat can be replaced by grouping the operations: $Sum(h * (T_j - T_i)) = h * (Sum(T) - N * T_i)$. This optimization allows an O(N) allocation approach without complex matrix dependencies. Also returning `Option<Vec<f64>>` to prevent an array of zeroes avoids redundant `VectorField` instantiations in the hot loop.
**Action:** In multi-zone coupling, evaluate substituting nested $T_j - T_i$ calculations with a singular precomputed sum array implementation. Return Option<> wrappers to skip zero-effect `Vec` operations in dense iteration flows.
