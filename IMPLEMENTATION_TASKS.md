# Implementation Tasks for Open GitHub Issues

## Task List

- [ ] Issue #103: Complete all 18 ASHRAE 140 test cases
- [ ] Issue #104: Add reference range validation with pass/fail criteria  
- [ ] Issue #105: Implement multi-zone thermal model for Case 960
- [ ] Issue #106: Phase 5 - Retrain surrogate with ASHRAE physics
- [ ] Issue #107: Add CI integration for ASHRAE 140 validation
- [ ] Issue #108: Physics model improvements (free-floating, HVAC scheduling)
- [ ] Issue #109: Fix README badge (update to accurate status)

## Parallel Execution Strategy

Issues that can be worked in parallel (minimal dependencies):
- #103 + #104 + #107 + #109 (validation framework, docs, CI)
- #108 (physics improvements - foundational for #103)

Issues with dependencies:
- #105 depends on #103/#108 (needs free-floating and multi-zone support)
- #106 depends on #103/#104 (needs validated physics first)

## Current Branch Status
- Currently on: feature/night-ventilation-fresh
- Issues 103-109: No feature branches exist yet
