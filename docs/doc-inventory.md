# Doc Inventory

Self-healing inventory of all documentation in the Fluxion repository. Each doc carries a 7-line summary (lines 2-8) for rapid context-setting in AI sessions.

| Doc | Purpose | Status |
|-----|---------|--------|
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | Physics module boundaries, I/O contracts, Mermaid diagram | ✅ Has summary |
| [RULES.md](../../RULES.md) | Coding rules, hard constraints, must-always rules | 📝 Needs summary |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Contribution guide, PR workflow, hotfix process | 📝 Needs summary |
| [CODEBASE_MAP.md](../../CODEBASE_MAP.md) | Code navigation, module dependency graph, Rust/Python/JS overview | 📝 Needs summary |
| [FIX.md](../../FIX.md) | Known bugs placeholder, ASHRAE 140 CI gate fixes | 📝 Needs summary |
| [docs/KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md) | Known systematic issues, ASHRAE 140 validation issues | 📝 Needs summary |
| [documentation/performance_guide.md](../../documentation/performance_guide.md) | Performance validation user guide, CLI usage | 📝 Needs summary |
| [documentation/performance.md](../../documentation/performance.md) | Performance benchmarks, optimization, validation targets | 📝 Needs summary |
| [validation_report.md](../../validation_report.md) | ASHRAE 140 validation results, pass/fail rates | 📝 Needs summary |
| [docs/worksheets/README.md](./README.md) | Index of worksheets by issue/tag | 🆕 New |

## 7-Line Summary Convention

Every doc MUST have a 7-line summary at lines 2-8:

```
# Doc Title

<!-- Exactly 6 lines of summary context — one line per concept -->
<!-- Line 1: What this doc is about -->
<!-- Line 2: Who should read it -->
<!-- Line 3: Key concepts covered -->
<!-- Line 4: How it relates to other docs -->
<!-- Line 5: Current status / freshness -->
<!-- Line 6: Any action required -->

## Rest of document...
```

**Agent Instruction**: After ANY change to a module, update the 7-line summary of the relevant doc.

## Maintenance

This inventory is self-healing: run `scripts/doc_inventory_check.sh` to verify all docs have 7-line summaries and the table is accurate.
