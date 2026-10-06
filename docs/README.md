# docs/ Index

<!-- 7-line summary -->
<!-- Line 1: Canonical entry points for Fluxion documentation. -->
<!-- Line 2: Start here if you are new: QUICKSTART for setup, -->
<!-- Line 3: FEATURES for build flags, EXAMPLES for runnable demos. -->
<!-- Line 4: Validation: ASHRAE140_RESULTS (per-case numbers) and -->
<!-- Line 5: KNOWN_ISSUES (the public roadmap of structural gaps). -->
<!-- Line 6: Architecture: ARCHITECTURE_DEEP_DIVE and adr/. -->
<!-- Line 7: Everything else is indexed in doc-inventory.md. -->

A guide to what lives under `docs/`. The full auto-generated inventory of
every file (with summaries) is [`doc-inventory.md`](doc-inventory.md);
investigation notes are in [`investigations/`](investigations/), and
historical worklogs and superseded plans are in
[`archive/`](archive/README.md).

## Canonical entry points

| Doc | What it covers |
|-----|----------------|
| [QUICKSTART.md](QUICKSTART.md) | Install, first simulation, config format |
| [FEATURES.md](FEATURES.md) | Authoritative cargo feature-flag list and build commands |
| [EXAMPLES.md](EXAMPLES.md) | Runnable examples, incl. the ASHRAE 140 first-run check |
| [ASHRAE140_RESULTS.md](ASHRAE140_RESULTS.md) | Per-case ASHRAE 140-2023 validation numbers vs reference bands |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | The public roadmap: every known structural failure and its investigation |
| [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md) | Thermal network, solvers, surrogate swap points |
| [adr/](adr/) | Architecture decision records (numbered) |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Dev workflow, linker/memory settings, tooling |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common build/run failures and fixes |
| [investigations/](investigations/) | Per-issue and per-limit root-cause investigations |
| [archive/](archive/README.md) | Historical session notes, worklogs, superseded plans |
