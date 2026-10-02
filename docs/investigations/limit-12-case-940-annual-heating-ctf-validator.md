# LIMIT-12 — investigation history

Narrative history for **LIMIT-12**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-12` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-12: Case 940 annual heating CTF-validator vs blind-diagnostic path divergence — setback-recovery overshoot (Issue #3062)

**Description:** After PR #3042, Case 940 annual heating is 5,158 kWh on the CTF validator path versus 1,289.9 kWh on the blind diagnostic path (per the §LIMIT-05 UPDATE #2452 measurement table — canonical source-of-truth for the post-PR #3042 numbers). The remaining setback-recovery overshoot is structural: the CTF coupling overshoots during setback recovery windows, and the residual gap between the two paths is not closable without production-physics changes.

**Historical numbers (do NOT cite as current):** the original Issue #3062 framing cited 7,487.81 kWh on the CTF path (pre-§LIMIT-05-UPDATE snapshot from the §LIMIT-05 UPDATE investigation); the §LIMIT-05 UPDATE #2452 measurement re-ran the same path with post-#3042 code and found 5,158 kWh. Both numbers are correct for their respective code snapshots; this entry cites the latest (§LIMIT-05 UPDATE #2452) as the source of truth.

**Affected Cases:** Case 940 ( setback thermostat, low-mass).

**Affected Metrics:** Annual heating (kWh) — CTF validator path vs blind diagnostic path divergence (5,158 vs 1,289.9 kWh; per §LIMIT-05 UPDATE #2452 measurement table).

**Severity:** Medium — single-case metric divergence; no energy-balance violation.

**GitHub Issue:** #3062 (this LIMIT entry's tracker).

**Status:** Tracked without a production-physics change per AGENTS.md / RULES.md / ADR-0001 (no parameter tuning to force path convergence).

**Resolution Notes:** Structural fix routed to the GaugeSolver production-path work (#1465 / #1462), same cohort as LIMIT-13 / #3063, LIMIT-14 / #3061, LIMIT-16 / #3059, LIMIT-17 / #3058, LIMIT-18 / #3104. Sibling cohort tracking: Issue #3072 (aggressive-baseline cohort). Section header added by Issue #3397 (the entry previously existed only in the intro text without a `### LIMIT-12:` anchor, breaking cross-reference jumps from LIMIT-14/16/18/21 and the SOAK workflow docs).
