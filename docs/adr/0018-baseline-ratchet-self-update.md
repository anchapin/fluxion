# ADR-0018: Self-updating baseline/ratchet enforcement for CI gates

> **Summary 1/4:** Decision: decouple ratchet/baseline enforcement DATA from script LOGIC by introducing a canonical LIMITS data file (`scripts/data/module_size_limits.json`) for `check_module_size.py`. The script reads LIMITS from the file, enforces the LIMITS-list drift against a `baseline` section stored in the same file (updated atomically by `--write-baseline`), and keeps per-file ceiling + ratchet-maximum enforcement byte-identical. The refactor eliminates per-PR mechanical script edits for ratchet/baseline changes while preserving the enforcement semantics.
> **Summary 2/4:** For `check_module_size.py`: the LIMITS table moves from hardcoded Python to a JSON data file; the freeze snapshot (`_BASELINE_MODULE_SIZE_LIMITS_SET`) and count (`BASELINE_MODULE_SIZE_LIMITS`) are read from the data file's `baseline` section (NOT derived from the live list — deriving them from the live list would make the drift check vacuous); `--write-baseline [--reason ...]` writes the data file + baseline atomically with a history entry; `--remove-entries [--delete-ratchets]` handles decomposition cleanup; the gate logic (`check()`, `count_lines()`, `update_ratchet()`) is unchanged.
> **Summary 3/4:** NOT implemented in this PR (future work): deriving `check_test_inventory_drift.py`'s `BASELINE_*` constants from `tests/reference_data/test_inventory_baseline.json::ratchet` at runtime (D2 below was drafted but not built — the script still requires the manual constant bump alongside the baseline JSON); AGENTS.md test-count generation (D5 below).
> **Summary 4/4:** Rejected alternatives: (a) auto-updating baselines on every PR (weakens gates — #3442/#3457 ratchets exist to catch unintended growth); (b) auto-updating in CI on merge (adds complexity and latency); (c) reading LIMITS from ratchet JSONs directly without a canonical data file (ratchet JSONs track history, not the active ceiling; the `max_lines` in LIMITS may be below the ratchet to enforce tighter bounds); (d) deriving the freeze snapshot from the live LIMITS list at runtime (makes `check_baseline_drift()` vacuous — it would compare the list against itself and could never fire; the stored `baseline` section exists precisely to avoid this).

- **Status:** Proposed
- **Date:** 2026-09-29
- **Deciders:** Fluxion maintainers
- **Issues:** Parent #4243

## Context

Issue #4243 documents that CI enforcement files require manual updates on nearly every physics PR:

1. **`scripts/check_test_inventory_drift.py`** hardcodes `BASELINE_LIB_TESTS` / `BASELINE_WORKSPACE_TESTS` / etc. as Python constants — every PR that adds tests must bump them alongside the baseline JSON.
2. **`scripts/check_module_size.py`** hardcodes `BASELINE_MODULE_SIZE_LIMITS`, `_BASELINE_MODULE_SIZE_LIMITS_SET`, and the full `LIMITS` table — every PR that grows a ratcheted module must bump them.
3. **`tests/reference_data/test_inventory_baseline.json`** must be regenerated per PR.
4. **`AGENTS.md`** hardcodes test counts that must be hand-synced with the inventory.
5. **`.github/workflows/`** changes require the `workflow` OAuth scope.

Tonight's #4155 session needed 4 separate PATs, most for baseline/ratchet bumps rather than real code. This is toil, not engineering.

Constraints from the brief:
- No weakening of the gates (ratchets exist to catch unintended growth — issues #3442, #2878, #3457)
- No auto-tuning of ASHRAE bands or tolerances
- Physics integrity is non-negotiable
- Do NOT modify files under `.github/workflows/`

## Decision

### D1 — `check_module_size.py`: canonical LIMITS data file

Introduce `scripts/data/module_size_limits.json` as the single source of truth for the module-size LIMITS table:

```json
{
  "schema_version": 1,
  "limits": [
    {
      "path": "src/sim/thermal_model_data.rs",
      "max_lines": 200,
      "reason": "Issue #2878: drop ThermalModelData below 200 lines..."
    },
    ...
  ],
  "history": [
    {"at": "2026-09-29", "len": 11, "action": "seed from check_module_size.py LIMITS table"}
  ]
}
```

The script:
- Reads LIMITS from this file at runtime (falls back to the committed default if the file is absent)
- Reads the freeze snapshot (`_BASELINE_MODULE_SIZE_LIMITS_SET`) and count (`BASELINE_MODULE_SIZE_LIMITS`) from the file's `baseline` section — deliberately NOT derived from the live `limits` array, which would make the drift check vacuous
- Adds `--write-baseline [--reason ...]` flag that writes the data file, updates the `baseline` section atomically, and appends a `history` entry

The ratchet JSONs (`tests/reference_data/module_size/*.json`) remain as-is: they track historical maxima independently. The LIMITS data file's `max_lines` is the active ceiling; the ratchet's `max_lines` is the historical max (used by `effective_max()`).

### D2 — `check_test_inventory_drift.py`: derive BASELINE_* from baseline JSON (NOT IMPLEMENTED)

Drafted but deliberately left unbuilt in this PR: at runtime, derive the
`BASELINE_*` constants from
`tests/reference_data/test_inventory_baseline.json::ratchet` instead of
hardcoding them in Python. The baseline JSON already contains a `ratchet`
section with all `BASELINE_*` values, and the existing `--update-baseline`
flag already writes it. The script would read those values (keeping
hardcoded values as documentation defaults with `# DO NOT EDIT DIRECTLY —
source of truth is baseline JSON` comments).

This is NOT a weakening: the baseline JSON is itself commit-protected;
anyone who updates it must still go through the PR process. Left as
follow-up work because the module-size gate was the higher-toil target and
the inventory script's dual-update (constant + JSON) is a smaller,
well-understood chore.

### D3 — `check_module_size.py`: add `--write-baseline` flag

Mirrors `--update-baseline` from `check_test_inventory_drift.py`:
- Writes current LIMITS to `scripts/data/module_size_limits.json`
- Updates the `baseline` section (count + path freeze) atomically so the
  drift check stays meaningful
- Appends a `history` entry; `--reason` records the tracking issue
  (replacing the old "documenting comment naming the tracking issue"
  requirement)
- Prints the diff vs. the previous version

### D4 — `check_module_size.py`: add `--update-ratchets` flag for post-decomposition cleanup

When a gated file is decomposed (e.g., `src/validation/report.rs` → `src/validation/report/`), the cleanup PR needs to:
1. Remove the entry from LIMITS
2. Lower `BASELINE_MODULE_SIZE_LIMITS` by one
3. Delete the associated ratchet JSON

The `--update-ratchets` flag handles this: given a list of removed paths, it:
- Filters them from the LIMITS list
- Writes the updated LIMITS to the data file
- Optionally deletes the associated ratchet JSONs (with `--delete-ratchets`)
- Prints the new `BASELINE_MODULE_SIZE_LIMITS` value for manual bump confirmation

### D5 — `AGENTS.md`: add a generate command

The AGENTS.md test-count citations are derived from `tests/test_inventory.json`. A `generate_agents_md_test_counts()` helper (or standalone script) can extract the numbers and emit the table. This is a separate follow-up (tracked in Issue #4243 §"AGENTS.md automation").

## Consequences

**Positive:**
- Mechanical baseline/ratchet bumps no longer require script edits (they become data-file updates via `--write-baseline`); data-file changes don't need the `workflow` OAuth scope, so fewer one-time PATs
- Single source of truth eliminates the dual-maintenance problem (LIMITS table vs. freeze constants in the same script)
- Enforcement semantics preserved: per-file ceilings and ratchet-maximum checks are byte-identical logic; the LIMITS-list drift check still fires on un-baselined additions (verified: doctored data file → exit 1, clean → exit 0) — only the update mechanism changed
- Decomposition cleanup PRs can use `--remove-entries` + `--write-baseline` to handle all ratchet changes in one command

**Negative:**
- The LIMITS data file adds a new file to maintain
- The `--write-baseline` / `--remove-entries` flags are new code paths that need testing
- A subtle trap was fixed during review: deriving the freeze snapshot from the *live* LIMITS list (instead of a stored baseline section) makes the drift check vacuous — see rejected alternative (d)

**Neutral:**
- The baseline JSON for test inventory is already maintained via `--update-baseline`; deriving the script's constants from it is drafted as D2 but not built here
- The workflow-file issue (requires `workflow` OAuth scope) is documented but not addressed (tracked separately)

## Companion artifacts

- **Scripts data directory:** `scripts/data/module_size_limits.json` — canonical LIMITS table
- **Hermetic tests:** `scripts/ci/test_check_module_size.py` extended with tests for the new `--write-baseline` / `--update-ratchets` flags
- **Follow-up:** AGENTS.md automation (Issue #4243 §"AGENTS.md automation")

## Rejected alternatives

1. **Auto-updating baselines on every PR**: This would weaken the gates by allowing growth without review. The ratchets exist to catch unintended growth (#3442, #3457); auto-updating defeats their purpose.

2. **Auto-updating in CI on merge**: This adds complexity (CI needs write access, merge-queue coordination, race conditions between concurrent merges) and latency (baseline update becomes a blocking step). The `--update-baseline` / `--write-baseline` flags achieve the same result with less complexity.

3. **Reading LIMITS from ratchet JSONs directly**: The ratchet JSONs track history (for audit purposes), not the active ceiling. The LIMITS `max_lines` may be below the ratchet to enforce tighter bounds. A separate data file keeps the concerns separate.

4. **Moving LIMITS to the baseline JSON**: The module-size LIMITS are independent from the test-inventory baseline. Mixing them would create coupling between unrelated enforcement mechanisms.

## Workflow-file changes (not implemented in this ADR)

The brief notes that `.github/workflows/` changes require the `workflow` OAuth scope. Potential changes that could move to `scripts/`:
- Workflow file validation logic (e.g., `check_workflow_pin.py` could be extended)
- Workflow configuration generation (e.g., `check_required_checks_sync.py`'s reverse operation)
- Branch protection application (partially addressed by `apply_branch_protection.py`)

These are tracked separately and are NOT in scope for this ADR.
