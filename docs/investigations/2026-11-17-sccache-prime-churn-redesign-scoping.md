# sccache prime churn — redesign scoping (2026-11-17, loop round 152)

Scoping only, per Alex's release ("do the arm retirement and scope the prime redesign", 2026-11-16 24:19 ET): what the nightly sccache prime actually caches, why the entry pile rewrites daily, and the ranked candidate fixes. Nothing in this document was acted on — no workflow edit, no cron change; the fix itself waits for Alex's word on the design.
Store state at measurement (2026-11-17 05:20 UTC, GitHub cache/usage endpoint): 11.37 GB / 2,750 caches vs the 10 GB cap — over cap, LRU eviction active, watch mode.
The store's count problem and its size problem are different piles: the sccache shards dominate COUNT, the per-workflow cargo archives dominate SIZE.
Churn engine (measured): the sccache GHA backend writes one cache entry per compilation (~0.6 MiB avg); with the store over cap the LRU sheds never-restored entries, and the next build that needs them recompiles and re-saves them as brand-new entries — the pile perpetually rewrites itself.
Measurement window: entries created Nov 15 (761 surviving, all later accessed) and Nov 16 (1,384 created, zero accessed yet at census time); the Nov 16 06:00Z schedule prime did NOT fire (scheduler miss — the same day the nightly β-soak gauge also silently missed), so the Nov 16 writes came from the workflow_run-triggered shard builds on a PR head (02:35–04:01Z), confirming the churn is repo-wide, not prime-only.
Ranked fixes below: stop the date-keyed bleed (cheap, trivial risk), consolidate the sccache pile into one rotating archive entry (main lever), leaner prime scope (helps, not sufficient alone), composite keys (weak alone), weekly prime (content-addressed hits don't decay with age). Estimated post-fix steady state ~9–9.5 GB, under the cap, ~80–120 entries.
Cheap first move recorded, not acted: the `cache-trivy-<date>` pile is the only date-keyed family (~95 MiB/day of orphan feed, first recorded r149) — a stable/weekly-rotating key stops it in one workflow line.

## 1. What the 06:00Z prime actually caches

- Workflow: `.github/workflows/ashrae-shard.yml` ("ASHRAE 140 Sharded Validation", Issue #4007). Job `cache-prime`, trigger `schedule: cron '0 6 * * *'`, checkout `ref: develop`, then `cargo build --release --tests` (all test targets, release profile) with sccache enabled.
- sccache is enabled by DEFAULT: `.github/actions/setup-rust-env/action.yml` input `sccache` defaults `'true'` (sets `RUSTC_WRAPPER=sccache`, `SCCACHE_GHA_ENABLED=true`; when on, the `target/` directory is NOT cached separately). Every job in every workflow that uses this composite therefore compiles through sccache's GitHub Actions cache backend — there is no opt-out line anywhere in `.github/workflows/` (grep for `sccache:` inputs shows only the composite's own default).
- Backend granularity: sccache's GHA backend stores ONE cache entry per compilation unit, content-addressed as `sccache/<40-hex-hash>`. Measured: 2,145 live entries, 1.281 GiB total, **0.61 MiB average** — a count-dominant, size-light pile.
- Purpose: warm the compile cache so the next day's six ASHRAE 140 shards (`cargo test --test all_tests <filter> --release`, each behind a `needs: [build]` warm-build job on PRs) skip cold compilation of the dependency graph.

## 2. Measured churn over the most recent night (census 2026-11-17 05:20Z)

| Family | Entries | Size | Notes |
|---|---|---|---|
| `sccache/*` | 2,145 | 1.281 GiB | avg 0.61 MiB; created Nov 15: 761 (all later accessed = hit by later builds), Nov 16: 1,384 (zero accessed yet) |
| non-sccache | 82 | 8.645 GiB | the size bulk: `Linux-cargo-nightly-hvac-*` 1.370 GiB, `Linux-cargo-nightly-ashrae-*` 1.068 GiB (both keyed on the rustc hash `0361a2fd…`), `v0-rust-*` (Swatinem rust-cache) ~0.6 GiB each, `surrogate-drift-gate-*` 0.611 GiB, `buildkit-blob-*` 59 entries / 2.199 GiB (docker buildx `type=gha` blobs) |
| `cache-trivy-*` + `trivy-*` | 2 | 0.134 GiB | the only date-keyed family; the pile is small at census time precisely because eviction eats the previous day's entry |

Caveats on the numbers: the REST list API returned 2,227 rows against the usage endpoint's 2,750 caches — the list caps, so the per-entry table above covers the listed subset; sizes are as listed. The r149 measurement (2,856 sccache entries in one day) and the r150 cohort deletion (90 tag-scope buildkit entries / 2.908 GiB) are the recorded history this census sits after.

Delivery reality of the prime itself: the Nov 15 fire ran at 13:31Z on a `0 6 * * *` cron (≈7.5 h late — GitHub scheduler drift) and succeeded; the Nov 16 fire did not run at all (same scheduler miss that silently skipped the nightly β-soak gauge that day). The pile's Nov 16 growth came from the PR-path shard builds — the churn does not depend on the prime firing.

## 3. Why ~2.8k per-object shards rewrite daily

1. The store is over its 10 GB cap (11.37 GB / 2,750 caches). GitHub evicts by last ACCESS (7-day no-access rule + LRU); a freshly saved entry that is never restored is unprotected — it is shed as soon as pressure demands.
2. Each shed `sccache/<hash>` entry is a compilation result. The next build that needs that compilation (the next prime, or any PR-path build job — they all compile through sccache) misses on lookup, recompiles, and RE-SAVES the result under the same content key as a brand-new entry (new entry id, new `created_at`, access clock reset).
3. That re-save is new store pressure, which pushes more coldest entries out — including the previous night's own shards — so the next build misses again. The pile is a self-sustaining rewrite loop: ~2.8k entries/day (r149 measurement), 1,384 of them in the last 24 h at this census, with the nightly prime's intended role (warm once, rest warm) defeated by the eviction it itself feeds.
4. The count pressure is what trips the cap-keeping machinery; the size pressure sits in the non-sccache families (8.6 GiB in 82 entries). Any redesign that only trims SIZE (e.g. deleting buildkit cohorts) does not stop the daily rewrite; any redesign that only trims COUNT does not free much space (the pile is 1.3 GiB of 11.37). The two levers are independent.

## 4. Candidate fixes, ranked (scoping only — none acted on)

### 4.1 Stop the date-keyed bleed — cheap first move (recorded, not acted)

`cache-trivy-<date>` keys a new entry every day; the previous day's entry becomes orphan feed for the LRU (~95 MiB/day, first recorded in r149). Fix: rotate the date component weekly (or use `key: cache-trivy-<week>` with a `restore-keys:` fallback), so the DB refreshes at a bounded cadence and orphans stop being minted daily. Expected effect: removes the one purely date-driven growth family. Risk to hit rates: effectively none — trivy DB staleness is bounded by the rotation period by construction. Store delta: −~0.7 GiB/week of pressure, a handful of entries. One workflow line.

### 4.2 Consolidate the sccache pile into one rotating archive entry — the main lever

Replace the per-compilation GHA backend entries with a single archive entry restored/saved by the cache action (`key: sccache-archive-<date>` + `restore-keys: sccache-archive-`), i.e. treat sccache's local store (~1.3 GiB) the way Swatinem/rust-cache already treats `target/`: one entry per day, yesterday's restored via prefix, today's saved under the rotated key. Expected effect: entry count for the pile drops 2,145 → ≤7 (one per rotation day, old ones age out); the daily rewrite loop stops because a restore of yesterday's archive re-warms every compilation it contains in one lookup, and same-day rebuilds hit in-archive; store size from the pile stays ~1.3 GiB but stops churning entry ids. Risk to hit rates: low — identical content, one bulk restore instead of thousands of per-key lookups (also faster: one archive transfer vs 2.8k API round-trips). The rotation is required (a same-key save is deduplicated by GitHub — the archive would freeze), so old entries do become LRU food; bounded by the rotation cadence instead of unbounded per-object churn. Implementation site: the composite action's sccache branch — a workflow/composite change, hence out of this round's scope.

### 4.3 Leaner prime scope — helps, not sufficient alone

`cache-prime` runs `cargo build --release --tests` (every test harness binary, 78 of them) while the six shards only ever run `--test all_tests`. Priming `--test all_tests` writes strictly fewer compilations. Expected effect: fewer entries minted per night by the prime itself. Risk to hit rates: none for the shard path (its objects are still warmed); PR-path builds needing other binaries compile them locally with the lib graph already warm — and linking was never sccache-cached anyway. Caveat: because sccache is default-on for every job, PR builds mint entries too (the Nov 16 growth was PR-path writes), so priming scope alone does not bound the pile — combine with 4.2.

### 4.4 Multi-object composite keys — weak alone

Keep per-object entries but make each key composite over its exact inputs so unchanged compilations dedup to yesterday's entry (same-key saves are skipped, no new entry minted). Expected effect: only materializes when develop is bit-identical day-over-day — any source change re-saves the affected graph regardless. Risk: low, but the content-addressed sccache key already IS the composite; wrapping it adds a layer without changing the miss behavior. Not recommended as the primary fix.

### 4.5 Schedule changes — prime weekly, not daily

sccache hits are content-addressed: an entry's hit-ability does not decay with age, only with input invalidation. A weekly prime mints 1/7 the entries while keeping hits for everything unchanged. Expected effect: churn 2.8k/day → ~2.8k/week, and only until 4.2 removes the pile. Risk: the first build after a mid-week dependency bump runs its new graph cold (the PR-path build job warms it for the shards anyway). Note: this is a cron change — explicitly out of scope this round; listed only for the design decision.

### 4.6 Adjacent size levers (separate decision, recorded not acted)

The size bulk is not sccache: `Linux-cargo-nightly-*` 2.44 GiB (two entries, stable rustc-hash keys), `v0-rust-*` rust-cache family ~2.3 GiB across entries, 59 `buildkit-blob-*` docker blobs 2.199 GiB on the develop scope (the tag-scope cohort was already deleted with Alex's approval in r150; these 59 sit on develop and regenerate on the next docker build — deletion is Alex's call per the standing rule). Trimming these frees real gigabytes but does not touch the count churn of §3.

## 5. Estimated post-fix store vs the 10 GB cap

With 4.1 + 4.2 landed: the sccache pile holds ~1.3 GiB in ≤7 entries; the daily mint of ~2.8k entries and their eviction-driven re-saves stops; LRU pressure drops to the point where the stable families (cargo caches, rust-cache, trivy) stop being shed and re-uploaded — which is itself size churn (each shed 1.37 GiB cargo archive is re-fetched and re-saved in full on the next needing job). Steady-state estimate: ~9–9.5 GB total, roughly 80–120 entries, under the cap with headroom for nightly variation. Without 4.2, no combination of 4.3/4.5 bounds the pile: PR-path writes alone keep minting entries while over cap.

## 6. Scope note

This document is the deliverable. No workflow file was edited, no cron changed, no cache was created or deleted, and the storage-limit setting was not touched (watch mode continues; the cap stays Alex's call). The redesign itself waits for Alex's word on the design chosen.

## 7. Landed (2026-11-17, loop round 153 — Alex's release "1. Land the prime redesign", 18:36 ET)

Alex released the landing. Executed in PR form under the loop's full discipline; §4.2 is the lever this section executes.

**§4.2 — consolidated rotating archive entry (implemented).** One site: `.github/actions/setup-rust-env/action.yml` (every job in every workflow inherits it; sccache is referenced nowhere else in the tree). The per-object GHA backend is off: `SCCACHE_GHA_ENABLED=true` is no longer set. Instead the composite computes a daily rotation key, points sccache at a bounded local disk store (`SCCACHE_DIR=$RUNNER_TEMP/sccache-local-store`, `SCCACHE_CACHE_SIZE=3G` so the archive stays ~pile-sized), and restores/saves it through `actions/cache@v4` (the same SHA pin the composite already uses for the cargo-registry cache — whose saves landing in the store are the in-repo proof that a composite-internal cache action's post phase fires) under `sccache-archive-<OS>-<YYYYMMDD>` with prefix `sccache-archive-<OS>-`. One entry per day+OS replaces the per-object pile; a restore re-warms every compilation it contains in one lookup; same-day saves by later jobs deduplicate (first finisher wins). mozilla-actions/sccache-action and its 3-attempt retry, the `sccache: 'false'` target/ fallback, and the cargo-registry cache are untouched. No gate weakened: restore-keys preserve cross-day warm starts, and the validation receipt below is the gate's own evidence.

**§4.1 — trivy date-key bleed (already-achieved state; no edit).** The current tree carries no `cache-trivy-*` cache key anywhere (git grep across `.github/` and tracked files: zero rows; the last workflow line defining one left with the 2026-09-28 hardening pass aaff8f72). The live census (2026-11-17 23:40Z) holds zero trivy entries — the pile self-cleared via LRU and nothing re-mints it. Recorded rather than edited.

**Store at execution (census 2026-11-17 23:40Z, before landing):** 9.94 GB (usage endpoint, vs the 10 GB cap — under cap after the LRU shed since this document's 05:20Z measurement), 2,977 listed entries all develop-scope: sccache shards 2,960 / 1.78 GB, plus the stable per-workflow families. Zero buildkit-blob entries (the r152-noted 59 / 2.199 GB develop-scope cohort is already LRU-evicted — recorded, nothing deleted), zero trivy entries, zero `ashrae140-weather-*` at census time (re-seeded 23:43Z, run 44201849276, exact key back in develop scope).

**Validation receipt (the gate's own evidence, r149-style).** After merge: one dispatch of ashrae-shard.yml on develop. Required observations: the prime job's archive restore/save behavior; the six shard jobs restore-hitting the same-day archive (restore receipt) and skipping duplicate saves (save-nothing receipt); the census delta ≈ +1 entry with zero new `sccache/*` per-object entries. If the consolidated entry cannot restore without per-object granularity, the PR records why and reverts rather than leaving builds cold.
