# LIMIT-45 — Table 5-138 per-class solar routing is the engine default; the round-83 "inert" verdict was an artifact of the recording-only D5 mass state
1. Observation: promoting the ASHRAE 140-2017 Table 5-138 constant interior-distribution fractions at the step_5r1c class solar term (FLX_R91_T5138, flip 3 of the round-91 bands-as-factor audit) re-enters 600/H, 600/C, 640/H and 650/C into their published bands, halves the 610-630 gaps, halves both FF Tmax gaps, and moves 650FF Tmin 0.43 K AWAY from the band edge flip 2 left it 0.03 K from — measured on 60491819 with flag-off bit-identical to the round-91b record (strict checker 0 REGRESSION, output-diff 42/42 at +0.0000%).
2. Mechanism: 5.2.1.12.2 publishes the constant fractions (Floor 0.642, Ceiling 0.1678, E 0.038, W 0.038, N 0.053, S 0.0276 — surface-row sum 0.9664; the 0.035 window-loss row is flip 1's retransmission sink) for programs that do not compute the interior distribution internally; the engine maps them to surface classes and renormalizes over the surface-row sum, so sum_c phi_c = phi_m_i*w + phi_st exactly (energy-conserving by construction; internal-gain mass share untouched — Table 5-138 routes transmitted solar, not internal gains).
3. Why the round-83 discriminator called this inert: at that time the D5 per-surface interior network's mass state was recording-only — the coupled sub-step overrode the hourly class advance with the lumped R76 route, so no routing of phi_st across classes could bind committed state. Since the round-84/85 per-class coupled sub-step default (#4531/#4533) the class drive binds the committed mass state, and the routing is live physics; the round-83 verdict was an artifact of that recording-only state, not a property of the routing.
4. Classification: spec-shaped physics adopted on its own merits under the standing 2026-10-22 criteria (known, verifiable, tested); band movement recorded, not weighted. The ISO 13790 C.3 conductance-share partition it replaces remains reachable via the falsy FLX_R91_T5138 token for A/B archaeology.
5. Evidence: strict table 16 PASS / 26 KNOWN-FAIL / 0 REGRESSION (_doc_round91c); 800/C equality-checksum gap 532.45 -> 1464.37 pp with the LIMIT-44 by-construction equality holding exactly (800 prints 600's 5.962 MWh); 9R4C family and the inert families byte-identical; 39/42 output-diff metrics moved (all low-mass or FF); full battery green (fabric, SS6/SS7 pins + CSVs, hotloop goldens, grid baseline, contracts 14/14 x4 runs, blind exact).
6. Consequence: the free-float cohort floor moves from "no reachable in-network lever" to a measured live routing; the remaining FF Tmax residuals (~15.5-15.8 pp) and the conditioned 610-630 known-fails are recorded operating points for the input-discrepancy sweep Alex chartered, not stopping points.
7. History: first measured as the round-81 partitioning discriminator (air/surface/mass split, refuted in direction), ruled inert in round-83 (recording-only mass state), re-measured live and flipped 2026-10-23 (loop round 91 flip 3, branch physics/r91-t5138).

## Detail

The routing replaces the C.3 conductance-share partition (s_c*phi_st) at the
same single phi_c site the round-83 discriminator instrumented, so the
flag-off route is exactly the pre-flip engine. The fractions are constant
per class — they do not depend on orientation-dependent conductances — which
is why they move the 5R1C family so much more than the conductance-share
split did: the floor class takes 0.642/0.9664 = 66.4% of transmitted solar
regardless of its conductance share, charging the high-capacitance floor
slab directly. The direction of every conditioned-row move is the physics:
more solar stored in mass means less heating energy metered and more stored
heat to remove during cooling hours. The falsy token restores the C.3 arm
bit-identically; nothing else keys on it.

## UPDATE — loop round 93 (2026-10-24, item 1c): the §C.3 lag-site air-share term retires under this route

1. Observation: under the flip-3 default the composed network over-subscribed
   the transmitted pool by the r88 lag-site air-share term
   `(1 − s_share)·h_tr_is·phi_st` — the class drives carry `1.0·phi_st`
   (item 2 above) while the air ODE kept the §C.3-partition air share
   (`step_5r1c.rs:1633-1645`, `FLX_R88_PHIST_MASS_SHARE` default on). The
   round-92 free-float energy-balance audit recorded this as the structural
   candidate over-routing ("recorded, not repaired"); round 93 verified the
   premise and flipped it.

2. Primary sources (round-93 verification, item 1b): ASHRAE 140-2017
   §5.2.1.12.1/2 + Table 5-138 (extracted from `docs/140_2017_a_20200901.pdf`)
   distribute transmitted solar *among interior surfaces* (floor 0.642 /
   ceiling 0.1678 / E 0.038 / W 0.038 / N 0.053 / S 0.0276 — surface-row sum
   0.9664) plus a 0.035 solar-lost-through-windows row — **no air row**. ISO
   13790 Annex C (ISO/FDIS 13790:2007 draft, eq. C.1 / C.10 / C.11) feeds
   `Φst` to the surface node with `Φia = 0,5·Φint` at the air node — no
   direct air-side solar anywhere in §C.3's network. The air-share term is
   therefore the air side of the §C.3 partition *route* (air `(1 − s_share)`
   + mass `s_share` = phi_st exactly, the r88 arm), and it belongs inactive
   wherever the T5138 fractions bind the mass side.

3. Flip (branch physics/r93-t5138-airshare): the lag-site input retires
   ONLY where the T5138 class route binds (same structural conditions as the
   class drive site: per-class arm truthy, class tensors present at the
   zone, positive class areas). Lumped/tensor-less zones keep the intact
   `s_share` partition. Falsy `FLX_R93_T5138_AIRSHARE` (or falsy
   `FLX_R91_T5138`) restores the pre-r93 composed route bit-identically.
   Contract 19 (`t5138_airshare_retired_default_and_falsy_restore`) pins the
   default lag state identically 0 and both falsy restores.

4. Instrumentation (item 1a, PR #4579, merged as 189148a1): `FFAUDIT_Z`
   gained `h_tr_is` + `s_share`. Exact annual air-side term from the live
   dump, formula `Σ (1 − s_share)·h_tr_is·phi_st·0.0036`:
   600FF = 650FF = 3,763,660 MJ/year (identical — shared window geometry,
   consistent with the r92 audit); mean `s_share` 0.389. For scale under the
   r92 book convention `(1 − s_share)·phi_st`: ≈ 22.7 GJ/year case-wide;
   day-18 ≈ 247 MJ vs the r92 harness's +56.22 MJ bucket — the r92 number is
   that harness's own book convention, not the as-consumed ODE input; both
   recorded with formulas. The decisive quantities remain the verified
   premise and the re-record movement below.

5. Re-record (provenance _doc_round93, captured at 3af347be):
   strict 42 rows re-measured on 3af347be: 600/H 4.692 -> 5.462 MWh (gap 0.00 -> 13.56 pp, honestly leaves band), 600/C 5.962 -> 3.965 (0.00 -> 16.59, leaves); 620/C 5.057 -> 3.689 (7.67 -> 0.00, ENTERS) and 630/C 3.653 -> 2.722 (14.38 -> 0.00, ENTERS); 640/H 2.699 -> 3.366 (0.00 -> 17.38, leaves), 640/C 5.962 -> 3.965 (0.00 -> 13.75, leaves), 650/C 4.368 -> 2.701 (0.00 -> 25.85, leaves); 610-630/H widen (11.04->29.57, 30.50->47.48, 35.87->49.98), 610/C 3.22 -> 29.02; 800/C equality-checksum gap 1464.37 -> 935.36 pp (the LIMIT-44 equality holds exactly); 600FF Tmin -12.22 -> -12.35 (PASS holds), Tmax 55.06 -> 49.42 (15.81 -> 24.43 pp); 650FF Tmin -17.16 -> -17.25 (PASS holds, 0.55 K from the edge), Tmax 54.04 -> 48.71 (15.50 -> 23.83 pp); 900FF/950FF and the high-mass 9R4C family byte-identical. Checker: 13 PASS / 29 KNOWN-FAIL / 0 REGRESSION. Fabric case_600_5r1c 4.692/5.962 -> 5.462/3.965 with parity ratios re-recorded. WD-case load pins moved (peaks/occurrences included; solar/sky pins byte-identical), hotloop analytical [45.73,115.14,152.67] -> [64.35,138.55,180.73] and CPU-surrogate [235.94,251.56,263.34] -> [237.36,252.75,264.42] goldens, grid thermal 146.99 -> 148.08 kWh (PV/battery/SoC unchanged). (strict rows old→new with the FF/conditioned moves;
   9R4C family and inert families byte-identical; falsy-token legs
   bit-identical: FLX_R93_T5138_AIRSHARE=1 ≡ the _doc_round91c record —
   strict checker 16/26/0 + output-diff 42/42 — and FLX_R91_T5138=0 at the
   fix tip byte-identical to the same-env develop run).

6. Classification: spec-shaped physics adopted on its own merits under the
   standing 2026-10-22 criteria (known, verifiable, tested); band movement
   recorded, not weighted. History: recorded-not-repaired in the round-92
   audit; verified + flipped 2026-10-24 (loop round 93).
