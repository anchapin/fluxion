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
