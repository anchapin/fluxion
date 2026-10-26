//! Per-surface multi-node conduction with distributed (per-surface,
//! absorptance-weighted) solar absorption.
//!
//! **Issue #3983 — D5 slice 1 of the ADR-0017 "conduction first" migration.**
//!
//! # What this module is
//!
//! A *standalone*, **non-default** physics formulation of the building
//! envelope in which
//!
//! * every surface carries **two thermal nodes with their own state** —
//!   an interior **surface node** `T_s,k` and a **depth (mass) node**
//!   `T_m,k` — both integrated as ODE states (backward Euler), *not*
//!   algebraically pinned to a weighted average; and
//! * the solar flux incident on the envelope is **distributed across the
//!   surfaces** by `area × absorptance` (ISO 13790-style nodal
//!   distribution) instead of being injected into a single lumped mass node
//!   behind a fixed split fraction.
//!
//! ```text
//!                 Φ_abs,k (α_k · I_k · A_k, per surface)
//!                            │
//!                            ▼  (sol-air offset ΔT = Φ_abs,k / h_tr_em,k)
//!   T_ext,k ──[h_tr_em,k]──▶ M_k ──[h_tr_ms,k]──▶ S_k ──[h_tr_is,k]──▶ T_air
//!   (exterior)             (depth node)         (surface node)
//!                                               ▲
//!                                               │ Φ_tr,k (glazing-transmitted
//!                                               │   solar, distributed by
//!                                               │   area × interior absorptance)
//! ```
//!
//! The topology is the ISO 13790 5R1C network **per surface** (`R_se+R_1`
//! → mass node → `R_2` → surface node → `R_si` → zone air) with the single
//! algebraic 5R1C surface node replaced by a **stateful** one. That single
//! change is the dynamic order the 600-series peaks need: the lumped 5R1C
//! network has exactly one thermal state per zone, so its interior surface
//! temperature can only follow the (slow) mass node, which is the
//! `#1522` bidirectional-signature limitation documented in
//! `docs/KNOWN_ISSUES.md` §LIMIT-28 (rounds 66–70) and `docs/FAQ.md` Q4.
//!
//! # Why the solar distribution matters
//!
//! The lumped 5R1C network has a single mass node, so *all* absorbed solar
//! (roof + walls + floor) lands in one state behind one fixed split
//! fraction. The per-surface formulation instead apportions the incident
//! flux by `α_k · A_k` and lets each surface's own two-node network decide
//! — dynamically, through its conductances and capacitances — how much of
//! that flux is stored in its depth node and how much reaches the zone.
//! Nothing in the split is a tuned constant: the split is a *consequence*
//! of the per-surface network.
//!
//! # Relationship to the existing modules
//!
//! | Module | Nodes | Role |
//! |---|---|---|
//! | [`crate::physics::five_r1c_solver`] | 1 state / zone (mass) + algebraic surface | lumped reference (unchanged) |
//! | [`crate::physics::multi_node_solver`] (9R4C) | 4 states / zone, **shared** surface node | heavy-mass path (unchanged) |
//! | `fluxion_core::per_surface_conduction` | 2 states / surface, but **shared** `T_sky` / `T_mass` update | substrate reused for [`SurfaceKind`] |
//! | **this module** | 2 states / surface, **per-surface** boundaries + per-surface solar | new, non-default formulation |
//!
//! The `fluxion_core::per_surface_conduction` substrate is *not* duplicated:
//! its [`SurfaceKind`] vocabulary is reused, but its `MassNode::update` /
//! `SurfaceNode::update` helpers drive every surface from a **shared**
//! `T_sky` / `mass_temperature`, which is precisely the lumped split this
//! issue removes — hence the separate stepping code here.
//!
//! # Status / scope
//!
//! This slice delivers the **formulation plus formulation-level evidence
//! only**. It is deliberately *not* wired into the solver dispatch: no
//! production path (`step_dispatcher`, `ThermalSelector`, the 5R1C / 9R4C
//! arms) is touched, and no parameter is tuned. Production wiring and the
//! 600-series scoring are later slices of #3983.
//!
//! # Numerical scheme
//!
//! Each surface is advanced by an **implicit (backward Euler)** solve of the
//! coupled 2-node system, and the zone air node is solved implicitly in the
//! same step through a Schur complement. The scheme is therefore
//! unconditionally stable and **energy conserving to machine precision**
//! (see [`EnergyLedger::residual`]).
//!
//! # References
//!
//! * ISO 13790:2008 §C — 5R1C network, half-insulation rule, sol-air
//!   temperature.
//! * `docs/KNOWN_ISSUES.md` §LIMIT-28, loop rounds 66–70 — the measurements
//!   that established that the *interior* per-surface network carries the
//!   dynamic order the exterior split alone could not produce.
//! * `docs/adr/0017-equation-based-dae-teacher-architecture.md` — D5
//!   migration sequence (conduction → zone air → HVAC).

use fluxion_core::per_surface_conduction::SurfaceKind;

pub mod solar;

#[cfg(test)]
mod tests;

pub use solar::{absorbed_flux, distribute_by_area_absorptance, distribute_transmitted};

/// Stateless description of one envelope surface.
///
/// All conductances are **total** (surface-wide) values in W/K and all
/// capacitances are **total** (surface-wide) values in J/K, so a surface
/// can be built from per-area data via [`SurfaceSpec::from_areal`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceSpec {
    /// Unique surface identifier within the zone.
    pub id: usize,
    /// Surface classification (wall / roof / floor).
    pub kind: SurfaceKind,
    /// Surface area [m²].
    pub area: f64,
    /// Exterior solar absorptance [-] (0..1).
    pub absorptance: f64,
    /// Interior-finish solar absorptance [-] (0..1). Weights the share of
    /// glazing-transmitted solar that lands on this surface's interior face.
    pub interior_absorptance: f64,
    /// Exterior ↔ depth-node conductance [W/K] (exterior film `R_se` plus
    /// the exterior half of the construction resistance `R_1`).
    pub h_tr_em: f64,
    /// Depth-node ↔ surface-node conductance [W/K] (the interior half of the
    /// construction resistance `R_2`).
    pub h_tr_ms: f64,
    /// Surface-node ↔ zone-air conductance [W/K] (interior film `R_si`).
    pub h_tr_is: f64,
    /// Interior surface-node capacitance [J/K].
    pub c_surface: f64,
    /// Depth (mass) node capacitance [J/K].
    pub c_mass: f64,
}

impl SurfaceSpec {
    /// Build a surface spec from ISO 13790-style **per-area** quantities.
    ///
    /// The construction is split with the ISO 13790 §C.3 **half-insulation
    /// rule**: half of the material resistance sits on the exterior side of
    /// the depth node (`R_1`) and half on the interior side (`R_2`). The
    /// material **capacitance** is split the same way (half at the depth
    /// node, half at the interior surface node) — the documented structural
    /// assumption that gives the surface node its own state; it is stated
    /// here rather than hidden in a tuned constant.
    ///
    /// # Arguments
    /// * `id` — surface identifier
    /// * `kind` — surface classification
    /// * `area` — surface area [m²]
    /// * `u_value` — overall transmittance including films [W/(m²·K)]
    /// * `c_areal` — areal heat capacity of the construction [J/(m²·K)]
    /// * `h_exterior` — exterior film coefficient [W/(m²·K)]
    /// * `h_interior` — interior film coefficient [W/(m²·K)]
    /// * `absorptance` — exterior solar absorptance [-]
    #[allow(clippy::too_many_arguments)]
    pub fn from_areal(
        id: usize,
        kind: SurfaceKind,
        area: f64,
        u_value: f64,
        c_areal: f64,
        h_exterior: f64,
        h_interior: f64,
        absorptance: f64,
    ) -> Self {
        let r_total = if u_value > 0.0 { 1.0 / u_value } else { 0.0 };
        let r_se = if h_exterior > 0.0 { 1.0 / h_exterior } else { 0.0 };
        let r_si = if h_interior > 0.0 { 1.0 / h_interior } else { 0.0 };
        let r_material = (r_total - r_se - r_si).max(0.0);
        let r_1 = 0.5 * r_material;
        let r_2 = 0.5 * r_material;

        let h_tr_em = if r_se + r_1 > 0.0 { 1.0 / (r_se + r_1) } else { 0.0 };
        let h_tr_ms = if r_2 > 0.0 { 1.0 / r_2 } else { 0.0 };
        let h_tr_is = if r_si > 0.0 { 1.0 / r_si } else { 0.0 };

        let c_total = c_areal * area;
        Self {
            id,
            kind,
            area,
            absorptance,
            interior_absorptance: absorptance,
            h_tr_em: h_tr_em * area,
            h_tr_ms: h_tr_ms * area,
            h_tr_is: h_tr_is * area,
            c_surface: 0.5 * c_total,
            c_mass: 0.5 * c_total,
        }
    }

    /// Override the interior-finish absorptance (defaults to the exterior
    /// absorptance set at construction).
    pub fn with_interior_absorptance(mut self, alpha_int: f64) -> Self {
        self.interior_absorptance = alpha_int;
        self
    }
}

/// Runtime state of one envelope surface (two ODE states).
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceState {
    /// The surface's stateless construction description.
    pub spec: SurfaceSpec,
    /// Interior surface-node temperature [°C] — ODE state.
    pub t_surface: f64,
    /// Depth (mass) node temperature [°C] — ODE state.
    pub t_mass: f64,
    /// Net heat flow into the surface node during the last step [W].
    pub flux_surface: f64,
    /// Net heat flow into the depth node during the last step [W].
    pub flux_mass: f64,
}

impl SurfaceState {
    /// Interior surface-node temperature [°C].
    pub fn t_surface(&self) -> f64 {
        self.t_surface
    }

    /// Depth (mass) node temperature [°C].
    pub fn t_mass(&self) -> f64 {
        self.t_mass
    }
}

/// Zone air-node parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoneAirSpec {
    /// Air-node heat capacity [J/K].
    pub c_air: f64,
    /// Ventilation + infiltration conductance [W/K].
    pub h_ve: f64,
}

impl ZoneAirSpec {
    /// Build zone air-node parameters.
    pub fn new(c_air: f64, h_ve: f64) -> Self {
        Self { c_air, h_ve }
    }
}

/// Boundary conditions for one implicit step.
#[derive(Debug, Clone, Copy)]
pub struct StepInputs<'a> {
    /// Per-surface exterior temperature [°C], indexed like the surfaces.
    ///
    /// This is the **non-solar** exterior boundary (air / sky / ground); the
    /// absorbed solar is injected separately as a heat flux so that it is
    /// never counted twice.
    pub exterior_temperatures: &'a [f64],
    /// Outdoor air temperature for the ventilation path [°C].
    pub t_outdoor: f64,
    /// Convective + air-side gains [W] (internal gains, air-side solar).
    pub phi_ia: f64,
    /// Per-surface absorbed exterior solar [W], indexed like the surfaces.
    pub absorbed_solar: &'a [f64],
    /// Per-surface glazing-transmitted solar landing on the interior face
    /// [W], indexed like the surfaces.
    pub transmitted_solar: &'a [f64],
}

/// Per-step energy ledger of the whole zone [W].
///
/// The ledger is the first-law bookkeeping of one implicit step; its
/// [`EnergyLedger::residual`] is zero up to floating-point round-off.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyLedger {
    /// Heat exchanged from the envelope surfaces with the zone air [W]
    /// (positive = into the air).
    pub surface_to_air: f64,
    /// Heat exchanged with the outdoor air through ventilation [W]
    /// (positive = into the zone).
    pub ventilation: f64,
    /// Convective + air-side gains [W].
    pub internal: f64,
    /// Heat entering the envelope from the exterior, *including* the absorbed
    /// solar [W] (positive = into the envelope).
    pub exterior_to_envelope: f64,
    /// Glazing-transmitted solar landing on the interior faces [W].
    pub transmitted_solar: f64,
    /// Rate of energy stored in the envelope nodes [W] (positive = charging).
    pub envelope_storage: f64,
    /// Rate of energy stored in the zone air node [W] (positive = charging).
    pub air_storage: f64,
}

impl EnergyLedger {
    /// First-law residual of the step [W] (zero up to round-off).
    pub fn residual(&self) -> f64 {
        let inflow = self.exterior_to_envelope
            + self.transmitted_solar
            + self.ventilation
            + self.internal;
        let storage = self.envelope_storage + self.air_storage;
        inflow - storage
    }
}

/// Per-surface multi-node conduction with distributed solar.
///
/// See the [module documentation](self) for the network topology, the solar
/// distribution rule and the scope of this formulation.
#[derive(Debug, Clone)]
pub struct PerSurfaceMultiNodeConduction {
    surfaces: Vec<SurfaceState>,
    zone: ZoneAirSpec,
    t_air: f64,
    ledger: EnergyLedger,
    dt_last: f64,
}

impl PerSurfaceMultiNodeConduction {
    /// Build the formulation from a set of surfaces.
    ///
    /// All node temperatures are initialised to `t_air` (uniform equilibrium
    /// seed); the first [`PerSurfaceMultiNodeConduction::step`] then relaxes
    /// the network towards the imposed boundary conditions.
    pub fn new(surfaces: Vec<SurfaceSpec>, zone: ZoneAirSpec, t_air: f64) -> Self {
        let states = surfaces
            .into_iter()
            .map(|spec| SurfaceState {
                spec,
                t_surface: t_air,
                t_mass: t_air,
                flux_surface: 0.0,
                flux_mass: 0.0,
            })
            .collect();
        Self {
            surfaces: states,
            zone,
            t_air,
            ledger: EnergyLedger::default(),
            dt_last: 0.0,
        }
    }

    /// Number of surfaces in the formulation.
    pub fn len(&self) -> usize {
        self.surfaces.len()
    }

    /// True when the formulation carries no surface.
    pub fn is_empty(&self) -> bool {
        self.surfaces.is_empty()
    }

    /// Current zone air temperature [°C].
    pub fn t_air(&self) -> f64 {
        self.t_air
    }

    /// The surface states, in construction order.
    pub fn surfaces(&self) -> &[SurfaceState] {
        &self.surfaces
    }

    /// The state of the surface with the given identifier.
    pub fn surface(&self, id: usize) -> Option<&SurfaceState> {
        self.surfaces.iter().find(|s| s.spec.id == id)
    }

    /// Interior surface-node temperatures [°C], in construction order.
    pub fn surface_temperatures(&self) -> Vec<f64> {
        self.surfaces.iter().map(|s| s.t_surface).collect()
    }

    /// Depth-node temperatures [°C], in construction order.
    pub fn mass_temperatures(&self) -> Vec<f64> {
        self.surfaces.iter().map(|s| s.t_mass).collect()
    }

    /// The energy ledger of the most recent step.
    pub fn energy_ledger(&self) -> EnergyLedger {
        self.ledger
    }

    /// First-law residual of the most recent step [W].
    pub fn energy_residual(&self) -> f64 {
        self.ledger.residual()
    }

    /// Timestep of the most recent step [s].
    pub fn dt_last(&self) -> f64 {
        self.dt_last
    }

    /// Advance the formulation by one implicit (backward Euler) step.
    ///
    /// Returns the new zone air temperature [°C].
    ///
    /// The step solves, simultaneously and implicitly:
    ///
    /// ```text
    /// C_m,k · dT_m,k/dt = h_tr_ms,k · (T_s,k − T_m,k)
    ///                     + h_tr_em,k · (T_solair,k − T_m,k)
    /// C_s,k · dT_s,k/dt = h_tr_ms,k · (T_m,k − T_s,k)
    ///                     + h_tr_is,k · (T_air − T_s,k) + Φ_tr,k
    /// C_air  · dT_air/dt = Σ_k h_tr_is,k · (T_s,k − T_air)
    ///                     + h_ve · (T_out − T_air) + φ_ia
    /// ```
    ///
    /// with the per-surface sol-air temperature
    /// `T_solair,k = T_ext,k + Φ_abs,k / h_tr_em,k` — i.e. the absorbed
    /// solar enters at the **exterior face** of that surface, which is where
    /// it is physically absorbed.
    pub fn step(&mut self, dt: f64, inputs: &StepInputs<'_>) -> f64 {
        assert!(dt > 0.0, "step requires a positive timestep, got {dt}");
        let n = self.surfaces.len();

        let t_air_old = self.t_air;
        let c_air = self.zone.c_air;
        let h_ve = self.zone.h_ve;

        // Per-surface elimination: solve the 2x2 node system for
        //   T_s,k = κ_k · T_air(new) + b_k
        // (the Schur complement of the depth node), then use the zone air
        // balance to close the system.
        let mut kappa = vec![0.0f64; n];
        let mut b = vec![0.0f64; n];
        let mut rhs1 = vec![0.0f64; n];
        let mut det = vec![0.0f64; n];
        let mut coef_a = vec![0.0f64; n];
        let mut coef_c = vec![0.0f64; n];
        let mut coef_solair = vec![0.0f64; n];
        let mut phi_tr = vec![0.0f64; n];

        for (k, state) in self.surfaces.iter().enumerate() {
            let spec = &state.spec;
            let h_em = spec.h_tr_em;
            let h_ms = spec.h_tr_ms;
            let h_is = spec.h_tr_is;
            let c_s = spec.c_surface;
            let c_m = spec.c_mass;

            let t_ext = inputs.exterior_temperatures.get(k).copied().unwrap_or(0.0);
            let phi_abs = inputs.absorbed_solar.get(k).copied().unwrap_or(0.0);
            let phi_t = inputs.transmitted_solar.get(k).copied().unwrap_or(0.0);
            phi_tr[k] = phi_t;

            // Sol-air offset: the absorbed solar enters at the exterior face.
            let solair = if h_em > 0.0 { t_ext + phi_abs / h_em } else { t_ext };
            coef_solair[k] = solair;

            // 2x2 system  [[A, −h_ms], [−h_ms, Cc]] · [T_m, T_s]ᵀ = [rhs1, rhs2]ᵀ
            //
            // The depth node is connected to the exterior (h_tr_em) *and* to
            // the surface node (h_tr_ms), so the exterior conductance belongs
            // in the depth node's diagonal.
            let a = c_m / dt + h_ms + h_em;
            let cc = c_s / dt + h_ms + h_is;
            let d = a * cc - h_ms * h_ms;
            let r1 = c_m / dt * state.t_mass + h_em * solair;
            coef_a[k] = a;
            coef_c[k] = cc;
            rhs1[k] = r1;
            det[k] = d;

            if d > 1e-12 {
                kappa[k] = a * h_is / d;
                b[k] = (h_ms * r1 + a * (c_s / dt * state.t_surface + phi_t)) / d;
            } else {
                // Degenerate surface (no conductance and no capacitance):
                // fall back to an explicit, decoupled update.
                kappa[k] = 0.0;
                b[k] = if cc > 0.0 {
                    (c_s / dt * state.t_surface + h_is * t_air_old + phi_t) / cc
                } else {
                    state.t_surface
                };
            }
        }

        // Zone air node (implicit), with the surface temperatures eliminated
        // through the Schur complement.
        let mut denom = c_air / dt + h_ve;
        let mut numer = c_air / dt * t_air_old + h_ve * inputs.t_outdoor + inputs.phi_ia;
        for (k, state) in self.surfaces.iter().enumerate() {
            let h_is = state.spec.h_tr_is;
            denom += h_is * (1.0 - kappa[k]);
            numer += h_is * b[k];
        }
        let t_air_new = if denom > 1e-12 { numer / denom } else { t_air_old };

        // Back-substitution: update the two node states of every surface.
        let mut surface_to_air = 0.0;
        let mut exterior_to_envelope = 0.0;
        let mut envelope_storage = 0.0;
        for (k, state) in self.surfaces.iter_mut().enumerate() {
            let spec = &state.spec;
            let h_em = spec.h_tr_em;
            let h_ms = spec.h_tr_ms;
            let h_is = spec.h_tr_is;
            let c_s = spec.c_surface;
            let c_m = spec.c_mass;

            let t_s_old = state.t_surface;
            let t_m_old = state.t_mass;
            let a = coef_a[k];
            let cc = coef_c[k];
            let d = det[k];
            let r1 = rhs1[k];
            let phi_t = phi_tr[k];

            let (t_s_new, t_m_new) = if d > 1e-12 {
                let r2 = c_s / dt * t_s_old + h_is * t_air_new + phi_t;
                let t_s = (h_ms * r1 + a * r2) / d;
                let t_m = (cc * r1 + h_ms * r2) / d;
                (t_s, t_m)
            } else {
                // Decoupled fallback (see the forward elimination above).
                let t_s = if cc > 0.0 {
                    (c_s / dt * t_s_old + h_is * t_air_new + phi_t) / cc
                } else {
                    t_s_old
                };
                let t_m = if c_m / dt + h_em > 1e-12 {
                    (c_m / dt * t_m_old + h_em * coef_solair[k]) / (c_m / dt + h_em)
                } else {
                    t_m_old
                };
                (t_s, t_m)
            };

            state.t_surface = t_s_new;
            state.t_mass = t_m_new;
            state.flux_surface = c_s / dt * (t_s_new - t_s_old) + h_is * (t_air_new - t_s_new);
            state.flux_mass = c_m / dt * (t_m_new - t_m_old) + h_em * (coef_solair[k] - t_m_new);

            surface_to_air += h_is * (t_s_new - t_air_new);
            exterior_to_envelope += h_em * (coef_solair[k] - t_m_new);
            envelope_storage += c_s / dt * (t_s_new - t_s_old) + c_m / dt * (t_m_new - t_m_old);
        }

        let ventilation = h_ve * (inputs.t_outdoor - t_air_new);
        let air_storage = c_air / dt * (t_air_new - t_air_old);
        let transmitted_total: f64 = phi_tr.iter().sum();

        self.ledger = EnergyLedger {
            surface_to_air,
            ventilation,
            internal: inputs.phi_ia,
            exterior_to_envelope,
            transmitted_solar: transmitted_total,
            envelope_storage,
            air_storage,
        };
        self.t_air = t_air_new;
        self.dt_last = dt;
        t_air_new
    }
}
