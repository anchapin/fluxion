//! Physics scratch buffer pooling for thermal model timestep solvers.
//!
//! Issue #1966: Store `PhysicsScratch*` buffers as reusable fields in
//! `ThermalModelData` (via `PhysicsScratchPool`) to eliminate per-timestep
//! heap allocations. Each timestep, `fill_zero()` reuses the existing
//! capacity and just zero-fills in-place.
//!
//! Issue #2687: every scratch field is now a `SmallVec<[f64; 4]>`, so
//! `PhysicsScratch*rYc::new(num_zones)` performs **no heap allocation** for
//! `num_zones <= 4` (the common BatchOracle / ASHRAE 140 single- and
//! few-zone case). The fields are scratch buffers — no arithmetic lives
//! here — so changing the container type is bit-identical to the prior
//! `Vec<f64>`. The `9r4c` `inter` buffer is `num_zones * 7`, which exceeds
//! the inline capacity for ≥1 zone and spills transparently (still correct,
//! just heap-backed).
//!
//! Issue #2756: `PhysicsScratchPool` is now *live* in the `step_physics_*`
//! hot path. Each step performs `checkout_*` (owned take from the pool;
//! allocates only on the very first timestep) → `fill_zero()` (resizes every
//! field back to `num_zones`, reproducing the post-`new(num_zones)` state)
//! → … physics … → `return_*` (restore the owned scratch into the pool).
//! The owned checkout/return pair is what lets the scratch coexist with
//! `&mut self` method calls (e.g. `self.prepare_solvers_and_sol_air`) — a
//! borrowed `&mut scratch` held across the whole step re-introduces the
//! borrow conflict that sank the #1436 WIP and forced #1966 to construct
//! locally. Steady-state (timestep ≥ 2) is now allocation-free for the
//! scratch buffers across all of 5R1C / 6R2C / 9R4C.

use smallvec::SmallVec;

/// Inline capacity shared with [`crate::physics::cta::VectorField`] so a
/// scratch field and the VectorField it feeds stay on the stack together.
const SCRATCH_INLINE_CAPACITY: usize = 4;

pub(crate) struct PhysicsScratch5r1c {
    pub num_zones: usize,
    pub phi_ia: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_st: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_m: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub t_i_act: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub t_s_act: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub new_mass: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub wall_surface_new: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub wall_surface_correction: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
}

impl PhysicsScratch5r1c {
    pub fn new(num_zones: usize) -> Self {
        Self {
            num_zones,
            phi_ia: SmallVec::from_elem(0.0, num_zones),
            phi_st: SmallVec::from_elem(0.0, num_zones),
            phi_m: SmallVec::from_elem(0.0, num_zones),
            t_i_act: SmallVec::from_elem(0.0, num_zones),
            t_s_act: SmallVec::from_elem(0.0, num_zones),
            new_mass: SmallVec::from_elem(0.0, num_zones),
            wall_surface_new: SmallVec::from_elem(0.0, num_zones),
            wall_surface_correction: SmallVec::from_elem(0.0, num_zones),
        }
    }

    /// Resize every field back to `num_zones` and zero-fill in place.
    ///
    /// Called by `step_physics_*` right after [`PhysicsScratchPool::checkout_5r1c`]
    /// to reproduce the exact post-`new(num_zones)` state. Reusing a pooled
    /// scratch without this would panic: several fields are emptied by
    /// `mem::take` during the previous step and then written by `[i] = …`.
    pub fn fill_zero(&mut self) {
        let n = self.num_zones;
        self.phi_ia.resize(n, 0.0);
        self.phi_st.resize(n, 0.0);
        self.phi_m.resize(n, 0.0);
        self.t_i_act.resize(n, 0.0);
        self.t_s_act.resize(n, 0.0);
        self.new_mass.resize(n, 0.0);
        self.wall_surface_new.resize(n, 0.0);
        self.wall_surface_correction.resize(n, 0.0);
        for v in &mut self.phi_ia {
            *v = 0.0;
        }
        for v in &mut self.phi_st {
            *v = 0.0;
        }
        for v in &mut self.phi_m {
            *v = 0.0;
        }
        for v in &mut self.t_i_act {
            *v = 0.0;
        }
        for v in &mut self.t_s_act {
            *v = 0.0;
        }
        for v in &mut self.new_mass {
            *v = 0.0;
        }
        for v in &mut self.wall_surface_new {
            *v = 0.0;
        }
        for v in &mut self.wall_surface_correction {
            *v = 0.0;
        }
    }
}

pub(crate) struct PhysicsScratch6r2c {
    pub num_zones: usize,
    pub phi_ia: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_st: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_m_env: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_m_int: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub ground_coeff: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub den: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub num_rest: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub t_i_act: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub t_s: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub new_env: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub new_int: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
}

impl PhysicsScratch6r2c {
    pub fn new(num_zones: usize) -> Self {
        Self {
            num_zones,
            phi_ia: SmallVec::from_elem(0.0, num_zones),
            phi_st: SmallVec::from_elem(0.0, num_zones),
            phi_m_env: SmallVec::from_elem(0.0, num_zones),
            phi_m_int: SmallVec::from_elem(0.0, num_zones),
            ground_coeff: SmallVec::from_elem(0.0, num_zones),
            den: SmallVec::from_elem(0.0, num_zones),
            num_rest: SmallVec::from_elem(0.0, num_zones),
            t_i_act: SmallVec::from_elem(0.0, num_zones),
            t_s: SmallVec::from_elem(0.0, num_zones),
            new_env: SmallVec::from_elem(0.0, num_zones),
            new_int: SmallVec::from_elem(0.0, num_zones),
        }
    }

    /// Resize every field back to `num_zones` and zero-fill in place. See
    /// [`PhysicsScratch5r1c::fill_zero`] for the rationale.
    pub fn fill_zero(&mut self) {
        let n = self.num_zones;
        self.phi_ia.resize(n, 0.0);
        self.phi_st.resize(n, 0.0);
        self.phi_m_env.resize(n, 0.0);
        self.phi_m_int.resize(n, 0.0);
        self.ground_coeff.resize(n, 0.0);
        self.den.resize(n, 0.0);
        self.num_rest.resize(n, 0.0);
        self.t_i_act.resize(n, 0.0);
        self.t_s.resize(n, 0.0);
        self.new_env.resize(n, 0.0);
        self.new_int.resize(n, 0.0);
        for v in &mut self.phi_ia {
            *v = 0.0;
        }
        for v in &mut self.phi_st {
            *v = 0.0;
        }
        for v in &mut self.phi_m_env {
            *v = 0.0;
        }
        for v in &mut self.phi_m_int {
            *v = 0.0;
        }
        for v in &mut self.ground_coeff {
            *v = 0.0;
        }
        for v in &mut self.den {
            *v = 0.0;
        }
        for v in &mut self.num_rest {
            *v = 0.0;
        }
        for v in &mut self.t_i_act {
            *v = 0.0;
        }
        for v in &mut self.t_s {
            *v = 0.0;
        }
        for v in &mut self.new_env {
            *v = 0.0;
        }
        for v in &mut self.new_int {
            *v = 0.0;
        }
    }
}

pub(crate) struct PhysicsScratch9r4c {
    pub n: usize,
    pub inter: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_ia: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_st: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub phi_m: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub t_i_free: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub hvac: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub t_i_act: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
    pub new_mass: SmallVec<[f64; SCRATCH_INLINE_CAPACITY]>,
}

impl PhysicsScratch9r4c {
    const NSLOTS: usize = 7;

    pub fn new(num_zones: usize) -> Self {
        Self {
            n: num_zones,
            inter: SmallVec::from_elem(0.0, num_zones * Self::NSLOTS),
            phi_ia: SmallVec::from_elem(0.0, num_zones),
            phi_st: SmallVec::from_elem(0.0, num_zones),
            phi_m: SmallVec::from_elem(0.0, num_zones),
            t_i_free: SmallVec::from_elem(0.0, num_zones),
            hvac: SmallVec::from_elem(0.0, num_zones),
            t_i_act: SmallVec::from_elem(0.0, num_zones),
            new_mass: SmallVec::from_elem(0.0, num_zones),
        }
    }

    /// Resize every field back to `num_zones` and zero-fill in place. See
    /// [`PhysicsScratch5r1c::fill_zero`] for the rationale. The `inter`
    /// buffer is resized to `num_zones * 7`.
    pub fn fill_zero(&mut self) {
        let n = self.n;
        self.inter.resize(n * Self::NSLOTS, 0.0);
        self.phi_ia.resize(n, 0.0);
        self.phi_st.resize(n, 0.0);
        self.phi_m.resize(n, 0.0);
        self.t_i_free.resize(n, 0.0);
        self.hvac.resize(n, 0.0);
        self.t_i_act.resize(n, 0.0);
        self.new_mass.resize(n, 0.0);
        for v in &mut self.inter {
            *v = 0.0;
        }
        for v in &mut self.phi_ia {
            *v = 0.0;
        }
        for v in &mut self.phi_st {
            *v = 0.0;
        }
        for v in &mut self.phi_m {
            *v = 0.0;
        }
        for v in &mut self.t_i_free {
            *v = 0.0;
        }
        for v in &mut self.hvac {
            *v = 0.0;
        }
        for v in &mut self.t_i_act {
            *v = 0.0;
        }
        for v in &mut self.new_mass {
            *v = 0.0;
        }
    }

    #[inline]
    pub fn t_sol_air(&self) -> &[f64] {
        &self.inter[0..self.n]
    }
    #[inline]
    pub fn pg_wall(&self) -> &[f64] {
        &self.inter[self.n..(2 * self.n)]
    }
    #[inline]
    pub fn pg_roof(&self) -> &[f64] {
        &self.inter[(2 * self.n)..(3 * self.n)]
    }
    #[inline]
    pub fn pg_floor(&self) -> &[f64] {
        &self.inter[(3 * self.n)..(4 * self.n)]
    }
    #[inline]
    pub fn pm_wall(&self) -> &[f64] {
        &self.inter[(4 * self.n)..(5 * self.n)]
    }
    #[inline]
    pub fn pm_roof(&self) -> &[f64] {
        &self.inter[(5 * self.n)..(6 * self.n)]
    }
    #[inline]
    pub fn pm_floor(&self) -> &[f64] {
        &self.inter[(6 * self.n)..(7 * self.n)]
    }

    #[inline]
    pub fn t_sol_air_mut(&mut self) -> &mut [f64] {
        &mut self.inter[0..self.n]
    }
    #[inline]
    pub fn pg_wall_mut(&mut self) -> &mut [f64] {
        &mut self.inter[self.n..(2 * self.n)]
    }
    #[inline]
    pub fn pg_roof_mut(&mut self) -> &mut [f64] {
        &mut self.inter[(2 * self.n)..(3 * self.n)]
    }
    #[inline]
    pub fn pg_floor_mut(&mut self) -> &mut [f64] {
        &mut self.inter[(3 * self.n)..(4 * self.n)]
    }
    #[inline]
    pub fn pm_wall_mut(&mut self) -> &mut [f64] {
        &mut self.inter[(4 * self.n)..(5 * self.n)]
    }
    #[inline]
    pub fn pm_roof_mut(&mut self) -> &mut [f64] {
        &mut self.inter[(5 * self.n)..(6 * self.n)]
    }
    #[inline]
    pub fn pm_floor_mut(&mut self) -> &mut [f64] {
        &mut self.inter[(6 * self.n)..(7 * self.n)]
    }
}

/// Per-instance pool of reusable `PhysicsScratch*rYc` buffers.
///
/// Lives on `ThermalModelData::scratch_pool` and is checked out / returned
/// by the `step_physics_*` hot path (Issue #2756). The pool is **not**
/// cloned — `ThermalModelData::clone()` gets a fresh empty pool (cloning is
/// a cold-path operation that does not need scratch reuse).
///
/// The checkout/return API returns *owned* scratch (not `&mut`) so the
/// caller can hold the scratch as a local variable across `&mut self`
/// method calls without the borrow-checker conflict that sank the #1436 WIP
/// and forced #1966 to construct locally (see module docs).
pub(crate) struct PhysicsScratchPool {
    pub r5r1c: Option<PhysicsScratch5r1c>,
    pub r6r2c: Option<PhysicsScratch6r2c>,
    pub r9r4c: Option<PhysicsScratch9r4c>,
}

impl PhysicsScratchPool {
    pub fn new() -> Self {
        Self {
            r5r1c: None,
            r6r2c: None,
            r9r4c: None,
        }
    }

    /// Take the 5R1C scratch out of the pool (owned). Allocates only when the
    /// pool is empty (first timestep); subsequent calls reuse the buffer that
    /// [`PhysicsScratchPool::return_5r1c`] restored on the previous step.
    /// The caller MUST call `fill_zero()` before indexing into any field —
    /// several fields are emptied by `mem::take` during the previous step.
    /// The caller MUST return the scratch via `return_5r1c` on every exit
    /// path (including early return), else the pool re-allocates next call.
    pub fn checkout_5r1c(&mut self, num_zones: usize) -> PhysicsScratch5r1c {
        self.r5r1c
            .take()
            .unwrap_or_else(|| PhysicsScratch5r1c::new(num_zones))
    }

    /// Restore a checked-out 5R1C scratch into the pool for reuse on the next
    /// timestep. Cheap (one `Option` write); does not reallocate.
    pub fn return_5r1c(&mut self, scratch: PhysicsScratch5r1c) {
        self.r5r1c = Some(scratch);
    }

    /// Take the 6R2C scratch out of the pool (owned). See
    /// [`PhysicsScratchPool::checkout_5r1c`].
    pub fn checkout_6r2c(&mut self, num_zones: usize) -> PhysicsScratch6r2c {
        self.r6r2c
            .take()
            .unwrap_or_else(|| PhysicsScratch6r2c::new(num_zones))
    }

    /// Restore a checked-out 6R2C scratch. See
    /// [`PhysicsScratchPool::return_5r1c`].
    pub fn return_6r2c(&mut self, scratch: PhysicsScratch6r2c) {
        self.r6r2c = Some(scratch);
    }

    /// Take the 9R4C scratch out of the pool (owned). See
    /// [`PhysicsScratchPool::checkout_5r1c`].
    pub fn checkout_9r4c(&mut self, num_zones: usize) -> PhysicsScratch9r4c {
        self.r9r4c
            .take()
            .unwrap_or_else(|| PhysicsScratch9r4c::new(num_zones))
    }

    /// Restore a checked-out 9R4C scratch. See
    /// [`PhysicsScratchPool::return_5r1c`].
    pub fn return_9r4c(&mut self, scratch: PhysicsScratch9r4c) {
        self.r9r4c = Some(scratch);
    }
}

impl Default for PhysicsScratchPool {
    fn default() -> Self {
        Self::new()
    }
}
