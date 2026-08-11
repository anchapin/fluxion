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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

#[allow(dead_code)]
pub(crate) struct PhysicsScratchPool {
    #[allow(dead_code)]
    pub r5r1c: Option<PhysicsScratch5r1c>,
    #[allow(dead_code)]
    pub r6r2c: Option<PhysicsScratch6r2c>,
    #[allow(dead_code)]
    pub r9r4c: Option<PhysicsScratch9r4c>,
}

impl PhysicsScratchPool {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            r5r1c: None,
            r6r2c: None,
            r9r4c: None,
        }
    }

    #[allow(dead_code)]
    pub fn get_5r1c(&mut self, num_zones: usize) -> &mut PhysicsScratch5r1c {
        if self.r5r1c.is_none() {
            self.r5r1c = Some(PhysicsScratch5r1c::new(num_zones));
        }
        self.r5r1c.as_mut().unwrap()
    }

    #[allow(dead_code)]
    pub fn get_6r2c(&mut self, num_zones: usize) -> &mut PhysicsScratch6r2c {
        if self.r6r2c.is_none() {
            self.r6r2c = Some(PhysicsScratch6r2c::new(num_zones));
        }
        self.r6r2c.as_mut().unwrap()
    }

    #[allow(dead_code)]
    pub fn get_9r4c(&mut self, num_zones: usize) -> &mut PhysicsScratch9r4c {
        if self.r9r4c.is_none() {
            self.r9r4c = Some(PhysicsScratch9r4c::new(num_zones));
        }
        self.r9r4c.as_mut().unwrap()
    }
}

impl Default for PhysicsScratchPool {
    fn default() -> Self {
        Self::new()
    }
}
