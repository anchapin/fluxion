//! Physics scratch buffer pooling for thermal model timestep solvers.
//!
//! Issue #1966: Store `PhysicsScratch*` buffers as reusable fields in
//! `ThermalModelData` (via `PhysicsScratchPool`) to eliminate per-timestep
//! heap allocations. Each timestep, `fill_zero()` reuses the existing
//! capacity and just zero-fills in-place.

pub(crate) struct PhysicsScratch5r1c {
    pub num_zones: usize,
    pub phi_ia: Vec<f64>,
    pub phi_st: Vec<f64>,
    pub phi_m: Vec<f64>,
    pub t_i_act: Vec<f64>,
    pub t_s_act: Vec<f64>,
    pub new_mass: Vec<f64>,
    pub wall_surface_new: Vec<f64>,
    pub wall_surface_correction: Vec<f64>,
}

impl PhysicsScratch5r1c {
    pub fn new(num_zones: usize) -> Self {
        Self {
            num_zones,
            phi_ia: vec![0.0; num_zones],
            phi_st: vec![0.0; num_zones],
            phi_m: vec![0.0; num_zones],
            t_i_act: vec![0.0; num_zones],
            t_s_act: vec![0.0; num_zones],
            new_mass: vec![0.0; num_zones],
            wall_surface_new: vec![0.0; num_zones],
            wall_surface_correction: vec![0.0; num_zones],
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
    pub phi_ia: Vec<f64>,
    pub phi_st: Vec<f64>,
    pub phi_m_env: Vec<f64>,
    pub phi_m_int: Vec<f64>,
    pub ground_coeff: Vec<f64>,
    pub den: Vec<f64>,
    pub num_rest: Vec<f64>,
    pub t_i_act: Vec<f64>,
    pub t_s: Vec<f64>,
    pub new_env: Vec<f64>,
    pub new_int: Vec<f64>,
}

impl PhysicsScratch6r2c {
    pub fn new(num_zones: usize) -> Self {
        Self {
            num_zones,
            phi_ia: vec![0.0; num_zones],
            phi_st: vec![0.0; num_zones],
            phi_m_env: vec![0.0; num_zones],
            phi_m_int: vec![0.0; num_zones],
            ground_coeff: vec![0.0; num_zones],
            den: vec![0.0; num_zones],
            num_rest: vec![0.0; num_zones],
            t_i_act: vec![0.0; num_zones],
            t_s: vec![0.0; num_zones],
            new_env: vec![0.0; num_zones],
            new_int: vec![0.0; num_zones],
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
    pub inter: Vec<f64>,
    pub phi_ia: Vec<f64>,
    pub phi_st: Vec<f64>,
    pub phi_m: Vec<f64>,
    pub t_i_free: Vec<f64>,
    pub hvac: Vec<f64>,
    pub t_i_act: Vec<f64>,
    pub new_mass: Vec<f64>,
}

impl PhysicsScratch9r4c {
    const NSLOTS: usize = 7;

    pub fn new(num_zones: usize) -> Self {
        Self {
            n: num_zones,
            inter: vec![0.0; num_zones * Self::NSLOTS],
            phi_ia: vec![0.0; num_zones],
            phi_st: vec![0.0; num_zones],
            phi_m: vec![0.0; num_zones],
            t_i_free: vec![0.0; num_zones],
            hvac: vec![0.0; num_zones],
            t_i_act: vec![0.0; num_zones],
            new_mass: vec![0.0; num_zones],
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

use std::cell::RefCell;

#[allow(dead_code)]
pub(crate) struct PhysicsScratchPool {
    #[allow(dead_code)]
    pub r5r1c: RefCell<Option<PhysicsScratch5r1c>>,
    #[allow(dead_code)]
    pub r6r2c: RefCell<Option<PhysicsScratch6r2c>>,
    #[allow(dead_code)]
    pub r9r4c: RefCell<Option<PhysicsScratch9r4c>>,
}

impl PhysicsScratchPool {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            r5r1c: RefCell::new(None),
            r6r2c: RefCell::new(None),
            r9r4c: RefCell::new(None),
        }
    }

    #[allow(dead_code)]
    pub fn get_5r1c(&self, num_zones: usize) -> RefMut<'_, PhysicsScratch5r1c> {
        if self.r5r1c.borrow().is_none() {
            *self.r5r1c.borrow_mut() = Some(PhysicsScratch5r1c::new(num_zones));
        }
        RefCell::borrow_mut(&self.r5r1c)
    }

    #[allow(dead_code)]
    pub fn get_6r2c(&self, num_zones: usize) -> RefMut<'_, PhysicsScratch6r2c> {
        if self.r6r2c.borrow().is_none() {
            *self.r6r2c.borrow_mut() = Some(PhysicsScratch6r2c::new(num_zones));
        }
        RefCell::borrow_mut(&self.r6r2c)
    }

    #[allow(dead_code)]
    pub fn get_9r4c(&self, num_zones: usize) -> RefMut<'_, PhysicsScratch9r4c> {
        if self.r9r4c.borrow().is_none() {
            *self.r9r4c.borrow_mut() = Some(PhysicsScratch9r4c::new(num_zones));
        }
        RefCell::borrow_mut(&self.r9r4c)
    }
}

impl Default for PhysicsScratchPool {
    fn default() -> Self {
        Self::new()
    }
}
