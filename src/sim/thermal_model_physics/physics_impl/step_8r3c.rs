//! 8R3C physics step implementation for `ThermalModel`.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::thermal_model_core::ThermalModel;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    pub(crate) fn step_physics_8r3c(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        dt_seconds: f64,
    ) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // Get ground temperature at this timestep (unused in simplified 8R3C)
        let _t_g = self
            .0
            .conduction
            .ground_temperature
            .ground_temperature(timestep);

        // Use 5R1C solve for simplicity (Phase 20 evaluation)
        // In a full implementation, this would be a proper 8R3C algebraic system
        let energy = self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds);

        // Update 8R3C mass temperatures using simple relaxation (for evaluation)
        // In a full implementation, these would be coupled with Ti_free calculation
        let t_i = self.0.setpoints.temperatures.clone();

        // Validate 8R3C fields are initialized (precondition for 8R3C physics step)
        let ceiling_mass = self
            .0
            .mass
            .ceiling_mass_temperatures
            .as_mut()
            .expect("ceiling_mass_temperatures must be initialized for 8R3C model");
        let floor_mass = self
            .0
            .mass
            .floor_mass_temperatures
            .as_mut()
            .expect("floor_mass_temperatures must be initialized for 8R3C model");
        let partition_mass = self
            .0
            .mass
            .partition_mass_temperatures
            .as_mut()
            .expect("partition_mass_temperatures must be initialized for 8R3C model");
        let ceiling_cap = self
            .0
            .mass
            .ceiling_thermal_capacitance
            .as_ref()
            .expect("ceiling_thermal_capacitance must be initialized for 8R3C model");
        let floor_cap = self
            .0
            .mass
            .floor_thermal_capacitance
            .as_ref()
            .expect("floor_thermal_capacitance must be initialized for 8R3C model");
        let partition_cap = self
            .0
            .mass
            .partition_thermal_capacitance
            .as_ref()
            .expect("partition_thermal_capacitance must be initialized for 8R3C model");
        let h_tr_ceiling = self
            .0
            .mass
            .h_tr_ceiling
            .as_ref()
            .expect("h_tr_ceiling must be initialized for 8R3C model");
        let h_tr_floor_mass = self
            .0
            .mass
            .h_tr_floor_mass
            .as_ref()
            .expect("h_tr_floor_mass must be initialized for 8R3C model");
        let h_tr_partition = self
            .0
            .mass
            .h_tr_partition
            .as_ref()
            .expect("h_tr_partition must be initialized for 8R3C model");

        // Update ceiling mass temperature
        for i in 0..self.0.hvac.num_zones {
            let dtm_ceiling = (t_i.as_ref()[i] - ceiling_mass.as_ref()[i])
                / (ceiling_cap.as_ref()[i] / (h_tr_ceiling.as_ref()[i] * dt));
            ceiling_mass.as_mut()[i] += dtm_ceiling;
        }

        // Update floor mass temperature
        for i in 0..self.0.hvac.num_zones {
            let dtm_floor = (t_i.as_ref()[i] - floor_mass.as_ref()[i])
                / (floor_cap.as_ref()[i] / (h_tr_floor_mass.as_ref()[i] * dt));
            floor_mass.as_mut()[i] += dtm_floor;
        }

        // Update partition mass temperature
        for i in 0..self.0.hvac.num_zones {
            let dtm_partition = (t_i.as_ref()[i] - partition_mass.as_ref()[i])
                / (partition_cap.as_ref()[i] / (h_tr_partition.as_ref()[i] * dt));
            partition_mass.as_mut()[i] += dtm_partition;
        }

        // Issue #1966: scratch is now created locally to avoid borrow conflicts
        energy
    }
}
