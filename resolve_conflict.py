import re

with open('src/sim/engine.rs', 'r') as f:
    content = f.read()

# Conflict 1: imports (lines around 7-15)
# Keep both: ventilation from PR94 AND schedule/shading from main
old_imports = '''<<<<<<< HEAD
use crate::sim::ventilation::{ConstantVentilation, ScheduledVentilation, VentilationSchedule};
use crate::validation::ashrae_140_cases::CaseSpec;
=======
use crate::sim::schedule::DailySchedule;
use crate::sim::shading::{Overhang, ShadeFin, Side};
use crate::validation::ashrae_140_cases::{CaseSpec, ShadingType};
>>>>>>> origin/main'''

new_imports = '''use crate::sim::ventilation::{ConstantVentilation, ScheduledVentilation, VentilationSchedule};
use crate::sim::schedule::DailySchedule;
use crate::sim::shading::{Overhang, ShadeFin, Side};
use crate::validation::ashrae_140_cases::{CaseSpec, ShadingType};'''

content = content.replace(old_imports, new_imports)

# Conflict 2: struct fields (around line 97)
old_struct = '''<<<<<<< HEAD

    // Inter-zone coupling (Issue #66)
    pub h_interzone: Vec<Vec<f64>>, // Matrix of inter-zone conductances (W/K)
    pub ventilation_schedule: Box<dyn VentilationSchedule>, // Ventilation schedule
=======
>>>>>>> origin/main'''

new_struct = '''

    // Inter-zone coupling (Issue #66)
    pub h_interzone: Vec<Vec<f64>>, // Matrix of inter-zone conductances (W/K)
    pub ventilation_schedule: Box<dyn VentilationSchedule>, // Ventilation schedule'''

content = content.replace(old_struct, new_struct)

# Conflict 3: Clone impl (around line 127)
old_clone = '''<<<<<<< HEAD
            h_interzone: self.h_interzone.clone(),
            ventilation_schedule: self.ventilation_schedule.clone_box(),
=======
>>>>>>> origin/main'''

new_clone = '''            h_interzone: self.h_interzone.clone(),
            ventilation_schedule: self.ventilation_schedule.clone_box(),'''

content = content.replace(old_clone, new_clone)

# Conflict 4: ThermalModel::new (around line 366)
old_new = '''<<<<<<< HEAD
            h_interzone: vec![vec![0.0; num_zones]; num_zones],
            ventilation_schedule: Box::new(ConstantVentilation::new(0.5)),      // Default 0.5 ACH
=======
>>>>>>> origin/main'''

new_new = '''            h_interzone: vec![vec![0.0; num_zones]; num_zones],
            ventilation_schedule: Box::new(ConstantVentilation::new(0.5)),      // Default 0.5 ACH'''

content = content.replace(old_new, new_new)

with open('src/sim/engine.rs', 'w') as f:
    f.write(content)

print("Done!")
