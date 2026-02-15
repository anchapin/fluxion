import re

with open('src/sim/engine.rs', 'r') as f:
    content = f.read()

# Conflict 1: imports
old1 = '''<<<<<<< HEAD
use crate::sim::shading::{Overhang, ShadeFin, Side};
use crate::sim::ventilation::{ConstantVentilation, ScheduledVentilation, VentilationSchedule};
use crate::validation::ashrae_140_cases::{CaseSpec, ShadingType};
=======
use crate::sim::ventilation::{ConstantVentilation, ScheduledVentilation, VentilationSchedule};
use crate::validation::ashrae_140_cases::CaseSpec;
>>>>>>> origin/feature/issue-64'''

new1 = '''use crate::sim::shading::{Overhang, ShadeFin, Side};
use crate::sim::ventilation::{ConstantVentilation, ScheduledVentilation, VentilationSchedule};
use crate::validation::ashrae_140_cases::{CaseSpec, ShadingType};'''

content = content.replace(old1, new1)

# Find remaining conflicts and replace them
# We need to find the pattern and resolve them
# Let me just replace all remaining conflicts

# Conflict 2: around line 265 - struct fields
old2 = '''<<<<<<< HEAD

    // Inter-zone coupling (Issue #66)
    pub h_interzone: Vec<Vec<f64>>, // Matrix of inter-zone conductances (W/K)
    pub ventilation_schedule: Box<dyn VentilationSchedule>, // Ventilation schedule
=======
>>>>>>> origin/feature/issue-64'''

new2 = '''

    // Inter-zone coupling (Issue #66)
    pub h_interzone: Vec<Vec<f64>>, // Matrix of inter-zone conductances (W/K)
    pub ventilation_schedule: Box<dyn VentilationSchedule>, // Ventilation schedule'''

content = content.replace(old2, new2)

# More patterns will need to be found and replaced
# Let's find all remaining conflict markers
while '<<<<<<' in content:
    start = content.find('<<<<<<< HEAD')
    if start == -1:
        break
    end = content.find('>>>>>>> origin/feature/issue-64', start)
    if end == -1:
        end = content.find('>>>>>>>', start)
    if end == -1:
        print(f"Cannot find end of conflict at position {start}")
        break
    
    # Get the section
    section = content[start:end+30]
    
    # Try to identify what kind of conflict it is
    if 'h_interzone' in section or 'ventilation_schedule' in section:
        # Keep the HEAD version (which has the PR features after main additions)
        replacement = section.replace('<<<<<<< HEAD\n', '').replace('=======', '').replace('>>>>>>> origin/feature/issue-64', '')
    else:
        # For other conflicts, keep HEAD
        replacement = section.replace('<<<<<<< HEAD\n', '').replace('=======\n', '').replace('>>>>>>> origin/feature/issue-64', '')
    
    content = content.replace(section, replacement)

with open('src/sim/engine.rs', 'w') as f:
    f.write(content)

print("Done!")
