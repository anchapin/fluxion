---
phase: M2-zone-hvac-controls
plan: 03
type: execute
wave: 2
depends_on: [M2-01, M2-02]
files_modified:
  - src/cli/hvac_commands.rs
  - src/cli/multi_zone.rs
  - Cargo.toml
autonomous: true
requirements: [MZ-10]

must_haves:
  truths:
    - "CLI supports multi-zone HVAC simulation commands"
    - "CLI commands integrate with Python API"
    - "Help documentation includes HVAC commands"
  artifacts:
    - path: "src/cli/hvac_commands.rs"
      provides: "CLI commands for HVAC operations"
      min_lines: 250
    - path: "src/cli/multi_zone.rs"
      provides: "Multi-zone CLI integration"
      min_lines: 150
  key_links:
    - from: "src/cli/hvac_commands.rs"
      to: "src/hvac/zone_control.rs"
      via: "CLI integration"
      pattern: "ZoneControl::new"
    - from: "src/cli/multi_zone.rs"
      to: "src/cli/hvac_commands.rs"
      via: "subcommand registration"
      pattern: "subcommand"
---

<objective>
Add CLI support for multi-zone HVAC operations

Purpose: Enable command-line control and simulation of zone-level HVAC systems
Output: CLI commands for HVAC configuration, control, and simulation with comprehensive help
</objective>

<execution_context>
@/home/alex/.config/opencode/get-shit-done/workflows/execute-plan.md
@/home/alex/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/M1-multi-zone-foundation/M1-03-SUMMARY.md
@src/cli/multi_zone.rs
@src/hvac/zone_setpoints.rs
@src/hvac/zone_control.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create HVAC CLI commands module</name>
  <files>src/cli/hvac_commands.rs</files>
  <action>
    Create comprehensive HVAC CLI module using clap:

    #[derive(Subcommand, Debug)]
    pub enum HvacCommand {
        /// Configure zone setpoints
        Setpoints {
            #[command(subcommand)]
            action: SetpointAction,
        },
        /// Run HVAC simulation
        Simulate {
            /// Number of simulation steps
            #[arg(long, default_value_t = 100)]
            steps: usize,
            /// Output file (CSV format)
            #[arg(short, long)]
            output: Option<PathBuf>,
        },
        /// Show current HVAC status
        Status,
    }

    Implement SetpointAction subcommands:
    - SetHeating { zone_id: usize, temperature: f64 }
    - SetCooling { zone_id: usize, temperature: f64 }
    - SetDeadband { zone_id: usize, deadband: f64 }
    - Show { zone_id: Option<usize> }

    Create command handlers:
    - handle_setpoints() -> Result<()>
    - handle_simulate() -> Result<()>
    - handle_status() -> Result<()>

    Add proper error handling and validation.
  </action>
  <verify>
    <automated>cargo build --release</automated>
  </verify>
  <done>
    HVAC commands module compiles
    All subcommands defined
    Error handling implemented
  </done>
</task>

<task type="auto">
  <name>Task 2: Integrate HVAC commands with multi-zone CLI</name>
  <files>src/cli/multi_zone.rs</files>
  <action>
    Extend multi-zone CLI to include HVAC subcommands:

    #[derive(Subcommand, Debug)]
    pub enum MultiZoneCommand {
        /// Thermal simulation commands
        Thermal(ThermalCommand),
        /// HVAC control commands
        Hvac(HvacCommand),
        /// Validation commands
        Validate(ValidateCommand),
    }

    Update main multi-zone handler:
    - Add match arm for MultiZoneCommand::Hvac(cmd)
    - Route to hvac_commands::handle_command(cmd)
    - Ensure proper error propagation

    Update help documentation:
    - Add HVAC section to --help output
    - Include examples for common HVAC operations
    - Document CSV output format for simulation
  </action>
  <verify>
    <automated>cargo run -- --help | grep -i hvac</automated>
  </verify>
  <done>
    HVAC commands integrated into CLI
    Help documentation updated
    Command routing working
  </done>
</task>

<task type="auto">
  <name>Task 3: Add CLI features to Cargo.toml and test integration</name>
  <files>Cargo.toml</files>
  <action>
    Update Cargo.toml to ensure CLI features are properly configured:
    - Add clap feature for HVAC commands
    - Ensure serde feature for CSV output
    - Add csv feature for simulation output

    Test CLI integration:
    - fluxion multi-zone hvac --help (shows HVAC commands)
    - fluxion multi-zone hvac status (shows current status)
    - fluxion multi-zone hvac setpoints show (shows setpoints)
    - fluxion multi-zone hvac simulate --steps 10 (runs simulation)

    Verify error handling:
    - Invalid zone IDs return proper error messages
    - Invalid temperatures show validation errors
    - Missing output files handled gracefully
  </action>
  <verify>
    <automated>cargo run -- multi-zone hvac --help && echo "CLI integration successful"</automated>
  </verify>
  <done>
    Cargo.toml features updated
    CLI commands working end-to-end
    Error handling validated
  </done>
</task>

</tasks>

<verification>
- CLI HVAC commands compile and run
- Help documentation includes HVAC section
- Setpoint configuration working
- Simulation commands functional
</verification>

<success_criteria>
- cargo build --release succeeds
- fluxion multi-zone hvac --help shows all commands
- fluxion multi-zone hvac status runs without errors
- fluxion multi-zone hvac setpoints show displays setpoints
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-03-SUMMARY.md`
</output>
