# Fluxion User Personas

> **Purpose:** These personas describe the three primary audiences for Fluxion. Use them to guide documentation priorities, feature prioritization, API design, and onboarding flow decisions.

---

## Persona 1 — The Building Scientist

**"I need a fast, validated engine I can trust for research and compliance work."**

### Profile
- **Background:** Mechanical or architectural engineer with expertise in building thermodynamics and HVAC. May also be an energy modeler at a national lab, utility, or consulting firm.
- **Tools they already use:** EnergyPlus, OpenStudio, TRNSYS, ESP-r, Python (NumPy/Pandas), Excel
- **Relationship to code:** Comfortable running scripts and editing config files; not a software developer by trade. Does not write Rust.

### Goals
- Run ASHRAE 140 / BESTEST validation cases and compare results against EnergyPlus/ESP-r/TRNSYS reference ranges
- Simulate real building configurations (multi-zone, high-mass, HVAC-controlled) and get physically meaningful outputs
- Run large parametric studies (e.g., 10,000 design variants) faster than EnergyPlus allows
- Produce results they can cite — traceable to standards, reproducible, clearly documented

### Pain Points
- EnergyPlus is slow for large parametric sweeps — a 10,000-run sensitivity study takes hours
- Unfamiliar with Rust; build toolchains are a barrier if Python install is non-trivial
- ASHRAE 140 validation pass/fail status is critical for credibility; unclear failure modes erode trust
- Does not want to interpret raw EUI values without units and calibration context
- Wants to know *what assumptions* the physics engine makes (CTF solver, thermal mass model, etc.)

### Entry Point into Fluxion
- Arrives via academic citation, national lab recommendation, or GitHub search for "ASHRAE 140 Rust"
- First question: *"Does this pass ASHRAE 140?"*
- Primary interface: Python (`fluxion.Model`, `BatchOracle`) or CLI (`fluxion run`)
- Key docs: `docs/QUICKSTART.md`, `docs/ASHRAE140_VALIDATION.md`, `docs/PHYSICAL_CONSTANTS.md`, `docs/KNOWN_ISSUES.md`

### Definition of Success
- Can install Fluxion via `pip install fluxion`, run a simulation, and get a physically sensible EUI in under 10 minutes
- Can run the ASHRAE 140 validation suite and understand the pass/fail report
- Can configure a multi-zone building and trust the inter-zone heat transfer behavior

---

## Persona 2 — The Python Application Developer

**"I need a fast, embeddable BEM engine I can call from Python to power an optimization or design tool."**

### Profile
- **Background:** Software engineer or data scientist building energy analysis tools, optimization pipelines, digital twins, or building design automation products. May work at a proptech startup, energy software company, or research group.
- **Tools they already use:** Python, NumPy, SciPy, D-Wave, PyTorch, scikit-learn, FastAPI, Jupyter
- **Relationship to code:** Fluent Python developer, comfortable with pip installs, virtual environments, and calling native extensions. Does not need to understand Rust internals.

### Goals
- Embed Fluxion as a high-performance "physics oracle" inside a larger Python optimization loop (Bayesian optimization, genetic algorithms, D-Wave quantum annealing, etc.)
- Evaluate thousands to millions of building design candidates per second
- Get clean, typed return values they can feed directly into ML training pipelines or optimization objectives
- Keep deployment simple — `pip install fluxion` should be all that's needed

### Pain Points
- Existing BEM engines (EnergyPlus) have no Python-native batch evaluation API; wrapping them is fragile
- The current EUI output is a raw cumulative metric (not calibrated kWh/m²/year) — this is confusing without clear documentation
- API surface should feel Pythonic; `Model(num_zones=1)` is fine, but error messages and type hints matter
- Surrogate model integration (`load_surrogate()`) needs clear documentation on ONNX input/output tensor shapes

### Entry Point into Fluxion
- Arrives via PyPI search, GitHub, or a colleague's recommendation
- First question: *"Can I call this from Python in a tight optimization loop? How fast is it?"*
- Primary interface: `fluxion.Model`, `fluxion.BatchOracle`
- Key docs: `docs/QUICKSTART.md`, `docs/API_REFERENCE.md`, `docs/EXAMPLES.md`, `docs/SCHEMA.md`

### Definition of Success
- `pip install fluxion` works without requiring Rust or a build toolchain (pre-built wheels available for major platforms)
- Can evaluate a 10,000-candidate population in under 1 second
- Clear documentation on what EUI values mean and how to normalize them for comparison
- Type stubs / IDE autocomplete work correctly

---

## Persona 3 — The Rust / Systems Developer

**"I'm contributing to the engine or integrating Fluxion into a high-performance backend system."**

### Profile
- **Background:** Systems programmer, Rust developer, or performance engineer. May be a core contributor, a researcher building a custom solver on top of Fluxion, or an infrastructure engineer embedding Fluxion in a web service or desktop app.
- **Tools they already use:** Rust, Cargo, `rayon`, `pyo3`, `napi-rs`, GitHub Actions, Docker, `criterion` benchmarks
- **Relationship to code:** Reads and writes Rust. Understands `unsafe`, lifetimes, trait bounds. Cares about allocations, SIMD, and throughput numbers.

### Goals
- Understand the Rust module structure and solver architecture before making changes
- Build and test locally without surprises (pre-commit hooks, CI parity)
- Extend the physics engine (new solvers, new zone types, FMI co-simulation)
- Expose new Rust capabilities via the Python (`pyo3`) and Node.js (`napi-rs`) bindings
- Contribute quality PRs that pass CI on first try

### Pain Points
- Architecture documentation (`docs/ARCHITECTURE.md`) needs to clearly explain where each physics module lives, its public API, and its invariants
- Pre-commit hook setup and CI requirements not immediately obvious from README
- The boundary between the analytical physics core and the surrogate layer is not clearly documented
- Node.js binding (`napi-rs`) documentation is thin — behavior differences from the Python API are undocumented

### Entry Point into Fluxion
- Arrives via GitHub directly (contributor, fork, or stars search)
- First question: *"Where is the CTF solver? How do I add a new thermal zone type? How do I run CI locally?"*
- Primary interface: Rust source (`src/`), Cargo.toml, GitHub Actions
- Key docs: `docs/ARCHITECTURE.md`, `docs/CONTRIBUTING.md`, `docs/HVAC_ARCHITECTURE.md`, `docs/NAPI_BINDINGS.md`, `docs/FMI.md`

### Definition of Success
- `cargo build --release && cargo test` succeeds on a fresh clone without manual environment tweaks
- Architecture docs answer "what does this module own?" without needing to read source
- First PR submitted, CI passes, review feedback is actionable

---

## Secondary Persona — The Node.js / TypeScript Integrator

**"I need BEM simulation in a TypeScript service or web app."**

### Profile
- Full-stack or backend JavaScript/TypeScript developer building a web-based energy design or retrofit tool
- Does not use Python; works in Node.js ecosystem (npm, TypeScript, Vite, etc.)

### Goals
- Install via `npm install fluxion` and get >10,000 configs/sec throughput with full TypeScript types
- Call `BatchOracle` from a Node.js API server or edge function

### Pain Points
- `npm/README.md` exists but is thin on examples and does not document behavior differences from the Python API
- No clear guidance on what platforms the native `.node` binary supports

### Entry Point
- `npm/README.md`, GitHub releases page
- Key docs: `npm/README.md`, `docs/NAPI_BINDINGS.md`

---

## How to Use These Personas

| Decision type | Which persona(s) to check |
|---------------|---------------------------|
| Onboarding flow / README structure | All three primary |
| API naming and error messages | Persona 2 (Python Dev) |
| Physics documentation depth | Persona 1 (Building Scientist) |
| Build system / CI docs | Persona 3 (Rust Dev) |
| npm bindings documentation | Secondary (Node.js integrator) |
| ASHRAE 140 validation reporting | Persona 1 |
| Surrogate model documentation | Personas 1 & 2 |
