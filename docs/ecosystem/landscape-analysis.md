# BEM Ecosystem Landscape Analysis

**Issue:** #781 — Ecosystem landscape analysis: survey BEM tools, file formats, and integration patterns
**Parent Epic:** #777 — Ecosystem Integration and Interoperability
**Date:** 2026-06-11
**Status:** Research Complete

---

## Executive Summary

The Building Energy Modeling (BEM) ecosystem is dominated by a single simulation engine — **EnergyPlus** — which powers most commercial tools either directly or indirectly. The landscape is characterized by:

1. **EnergyPlus as the de facto standard engine** — used by OpenStudio, DesignBuilder, Trane TRACE 3D Plus, Carrier HAP, Honeybee, and others
2. **IFC and gbXML as the two primary BIM↔BEM interchange formats**, each with trade-offs
3. **OpenStudio as the dominant middleware/SDK** — provides the most mature programmatic interface to EnergyPlus
4. **A mature but fragmented tool ecosystem** — vendor-specific tools (TRACE, HAP) coexist with open-source platforms (OpenStudio, Ladybug Tools)
5. **FMI/FMU as the emerging co-simulation standard** — fluxion already has FMI 2.0 support via `src/interop/fmi/`

**Recommended integration priority for fluxion:**
1. EnergyPlus IDF/epJSON import (highest adoption, most data available)
2. OpenStudio OSM import (SDK maturity, workflow automation)
3. gbXML import (BIM tool bridge — Revit, ArchiCAD, DesignBuilder)
4. IFC geometry extraction (long-term BIM standard, complex but strategic)
5. eQUEST/DOE-2 BDL import (legacy support, declining usage)

---

## 1. Simulation Engines

### 1.1 EnergyPlus (DOE)

| Attribute | Detail |
|-----------|--------|
| **Developer** | U.S. DOE Building Technologies Office |
| **Current Version** | 26.1.0 (March 2026) |
| **Release Cadence** | Twice annually (March, September) |
| **License** | Open source (BSD-like) |
| **Language** | C++ core, Python/C API bindings |
| **Input Format** | IDF (text) / epJSON (JSON) |
| **Output Format** | ESO/CSV (timeseries), RDD (report dictionary) |
| **Download Volume** | ~43,000+ per release |
| **GitHub** | NatLabRockies/EnergyPlus |

**Key Capabilities:**
- Heat balance method (ASHRAE Handbook Chapter 18)
- Sub-hourly timesteps (user-configurable)
- Comprehensive HVAC system library
- EMS (Energy Management System) for custom controls
- FMU co-simulation import (FMI 2.0)
- Python API for programmatic control
- Radiance coupling for daylighting

**Integration Relevance for fluxion:**
- EnergyPlus is the validation baseline for most BEM tools
- IDF/epJSON are the most widely-used BEM input formats
- Direct import of IDF into fluxion would cover the majority of existing models
- EnergyPlus FMU export capability means fluxion could serve as a co-simulation partner

### 1.2 OpenStudio (NREL/DOE)

| Attribute | Detail |
|-----------|--------|
| **Developer** | NREL (NatLabRockies fork) |
| **Current Version** | 3.11.0 (January 2026) |
| **License** | Open source |
| **Language** | C++ SDK, Ruby/Python/C# bindings |
| **Native Format** | OSM (OpenStudio Model) |
| **Simulation Target** | EnergyPlus (via forward translation) |

**Key Capabilities:**
- Object-oriented abstraction over EnergyPlus IDF
- **Measures ecosystem** — 1000+ Ruby scripts on BCL (Building Component Library)
- CLI for automated workflows
- Forward/Reverse translators (OSM ↔ IDF)
- gbXML and IFC import for geometry
- Parametric Analysis Tool (PAT) for large-scale studies
- Standards library (openstudio-standards gem — ASHRAE 90.1 prototypes)

**Integration Relevance for fluxion:**
- OpenStudio is the most mature programmatic interface to EnergyPlus
- OSM format is higher-level than IDF — includes space types, constructions, schedules as abstractions
- Measures represent a massive body of reusable workflow logic
- Ruby interop is required for Measures; Python bindings exist but Measures are Ruby-first
- OpenStudio's forward translator (OSM→IDF) is a well-tested reference for how building models map to EnergyPlus

### 1.3 eQUEST / DOE-2

| Attribute | Detail |
|-----------|--------|
| **Developer** | James J. Hirsch & Associates |
| **Engine** | DOE-2.2 (legacy, C/Fortran) |
| **License** | Free (closed-source engine) |
| **Input Format** | BDL (Building Description Language) — .INP files |
| **UI Files** | .PD2 (wizard), .PRD (parametric), .SIM (output) |

**Key Capabilities:**
- Wizard-based model creation
- Quick energy analysis for code compliance
- Built-in utility rate structures
- Parametric runs for energy efficiency measures
- Widely used for California Title 24 compliance

**Integration Relevance for fluxion:**
- **Declining relevance** — DOE-2 is no longer actively developed by DOE
- BDL is a legacy format with limited documentation
- eQUEST remains in use for California Title 24 and some utility programs
- Low priority for fluxion unless targeting legacy model migration

### 1.4 DesignBuilder

| Attribute | Detail |
|-----------|--------|
| **Developer** | DesignBuilder Software Ltd (UK) |
| **Engine** | EnergyPlus (embedded) |
| **Current Version** | 2025.1 |
| **License** | Commercial |
| **Input Format** | Native XML (new in 2025.1), gbXML, IDF |

**Key Capabilities:**
- Most mature GUI for EnergyPlus
- ASHRAE 90.1 Appendix G PRM workflows
- gbXML import/export
- Python scripting (v3.4+)
- EMS runtime scripting
- New open XML format with third-party API documentation
- Parallel simulation management

**Integration Relevance for fluxion:**
- DesignBuilder is the most widely-used commercial GUI for EnergyPlus
- New XML format (2025.1) with published API docs could be a bridge format
- gbXML import/export is a key interop pathway
- Python scripting support enables potential programmatic integration

### 1.5 Trane TRACE

| Attribute | Detail |
|-----------|--------|
| **Developer** | Trane Technologies |
| **Versions** | TRACE 700 (legacy, phased out), TRACE 3D Plus (current) |
| **Engine** | EnergyPlus (3D Plus), proprietary (700) |
| **License** | Commercial |
| **Input** | gbXML geometry import, proprietary UI |

**Key Capabilities:**
- Load calculations + energy modeling + economic analysis
- VRF modeling, renewable energy, radiant systems
- Cloud simulation
- ASHRAE 90.1 compliance
- Integration with Autodesk Revit (planned)

**Integration Relevance for fluxion:**
- TRACE 3D Plus uses EnergyPlus engine — models are EnergyPlus-compatible at core
- gbXML import is the primary geometry pathway
- Vendor-specific tool — limited open integration opportunities
- TRACE 700 is legacy and being phased out

### 1.6 Carrier HAP

| Attribute | Detail |
|-----------|--------|
| **Developer** | Carrier Corporation |
| **Current Version** | HAP v6 |
| **Engine** | EnergyPlus |
| **License** | Commercial |

**Key Capabilities:**
- Load calculations + energy analysis
- Life cycle costing
- ASHRAE 90.1 compliance
- gbXML geometry import

**Integration Relevance for fluxion:**
- Uses EnergyPlus engine — similar to TRACE 3D Plus
- Smaller market share than TRACE
- Limited open integration opportunities

### 1.7 Ladybug Tools / Honeybee

| Attribute | Detail |
|-----------|--------|
| **Developer** | Ladybug Tools (open source community) |
| **Platform** | Rhino/Grasshopper, Dynamo, Python SDK |
| **Engines** | Radiance, EnergyPlus/OpenStudio, THERM |
| **License** | Open source |

**Key Capabilities:**
- Parametric environmental design workflows
- Direct Rhino/Grasshopper geometry → EnergyPlus models
- Python SDK (honeybee-core, honeybee-energy)
- honeybee-openstudio for OpenStudio translation
- Standards library (honeybee-energy-standards)
- Climate data analysis (Ladybug)
- Radiance daylighting coupling

**Integration Relevance for fluxion:**
- Honeybee is the most popular parametric BEM workflow tool
- Python SDK is well-documented and actively maintained
- honeybee-energy creates OpenStudio models that could be translated to fluxion
- Parametric study workflows (grasshopper) are a major use case for fluxion's BatchOracle
- Potential integration: fluxion as a faster simulation backend within Honeybee workflows

### 1.8 TRNSYS

| Attribute | Detail |
|-----------|--------|
| **Developer** | Transsolar + University of Wisconsin |
| **Engine** | Fortran (TRNDll) |
| **Input Format** | .dck (deck file), .bui (building description) |
| **License** | Commercial |
| **Key Component** | Type 56 (multizone building) |

**Key Capabilities:**
- Component-based simulation (Types)
- Solar/renewable energy focus
- Transient system simulation
- Co-simulation via BCVTB/FMI
- Flexible component library

**Integration Relevance for fluxion:**
- Niche tool — primarily academic and solar/renewable applications
- .dck format is TRNSYS-specific, limited adoption outside TRNSYS community
- FMI co-simulation is the primary interop pathway
- Low priority for fluxion unless targeting solar/renewable integration

---

## 2. File Formats

### 2.1 EnergyPlus IDF (Input Data File)

| Attribute | Detail |
|-----------|--------|
| **Extension** | `.idf` |
| **Schema** | `.idd` (Input Data Dictionary) — version-specific |
| **Structure** | Text-based, object-oriented key-value pairs |
| **Complexity** | ★★★★☆ (hundreds of object types) |
| **Adoption** | ★★★★★ (dominant BEM format) |

**Description:**
IDF is the native input format for EnergyPlus. It uses a custom text format (not XML/JSON) defined by the IDD schema. Each EnergyPlus release ships with an updated IDD. Objects reference each other by name strings.

**Key Object Categories:**
- **Geometry:** Surface, SubSurface, BuildingSurface:Detailed, FenestrationSurface:Detailed
- **Constructions:** Construction, Construction:WindowEquivalentLayer, Material, MaterialProperty
- **Zones:** Zone, ZoneList, Connector:ZoneReturnAirPath
- **Schedules:** Schedule:Day, Schedule:Week, Schedule:Year, Schedule:Compact
- **HVAC:** AirLoopHVAC, ZoneHVAC, Coil:Heating, Coil:Cooling, Boiler, Chiller, etc.
- **Output:** Output:Variable, Output:Meter, Output:Table:Summary
- **Weather:** WeatherData (via .epw files)

**Integration Complexity:**
- IDF parsing requires IDD-aware parser (field types, references, defaults)
- OpenStudio provides mature IDF read/write C++ library
- Python libraries: `eppy`, `eplusr`, `eppd` provide programmatic access
- **Recommended approach:** Use existing parsers (eppy/eplusr) or port OpenStudio's IDF parser logic

### 2.2 EnergyPlus epJSON

| Attribute | Detail |
|-----------|--------|
| **Extension** | `.epJSON` |
| **Schema** | `.epJSON.schema` (JSON Schema) |
| **Structure** | JSON representation of IDF objects |
| **Complexity** | ★★★☆☆ (JSON is easier to parse) |
| **Adoption** | ★★☆☆☆ (secondary to IDF) |

**Description:**
epJSON is a JSON representation of EnergyPlus IDF files. It was introduced to provide a more machine-readable format. The mapping between IDF and epJSON is bidirectional but not always intuitive due to the IDF's custom syntax.

**Integration Relevance:**
- JSON is trivially parseable in any language
- Less tooling support than IDF
- Good candidate for fluxion's native format if targeting EnergyPlus interoperability

### 2.3 OpenStudio OSM (OpenStudio Model)

| Attribute | Detail |
|-----------|--------|
| **Extension** | `.osm` |
| **Schema** | IDD-based (similar to EnergyPlus IDD) |
| **Structure** | Text-based, object-oriented |
| **Complexity** | ★★★★☆ (similar to IDF) |
| **Adoption** | ★★★★☆ (dominant in OpenStudio workflows) |

**Description:**
OSM is OpenStudio's native model format. It's a higher-level abstraction over EnergyPlus IDF, including concepts like SpaceTypes, ConstructionSets, and LoadObjects that don't exist in raw IDF. OSM can be translated to/from IDF via OpenStudio's ForwardTranslator/ReverseTranslator.

**Key Abstractions (not in IDF):**
- **SpaceType** — reusable space definitions (lighting, equipment, schedules)
- **ConstructionSet** — grouped construction definitions
- **ModelObject** — typed objects with UUID references instead of name strings
- **Relationships** — explicit parent-child relationships

**Integration Complexity:**
- OSM parsing requires OpenStudio SDK or understanding the IDD schema
- Forward translation (OSM→IDF) is well-documented in OpenStudio source
- **Recommended approach:** Use OpenStudio SDK as middleware, or implement OSM reader based on IDD

### 2.4 gbXML (Green Building XML)

| Attribute | Detail |
|-----------|--------|
| **Extension** | `.xml` (gbXML namespace) |
| **Schema** | gbXML schema (xsd) |
| **Structure** | XML |
| **Complexity** | ★★★☆☆ (well-defined schema) |
| **Adoption** | ★★★★☆ (primary BIM→BEM bridge) |

**Description:**
gbXML is an XML schema developed by Green Building Studio (now part of Autodesk) for exchanging building data between BIM tools and energy analysis tools. It focuses on geometry, thermal zones, and constructions. Most BIM tools (Revit, ArchiCAD, IES VE, DesignBuilder) can export gbXML.

**Key Elements:**
- **Surface** — building surfaces (walls, roofs, floors) with adjacency
- **Space** — thermal zones
- **BuildingStory** — floor levels
- **Construction** — layered constructions
- **Layer** — material layers
- **Opening** — windows and doors
- **ClimateZone** — ASHRAE climate zone

**Known Issues:**
- Centre-line geometry representation causes area/volume discrepancies
- Complex geometry can cause data loss during export
- Schema variations between versions
- Not all tools implement the full schema

**Integration Relevance:**
- gbXML is the primary bridge between BIM tools and BEM engines
- XML parsing is straightforward in Rust (quick-xml, roxmltree)
- **Recommended approach:** Implement gbXML reader for geometry/zone extraction
- Consider as the primary import path for architectural models

### 2.5 IFC (Industry Foundation Classes)

| Attribute | Detail |
|-----------|--------|
| **Extension** | `.ifc` |
| **Schema** | IFC4 (ISO 16739-1:2024) |
| **Structure** | STEP Physical File Format (SPF) |
| **Complexity** | ★★★★★ (very large, complex schema) |
| **Adoption** | ★★★★☆ (BIM standard, growing in BEM) |

**Description:**
IFC is the open BIM standard developed by buildingSMART International. It covers the entire building lifecycle, not just energy. For BEM, the key subset is IfcRelSpaceBoundary2ndLevel which defines thermal zones and surface boundaries.

**Key IFC Entities for BEM:**
- **IfcBuilding, IfcBuildingStorey** — building hierarchy
- **IfcSpace** — thermal zones
- **IfcWall, IfcSlab, IfcRoof** — building elements
- **IfcWindow, IfcDoor** — openings
- **IfcRelSpaceBoundary** — surface boundaries (critical for BEM)
- **IfcMaterial, IfcMaterialLayer** — construction materials

**Known Issues:**
- Space boundary definition is inconsistent across BIM tools
- IFC4 Reference View is needed for proper BEM extraction
- Geometry representation differs from BEM requirements
- BIM2SIM project (IEA EBC Annex 60) has developed IFC→IDF/epJSON tools

**Integration Relevance:**
- IFC is the strategic long-term BIM standard
- Very complex to parse fully — recommend focusing on the BEM-relevant subset
- Open-source parsers exist (IfcOpenShell, BIMserver)
- **Recommended approach:** Extract geometry and thermal zones from IFC4, map to fluxion model
- Lower priority than IDF/OSM/gbXML due to complexity

### 2.6 DOE-2 BDL (Building Description Language)

| Attribute | Detail |
|-----------|--------|
| **Extension** | `.inp` |
| **Structure** | Custom text format |
| **Complexity** | ★★★☆☆ |
| **Adoption** | ★★☆☆☆ (legacy, declining) |

**Description:**
BDL is the input format for DOE-2, the predecessor to EnergyPlus. It's used by eQUEST. The format is proprietary and poorly documented outside of DOE-2 manuals.

**Integration Relevance:**
- Legacy format — only needed for migrating eQUEST models
- Low priority unless targeting specific utility program workflows

### 2.7 Weather Files

| Format | Description | Usage |
|--------|-------------|-------|
| **EPW** (.epw) | EnergyPlus Weather | Universal — used by EnergyPlus, OpenStudio, DesignBuilder, Honeybee |
| **DDY** (.ddy) | Design Day | ASHRAE design day files (used with EPW) |
| **TMY3** (.epw) | Typical Meteorological Year | Standard weather data format |
| **CWEC** (.epw) | Canadian Weather for Energy Calculations | Canadian climate data |
| **ISM** (.ism) | Indian Standard Meteorological | Indian climate data |

**Integration Relevance:**
- EPW is the universal weather format for BEM tools
- EPW parsing is well-documented and straightforward
- Fluxion should support EPW import as a baseline requirement

### 2.8 FMI/FMU (Functional Mock-up Interface/Unit)

| Attribute | Detail |
|-----------|--------|
| **Standard** | FMI 2.0 (FMI 3.0 emerging) |
| **Extension** | `.fmu` (zip archive) |
| **Structure** | modelDescription.xml + shared libraries + resources |
| **Complexity** | ★★★★☆ |
| **Adoption** | ★★★☆☆ (growing, especially for co-simulation) |

**Description:**
FMI is an open standard for exchanging dynamic simulation models. An FMU contains a model description (XML), source code or shared libraries, and resources. It supports Co-Simulation (time-stepped) and Model Exchange (equation-based) modes.

**Integration Relevance:**
- Fluxion already implements FMI 2.0 export (`src/interop/fmi/`)
- FMU co-simulation is the standard for coupling different simulation tools
- EnergyPlus supports FMU import (as co-simulation master)
- Fluxion could be exported as FMU for use in EnergyPlus, Modelica, MATLAB/Simulink workflows
- **Strategic value:** FMU export makes fluxion usable in any FMI-compatible toolchain

---

## 3. Integration Patterns

### 3.1 Direct Model Translation (Fluxion as Import Target)

```
┌─────────────┐    IDF/OSM/gbXML    ┌─────────────┐
│  BIM Tools  │ ──────────────────→ │   fluxion   │
│  (Revit,    │                     │  (importer) │
│  ArchiCAD)  │                     └─────────────┘
└─────────────┘
```

**Pattern:** Parse existing BEM/BIM files directly into fluxion's internal model representation.

**Pros:**
- Lowest friction for users — bring existing models directly
- No middleware dependency
- Fastest simulation (no translation overhead)

**Cons:**
- Must implement parsers for each format (IDF, OSM, gbXML, IFC)
- Lossy translation possible — not all EnergyPlus features map to fluxion
- Version compatibility concerns (IDD changes between EnergyPlus versions)

**Recommended Priority:** IDF import (highest), then OSM, then gbXML

### 3.2 OpenStudio Middleware Pattern

```
┌─────────────┐    gbXML/IFC    ┌──────────────┐    OSM    ┌─────────────┐
│  BIM Tools  │ ──────────────→ │  OpenStudio  │ ────────→ │   fluxion   │
│             │                 │  (middleware)│           │  (OSM import)│
└─────────────┘                 └──────────────┘           └─────────────┘
                                     ↕
                                Measures (Ruby)
                                BCL Library
```

**Pattern:** Use OpenStudio as the model translation layer. BIM tools export to gbXML/IFC, OpenStudio translates to OSM, fluxion imports OSM.

**Pros:**
- Leverages OpenStudio's mature translation infrastructure
- Access to 1000+ BCL Measures for model manipulation
- gbXML/IFC import is handled by OpenStudio
- OSM is a well-defined, stable format

**Cons:**
- Requires OpenStudio SDK as a dependency (large C++ library)
- Ruby runtime needed for Measures
- Adds complexity to the build/distribution

**Recommended for:** Advanced workflows where users need Measure automation

### 3.3 Co-Simulation via FMI/FMU

```
┌─────────────┐                    ┌─────────────┐
│  EnergyPlus │ ←── FMU (fluxion) │   fluxion   │
│  (master)   │                    │  (slave)    │
└─────────────┘                    └─────────────┘
       ↕
┌─────────────┐
│  BCVTB /    │
│  Modelica   │
│  MATLAB     │
└─────────────┘
```

**Pattern:** Export fluxion as FMU for co-simulation with EnergyPlus or other tools. Fluxion handles specific sub-systems (e.g., envelope thermal network) while EnergyPlus handles HVAC.

**Pros:**
- Standard interop mechanism (FMI 2.0)
- Fluxion's fast thermal network runs as a component in larger simulations
- No format translation needed — runtime coupling
- Already implemented in fluxion (`src/interop/fmi/`)

**Cons:**
- Co-simulation adds complexity (time step synchronization, convergence)
- Limited to what FMI variables expose
- Not suitable for standalone model translation

**Recommended for:** Hybrid simulations where fluxion's speed advantage is leveraged for specific sub-systems

### 3.4 Python API Integration

```
┌─────────────────┐     Python API     ┌─────────────┐
│  Honeybee /     │ ──────────────────→ │   fluxion   │
│  Ladybug Tools  │                     │  (PyO3)     │
│  eppy / eplusr  │                     └─────────────┘
└─────────────────┘
```

**Pattern:** Expose fluxion via Python bindings (PyO3) for integration with Python-based BEM workflows.

**Pros:**
- Python is the dominant language for BEM scripting
- Honeybee, eppy, eplusr all use Python
- Fluxion already has PyO3 bindings (`src/python/`)
- Lowest barrier for ecosystem tool integration

**Cons:**
- Python overhead for tight loops (mitigated by BatchOracle pattern)
- Requires users to install Python package

**Recommended for:** Programmatic access, batch studies, workflow automation

### 3.5 CLI Compatibility Pattern

```
┌─────────────────────────────────────────┐
│  User CLI Workflow                       │
│                                          │
│  fluxion run --input model.idf \        │
│              --weather denver.epw \     │
│              --output results.csv       │
│                                          │
│  (Similar to: energyplus --weather ...  │
│               --output-directory ...    │
│               model.idf)                │
└─────────────────────────────────────────┘
```

**Pattern:** Provide CLI interfaces that mirror EnergyPlus/OpenStudio conventions.

**Pros:**
- Familiar workflow for existing EnergyPlus users
- Easy to integrate into CI/CD pipelines
- Scriptable and automatable

**Cons:**
- Must maintain compatibility with evolving EnergyPlus CLI patterns

**Recommended for:** All users — CLI should be a first-class interface

---

## 4. Standards Landscape

### 4.1 Current fluxion Target

| Standard | Status | Description |
|----------|--------|-------------|
| **ASHRAE 140-2023** | Active | Building thermal envelope validation — primary target |

### 4.2 Planned Standards (from `docs/compliance/standards-roadmap.md`)

| Standard | Priority | Market | Key Gap |
|----------|----------|--------|---------|
| **ASHRAE 90.1-2022 Appendix G** | 2 | US Commercial | Multi-zone HVAC, equipment curves, LPD scheduling |
| **RESNET HERS (ANSI/RESNET/ICC 301-2022)** | 3 | US Residential | Infiltration model, ventilation, duct leakage, DHW |
| **California Title 24 ACM** | 4 | California | Specific compliance pathways |
| **ISO 52016-2017 / EN 13790** | 5 | EU | ISO 13790 5R1C already implemented in fluxion |

### 4.3 Ecosystem Standards for Interop

| Standard | Relevance | fluxion Status |
|----------|-----------|----------------|
| **FMI 2.0** | Co-simulation interop | Implemented (src/interop/fmi/) |
| **IFC4 (ISO 16739)** | BIM interchange | Not implemented |
| **gbXML** | BIM→BEM bridge | Not implemented |
| **EPW** | Weather data | Not implemented (uses synthetic weather) |
| **ASHRAE 136** | Infiltration calculation | Not implemented (needed for RESNET) |
| **ASHRAE 62.2** | Ventilation | Not implemented (needed for RESNET) |

---

## 5. Adoption Priority Matrix

| Format/Tool | Adoption | Integration Value | Implementation Effort | Priority |
|-------------|----------|-------------------|----------------------|----------|
| **EnergyPlus IDF** | ★★★★★ | ★★★★★ | ★★★☆☆ | **P0 — Do First** |
| **EPW Weather** | ★★★★★ | ★★★★★ | ★★☆☆☆ | **P0 — Do First** |
| **OpenStudio OSM** | ★★★★☆ | ★★★★☆ | ★★★★☆ | **P1 — Do Second** |
| **gbXML** | ★★★★☆ | ★★★★☆ | ★★★☆☆ | **P1 — Do Second** |
| **Python API (PyO3)** | ★★★★☆ | ★★★★★ | ★★☆☆☆ | **P0 — Already exists** |
| **CLI Interface** | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | **P1 — Enhance existing** |
| **FMI/FMU Export** | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | **P1 — Already exists** |
| **FMI/FMU Import** | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | **P2 — Future** |
| **IFC Geometry** | ★★★★☆ | ★★★☆☆ | ★★★★★ | **P2 — Future** |
| **eQUEST BDL** | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | **P3 — Legacy only** |
| **TRNSYS .dck** | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★☆ | **P4 — Not recommended** |

---

## 6. Recommended Implementation Order

Based on the adoption priority matrix and fluxion's current capabilities:

### Phase 1: Core Import (Issue #777 sub-tasks)

1. **EPW Weather Import** — Replace synthetic weather with real TMY data
   - Straightforward parsing (well-documented format)
   - Required for ASHRAE 140 compliance (DEV-004)
   - Affects all validation metrics

2. **EnergyPlus IDF Import** — Parse IDF files into fluxion model
   - Highest adoption format — covers most existing BEM models
   - Use eppy/eplusr as reference parsers
   - Focus on geometry, constructions, schedules, and zone definitions
   - Defer full HVAC import initially (fluxion has its own HVAC)

3. **Python API Enhancement** — Improve PyO3 bindings for ecosystem access
   - Already exists via maturin/PyO3
   - Add model construction API (not just simulation)
   - Enable Honeybee/eppy integration

### Phase 2: BIM Bridge

4. **gbXML Import** — Enable BIM tool export to fluxion
   - XML parsing with quick-xml or roxmltree
   - Extract geometry, zones, constructions from gbXML
   - Bridge from Revit/ArchiCAD/DesignBuilder

5. **OpenStudio OSM Import** — Leverage OpenStudio's translation ecosystem
   - Most complex but most powerful
   - Enables access to BCL Measures and standards library
   - Consider using OpenStudio SDK as optional dependency

### Phase 3: Advanced Integration

6. **IFC Geometry Extraction** — Long-term BIM standard
   - Focus on IfcRelSpaceBoundary2ndLevel subset
   - Use IfcOpenShell or similar parser
   - Extract thermal zones and surface boundaries

7. **FMI Enhancement** — Expand co-simulation capabilities
   - FMU import for coupling with external tools
   - Multi-zone FMU export (currently single-zone only)
   - BCVTB integration patterns

---

## 7. Key Findings

### Finding 1: EnergyPlus is the Hub
Every major BEM tool either uses EnergyPlus as its engine or can export to EnergyPlus format. This makes IDF/epJSON the most valuable import format for fluxion.

### Finding 2: OpenStudio is the Gateway
OpenStudio provides the most mature programmatic interface to EnergyPlus, with a rich Measures ecosystem. However, it's a large C++ dependency with Ruby runtime requirements.

### Finding 3: gbXML is the BIM Bridge
For architectural modelers using Revit/ArchiCAD, gbXML is the primary export path to BEM tools. Fluxion should support gbXML import to capture this workflow.

### Finding 4: IFC is Strategic but Complex
IFC is the long-term BIM standard, but its complexity and inconsistent implementation across tools make it a lower-priority integration target.

### Finding 5: FMI is Already Implemented
Fluxion's existing FMI 2.0 export capability is a significant advantage. Expanding this to multi-zone support and FMU import would open co-simulation workflows.

### Finding 6: Python is the Ecosystem Language
The BEM scripting ecosystem (Honeybee, eppy, eplusr) is Python-first. Fluxion's PyO3 bindings position it well for programmatic integration.

### Finding 7: Vendor Tools are EnergyPlus-Based
Trane TRACE 3D Plus and Carrier HAP both use EnergyPlus engines, meaning their models are EnergyPlus-compatible at core. This reinforces IDF as the priority import format.

---

## 8. Concerns and Risks

### 8.1 IDD Version Compatibility
EnergyPlus IDF schema (IDD) changes with each release. Fluxion's IDF parser must handle version-specific schemas or focus on a subset.

**Mitigation:** Implement a version-tolerant parser; focus on core objects (geometry, constructions, schedules) that are stable across versions.

### 8.2 Translation Loss
Not all EnergyPlus features map to fluxion's physics model. HVAC systems, EMS scripts, and advanced controls may not be directly translatable.

**Mitigation:** Clearly document supported/unsupported objects; provide warnings for untranslated elements; implement a "basic import" mode that captures envelope and loads.

### 8.3 OpenStudio Dependency Weight
OpenStudio SDK is a large C++ library (~100MB+). Making it a required dependency would significantly increase fluxion's distribution size.

**Mitigation:** Make OpenStudio integration optional (feature flag); provide direct IDF import as the primary path; consider OpenStudio as a separate "adapter" package.

### 8.4 gbXML Geometry Inconsistencies
gbXML center-line geometry causes area/volume discrepancies. Different BIM tools produce different gbXML quality.

**Mitigation:** Implement geometry validation and correction; provide import options for surface adjustment; document known issues.

### 8.5 IFC Space Boundary Quality
IFC space boundaries are inconsistently defined across BIM tools, making automated BEM extraction unreliable.

**Mitigation:** Focus on IFC4 Reference View; implement validation and correction algorithms (following BIM2SIM approach); provide manual override options.

---

## 9. Existing fluxion Ecosystem Capabilities

| Capability | Status | Location |
|------------|--------|----------|
| FMI 2.0 Export | Implemented | `src/interop/fmi/` |
| PyO3 Python Bindings | Implemented | `src/python/`, maturin |
| NAPI Node.js Bindings | Implemented | `src/napi/` |
| CLI Interface | Implemented | `src/cli/` |
| ASHRAE 140 Validation | Active | `docs/ashrae_140/` |
| HVAC Architecture | Designed | `src/hvac/`, `docs/HVAC_ARCHITECTURE.md` |
| EPW Weather Import | Not implemented | — |
| IDF Import | Not implemented | — |
| gbXML Import | Not implemented | — |
| OSM Import | Not implemented | — |
| IFC Import | Not implemented | — |

---

## 10. Appendix: Tool Comparison Matrix

| Feature | EnergyPlus | OpenStudio | DesignBuilder | eQUEST | TRACE 3D+ | HAP | Honeybee |
|---------|-----------|------------|---------------|--------|-----------|-----|----------|
| **Open Source** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| **GUI** | Minimal | Moderate | Excellent | Good | Good | Good | Rhino/Grasshopper |
| **Engine** | Own | EnergyPlus | EnergyPlus | DOE-2 | EnergyPlus | EnergyPlus | EnergyPlus/OpenStudio |
| **IDF Support** | Native | Native | Native | ❌ | Via EnergyPlus | Via EnergyPlus | Via OpenStudio |
| **gbXML Import** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **IFC Import** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Python API** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Ruby API** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **FMI/FMU** | Import | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Measures/BCL** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **CLI** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 11. References

- EnergyPlus Documentation: https://energyplus.net/documentation
- OpenStudio SDK Documentation: https://openstudio-sdk-documentation.s3.amazonaws.com/index.html
- BCL (Building Component Library): https://bcl.nrel.gov/
- gbXML Schema: https://www.gbxml.org/
- buildingSMART IFC: https://standards.buildingsmart.org/
- FMI Standard: https://fmi-standard.org/
- Ladybug Tools: https://www.ladybug.tools/
- Honeybee: https://www.ladybug.tools/honeybee.html
- BIM2SIM Project: https://www.bsim2sim.de/
- ASHRAE Standards: https://www.ashrae.org/technical-resources/standards-and-guidelines
- DOE Building Energy Software Tools Directory: https://www.energy.gov/eere/buildings/building-energy-software-tools-directory
