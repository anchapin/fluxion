/**
 * TypeScript mirror of `schemas/topology_v1.schema.json` (topology document v1,
 * emitted by `fluxion topology export` — Issue #3963).
 *
 * ⚠ Hand-written and kept in sync MANUALLY with the JSON Schema. When the
 * schema gains a field or a kind, update both this file and the styling maps
 * in `src/topology/styling.ts`. `tests/topology-viewmodel.test.ts` asserts
 * exhaustiveness against the kind unions to catch drift.
 */

export const NODE_KINDS = [
  "outdoor_ambient",
  "exterior_surface",
  "wall_layer",
  "interior_surface",
  "internal_mass",
  "zone_air",
  "internal_gain",
  "hvac_terminal",
] as const;

export type NodeKind = (typeof NODE_KINDS)[number];

export const EDGE_KINDS = [
  "conduction",
  "convection_exterior",
  "convection_interior",
  "longwave_radiation",
  "shortwave_solar_direct",
  "shortwave_solar_diffuse",
  "air_exchange",
  "internal_gain_split",
  "hvac_sensible",
] as const;

export type EdgeKind = (typeof EDGE_KINDS)[number];

/** A thermal-network node (schema `$defs/node`). Numeric attributes are
 *  absent (not null) when not applicable to the node kind. */
export interface TopologyNode {
  id: string;
  kind: NodeKind;
  /** Owning zone air node id, when the node belongs to a zone. */
  zone_id?: string;
  name: string;
  /** Thermal capacitance (J/K). */
  capacitance_j_per_k?: number;
  /** Reference surface area (m²). */
  area_m2?: number;
  /** Reference volume (m³). */
  volume_m3?: number;
  /** Surface azimuth, degrees clockwise from North (absent for flat/ground). */
  azimuth_deg?: number;
  /** Surface tilt from horizontal (90 = vertical wall, 0 = up roof, 180 = down floor). */
  tilt_deg?: number;
  /** Node elevation above grade (m). */
  elevation_m?: number;
  /** Extensible numeric attributes (HVAC setpoints, layer properties, …). */
  attributes?: Readonly<Record<string, number>>;
}

/** A heat-transfer coupling between two nodes (schema `$defs/edge`). */
export interface TopologyEdge {
  source_id: string;
  target_id: string;
  coupling_type: EdgeKind;
  /** Coupling conductance (W/K); absent for non-conductive couplings. */
  conductance_w_per_k?: number;
  /** Coupling fraction (e.g. SHGC, gain split); absent when not applicable. */
  fraction?: number;
  bidirectional: boolean;
  /** Extensible numeric attributes (ACH, schedule hours, window areas, …). */
  attributes?: Readonly<Record<string, number>>;
}

export interface TopologyMetadata {
  schema_version: "1.0.0";
  tool_version: string;
  model_name: string;
  model_source: string;
  timestamp?: string;
  node_count: number;
  edge_count: number;
}

/** Root topology document (schema top level). */
export interface TopologyDocument {
  metadata: TopologyMetadata;
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

/**
 * A single finding from `fluxion topology lint` (Issue #3964). The report
 * envelope is `{ clean, findings, source, strict }`; the viewer tolerates
 * missing fields so partial/partially-versioned reports still render.
 */
export interface LintFinding {
  code: string;
  severity?: "error" | "warning" | "info";
  node_id?: string;
  edge_key?: string;
  message?: string;
  remediation?: string;
}

export interface LintReport {
  clean?: boolean;
  findings?: LintFinding[];
  source?: string;
  strict?: boolean;
}

/** Diagnostic codes emitted by the #3964 linter; mirrored here for badge
 *  labels and client-side recomputation of the cheap structural checks. */
export const LINT_CODES = [
  "E001_ORPHAN_NODE",
  "E002_DANGLING_BOUNDARY",
  "E003_ORPHAN_MASS",
  "E004_RECIPROCAL_MISMATCH",
  "E005_INVALID_SPLIT_FRACTION",
  "E006_UNRESOLVED_RADIANT_TARGET",
  "E007_NON_POSITIVE_CAPACITY",
] as const;

export type LintCode = (typeof LINT_CODES)[number];
