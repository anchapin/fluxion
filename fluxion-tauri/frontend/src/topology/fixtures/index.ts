/**
 * Bundled sample topology documents (issue #3965).
 * Generated with:
 *   cargo run -q -p fluxion --bin fluxion -- topology export --case 600 --output case600.json
 *   cargo run -q -p fluxion --bin fluxion -- topology export --case 900 --output case900.json
 * from the #3963 exporter at worktree HEAD. Regenerate the same way when the
 * exporter changes.
 */

import type { TopologyDocument } from "../types";
import case600 from "./case600.json";
import case900 from "./case900.json";

export interface TopologyFixture {
  id: string;
  label: string;
  doc: TopologyDocument;
}

export const CASE_600: TopologyDocument = case600 as TopologyDocument;
export const CASE_900: TopologyDocument = case900 as TopologyDocument;

export const FIXTURES: readonly TopologyFixture[] = [
  { id: "case600", label: "ASHRAE 140 Case 600 — low mass", doc: CASE_600 },
  { id: "case900", label: "ASHRAE 140 Case 900 — high mass", doc: CASE_900 },
];

export function fixtureById(id: string): TopologyFixture | undefined {
  return FIXTURES.find((f) => f.id === id);
}
