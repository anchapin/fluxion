/**
 * Viewer rendering tests (issue #3965): SSR-strict markup assertions — the
 * component must render a meaningful SVG section view without a DOM (effects
 * and browser-only loading paths are skipped by design).
 *
 * Note on scope: interaction *handlers* are covered by the pure-logic suites
 * (viewModel/layout/diagnostics/panZoom). End-to-end browser tests
 * (Playwright) are out of scope here — the toolchain is not part of the
 * frontend devDependencies; see the issue-#3965 report note.
 */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import TopologyViewer from "../src/topology/TopologyViewer";
import { CASE_600 } from "../src/topology/fixtures";

function render(): string {
  return renderToStaticMarkup(createElement(TopologyViewer));
}

describe("TopologyViewer (static markup)", () => {
  const html = render();

  it("renders an SVG canvas with the default case-600 fixture", () => {
    expect(html).toContain('data-testid="topology-svg"');
    expect(html).toContain("<svg");
  });

  it("renders the zone air node with an accessible label", () => {
    expect(html).toContain("Zone 0 air");
    expect(html).toContain('role="button"');
    expect(html).toMatch(/aria-label="Zone 0 air, Zone air/);
  });

  it("renders all 34 nodes and a coupling path for each of the 47 edges", () => {
    const nodes = html.match(/<g[^>]*class="[^"]*topo-node/g) ?? [];
    expect(nodes.length).toBe(34);
    const paths = html.match(/<path[^>]*class="topo-edge/g) ?? [];
    expect(paths.length).toBe(47);
  });

  it("renders node boxes and labels for key kinds", () => {
    expect(html).toContain("Outdoor ambient");
    expect(html).toContain("exterior film");
    expect(html).toContain("HVAC terminal");
    // Every fixture node name (or its truncation) appears in the markup.
    for (const n of CASE_600.nodes) {
      expect(html).toContain(n.name.length > 22 ? `${n.name.slice(0, 21)}…` : n.name);
    }
  });

  it("renders the legend and the inspector default state", () => {
    expect(html).toContain("Legend");
    expect(html).toContain("Couplings");
    expect(html).toContain("Select a node or coupling");
    expect(html).toContain("34 nodes");
    expect(html).toContain("47 couplings");
  });

  it("renders the toolbar with fixture selector and overlay toggle", () => {
    expect(html).toContain('id="topo-fixture"');
    expect(html).toContain("ASHRAE 140 Case 600");
    expect(html).toContain("ASHRAE 140 Case 900");
    expect(html).toContain("Conductance overlay");
    expect(html).toContain("Load JSON…");
  });

  it("renders no diagnostic badges for the healthy case-600 export", () => {
    expect(html).not.toContain("topo-badge");
  });
});
