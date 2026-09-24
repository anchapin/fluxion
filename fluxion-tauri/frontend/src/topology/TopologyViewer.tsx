/**
 * Interactive 2D topology section viewer (issue #3965).
 *
 * Renders a topology_v1 document (from `fluxion topology export`, #3963) as a
 * hand-rolled SVG cross-section/graph: zones, surfaces, construction layers,
 * internal mass, gains and HVAC terminals as nodes; heat-transfer couplings
 * as curved edges colored by kind. Pan (drag), zoom (wheel), click-to-inspect
 * with upstream/downstream highlighting, kind legend, optional conductance
 * ("flux") overlay scaling, and diagnostic badges (#3964 linter codes +
 * client-side checks).
 *
 * Loads documents via the bundled fixtures, file picker, drag-and-drop, or
 * `?model=<url>` query parameter. A dropped/fetched JSON that looks like a
 * lint report (`findings` array, no `nodes`) is consumed as the lint report.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PanZoom } from "./panZoom";
import { fit as fitTo, pan as panBy, zoomAt } from "./panZoom";
import type { LintReport, TopologyDocument } from "./types";
import { buildViewModel, highlightFor, keyEdge, parseTopologyDocument } from "./viewModel";
import { TopologyInspector, type TopologySelection } from "./TopologyInspector";
import { FIXTURES, fixtureById } from "./fixtures";

function truncate(s: string, max = 22): string {
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

export default function TopologyViewer(): JSX.Element {
  const [doc, setDoc] = useState<TopologyDocument>(FIXTURES[0].doc);
  const [fixtureId, setFixtureId] = useState<string>(FIXTURES[0].id);
  const [docLabel, setDocLabel] = useState<string>(FIXTURES[0].label);
  const [lintReport, setLintReport] = useState<LintReport | undefined>(undefined);
  const [selection, setSelection] = useState<TopologySelection | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [fluxOverlay, setFluxOverlay] = useState<boolean>(false);
  const [transform, setTransform] = useState<PanZoom>({ x: 0, y: 0, scale: 0.55 });
  const [error, setError] = useState<string | null>(null);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ px: number; py: number; tx: PanZoom; moved: boolean } | null>(null);

  const vm = useMemo(() => buildViewModel(doc, { fluxOverlay, lintReport }), [doc, fluxOverlay, lintReport]);

  const effectiveSelection: TopologySelection | null =
    hover !== null ? { type: "node", id: hover } : selection;
  const highlight = useMemo(() => highlightFor(vm, effectiveSelection), [vm, effectiveSelection]);

  const loadText = useCallback((text: string, label: string) => {
    const { doc: parsed, error: parseError } = parseTopologyDocument(text);
    if (parsed) {
      setDoc(parsed);
      setFixtureId("");
      setDocLabel(label);
      setSelection(null);
      setHover(null);
      setError(null);
      return;
    }
    // Not a topology document — maybe a lint report from `fluxion topology lint`.
    let maybeReport: LintReport | undefined;
    try {
      const raw = JSON.parse(text) as Record<string, unknown>;
      if (Array.isArray(raw.findings)) maybeReport = raw as unknown as LintReport;
    } catch {
      /* not JSON at all */
    }
    if (maybeReport) {
      setLintReport(maybeReport);
      setError(null);
      return;
    }
    setError(parseError ?? "Unsupported file.");
  }, []);

  // Deep-link support: ?model=<url> (and optional ?lint=<url>), browser only.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const modelUrl = params.get("model");
    if (!modelUrl) return;
    let cancelled = false;
    void fetch(modelUrl)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((text) => {
        if (!cancelled) loadText(text, modelUrl);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(`Could not load ?model=${modelUrl}: ${(err as Error).message}`);
      });
    const lintUrl = params.get("lint");
    if (lintUrl) {
      void fetch(lintUrl)
        .then(async (res) => res.text())
        .then((text) => {
          const raw = JSON.parse(text) as Record<string, unknown>;
          if (Array.isArray(raw.findings) && !cancelled) setLintReport(raw as unknown as LintReport);
        })
        .catch(() => undefined);
    }
    return () => {
      cancelled = true;
    };
  }, [loadText]);

  // Fit the diagram into the viewport whenever the document changes.
  useEffect(() => {
    if (typeof window === "undefined" || !wrapRef.current) return;
    const el = wrapRef.current;
    setTransform(
      fitTo(
        { width: vm.layout.width, height: vm.layout.height },
        { width: el.clientWidth || 800, height: el.clientHeight || 600 },
      ),
    );
  }, [doc, vm.layout.width, vm.layout.height]);

  // Non-passive wheel zoom about the cursor.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      setTransform((t) => zoomAt(t, e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.15 : 1 / 1.15));
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>): void => {
    if (e.button !== 0) return;
    dragRef.current = { px: e.clientX, py: e.clientY, tx: transform, moved: false };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>): void => {
    const d = dragRef.current;
    if (!d) return;
    const dx = e.clientX - d.px;
    const dy = e.clientY - d.py;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (d.moved) setTransform(panBy(d.tx, dx, dy));
  };
  const onPointerUp = (): void => {
    dragRef.current = null;
  };

  const selectNode = useCallback((id: string) => setSelection({ type: "node", id }), []);
  const selectEdge = useCallback(
    (id: string) => setSelection((s) => (s && s.type === "edge" && s.id === id ? null : { type: "edge", id })),
    [],
  );

  const markerColors = useMemo(() => {
    const colors = new Set<string>();
    for (const e of vm.edges) colors.add(e.style.color);
    return [...colors];
  }, [vm.edges]);

  const nodeOpacity = (id: string): string =>
    effectiveSelection && !highlight.nodeIds.has(id) ? "0.22" : "1";
  const edgeOpacity = (idx: number): string =>
    effectiveSelection && !highlight.edgeIdxs.has(idx) ? "0.1" : "1";

  const onDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    void file.text().then((text) => loadText(text, file.name));
  };

  return (
    <div
      className="topo-viewer"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          setSelection(null);
          setHover(null);
        }
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
    >
      <div className="topo-toolbar" role="toolbar" aria-label="Topology viewer controls">
        <label className="topo-toolbar-label" htmlFor="topo-fixture">
          Model
          <select
            id="topo-fixture"
            value={fixtureId}
            onChange={(e) => {
              const f = fixtureById(e.target.value);
              if (f) {
                setDoc(f.doc);
                setFixtureId(f.id);
                setDocLabel(f.label);
                setSelection(null);
                setHover(null);
                setError(null);
              }
            }}
          >
            {FIXTURES.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
            {fixtureId === "" && <option value="">{docLabel} (loaded)</option>}
          </select>
        </label>

        <label className="topo-toolbar-btn">
          Load JSON…
          <input
            type="file"
            accept="application/json,.json"
            aria-label="Load topology JSON file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void file.text().then((text) => loadText(text, file.name));
              e.target.value = "";
            }}
          />
        </label>

        <label className="topo-toolbar-check">
          <input
            type="checkbox"
            checked={fluxOverlay}
            onChange={(e) => setFluxOverlay(e.target.checked)}
          />
          Conductance overlay
        </label>

        <button
          type="button"
          onClick={() =>
            wrapRef.current &&
            setTransform(
              fitTo(
                { width: vm.layout.width, height: vm.layout.height },
                { width: wrapRef.current.clientWidth || 800, height: wrapRef.current.clientHeight || 600 },
              ),
            )
          }
        >
          Fit
        </button>
        <button type="button" onClick={() => setTransform({ x: 0, y: 0, scale: 1 })}>
          1:1
        </button>

        <span className="topo-toolbar-stats" aria-live="polite">
          {vm.nodeCount} nodes · {vm.edgeCount} couplings · {vm.diagnostics.length} findings
        </span>
      </div>

      {error && (
        <p className="topo-error" role="alert">
          {error}
        </p>
      )}

      <div className="topo-body">
        <div
          className="topo-canvas-wrap"
          ref={wrapRef}
          aria-label={`Topology section diagram for ${vm.modelName}. Drag to pan, scroll to zoom, click nodes or couplings to inspect.`}
          role="application"
        >
          <svg
            ref={svgRef}
            className="topo-svg"
            width="100%"
            height="100%"
            data-testid="topology-svg"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}
            onClick={(e) => {
              if (!dragRef.current?.moved && (e.target as Element).tagName.toLowerCase() === "svg") setSelection(null);
            }}
          >
            <defs>
              {markerColors.map((color) => (
                <marker
                  key={color}
                  id={`topo-arrow-${color.slice(1)}`}
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
                </marker>
              ))}
            </defs>
            <g transform={`translate(${transform.x.toFixed(1)} ${transform.y.toFixed(1)}) scale(${transform.scale.toFixed(3)})`}>
              {vm.edges.map((e, idx) => (
                <path
                  key={keyEdge(e.edge)}
                  d={e.path}
                  fill="none"
                  stroke={e.style.color}
                  strokeWidth={e.strokeWidth}
                  strokeDasharray={e.style.dash}
                  opacity={edgeOpacity(idx)}
                  markerEnd={`url(#topo-arrow-${e.style.color.slice(1)})`}
                  className={selection?.type === "edge" && selection.id === keyEdge(e.edge) ? "topo-edge-selected" : "topo-edge"}
                  tabIndex={-1}
                  aria-label={`${e.style.label} from ${vm.nodeViewsById.get(e.edge.source_id)?.node.name ?? e.edge.source_id} to ${vm.nodeViewsById.get(e.edge.target_id)?.node.name ?? e.edge.target_id}`}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    selectEdge(keyEdge(e.edge));
                  }}
                />
              ))}
              {vm.nodes.map((n) => (
                <g
                  key={n.node.id}
                  transform={`translate(${n.x.toFixed(1)} ${n.y.toFixed(1)})`}
                  opacity={nodeOpacity(n.node.id)}
                  className={
                    "topo-node" +
                    (selection?.type === "node" && selection.id === n.node.id ? " topo-node-selected" : "") +
                    (hover === n.node.id ? " topo-node-hover" : "")
                  }
                  role="button"
                  tabIndex={0}
                  aria-label={`${n.node.name}, ${n.style.label}${n.badges.length > 0 ? `, ${n.badges.length} diagnostic findings` : ""}`}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    selectNode(n.node.id);
                  }}
                  onKeyDown={(ev) => {
                    if (ev.key === "Enter" || ev.key === " ") {
                      ev.preventDefault();
                      selectNode(n.node.id);
                    }
                  }}
                  onMouseEnter={() => setHover(n.node.id)}
                  onMouseLeave={() => setHover((h) => (h === n.node.id ? null : h))}
                >
                  <rect width={n.w} height={n.h} rx={n.role === "zone_air" ? 10 : 5} fill={n.style.fill} stroke={n.style.stroke} />
                  <text x={n.w / 2} y={n.h / 2 - 3} textAnchor="middle" className="topo-node-label">
                    {truncate(n.node.name)}
                  </text>
                  <text x={n.w / 2} y={n.h / 2 + 13} textAnchor="middle" className="topo-node-sublabel">
                    {n.role === "zone_air" && n.node.volume_m3 !== undefined
                      ? `${Math.round(n.node.volume_m3)} m³`
                      : n.node.area_m2 !== undefined
                        ? `${Math.round(n.node.area_m2)} m²`
                        : n.style.label}
                  </text>
                  {n.badges.length > 0 && (
                    <g className="topo-badge">
                      <circle cx={n.w - 2} cy={2} r={9} />
                      <text x={n.w - 2} y={6} textAnchor="middle">
                        {n.badges.length}
                      </text>
                      <title>{n.badges.map((b) => `${b.code}: ${b.message}`).join(" | ")}</title>
                    </g>
                  )}
                </g>
              ))}
            </g>
          </svg>

          <div className="topo-legend" aria-label="Legend">
            <h4>Nodes</h4>
            <ul>
              {vm.legendNodeKinds.map((k) => (
                <li key={k}>
                  <span className="topo-swatch" style={{ background: vm.nodes.find((n) => n.node.kind === k)?.style.stroke }} aria-hidden="true" />
                  {vm.nodes.find((n) => n.node.kind === k)?.style.label ?? k}
                </li>
              ))}
            </ul>
            <h4>Couplings</h4>
            <ul>
              {vm.legendEdgeKinds.map((k) => (
                <li key={k}>
                  <span className="topo-swatch topo-swatch-line" style={{ background: vm.edges.find((e) => e.edge.coupling_type === k)?.style.color }} aria-hidden="true" />
                  {vm.edges.find((e) => e.edge.coupling_type === k)?.style.label ?? k}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <TopologyInspector vm={vm} selection={selection} onSelectNode={selectNode} />
      </div>
    </div>
  );
}
