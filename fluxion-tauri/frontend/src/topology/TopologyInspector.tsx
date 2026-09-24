/**
 * Inspector side panel for the topology viewer (issue #3965): shows the
 * selected node/edge metadata (areas, capacitances, conductances, h values,
 * split fractions, schedule/attribute tables), or a document summary when
 * nothing is selected. Upstream sources / downstream sinks are listed for the
 * current selection.
 */

import type { EdgeView, NodeView, TopologyViewModel } from "./viewModel";

export interface TopologySelection {
  type: "node" | "edge";
  id: string;
}

interface InspectorProps {
  vm: TopologyViewModel;
  selection: TopologySelection | null;
  onSelectNode: (id: string) => void;
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <tr>
      <th scope="row">{k}</th>
      <td>{v}</td>
    </tr>
  );
}

function attrRows(attributes: Readonly<Record<string, number>> | undefined): Array<{ k: string; v: string }> {
  if (!attributes) return [];
  return Object.entries(attributes)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => ({ k, v: formatNum(v) }));
}

export function formatNum(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 0.01 || abs >= 100000)) return v.toExponential(3);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(3);
}

function AttrTable({ attributes, title }: { attributes: Readonly<Record<string, number>> | undefined; title: string }) {
  const rows = attrRows(attributes);
  if (rows.length === 0) return null;
  return (
    <div className="topo-inspect-section">
      <h4>{title}</h4>
      <table className="topo-inspect-table">
        <tbody>
          {rows.map((r) => (
            <Row key={r.k} k={r.k} v={r.v} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NeighborList({
  title,
  ids,
  vm,
  onSelectNode,
}: {
  title: string;
  ids: string[];
  vm: TopologyViewModel;
  onSelectNode: (id: string) => void;
}) {
  if (ids.length === 0) return null;
  return (
    <div className="topo-inspect-section">
      <h4>{title}</h4>
      <ul className="topo-neighbor-list">
        {ids.map((id) => {
          const view = vm.nodeViewsById.get(id);
          return (
            <li key={id}>
              <button type="button" className="topo-neighbor-btn" onClick={() => onSelectNode(id)}>
                <span className="topo-swatch" style={{ background: view ? view.style.stroke : "#888" }} aria-hidden="true" />
                {view ? view.node.name : id}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function TopologyInspector({ vm, selection, onSelectNode }: InspectorProps) {
  if (!selection) {
    return (
      <aside className="topo-inspector" aria-label="Topology inspector">
        <h3>{vm.modelName}</h3>
        <p className="topo-inspect-muted">
          Select a node or coupling in the diagram to inspect its thermal metadata.
        </p>
        <div className="topo-inspect-section">
          <h4>Document</h4>
          <table className="topo-inspect-table">
            <tbody>
              <Row k="Source" v={vm.modelSource} />
              <Row k="Nodes" v={String(vm.nodeCount)} />
              <Row k="Couplings" v={String(vm.edgeCount)} />
              <Row k="Findings" v={String(vm.diagnostics.length)} />
            </tbody>
          </table>
        </div>
        <div className="topo-inspect-section">
          <h4>Node kinds</h4>
          <table className="topo-inspect-table">
            <tbody>
              {vm.legendNodeKinds.map((k) => (
                <Row key={k} k={k} v={String(vm.nodes.filter((n) => n.node.kind === k).length)} />
              ))}
            </tbody>
          </table>
        </div>
      </aside>
    );
  }

  if (selection.type === "node") {
    const view: NodeView | undefined = vm.nodeViewsById.get(selection.id);
    if (!view) return <aside className="topo-inspector" aria-label="Topology inspector"><p>Unknown node.</p></aside>;
    const n = view.node;
    return (
      <aside className="topo-inspector" aria-label="Topology inspector">
        <h3>{n.name}</h3>
        <p className="topo-inspect-kind" style={{ color: view.style.stroke }}>
          {view.style.label} · {n.kind}
        </p>
        {view.badges.length > 0 && (
          <div className="topo-inspect-section topo-badge-list" role="alert">
            {view.badges.map((b, i) => (
              <p key={i} className="topo-badge-note">
                <strong>{b.code}</strong> — {b.message}
              </p>
            ))}
          </div>
        )}
        <div className="topo-inspect-section">
          <table className="topo-inspect-table">
            <tbody>
              <Row k="ID" v={n.id} />
              {n.zone_id !== undefined && <Row k="Zone" v={n.zone_id} />}
              {n.area_m2 !== undefined && <Row k="Area" v={`${formatNum(n.area_m2)} m²`} />}
              {n.volume_m3 !== undefined && <Row k="Volume" v={`${formatNum(n.volume_m3)} m³`} />}
              {n.capacitance_j_per_k !== undefined && <Row k="Capacitance" v={`${formatNum(n.capacitance_j_per_k)} J/K`} />}
              {n.tilt_deg !== undefined && <Row k="Tilt" v={`${formatNum(n.tilt_deg)}°`} />}
              {n.azimuth_deg !== undefined && <Row k="Azimuth" v={`${formatNum(n.azimuth_deg)}°`} />}
              {n.elevation_m !== undefined && <Row k="Elevation" v={`${formatNum(n.elevation_m)} m`} />}
            </tbody>
          </table>
        </div>
        <AttrTable attributes={n.attributes} title="Attributes" />
        <NeighborList title="Upstream sources" ids={[...(vm.adjacency.upstream.get(n.id) ?? [])].sort()} vm={vm} onSelectNode={onSelectNode} />
        <NeighborList title="Downstream sinks" ids={[...(vm.adjacency.downstream.get(n.id) ?? [])].sort()} vm={vm} onSelectNode={onSelectNode} />
      </aside>
    );
  }

  const idx = vm.edges.findIndex((e) => `${e.edge.source_id}→${e.edge.target_id}:${e.edge.coupling_type}` === selection.id);
  if (idx < 0) return <aside className="topo-inspector" aria-label="Topology inspector"><p>Unknown coupling.</p></aside>;
  const ev: EdgeView = vm.edges[idx];
  const e = ev.edge;
  return (
    <aside className="topo-inspector" aria-label="Topology inspector">
      <h3>{ev.style.label}</h3>
      <p className="topo-inspect-kind" style={{ color: ev.style.color }}>
        {e.coupling_type}
      </p>
      <div className="topo-inspect-section">
        <table className="topo-inspect-table">
          <tbody>
            <Row k="From" v={vm.nodeViewsById.get(e.source_id)?.node.name ?? e.source_id} />
            <Row k="To" v={vm.nodeViewsById.get(e.target_id)?.node.name ?? e.target_id} />
            {e.conductance_w_per_k !== undefined && <Row k="Conductance" v={`${formatNum(e.conductance_w_per_k)} W/K`} />}
            {e.fraction !== undefined && <Row k="Fraction" v={formatNum(e.fraction)} />}
            <Row k="Bidirectional" v={e.bidirectional ? "yes" : "no"} />
          </tbody>
        </table>
      </div>
      <AttrTable attributes={e.attributes} title="Attributes" />
      <div className="topo-inspect-section">
        <h4>Endpoints</h4>
        <ul className="topo-neighbor-list">
          <li>
            <button type="button" className="topo-neighbor-btn" onClick={() => onSelectNode(e.source_id)}>
              <span className="topo-swatch" style={{ background: vm.nodeViewsById.get(e.source_id)?.style.stroke ?? "#888" }} aria-hidden="true" />
              {vm.nodeViewsById.get(e.source_id)?.node.name ?? e.source_id} (source)
            </button>
          </li>
          <li>
            <button type="button" className="topo-neighbor-btn" onClick={() => onSelectNode(e.target_id)}>
              <span className="topo-swatch" style={{ background: vm.nodeViewsById.get(e.target_id)?.style.stroke ?? "#888" }} aria-hidden="true" />
              {vm.nodeViewsById.get(e.target_id)?.node.name ?? e.target_id} (target)
            </button>
          </li>
        </ul>
      </div>
    </aside>
  );
}
