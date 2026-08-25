use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PortKind {
    Inlet,
    Outlet,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Port {
    pub id: usize,
    pub kind: PortKind,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentKind {
    HeatExchanger,
    Pump,
    Pipe,
    Valve,
    Tank,
    Boiler,
    Chiller,
    Junction,
    Split,
    Source,
    Sink,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node {
    pub id: NodeId,
    pub kind: ComponentKind,
    pub name: String,
    pub ports: Vec<Port>,
}

impl Node {
    pub fn new(id: NodeId, kind: ComponentKind, name: &str) -> Self {
        Self {
            id,
            kind,
            name: name.to_string(),
            ports: Vec::new(),
        }
    }

    pub fn with_ports(mut self, ports: Vec<Port>) -> Self {
        self.ports = ports;
        self
    }

    pub fn inlet_ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.iter().filter(|p| p.kind == PortKind::Inlet)
    }

    pub fn outlet_ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.iter().filter(|p| p.kind == PortKind::Outlet)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    pub id: EdgeId,
    pub from_node: NodeId,
    pub from_port: usize,
    pub to_node: NodeId,
    pub to_port: usize,
    pub fluid_type: String,
}

impl Edge {
    pub fn new(
        id: EdgeId,
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
        fluid_type: &str,
    ) -> Self {
        Self {
            id,
            from_node,
            from_port,
            to_node,
            to_port,
            fluid_type: fluid_type.to_string(),
        }
    }
}

pub struct FluidGraph {
    nodes: BTreeMap<NodeId, Node>,
    edges: BTreeMap<EdgeId, Edge>,
    adjacency: BTreeMap<NodeId, BTreeSet<EdgeId>>,
    reverse_adjacency: BTreeMap<NodeId, BTreeSet<EdgeId>>,
    next_node_id: usize,
    next_edge_id: usize,
}

impl FluidGraph {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            adjacency: BTreeMap::new(),
            reverse_adjacency: BTreeMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
        }
    }

    pub fn add_node(&mut self, kind: ComponentKind, name: &str) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = Node::new(id, kind, name);
        self.nodes.insert(id, node);
        self.adjacency.entry(id).or_default();
        self.reverse_adjacency.entry(id).or_default();
        id
    }

    pub fn add_port(&mut self, node_id: NodeId, kind: PortKind, name: &str) -> usize {
        let port_id = self.nodes.get(&node_id).map(|n| n.ports.len()).unwrap_or(0);
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.ports.push(Port {
                id: port_id,
                kind,
                name: name.to_string(),
            });
        }
        port_id
    }

    pub fn add_edge(
        &mut self,
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
        fluid_type: &str,
    ) -> Option<EdgeId> {
        if !self.nodes.contains_key(&from_node) || !self.nodes.contains_key(&to_node) {
            return None;
        }

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;

        let edge = Edge::new(id, from_node, from_port, to_node, to_port, fluid_type);
        self.edges.insert(id, edge.clone());

        self.adjacency.entry(from_node).or_default().insert(id);
        self.reverse_adjacency
            .entry(to_node)
            .or_default()
            .insert(id);

        Some(id)
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(&id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    pub fn outgoing_edges(&self, node_id: NodeId) -> impl Iterator<Item = &Edge> {
        self.adjacency
            .get(&node_id)
            .into_iter()
            .flatten()
            .filter_map(|eid| self.edges.get(eid))
    }

    pub fn incoming_edges(&self, node_id: NodeId) -> impl Iterator<Item = &Edge> {
        self.reverse_adjacency
            .get(&node_id)
            .into_iter()
            .flatten()
            .filter_map(|eid| self.edges.get(eid))
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn topological_sort(&self) -> Result<Vec<NodeId>, CycleError> {
        let mut in_degree: BTreeMap<NodeId, usize> = BTreeMap::new();
        for &node_id in self.nodes.keys() {
            in_degree.insert(node_id, 0);
        }

        for edge in self.edges.values() {
            if let Some(count) = in_degree.get_mut(&edge.to_node) {
                *count += 1;
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|&(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            if let Some(out_edges) = self.adjacency.get(&node_id) {
                for &edge_id in out_edges {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if let Some(degree) = in_degree.get_mut(&edge.to_node) {
                            *degree -= 1;
                            if *degree == 0 {
                                queue.push_back(edge.to_node);
                            }
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            let cycle_nodes: BTreeSet<NodeId> = result.iter().cloned().collect();
            let cycle: Vec<NodeId> = self
                .nodes
                .keys()
                .filter(|n| !cycle_nodes.contains(n))
                .cloned()
                .collect();
            Err(CycleError::CycleDetected(cycle))
        } else {
            Ok(result)
        }
    }

    pub fn find_cycles(&self) -> Vec<Vec<NodeId>> {
        let mut index = 0;
        let mut stack: Vec<NodeId> = Vec::new();
        let mut indices: BTreeMap<NodeId, Option<usize>> = BTreeMap::new();
        let mut low_links: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut on_stack: BTreeSet<NodeId> = BTreeSet::new();
        let mut sccs: Vec<Vec<NodeId>> = Vec::new();

        for &node_id in self.nodes.keys() {
            if !indices.contains_key(&node_id) {
                self.tarjan_scc(
                    node_id,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut low_links,
                    &mut on_stack,
                    &mut sccs,
                );
            }
        }

        sccs.into_iter().filter(|scc| scc.len() > 1).collect()
    }

    fn tarjan_scc(
        &self,
        node_id: NodeId,
        index: &mut usize,
        stack: &mut Vec<NodeId>,
        indices: &mut BTreeMap<NodeId, Option<usize>>,
        low_links: &mut BTreeMap<NodeId, usize>,
        on_stack: &mut BTreeSet<NodeId>,
        sccs: &mut Vec<Vec<NodeId>>,
    ) {
        indices.insert(node_id, Some(*index));
        low_links.insert(node_id, *index);
        *index += 1;
        stack.push(node_id);
        on_stack.insert(node_id);

        if let Some(out_edges) = self.adjacency.get(&node_id) {
            for &edge_id in out_edges {
                if let Some(edge) = self.edges.get(&edge_id) {
                    let successor = edge.to_node;
                    let succ_idx = indices.get(&successor).copied();
                    if succ_idx.is_none() {
                        self.tarjan_scc(
                            successor, index, stack, indices, low_links, on_stack, sccs,
                        );
                        let successor_low = *low_links.get(&successor).unwrap_or(&0);
                        let node_low = *low_links.get(&node_id).unwrap_or(&0);
                        low_links.insert(node_id, node_low.min(successor_low));
                    } else if let Some(Some(idx)) = succ_idx {
                        if on_stack.contains(&successor) {
                            let node_low = *low_links.get(&node_id).unwrap_or(&0);
                            low_links.insert(node_id, node_low.min(idx));
                        }
                    }
                }
            }
        }

        let node_low = *low_links.get(&node_id).unwrap_or(&0);
        let node_idx = indices.get(&node_id).and_then(|x| *x).unwrap_or(0);
        if node_low == node_idx {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w);
                if w == node_id {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    pub fn find_junctions(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|n| {
                matches!(n.kind, ComponentKind::Junction | ComponentKind::Split)
                    || (n.ports.len() > 2 && {
                        let in_ports = n.inlet_ports().count();
                        let out_ports = n.outlet_ports().count();
                        in_ports >= 1 && out_ports >= 2 || out_ports >= 1 && in_ports >= 2
                    })
            })
            .map(|n| n.id)
            .collect()
    }

    pub fn parallel_branches(&self, source: NodeId, sink: NodeId) -> Vec<Vec<NodeId>> {
        let mut all_paths: Vec<Vec<NodeId>> = Vec::new();
        let mut visited: BTreeSet<NodeId> = BTreeSet::new();
        self.dfs_paths(source, sink, vec![source], &mut visited, &mut all_paths);
        all_paths
    }

    fn dfs_paths(
        &self,
        current: NodeId,
        target: NodeId,
        path: Vec<NodeId>,
        visited: &mut BTreeSet<NodeId>,
        all_paths: &mut Vec<Vec<NodeId>>,
    ) {
        if current == target {
            all_paths.push(path);
            return;
        }

        visited.insert(current);

        if let Some(out_edges) = self.adjacency.get(&current) {
            for &edge_id in out_edges {
                if let Some(edge) = self.edges.get(&edge_id) {
                    let next = edge.to_node;
                    if !visited.contains(&next) {
                        let mut new_path = path.clone();
                        new_path.push(next);
                        self.dfs_paths(next, target, new_path, visited, all_paths);
                    }
                }
            }
        }

        visited.remove(&current);
    }

    pub fn has_algebraic_loops(&self) -> bool {
        !self.find_cycles().is_empty()
    }
}

impl Default for FluidGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum CycleError {
    CycleDetected(Vec<NodeId>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_thermal_system() -> FluidGraph {
        let mut graph = FluidGraph::new();

        let source = graph.add_node(ComponentKind::Source, "Boiler");
        graph.add_port(source, PortKind::Outlet, "HotWaterOut");

        let pump = graph.add_node(ComponentKind::Pump, "CirculationPump");
        graph.add_port(pump, PortKind::Inlet, "In");
        graph.add_port(pump, PortKind::Outlet, "Out");

        let hx = graph.add_node(ComponentKind::HeatExchanger, "HeatExchanger");
        graph.add_port(hx, PortKind::Inlet, "PrimaryIn");
        graph.add_port(hx, PortKind::Outlet, "PrimaryOut");
        graph.add_port(hx, PortKind::Inlet, "SecondaryIn");
        graph.add_port(hx, PortKind::Outlet, "SecondaryOut");

        let sink = graph.add_node(ComponentKind::Sink, "Radiator");
        graph.add_port(sink, PortKind::Inlet, "In");
        graph.add_port(sink, PortKind::Outlet, "Out");

        graph.add_edge(source, 0, pump, 0, "HotWater");
        graph.add_edge(pump, 1, hx, 0, "HotWater");
        graph.add_edge(hx, 1, sink, 0, "HotWater");
        graph.add_edge(sink, 1, hx, 2, "CooledWater");
        graph.add_edge(hx, 3, source, 0, "CooledWater");

        graph
    }

    fn create_simple_acyclic_system() -> FluidGraph {
        let mut graph = FluidGraph::new();

        let source = graph.add_node(ComponentKind::Source, "Boiler");
        graph.add_port(source, PortKind::Outlet, "Out");

        let pump = graph.add_node(ComponentKind::Pump, "Pump");
        graph.add_port(pump, PortKind::Inlet, "In");
        graph.add_port(pump, PortKind::Outlet, "Out");

        let hx = graph.add_node(ComponentKind::HeatExchanger, "HX");
        graph.add_port(hx, PortKind::Inlet, "In");
        graph.add_port(hx, PortKind::Outlet, "Out");

        let sink = graph.add_node(ComponentKind::Sink, "Sink");
        graph.add_port(sink, PortKind::Inlet, "In");

        graph.add_edge(source, 0, pump, 0, "Water");
        graph.add_edge(pump, 1, hx, 0, "Water");
        graph.add_edge(hx, 1, sink, 0, "Water");

        graph
    }

    fn create_system_with_cycle() -> FluidGraph {
        let mut graph = FluidGraph::new();

        let a = graph.add_node(ComponentKind::Source, "A");
        graph.add_port(a, PortKind::Outlet, "Out");

        let b = graph.add_node(ComponentKind::Pipe, "B");
        graph.add_port(b, PortKind::Inlet, "In");
        graph.add_port(b, PortKind::Outlet, "Out");

        let c = graph.add_node(ComponentKind::Pipe, "C");
        graph.add_port(c, PortKind::Inlet, "In");
        graph.add_port(c, PortKind::Outlet, "Out");

        graph.add_edge(a, 0, b, 0, "Water");
        graph.add_edge(b, 1, c, 0, "Water");
        graph.add_edge(c, 1, a, 0, "Water");

        graph
    }

    #[test]
    fn test_graph_construction() {
        let graph = create_simple_acyclic_system();

        assert_eq!(graph.num_nodes(), 4);
        assert_eq!(graph.num_edges(), 3);
    }

    #[test]
    fn test_topological_sort_acyclic() {
        let graph = create_simple_acyclic_system();

        let sorted = graph.topological_sort().expect("Should be acyclic");
        assert_eq!(sorted.len(), 4);

        let source_idx = sorted
            .iter()
            .position(|&n| graph.node(n).unwrap().name == "Boiler")
            .unwrap();
        let pump_idx = sorted
            .iter()
            .position(|&n| graph.node(n).unwrap().name == "Pump")
            .unwrap();
        let hx_idx = sorted
            .iter()
            .position(|&n| graph.node(n).unwrap().name == "HX")
            .unwrap();
        let sink_idx = sorted
            .iter()
            .position(|&n| graph.node(n).unwrap().name == "Sink")
            .unwrap();

        assert!(source_idx < pump_idx);
        assert!(pump_idx < hx_idx);
        assert!(hx_idx < sink_idx);
    }

    #[test]
    fn test_topological_sort_cyclic() {
        let graph = create_system_with_cycle();

        let result = graph.topological_sort();
        assert!(result.is_err());
    }

    #[test]
    fn test_cycle_detection() {
        let acyclic = create_simple_acyclic_system();
        assert!(acyclic.find_cycles().is_empty());
        assert!(!acyclic.has_algebraic_loops());

        let cyclic = create_system_with_cycle();
        let cycles = cyclic.find_cycles();
        assert!(!cycles.is_empty());
        assert!(cyclic.has_algebraic_loops());
    }

    #[test]
    fn test_junction_detection() {
        let mut graph = FluidGraph::new();

        let junction = graph.add_node(ComponentKind::Junction, "Junction");
        graph.add_port(junction, PortKind::Inlet, "In1");
        graph.add_port(junction, PortKind::Inlet, "In2");
        graph.add_port(junction, PortKind::Outlet, "Out");

        let split = graph.add_node(ComponentKind::Split, "Split");
        graph.add_port(split, PortKind::Inlet, "In");
        graph.add_port(split, PortKind::Outlet, "Out1");
        graph.add_port(split, PortKind::Outlet, "Out2");

        let junctions = graph.find_junctions();
        assert!(junctions.contains(&junction));
        assert!(junctions.contains(&split));
    }

    #[test]
    fn test_parallel_branches() {
        let mut graph = FluidGraph::new();

        let source = graph.add_node(ComponentKind::Source, "Source");
        let sink = graph.add_node(ComponentKind::Sink, "Sink");

        let pipe1 = graph.add_node(ComponentKind::Pipe, "Pipe1");
        graph.add_port(pipe1, PortKind::Inlet, "In");
        graph.add_port(pipe1, PortKind::Outlet, "Out");

        let pipe2 = graph.add_node(ComponentKind::Pipe, "Pipe2");
        graph.add_port(pipe2, PortKind::Inlet, "In");
        graph.add_port(pipe2, PortKind::Outlet, "Out");

        graph.add_edge(source, 0, pipe1, 0, "Water");
        graph.add_edge(source, 0, pipe2, 0, "Water");
        graph.add_edge(pipe1, 1, sink, 0, "Water");
        graph.add_edge(pipe2, 1, sink, 0, "Water");

        let branches = graph.parallel_branches(source, sink);
        assert_eq!(branches.len(), 2);
    }

    #[test]
    fn test_node_and_port_access() {
        let mut graph = FluidGraph::new();

        let node_id = graph.add_node(ComponentKind::HeatExchanger, "HX");
        graph.add_port(node_id, PortKind::Inlet, "In");
        graph.add_port(node_id, PortKind::Outlet, "Out");

        let node = graph.node(node_id).unwrap();
        assert_eq!(node.name, "HX");
        assert_eq!(node.inlet_ports().count(), 1);
        assert_eq!(node.outlet_ports().count(), 1);
    }
}
