use crate::helpers::modpack::ModManager;
use crate::helpers::positions::WorldPos;
use crate::world::buildings::buildings::BuildingId;
use crate::world::buildings::utilities::utilities::{
    Utility, UtilityId, UtilityRail, UtilityUsage,
};
use crate::world::roads::road_structs::{NodeId, SegmentId};
use crate::world::roads::road_subsystem::Roads;
use crate::world::roads::roads::{RoadRegion, RoadRegionId, RoadRegions, RoadStorage, Segment};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use tracing::error;

pub type UtilityRegionId = u32;

const FLOW_EPS: f64 = 1e-9;

#[derive(Clone, Debug)]
pub enum EndpointType {
    Building {
        building_id: BuildingId,
        segment_id: SegmentId,
    },
    UninitializedBuilding,
}
impl EndpointType {
    pub fn segment_id(&self) -> Option<SegmentId> {
        match self {
            EndpointType::Building { segment_id, .. } => Some(*segment_id),
            _ => None,
        }
    }
    pub fn building_id(&self) -> Option<BuildingId> {
        match self {
            EndpointType::Building { building_id, .. } => Some(*building_id),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UtilityEndpoint {
    pub endpoint_type: EndpointType,
    pub utility_usages: Vec<UtilityUsage>,
}
impl UtilityEndpoint {
    pub fn usage(&self, utility: UtilityId) -> Option<UtilityUsage> {
        self.utility_usages.get(utility as usize).copied()
    }
}
#[derive(Clone, Debug, Default)]
pub struct UtilityRegion {
    pub utility: UtilityId,
    pub road_region: RoadRegionId,
    pub nodes: Vec<NodeId>,
    pub segments: Vec<SegmentId>,
    pub supply: f32,
    pub demand: f32,
    pub delivered: f32,
}

#[derive(Clone, Debug)]
pub enum UtilitySuggestion {
    AddProduction { amount: f32 },
    ConnectToProducer,
    AddRails { count: u32 },
}

#[derive(Clone, Debug)]
pub enum UtilityComplaintKind {
    Shortage {
        supply: f32,
        demand: f32,
    },
    NoSource {
        demand: f32,
    },
    Bottleneck {
        capacity: f32,
        load: f32,
        unmet: f32,
    },
}

#[derive(Clone, Debug)]
pub struct UtilityComplaint {
    pub utility: UtilityId,
    pub region: UtilityRegionId,
    pub segment: Option<SegmentId>,
    pub pos: WorldPos,
    pub kind: UtilityComplaintKind,
    pub suggestion: UtilitySuggestion,
}

struct UtilityFlow {
    segment_load: Vec<f32>,
    segment_cut: Vec<bool>,
    delivered: Vec<(UtilityRegionId, f32)>,
}

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

struct FlowGraph {
    to: Vec<usize>,
    cap: Vec<f64>,
    adj: Vec<Vec<usize>>,
    level: Vec<i32>,
    cursor: Vec<usize>,
}

impl FlowGraph {
    fn new(nodes: usize) -> Self {
        Self {
            to: Vec::new(),
            cap: Vec::new(),
            adj: vec![Vec::new(); nodes],
            level: vec![-1; nodes],
            cursor: vec![0; nodes],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, forward: f64, backward: f64) -> usize {
        let index = self.to.len();
        self.to.push(to);
        self.cap.push(forward);
        self.adj[from].push(index);
        self.to.push(from);
        self.cap.push(backward);
        self.adj[to].push(index + 1);
        index
    }

    fn build_levels(&mut self, source: usize, sink: usize) -> bool {
        self.level.fill(-1);
        self.level[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            for &edge in &self.adj[node] {
                let next = self.to[edge];
                if self.cap[edge] > FLOW_EPS && self.level[next] < 0 {
                    self.level[next] = self.level[node] + 1;
                    queue.push_back(next);
                }
            }
        }
        self.level[sink] >= 0
    }

    fn augment(&mut self, source: usize, sink: usize) -> f64 {
        let mut path: Vec<usize> = Vec::new();
        let mut node = source;
        let mut total = 0.0;
        loop {
            if node == sink {
                let mut bottleneck = f64::INFINITY;
                for &edge in &path {
                    bottleneck = bottleneck.min(self.cap[edge]);
                }
                for &edge in &path {
                    self.cap[edge] -= bottleneck;
                    self.cap[edge ^ 1] += bottleneck;
                }
                total += bottleneck;
                path.clear();
                node = source;
                continue;
            }
            let mut advanced = false;
            while self.cursor[node] < self.adj[node].len() {
                let edge = self.adj[node][self.cursor[node]];
                let next = self.to[edge];
                if self.cap[edge] > FLOW_EPS && self.level[next] == self.level[node] + 1 {
                    path.push(edge);
                    node = next;
                    advanced = true;
                    break;
                }
                self.cursor[node] += 1;
            }
            if advanced {
                continue;
            }
            if node == source {
                break;
            }
            let edge = path.pop().unwrap();
            node = self.to[edge ^ 1];
            self.cursor[node] += 1;
        }
        total
    }

    fn max_flow(&mut self, source: usize, sink: usize) -> f64 {
        let mut total = 0.0;
        while self.build_levels(source, sink) {
            self.cursor.fill(0);
            total += self.augment(source, sink);
        }
        total
    }

    fn reachable_from(&self, source: usize) -> Vec<bool> {
        let mut seen = vec![false; self.adj.len()];
        let mut queue = VecDeque::new();
        seen[source] = true;
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            for &edge in &self.adj[node] {
                let next = self.to[edge];
                if self.cap[edge] > FLOW_EPS && !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        seen
    }
}

pub type UtilityEndpointId = u32;

#[derive(Clone, Default)]
pub struct UtilityNetwork {
    pub utilities: Vec<Utility>,
    pub endpoints: Vec<Option<UtilityEndpoint>>,
    free_endpoints: Vec<UtilityEndpointId>,
    pub regions: Vec<UtilityRegion>,
    node_region: Vec<Vec<Option<UtilityRegionId>>>,
    complaints: Vec<UtilityComplaint>,
    network_dirty: bool,
}

impl UtilityNetwork {
    pub fn new(mod_manager: &ModManager) -> UtilityNetwork {
        let mut net = UtilityNetwork::default();
        net.load(mod_manager);
        net.network_dirty = true;
        net
    }
    pub fn mark_dirty(&mut self) {
        self.network_dirty = true;
    }
    fn load(&mut self, mod_manager: &ModManager) {
        let mut utilities = Vec::new();

        let paths = mod_manager.utility_paths();

        for path in paths {
            let name = match path.file_stem().and_then(|stem| stem.to_str()) {
                Some(name) => name.to_owned(),
                None => {
                    error!(
                        "[Utilities] Failed to determine Utility name from file '{}'. Filename is not valid UTF-8.",
                        path.display()
                    );
                    continue;
                }
            };

            let contents = match std::fs::read_to_string(&path) {
                Ok(contents) => contents,
                Err(err) => {
                    error!(
                        "[Utilities] Failed to read Utility file '{}'. Error: {}",
                        path.display(),
                        err
                    );
                    continue;
                }
            };

            let utility = match serde_yaml::from_str::<Utility>(&contents) {
                Ok(utility) => utility,
                Err(err) => {
                    error!(
                        "[Utilities] Failed to deserialize Utility file '{}'. Expected a valid Utility YAML. Error: {}",
                        path.display(),
                        err
                    );
                    continue;
                }
            };

            let utility_name = utility.name.as_str();
            if utilities
                .iter()
                .any(|util: &Utility| util.name.as_str() == utility_name)
            {
                tracing::warn!(
                    "[Utilities] Warning: Duplicate Utility name '{}'. The Utility from file '{}' overwrote the previously loaded Utility with the same name.",
                    utility_name,
                    path.display()
                );
            } else {
                utilities.push(utility);
            }
        }
        println!("[Utilities] Loaded {} Utility files", utilities.len());
        println!("{:?}", utilities);
        self.utilities = utilities;
    }
    pub fn add_endpoint(&mut self, endpoint: UtilityEndpoint) -> UtilityEndpointId {
        let id = match self.free_endpoints.pop() {
            Some(id) => id,
            None => {
                let id = self.endpoints.len() as UtilityEndpointId;
                self.endpoints.push(None);
                id
            }
        };

        self.endpoints[id as usize] = Some(endpoint);
        id
    }

    pub fn remove_endpoint(&mut self, endpoint_id: UtilityEndpointId) -> bool {
        let Some(slot) = self.endpoints.get_mut(endpoint_id as usize) else {
            return false;
        };

        if slot.take().is_none() {
            return false;
        }

        self.free_endpoints.push(endpoint_id);
        true
    }
    pub fn update_endpoint(&mut self, endpoint_id: UtilityEndpointId, endpoint: UtilityEndpoint) {
        if let Some(slot) = self.endpoints.get_mut(endpoint_id as usize) {
            slot.replace(endpoint);
        }
    }
    pub fn drain_complaints(&mut self) -> Vec<UtilityComplaint> {
        std::mem::take(&mut self.complaints)
    }

    pub fn utility_region_of_node(
        &self,
        utility: UtilityId,
        node: NodeId,
    ) -> Option<UtilityRegionId> {
        self.node_region
            .get(utility as usize)?
            .get(node.index())
            .copied()
            .flatten()
    }

    pub fn are_connected(&self, utility: UtilityId, a: NodeId, b: NodeId) -> bool {
        match (
            self.utility_region_of_node(utility, a),
            self.utility_region_of_node(utility, b),
        ) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }

    pub fn region(&self, id: UtilityRegionId) -> Option<&UtilityRegion> {
        self.regions.get(id as usize)
    }

    /// Fossil
    // pub fn update_network(&mut self, roads: &mut Roads) {
    //     self.pathfind_network(roads);
    //     self.check_network(roads);
    // }

    fn segment_carries(segment: &Segment, utility: UtilityId) -> bool {
        if segment.utility_rails.is_empty() {
            !segment.lanes().is_empty()
        } else {
            segment
                .utility_rails
                .iter()
                .any(|rail| rail.utility_id == utility)
        }
    }

    fn segment_node_slots(storage: &RoadStorage, segment: &Segment) -> Option<(usize, usize)> {
        let valid = storage.node_safe(segment.start()).is_some()
            && storage.node_safe(segment.end()).is_some();
        valid.then(|| (segment.start().index(), segment.end().index()))
    }

    fn segment_pos(storage: &RoadStorage, segment: &Segment) -> WorldPos {
        let a = storage.node(segment.start()).pos();
        let b = storage.node(segment.end()).pos();
        a.add_vec3(a.delta_to(b) * 0.5)
    }

    pub fn pathfind_network(&mut self, roads: &mut Roads) {
        println!(
            "Pathfinding Utility Network. Time: {:?}",
            Instant::now().elapsed()
        );
        let storage = &mut roads.road_manager.roads;
        let slots = storage.nodes.len();

        let mut road_sets = DisjointSet::new(slots);
        for (_, segment) in storage.iter_segments() {
            if let Some((a, b)) = Self::segment_node_slots(storage, segment) {
                road_sets.union(a, b);
            }
        }

        let mut road_regions = RoadRegions::default();
        road_regions.node_to_region = vec![None; slots];
        let mut root_to_road_region: HashMap<usize, RoadRegionId> = HashMap::new();
        for (node_id, _) in storage.iter_nodes() {
            let root = road_sets.find(node_id.index());
            let region_id = match root_to_road_region.get(&root) {
                Some(&id) => id,
                None => {
                    let id = road_regions.regions.len() as RoadRegionId;
                    road_regions.regions.push(RoadRegion::new());
                    root_to_road_region.insert(root, id);
                    id
                }
            };
            road_regions.regions[region_id as usize].push_node(node_id);
            road_regions.node_to_region[node_id.index()] = Some(region_id);
        }

        self.regions.clear();
        self.node_region = vec![vec![None; slots]; self.utilities.len()];

        for utility_index in 0..self.utilities.len() {
            let utility = utility_index as UtilityId;

            let mut sets = DisjointSet::new(slots);
            for (_, segment) in storage.iter_segments() {
                if !Self::segment_carries(segment, utility) {
                    continue;
                }
                if let Some((a, b)) = Self::segment_node_slots(storage, segment) {
                    sets.union(a, b);
                }
            }

            let mut root_to_region: HashMap<usize, UtilityRegionId> = HashMap::new();
            for (segment_id, segment) in storage.iter_segments() {
                if !Self::segment_carries(segment, utility) {
                    continue;
                }
                let Some((a, _)) = Self::segment_node_slots(storage, segment) else {
                    continue;
                };
                let root = sets.find(a);
                let region_id = match root_to_region.get(&root) {
                    Some(&id) => id,
                    None => {
                        let id = self.regions.len() as UtilityRegionId;
                        self.regions.push(UtilityRegion {
                            utility,
                            road_region: road_regions.node_to_region[a].unwrap_or(0),
                            ..Default::default()
                        });
                        root_to_region.insert(root, id);
                        id
                    }
                };
                let region = &mut self.regions[region_id as usize];
                region.segments.push(segment_id);
                for node in [segment.start(), segment.end()] {
                    let slot = &mut self.node_region[utility_index][node.index()];
                    if slot.is_none() {
                        *slot = Some(region_id);
                        region.nodes.push(node);
                    }
                }
            }
        }

        storage.road_regions = road_regions;
        self.network_dirty = false;
    }

    fn solve_flow(
        storage: &RoadStorage,
        endpoints: &[Option<UtilityEndpoint>],
        node_region: &[Vec<Option<UtilityRegionId>>],
        utility: UtilityId,
        capacity_of: impl Fn(&Segment) -> f64,
    ) -> UtilityFlow {
        let node_slots = storage.nodes.len();
        let source = node_slots;
        let sink = node_slots + 1;
        let mut graph = FlowGraph::new(node_slots + 2);

        let mut segment_edge: Vec<Option<(usize, f64)>> = vec![None; storage.segments.len()];

        for (segment_id, segment) in storage.iter_segments() {
            let capacity = capacity_of(segment);

            if capacity <= FLOW_EPS {
                continue;
            }

            let Some((a, b)) = Self::segment_node_slots(storage, segment) else {
                continue;
            };

            let edge = graph.add_edge(a, b, capacity, capacity);
            segment_edge[segment_id.index()] = Some((edge, capacity));
        }

        let mut sink_arcs: Vec<(usize, f64, UtilityRegionId)> = Vec::new();

        for endpoint in endpoints {
            let Some(endpoint) = endpoint.as_ref() else {
                continue;
            };

            let Some(usage) = endpoint.usage(utility) else {
                continue;
            };

            let Some(segment_id) = endpoint.endpoint_type.segment_id() else {
                continue;
            };

            let Some(segment) = storage.segment_safe(segment_id) else {
                continue;
            };

            let Some(region) = node_region
                .get(utility as usize)
                .and_then(|regions| regions.get(segment.start().index()))
                .copied()
                .flatten()
            else {
                continue;
            };

            for node in [segment.start(), segment.end()] {
                if usage.production > 0.0 {
                    graph.add_edge(source, node.index(), usage.production as f64 * 0.5, 0.0);
                }

                if usage.consumption > 0.0 {
                    let capacity = usage.consumption as f64 * 0.5;
                    let edge = graph.add_edge(node.index(), sink, capacity, 0.0);
                    sink_arcs.push((edge, capacity, region));
                }
            }
        }

        graph.max_flow(source, sink);

        let delivered = sink_arcs
            .iter()
            .map(|&(edge, capacity, region)| (region, (capacity - graph.cap[edge]) as f32))
            .collect();

        let reachable = graph.reachable_from(source);

        let mut segment_load = vec![0.0f32; storage.segments.len()];
        let mut segment_cut = vec![false; storage.segments.len()];

        for (segment_id, segment) in storage.iter_segments() {
            let slot = segment_id.index();

            let Some((edge, capacity)) = segment_edge[slot] else {
                continue;
            };

            segment_load[slot] = (capacity - graph.cap[edge]).abs() as f32;
            segment_cut[slot] =
                reachable[segment.start().index()] != reachable[segment.end().index()];
        }

        UtilityFlow {
            segment_load,
            segment_cut,
            delivered,
        }
    }

    fn assign_rails(
        lane_count: usize,
        per_rail: &[f32],
        load_of: impl Fn(usize) -> f32,
    ) -> Vec<UtilityRail> {
        let utility_count = per_rail.len();
        let total = lane_count * utility_count;
        if total == 0 {
            return Vec::new();
        }

        let need: Vec<usize> = (0..utility_count)
            .map(|u| (load_of(u) / per_rail[u]).ceil() as usize)
            .collect();
        let mut counts = vec![1usize; utility_count];
        let mut remaining = total - utility_count;

        while remaining > 0 {
            let mut best = None;
            let mut best_gap = 0usize;
            for u in 0..utility_count {
                let gap = need[u].saturating_sub(counts[u]);
                if gap > best_gap {
                    best_gap = gap;
                    best = Some(u);
                }
            }
            let Some(u) = best else {
                break;
            };
            counts[u] += 1;
            remaining -= 1;
        }

        let mut cursor = 0;
        while remaining > 0 {
            counts[cursor % utility_count] += 1;
            cursor += 1;
            remaining -= 1;
        }

        let mut rails = Vec::with_capacity(total);
        for u in 0..utility_count {
            for _ in 0..counts[u] {
                rails.push(UtilityRail {
                    utility_id: u as UtilityId,
                    capacity: per_rail[u],
                    usage: 0.0,
                    secondary_value: 0.0,
                });
            }
        }
        rails
    }

    pub fn check_network(&mut self, roads: &mut Roads) {
        if self.network_dirty {
            self.pathfind_network(roads);
            self.network_dirty = false;
        }

        self.complaints.clear();

        let utility_count = self.utilities.len();
        if utility_count == 0 {
            return;
        }

        let storage = &mut roads.road_manager.roads;
        let segment_slots = storage.segments.len();

        let per_rail: Vec<f32> = self
            .utilities
            .iter()
            .map(|utility| utility.amount_per_rail.max(f32::EPSILON))
            .collect();

        for region in &mut self.regions {
            region.supply = 0.0;
            region.demand = 0.0;
            region.delivered = 0.0;
        }

        for endpoint in &self.endpoints {
            let Some(endpoint) = endpoint.as_ref() else {
                continue;
            };

            let Some(segment_id) = endpoint.endpoint_type.segment_id() else {
                continue;
            };

            let Some(segment) = storage.segment_safe(segment_id) else {
                continue;
            };

            for utility_index in 0..utility_count {
                let utility = utility_index as UtilityId;

                let Some(usage) = endpoint.usage(utility) else {
                    continue;
                };

                let Some(region) = self
                    .node_region
                    .get(utility_index)
                    .and_then(|regions| regions.get(segment.start().index()))
                    .copied()
                    .flatten()
                else {
                    continue;
                };

                let region = &mut self.regions[region as usize];
                region.supply += usage.production;
                region.demand += usage.consumption;
            }
        }

        let mut wanted: Vec<Vec<f32>> = Vec::with_capacity(utility_count);
        for utility_index in 0..utility_count {
            let rail_capacity = per_rail[utility_index] as f64;
            let all_rails = utility_count as f64;
            let flow = Self::solve_flow(
                &*storage,
                &self.endpoints,
                &self.node_region,
                utility_index as UtilityId,
                |segment| segment.lanes().len() as f64 * all_rails * rail_capacity,
            );
            wanted.push(flow.segment_load);
        }

        for slot in 0..segment_slots {
            let Some(lane_count) = storage.segments[slot].as_ref().map(|s| s.lanes().len()) else {
                continue;
            };
            let rails = Self::assign_rails(lane_count, &per_rail, |u| wanted[u][slot]);
            if let Some(segment) = storage.segments[slot].as_mut() {
                segment.utility_rails = rails;
            }
        }

        let mut loads: Vec<Vec<f32>> = Vec::with_capacity(utility_count);
        let mut cuts: Vec<Vec<bool>> = Vec::with_capacity(utility_count);
        for utility_index in 0..utility_count {
            let utility = utility_index as UtilityId;
            let flow = Self::solve_flow(
                &*storage,
                &self.endpoints,
                &self.node_region,
                utility,
                |segment| {
                    segment
                        .utility_rails
                        .iter()
                        .filter(|rail| rail.utility_id == utility)
                        .map(|rail| rail.capacity as f64)
                        .sum()
                },
            );
            for (region, amount) in &flow.delivered {
                self.regions[*region as usize].delivered += *amount;
            }
            loads.push(flow.segment_load);
            cuts.push(flow.segment_cut);
        }

        let mut rail_counts = vec![0usize; utility_count];
        for slot in 0..segment_slots {
            let Some(segment) = storage.segments[slot].as_mut() else {
                continue;
            };
            rail_counts.fill(0);
            for rail in &segment.utility_rails {
                rail_counts[rail.utility_id as usize] += 1;
            }
            for rail in &mut segment.utility_rails {
                let u = rail.utility_id as usize;
                rail.usage = (loads[u][slot] / rail_counts[u] as f32).min(rail.capacity);
            }
        }

        let mut producing = vec![false; utility_count];
        for region in &self.regions {
            if region.supply > 1e-6 {
                producing[region.utility as usize] = true;
            }
        }

        let mut region_deficit = vec![0.0f32; self.regions.len()];
        for (region_index, region) in self.regions.iter().enumerate() {
            if region.demand <= 1e-6 {
                continue;
            }
            let tolerance = (region.demand * 1e-4).max(1e-3);
            let pos = Self::segment_pos(&*storage, storage.segment(region.segments[0]));
            let region_id = region_index as UtilityRegionId;

            if region.supply <= 1e-6 {
                let suggestion = if producing[region.utility as usize] {
                    UtilitySuggestion::ConnectToProducer
                } else {
                    UtilitySuggestion::AddProduction {
                        amount: region.demand,
                    }
                };
                self.complaints.push(UtilityComplaint {
                    utility: region.utility,
                    region: region_id,
                    segment: None,
                    pos,
                    kind: UtilityComplaintKind::NoSource {
                        demand: region.demand,
                    },
                    suggestion,
                });
                continue;
            }

            if region.supply + tolerance < region.demand {
                self.complaints.push(UtilityComplaint {
                    utility: region.utility,
                    region: region_id,
                    segment: None,
                    pos,
                    kind: UtilityComplaintKind::Shortage {
                        supply: region.supply,
                        demand: region.demand,
                    },
                    suggestion: UtilitySuggestion::AddProduction {
                        amount: region.demand - region.supply,
                    },
                });
            }

            let deliverable = region.supply.min(region.demand);
            if region.delivered + tolerance < deliverable {
                region_deficit[region_index] = deliverable - region.delivered;
            }
        }

        for (segment_id, segment) in storage.iter_segments() {
            let slot = segment_id.index();
            for utility_index in 0..utility_count {
                if !cuts[utility_index][slot] {
                    continue;
                }
                let Some(region) = self.node_region[utility_index]
                    .get(segment.start().index())
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                let unmet = region_deficit[region as usize];
                if unmet <= 0.0 {
                    continue;
                }
                let utility = utility_index as UtilityId;
                let capacity: f32 = segment
                    .utility_rails
                    .iter()
                    .filter(|rail| rail.utility_id == utility)
                    .map(|rail| rail.capacity)
                    .sum();
                self.complaints.push(UtilityComplaint {
                    utility,
                    region,
                    segment: Some(segment_id),
                    pos: Self::segment_pos(&*storage, segment),
                    kind: UtilityComplaintKind::Bottleneck {
                        capacity,
                        load: loads[utility_index][slot],
                        unmet,
                    },
                    suggestion: UtilitySuggestion::AddRails {
                        count: ((unmet / per_rail[utility_index]).ceil() as u32).max(1),
                    },
                });
            }
        }
    }
}
