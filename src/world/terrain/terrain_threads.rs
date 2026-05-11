use crate::helpers::positions::{ChunkCoord, LodStep};
use crate::world::terrain::chunk_builder::{
    ChunkBuilder, ChunkHeightGrid, ChunkState, CpuChunkMesh,
};
use crate::world::terrain::terrain_editing::TerrainEdit;
use crate::world::terrain::terrain_gen::TerrainGenerator;
use crossbeam_channel::{Receiver, unbounded};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

#[derive(Clone)]
pub struct TerrainEditsSnapshot {
    pub edits: Arc<Vec<TerrainEdit>>,
    pub affected_chunks: Arc<HashSet<ChunkCoord>>,
}

impl TerrainEditsSnapshot {
    #[inline(always)]
    pub fn has_edits_on_chunk(&self, coord: ChunkCoord) -> bool {
        self.affected_chunks.contains(&coord)
    }
}

#[derive(Clone)]
pub struct LoadedChunkSnapshot {
    pub step: LodStep,
    pub height_grid: Arc<ChunkHeightGrid>,
}
// ─── Public request type ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PendingChunkRequest {
    pub coord: ChunkCoord,
    pub state: ChunkState,
    pub version: u64,
    pub version_atomic: Arc<AtomicU64>,
    pub has_edits: bool,
    pub priority: u64,
    pub in_progress: Arc<AtomicBool>,

    pub terrain_edits_snapshot: TerrainEditsSnapshot,
    pub loaded_snapshot: Arc<LoadedChunksSnapshot>,
}

impl PendingChunkRequest {
    #[inline]
    pub fn same_as(&self, other: &PendingChunkRequest) -> bool {
        self.coord == other.coord
            && self.state.same_as(&other.state)
            && self.has_edits == other.has_edits
    }
}

// ─── Priority heap entry ──────────────────────────────────────────────────────
//
// The heap stores only lightweight (priority, coord) pairs.  The actual request
// data lives in `WorkQueueInner::pending`, which is the canonical source of
// truth.  Heap entries can become stale (when a request is re-submitted with a
// new priority or removed entirely); they are discarded lazily the next time a
// worker reaches the top of the heap.  This avoids an O(n) heap search on every
// re-submission.

#[derive(Eq, PartialEq)]
struct HeapEntry {
    priority: u64,
    coord: ChunkCoord,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority value = closer to camera = pop first.
        // BinaryHeap is a max-heap, so this is correct as-is.
        self.priority
            .cmp(&other.priority)
            // Tiebreak deterministically so Ord is a total order.
            .then_with(|| self.coord.x.cmp(&other.coord.x))
            .then_with(|| self.coord.z.cmp(&other.coord.z))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ─── Work queue ───────────────────────────────────────────────────────────────

struct WorkQueueInner {
    /// Ordering structure.  May contain stale entries; always validate against
    /// `pending` before trusting a popped entry.
    heap: BinaryHeap<HeapEntry>,
    /// Canonical job data keyed by coord.  A coord present here with a given
    /// priority is the only valid job for that coord.
    pending: HashMap<ChunkCoord, PendingChunkRequest>,
    shutdown: bool,
}

struct WorkQueue {
    inner: Mutex<WorkQueueInner>,
    condvar: Condvar,
}

impl WorkQueue {
    fn new() -> Self {
        Self {
            inner: Mutex::new(WorkQueueInner {
                heap: BinaryHeap::new(),
                pending: HashMap::new(),
                shutdown: false,
            }),
            condvar: Condvar::new(),
        }
    }

    // ── Push ─────────────────────────────────────────────────────────────────
    //
    // Inserts or replaces a request.  If the same request is already pending,
    // it is a no-op.  Replacing a request with a new priority is O(log n):
    // we just insert a fresh heap entry; the old one becomes stale and will be
    // discarded for free the next time a worker drains the heap top.

    fn push(&self, req: PendingChunkRequest) {
        let mut inner = self.inner.lock().unwrap();

        if let Some(existing) = inner.pending.get(&req.coord) {
            // If same state and version, don't re-submit even if priority differs
            if existing.same_as(&req) && existing.version >= req.version {
                return;
            }
        }

        let priority = req.priority;
        let coord = req.coord;

        // Update the canonical record.  Any old heap entry with a different
        // priority is now stale and will be lazily discarded on the next pop.
        inner.pending.insert(coord, req);
        inner.heap.push(HeapEntry { priority, coord });

        // Wake exactly one sleeping worker — there is exactly one new job.
        self.condvar.notify_one();
    }

    // ── Pop ──────────────────────────────────────────────────────────────────
    //
    // Blocks cheaply (via Condvar) until a valid job is available or shutdown
    // is requested.  Holds the lock only for O(log n) heap operations; releases
    // it before the caller does any real work.

    fn pop(&self) -> Option<PendingChunkRequest> {
        let mut inner = self.inner.lock().unwrap();

        loop {
            // Drain stale entries from the top of the heap before deciding
            // whether there is real work to do.
            Self::drain_stale(&mut inner);

            if !inner.heap.is_empty() {
                // Top entry is valid — pop both the heap entry and the canonical
                // record and hand the job to the worker.
                let entry = inner.heap.pop().unwrap();

                let req = inner.pending.get(&entry.coord)?.clone();

                // Mark as in-progress (prevents re-submission)
                req.in_progress.store(true, AtomicOrdering::Relaxed);
                return Some(req);
            }

            if inner.shutdown {
                return None;
            }

            // No work available.  Release the lock and sleep until push() or
            // shutdown() wakes us.  This is a zero-CPU wait — no polling.
            inner = self.condvar.wait(inner).unwrap();
        }
    }

    // Remove stale heap entries (coord absent from `pending`, or priority
    // mismatch because the request was re-submitted with a different priority).
    #[inline]
    fn drain_stale(inner: &mut WorkQueueInner) {
        loop {
            match inner.heap.peek() {
                None => break,
                Some(entry) => {
                    let valid = inner
                        .pending
                        .get(&entry.coord)
                        .map_or(false, |r| r.priority == entry.priority);
                    if valid {
                        break; // top entry is current, stop draining
                    }
                    inner.heap.pop(); // stale — discard and check the next one
                }
            }
        }
    }

    // ── Other operations ─────────────────────────────────────────────────────

    fn has_request(&self, coord: ChunkCoord, state: &ChunkState) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.pending.get(&coord).map_or(false, |r| {
            r.state.same_as(state) && !r.in_progress.load(AtomicOrdering::Relaxed) // Ignore if in progress
        })
    }

    fn is_building(&self, coord: ChunkCoord) -> bool {
        let inner = self.inner.lock().unwrap();
        inner
            .pending
            .get(&coord)
            .map_or(false, |r| r.in_progress.load(AtomicOrdering::Relaxed))
    }

    /// Remove a pending request.  Any heap entry for this coord is now stale
    /// and will be discarded lazily — no O(n) heap search needed.
    fn remove(&self, coord: ChunkCoord) {
        let mut inner = self.inner.lock().unwrap();
        inner.pending.remove(&coord);
    }
    fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.heap.clear();
        inner.pending.clear();
    }

    fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.shutdown = true;
        // Wake all workers so they see the shutdown flag.
        self.condvar.notify_all();
    }

    fn pending_count(&self) -> usize {
        self.inner.lock().unwrap().pending.len()
    }
    fn iter_pending_chunks(&self) -> Vec<ChunkCoord> {
        let inner = self.inner.lock().unwrap();
        inner.pending.keys().map(|k| *k).collect()
    }
}
pub type LoadedChunksSnapshot = HashMap<ChunkCoord, LoadedChunkSnapshot>;
// ─── ChunkWorkerPool ──────────────────────────────────────────────────────────

pub struct ChunkWorkerPool {
    pub result_rx: Receiver<CpuChunkMesh>,

    queue: Arc<WorkQueue>,

    // Version tracking is main-thread only; workers access versions exclusively
    // through the Arc<AtomicU64> baked into each PendingChunkRequest, so reads
    // from worker threads never touch this map.
    versions: Arc<RwLock<HashMap<ChunkCoord, Arc<AtomicU64>>>>,
}

impl ChunkWorkerPool {
    pub fn new(worker_count: usize, terrain_gen: TerrainGenerator) -> Self {
        let (result_tx, result_rx) = unbounded::<CpuChunkMesh>();

        let queue = Arc::new(WorkQueue::new());
        let versions: Arc<RwLock<HashMap<ChunkCoord, Arc<AtomicU64>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        for _ in 0..worker_count {
            let result_tx = result_tx.clone();
            let terrain = terrain_gen.clone();
            let queue = queue.clone();

            std::thread::spawn(move || {
                loop {
                    // Blocks with zero CPU until there is real work.
                    // Returns None only on shutdown.
                    let Some(job) = queue.pop() else {
                        return;
                    };

                    // The job may already be superseded (a newer version was
                    // submitted while this one sat in the queue).  The version
                    // atomic is the lightweight cancellation mechanism; no lock
                    // needed here.
                    if job.version_atomic.load(AtomicOrdering::Relaxed) != job.version {
                        continue;
                    }

                    let build_result = ChunkBuilder::build_chunk_cpu(
                        job.coord,
                        job.state,
                        job.version,
                        &job.version_atomic,
                        &terrain,
                        &job.terrain_edits_snapshot,
                        &job.loaded_snapshot,
                    );
                    // ALWAYS remove from pending, regardless of outcome
                    queue.remove(job.coord);

                    // Send result only if build succeeded
                    if let Some(cpu) = build_result {
                        let _ = result_tx.send(cpu);
                    }
                }
            });
        }

        Self {
            result_rx,
            queue,
            versions,
        }
    }

    // ── Version helpers ───────────────────────────────────────────────────────

    pub fn is_current_version(&self, coord: ChunkCoord, version: u64) -> bool {
        let guard = self.versions.read().unwrap();
        guard
            .get(&coord)
            .map_or(false, |v| v.load(AtomicOrdering::Relaxed) == version)
    }

    #[inline(always)]
    pub fn still_current(v: &AtomicU64, expected: u64) -> bool {
        v.load(AtomicOrdering::Relaxed) == expected
    }

    pub fn new_version_for(&self, coord: ChunkCoord) -> (u64, Arc<AtomicU64>) {
        let atomic = {
            let mut g = self.versions.write().unwrap();
            g.entry(coord)
                .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                .clone()
        };
        let v = atomic.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        (v, atomic)
    }

    // ── Queue operations ──────────────────────────────────────────────────────

    pub fn submit_request(&self, req: PendingChunkRequest) {
        self.queue.push(req);
    }

    pub fn has_request(&self, coord: ChunkCoord, state: &ChunkState) -> bool {
        self.queue.has_request(coord, state)
    }
    pub fn is_building(&self, coord: ChunkCoord) -> bool {
        self.queue.is_building(coord)
    }
    /// Permanently remove a chunk: cancel any queued request and invalidate
    /// any in-flight build so its result is discarded on arrival.
    pub fn forget_chunk(&self, coord: ChunkCoord) {
        // Drop from queue first so no new job can start for this coord.
        self.queue.remove(coord);

        // Bump the version to invalidate any job already popped by a worker
        // and currently executing.  The worker checks the atomic mid-build.
        let mut versions = self.versions.write().unwrap();
        if let Some(v) = versions.get(&coord) {
            v.fetch_add(1, AtomicOrdering::Relaxed);
        }
        versions.remove(&coord);
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.versions.write().unwrap().clear();
    }

    /// Signal all worker threads to exit cleanly.  Call this on game shutdown.
    pub fn shutdown(&self) {
        self.queue.shutdown();
    }

    /// Number of requests currently waiting in the queue (debug / UI).
    pub fn pending_count(&self) -> usize {
        self.queue.pending_count()
    }

    pub fn pending_chunks(&self) -> Vec<ChunkCoord> {
        self.queue.iter_pending_chunks()
    }
}
