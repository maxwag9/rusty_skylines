// gpu_profiler.rs

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use wgpu::*;

use crate::ui::variables::UiVariableRegistry;

const MAX_PROFILE_PAIRS: u32 = 32;
const MAX_PROFILE_ENTRIES: u32 = MAX_PROFILE_PAIRS * 2;

#[macro_export]
macro_rules! gpu_timestamp {
    ($pass:expr, $profiler:expr, $label:expr, $body:block) => {{
        let __label: &str = $label;
        let (__start, __end) = $profiler.get_range(__label);
        $profiler.mark_written(__start);
        $pass.write_timestamp(&$profiler.query_set, __start);
        let __r = { $body };
        $pass.write_timestamp(&$profiler.query_set, __end);
        __r
    }};
}

struct Slot {
    resolve: Buffer,
    readback: Buffer,
    pending: Option<mpsc::Receiver<Result<(), BufferAsyncError>>>,
    /// Which pairs were written when this slot was submitted.
    written_pairs: Vec<bool>,
}

pub struct GpuProfiler {
    pub query_set: QuerySet,
    slots: Vec<Slot>,

    frame: u64,
    slot_just_written: Option<usize>,

    /// Tracks which pairs were written THIS frame (reset each frame via begin_frame).
    written_this_frame: Vec<bool>,

    sums_ms: HashMap<String, f64>,
    samples: u32,
    last_print: Instant,

    label_to_pair: HashMap<String, u32>,
    pair_to_label: Vec<String>,
    used_pairs: u32,
}

impl GpuProfiler {
    pub fn new(device: &Device, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight >= 3);

        let buffer_size = MAX_PROFILE_ENTRIES as u64 * size_of::<u64>() as u64;

        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: MAX_PROFILE_ENTRIES,
            ty: QueryType::Timestamp,
        });

        let mut slots = Vec::with_capacity(frames_in_flight);
        for i in 0..frames_in_flight {
            let resolve = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("Query Resolve Buffer {i}")),
                size: buffer_size,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("Query Readback Buffer {i}")),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            slots.push(Slot {
                resolve,
                readback,
                pending: None,
                written_pairs: Vec::new(),
            });
        }

        Self {
            query_set,
            slots,
            frame: 0,
            slot_just_written: None,
            written_this_frame: Vec::new(),
            sums_ms: HashMap::with_capacity(32),
            samples: 0,
            last_print: Instant::now(),
            label_to_pair: HashMap::new(),
            pair_to_label: Vec::new(),
            used_pairs: 0,
        }
    }

    /// Call at the start of each frame before any gpu_timestamp! calls.
    pub fn begin_frame(&mut self) {
        for w in self.written_this_frame.iter_mut() {
            *w = false;
        }
    }

    /// Returns (start_entry, end_entry) for the given label.
    pub fn get_range(&mut self, label: &str) -> (u32, u32) {
        let key = label.to_lowercase();

        if let Some(&pair) = self.label_to_pair.get(&key) {
            let start = pair * 2;
            return (start, start + 1);
        }

        let pair = self.used_pairs;
        let start = pair * 2;
        let end = start + 1;

        assert!(
            end < MAX_PROFILE_ENTRIES,
            "GpuProfiler: ran out of timestamp slots (max {MAX_PROFILE_PAIRS} pairs)"
        );

        self.label_to_pair.insert(key.clone(), pair);
        self.pair_to_label.push(key);
        self.used_pairs += 1;

        if self.written_this_frame.len() <= pair as usize {
            self.written_this_frame.resize(pair as usize + 1, false);
        }

        (start, end)
    }

    /// Mark a pair as written this frame.
    pub fn mark_written(&mut self, start_entry: u32) {
        let pair = (start_entry / 2) as usize;
        if pair >= self.written_this_frame.len() {
            self.written_this_frame.resize(pair + 1, false);
        }
        self.written_this_frame[pair] = true;
    }

    /// Call while encoding, before submit.
    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        if self.used_pairs == 0 {
            self.slot_just_written = None;
            return;
        }

        let write_slot = (self.frame as usize) % self.slots.len();

        if self.slots[write_slot].pending.is_some() {
            self.slot_just_written = None;
            return;
        }

        let num_pairs = self.used_pairs as usize;

        // Resolve only contiguous ranges of written pairs
        let mut i = 0;
        while i < num_pairs {
            if i >= self.written_this_frame.len() || !self.written_this_frame[i] {
                i += 1;
                continue;
            }

            let range_start = i;
            while i < num_pairs && i < self.written_this_frame.len() && self.written_this_frame[i] {
                i += 1;
            }
            let range_end = i;

            let entry_start = (range_start * 2) as u32;
            let entry_end = (range_end * 2) as u32;
            let offset = entry_start as u64 * size_of::<u64>() as u64;

            encoder.resolve_query_set(
                &self.query_set,
                entry_start..entry_end,
                &self.slots[write_slot].resolve,
                offset,
            );
        }

        // Copy resolve -> readback for the used portion
        let copy_size = self.used_pairs as u64 * 2 * size_of::<u64>() as u64;
        let slot = &self.slots[write_slot];
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, copy_size);

        // Snapshot which pairs were written
        self.slots[write_slot].written_pairs = self.written_this_frame.clone();
        self.slot_just_written = Some(write_slot);
    }

    /// Call once per frame AFTER `queue.submit()`.
    pub fn end_frame(
        &mut self,
        device: &Device,
        queue: &Queue,
        variables: &mut UiVariableRegistry,
    ) {
        let _ = device.poll(PollType::Poll);

        self.collect_ready(queue);

        if let Some(i) = self.slot_just_written.take() {
            let slot = &mut self.slots[i];
            if slot.pending.is_none() {
                let (tx, rx) = mpsc::channel();
                slot.readback
                    .slice(..)
                    .map_async(MapMode::Read, move |res| {
                        let _ = tx.send(res);
                    });
                slot.pending = Some(rx);
            }
        }

        self.frame += 1;
        if self.last_print.elapsed() >= Duration::from_secs(1) && self.samples > 0 {
            let inv_samples = 1.0 / self.samples as f64;

            for (label, sum) in self.sums_ms.iter() {
                let name = format!("{label}_frametime");
                variables.set_f32(&name, (*sum * inv_samples) as f32);
            }

            self.sums_ms.clear();
            self.samples = 0;
            self.last_print = Instant::now();
        }
    }

    fn collect_ready(&mut self, queue: &Queue) {
        if self.used_pairs == 0 {
            return;
        }
        let period = queue.get_timestamp_period() as f64;

        // We need to borrow pair_to_label and sums_ms while iterating slots,
        // so we collect results first then apply them.
        let mut results: Vec<(usize, Vec<(u32, f64)>)> = Vec::new();

        for (slot_idx, slot) in self.slots.iter_mut().enumerate() {
            let Some(rx) = slot.pending.as_ref() else {
                continue;
            };

            let done = match rx.try_recv() {
                Ok(Ok(())) => true,
                Ok(Err(_)) => {
                    slot.pending = None;
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => continue,
                Err(_) => {
                    slot.pending = None;
                    continue;
                }
            };

            if !done {
                continue;
            }

            let slice = slot.readback.slice(..);
            let mapped = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&mapped);

            let mut pairs = Vec::new();
            for (pair_idx, &written) in slot.written_pairs.iter().enumerate() {
                if !written {
                    continue;
                }

                let s = pair_idx * 2;
                let e = s + 1;
                if e >= timestamps.len() {
                    break;
                }

                let start = timestamps[s];
                let end = timestamps[e];

                if end >= start && start != 0 {
                    let ns = (end - start) as f64 * period;
                    let ms = ns / 1_000_000.0;
                    pairs.push((pair_idx as u32, ms));
                }
            }

            drop(mapped);
            slot.readback.unmap();
            slot.pending = None;

            results.push((slot_idx, pairs));
        }

        for (_slot_idx, pairs) in results {
            for (pair_idx, ms) in pairs {
                if let Some(label) = self.pair_to_label.get(pair_idx as usize) {
                    *self.sums_ms.entry(label.clone()).or_insert(0.0) += ms;
                }
            }
            self.samples += 1;
        }
    }
}
