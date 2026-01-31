use crate::ui::variables::UiVariableRegistry;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use wgpu::{
    Buffer, BufferAsyncError, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode,
    PollType, QuerySet, QuerySetDescriptor, QueryType, Queue,
};

#[macro_export]
macro_rules! gpu_timestamp {
    ($pass:expr, $profiler:expr, $label:literal, $body:block) => {{
        let (start, end) = $profiler.get_range($label);
        $pass.write_timestamp(&$profiler.query_set, start);
        let r = { $body };
        $pass.write_timestamp(&$profiler.query_set, end);
        r
    }};
}

struct Slot {
    resolve: Buffer,
    readback: Buffer,
    pending: Option<mpsc::Receiver<Result<(), BufferAsyncError>>>,
}

pub struct GpuProfiler {
    pub query_set: QuerySet,
    slots: Vec<Slot>,
    capacity_entries: u32,
    buffer_size: u64,

    frame: u64,
    slot_just_written: Option<usize>,

    sums_ms: HashMap<String, f64>,
    samples: u32,
    last_print: Instant,

    label_to_index: HashMap<String, u32>,
    index_to_label: Vec<String>,
    used_entries: u32,
}

impl GpuProfiler {
    pub fn new(device: &Device, num_systems: usize, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight >= 3);

        let num_entries = (num_systems * 2) as u32;
        let buffer_size = num_entries as u64 * size_of::<u64>() as u64;

        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: num_entries,
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
            });
        }

        Self {
            query_set,
            slots,
            capacity_entries: num_entries,
            buffer_size,
            frame: 0,
            slot_just_written: None,
            sums_ms: HashMap::with_capacity(num_systems),
            samples: 0,
            last_print: Instant::now(),
            label_to_index: HashMap::new(),
            index_to_label: vec![],
            used_entries: 0,
        }
    }
    pub fn get_range(&mut self, label: &str) -> (u32, u32) {
        let key = label.to_lowercase();

        if let Some(&start) = self.label_to_index.get(&key) {
            return (start, start + 1);
        }

        let start = self.used_entries;
        let end = start + 1;

        assert!(
            end < self.capacity_entries,
            "GpuProfiler: ran out of timestamp slots"
        );

        self.label_to_index.insert(key.clone(), start);
        self.index_to_label.push(key);
        self.used_entries += 2;

        (start, end)
    }

    /// Call while encoding, before submit.
    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        if self.used_entries == 0 {
            // Nothing to resolve, skip
            self.slot_just_written = None;
            return;
        }
        let write_slot = (self.frame as usize) % self.slots.len();

        // If still mapped/pending, skip profiling this frame (prevents submit validation error).
        if self.slots[write_slot].pending.is_some() {
            self.slot_just_written = None;
            return;
        }

        let slot = &self.slots[write_slot];
        encoder.resolve_query_set(&self.query_set, 0..self.capacity_entries, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, self.buffer_size);

        self.slot_just_written = Some(write_slot);
    }

    /// Call once per frame AFTER `queue.submit()` and `frame.present()`.
    pub fn end_frame(
        &mut self,
        device: &Device,
        queue: &Queue,
        variables: &mut UiVariableRegistry,
    ) {
        let _ = device.poll(PollType::Poll);

        self.collect_ready(queue);

        // Start mapping the slot we JUST wrote this frame
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
        if self.used_entries == 0 {
            return; // nothing to collect
        }
        let period = queue.get_timestamp_period() as f64;

        for slot in &mut self.slots {
            let Some(rx) = slot.pending.as_ref() else {
                continue;
            };

            let done = match rx.try_recv() {
                Ok(Ok(())) => true,
                Ok(Err(_)) => {
                    slot.pending = None;
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => false,
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

            let max_pairs = (self.used_entries / 2) as usize;
            let available_pairs = timestamps.len() / 2;
            let pair_count = max_pairs.min(available_pairs);

            for i in 0..pair_count {
                let s = i * 2;
                let e = s + 1;

                let start = timestamps[s];
                let end = timestamps[e];

                if end >= start {
                    let ns = (end - start) as f64 * period;
                    let ms = ns / 1_000_000.0;

                    if let Some(label) = self.index_to_label.get(i) {
                        *self.sums_ms.entry(label.clone()).or_insert(0.0) += ms;
                    }
                }
            }

            self.samples += 1;

            drop(mapped);
            slot.readback.unmap();
            slot.pending = None;
        }
    }
}
