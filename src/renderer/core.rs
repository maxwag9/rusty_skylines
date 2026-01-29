use crate::components::camera::*;
use crate::data::{DebugViewState, Settings};
use crate::mouse_ray::*;
use crate::paths::{shader_dir, texture_dir};
use crate::positions::WorldPos;
use crate::renderer::astronomy::*;
use crate::renderer::general_mesh_arena::GeneralMeshArena;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::pipelines::{DEPTH_FORMAT, Pipelines};
use crate::renderer::procedural_bind_group_manager::FullscreenPassType;
use crate::renderer::procedural_render_manager::{
    DepthDebugParams, PipelineOptions, RenderManager, create_color_attachment_load,
};
use crate::renderer::procedural_texture_manager::{MaterialKind, Params, TextureCacheKey};
use crate::renderer::render_passes::{
    RenderPassConfig, create_color_attachment, create_depth_attachment, create_normal_attachment,
};
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::shadows::{CSM_CASCADES, ShadowMatUniform, render_roads_shadows};
use crate::renderer::ui::{ScreenUniform, UiRenderer};
use crate::renderer::uniform_updates::UniformUpdater;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::resources::{TimeSystem, Uniforms};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_mesh_renderer::RoadRenderSubsystem;
use crate::terrain::sky::{STAR_COUNT, STARS_VERTEX_LAYOUT};
use crate::terrain::water::SimpleVertex;
use crate::ui::input::InputState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::variables::{UiVariableRegistry, update_ui_variables};
use crate::ui::vertex::{LineVtxRender, Vertex};
use crate::world::CameraBundle;
use glam::Mat4;
use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Rgba8UnormSrgb;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

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
    num_systems: usize,
    capacity_entries: u32,
    buffer_size: u64,

    frame: u64,
    slot_just_written: Option<usize>,

    sums_ms: HashMap<String, f64>,
    samples: u32,
    last_print: Instant,

    label_to_index: HashMap<String, u32>,
    index_to_label: Vec<String>,
    next_index: u32,
    used_entries: u32,
}

impl GpuProfiler {
    pub fn new(device: &Device, num_systems: usize, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight >= 3);

        let num_entries = (num_systems * 2) as u32;
        let buffer_size = num_entries as u64 * size_of::<u64>() as u64;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: num_entries,
            ty: QueryType::Timestamp,
        });

        let mut slots = Vec::with_capacity(frames_in_flight);
        for i in 0..frames_in_flight {
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Query Resolve Buffer {i}")),
                size: buffer_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("Query Readback Buffer {i}")),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | BufferUsages::MAP_READ,
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
            num_systems,
            capacity_entries: num_entries,
            buffer_size,
            frame: 0,
            slot_just_written: None,
            sums_ms: HashMap::with_capacity(num_systems),
            samples: 0,
            last_print: Instant::now(),
            label_to_index: HashMap::new(),
            index_to_label: vec![],
            next_index: 0,
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

pub struct RenderCore {
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub msaa_samples: u32,
    pub pipelines: Pipelines,
    ui_renderer: UiRenderer,
    pub terrain_renderer: TerrainRenderer,
    pub road_renderer: RoadRenderSubsystem,
    shader_watcher: Option<ShaderWatcher>,
    encoder: Option<CommandEncoder>,
    arena: GeneralMeshArena,
    render_manager: RenderManager,
    pub gizmo: Gizmo,
    astronomy: AstronomyState,
    profiler: GpuProfiler,
}

impl RenderCore {
    pub fn new(window: Arc<Window>, settings: &Settings, camera: &Camera) -> Self {
        let (surface, adapter, size) = create_surface_and_adapter(window);
        let (config, msaa_samples) = create_surface_config(&surface, &adapter, settings, size);
        let (device, queue) = create_device(&adapter);
        surface.configure(&device, &config);

        let shader_watcher = ShaderWatcher::new().ok();

        let pipelines = Pipelines::new(
            &device,
            &config,
            msaa_samples,
            camera,
            settings.shadow_map_size,
        )
        .expect("Failed to create render pipelines");
        let ui_renderer = UiRenderer::new(&device, config.format, size, msaa_samples)
            .expect("Failed to create UI pipelines");
        let terrain_renderer = TerrainRenderer::new(&device, settings);
        let road_renderer = RoadRenderSubsystem::new(&device);
        let arena = GeneralMeshArena::new(&device, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let render_manager = RenderManager::new(&device, &queue, config.format, texture_dir());
        let gizmo = Gizmo::new(&device, terrain_renderer.chunk_size);
        let profiler = GpuProfiler::new(&device, 5, 3);
        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_samples,
            ui_renderer,
            terrain_renderer,
            road_renderer,
            shader_watcher,
            encoder: None,
            arena,
            render_manager,
            gizmo,
            astronomy: AstronomyState::default(),
            profiler,
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.pipelines.resize(&self.config, self.msaa_samples);
        self.render_manager.invalidate_resize_bind_groups();
    }

    pub(crate) fn render(
        &mut self,
        camera_bundle: &mut CameraBundle,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
    ) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        self.update_render(
            camera_bundle,
            ui_loader,
            time,
            input_state,
            settings,
            aspect,
        );

        let camera = &camera_bundle.camera;
        let Some(frame) = acquire_frame(&self.surface, &self.device, &self.config) else {
            return;
        };

        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.execute_shadow_pass(&mut encoder, camera, aspect, time, settings);
        self.execute_main_pass(
            &mut encoder,
            &surface_view,
            &RenderPassConfig::from_settings(settings),
            camera,
            aspect,
            time,
            input_state,
            ui_loader,
            settings,
            settings.show_world,
        );

        self.profiler.resolve(&mut encoder);

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        self.profiler
            .end_frame(&self.device, &self.queue, &mut ui_loader.variables);
    }

    pub fn update_render(
        &mut self,
        camera_bundle: &mut CameraBundle,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
        aspect: f32,
    ) {
        let camera = &mut camera_bundle.camera;
        self.check_shader_changes(ui_loader);

        let time_scales = TimeScales::from_game_time(time.total_game_time, settings.always_day);
        let observer = ObserverParams::new(time_scales.day_angle);
        let astronomy = compute_astronomy(&time_scales);

        update_ui_variables(ui_loader, &time_scales, &astronomy, observer.obliquity);

        let (view, proj, view_proj) = camera.matrices(aspect);
        let ray = WorldRay::from_mouse(
            glam::Vec2::new(input_state.mouse.pos.x, input_state.mouse.pos.y),
            self.config.width as f32,
            self.config.height as f32,
            view,
            proj,
            camera.eye_world(),
            self.terrain_renderer.chunk_size,
        );
        self.terrain_renderer.pick_terrain_point(ray);
        self.render_manager
            .update_depth_params_buffer(DepthDebugParams {
                near: camera.near,
                far: camera.far,
                power: 20.0,
                reversed_z: 0, // or 1? Not yet
                msaa_samples: self.msaa_samples,
                _pad0: 0,
                _pad1: 0,
            });
        let (new_uniforms, light_mats, splits) = self.update_uniforms(
            camera, view, proj, view_proj, &astronomy, time, aspect, settings,
        );
        self.pipelines.uniforms_cpu = new_uniforms;
        self.pipelines.cascaded_shadow_map.light_mats = light_mats;
        self.pipelines.cascaded_shadow_map.splits = splits;

        // upload per-cascade shadow uniforms ONCE (outside encoder)
        for i in 0..CSM_CASCADES {
            let smu = ShadowMatUniform {
                light_view_proj: self.pipelines.cascaded_shadow_map.light_mats[i]
                    .to_cols_array_2d(),
            };
            self.queue.write_buffer(
                &self.pipelines.cascaded_shadow_map.shadow_mat_buffers[i],
                0,
                bytemuck::bytes_of(&smu),
            );
        }
        self.update_subsystems(
            camera,
            aspect,
            settings,
            input_state,
            time,
            ui_loader,
            &astronomy,
        );
        self.astronomy = astronomy;
    }

    fn update_uniforms(
        &mut self,
        camera: &Camera,
        view: Mat4,
        proj: Mat4,
        view_proj: Mat4,
        astronomy: &AstronomyState,
        time: &TimeSystem,
        aspect: f32,
        settings: &Settings,
    ) -> (Uniforms, [Mat4; CSM_CASCADES], [f32; 4]) {
        let updater = UniformUpdater::new(&self.queue, &self.pipelines);
        let (new_uniforms, light_mats, splits) = updater.update_camera_uniforms(
            view,
            proj,
            view_proj,
            astronomy,
            camera,
            time.total_time,
            aspect,
        );
        updater.update_fog_uniforms(&self.config, camera);
        updater.update_sky_uniforms(astronomy.moon_phase);
        updater.update_water_uniforms();
        updater.update_tonemapping_uniforms(&settings.tonemapping_state);
        (new_uniforms, light_mats, splits)
    }

    fn update_subsystems(
        &mut self,
        camera: &mut Camera,
        aspect: f32,
        settings: &Settings,
        input_state: &mut InputState,
        time: &TimeSystem,
        ui_loader: &mut UiButtonLoader,
        astronomy: &AstronomyState,
    ) {
        let eye = camera.target;
        let target_pos_render = eye.to_render_pos(WorldPos::zero(), camera.chunk_size);
        ui_loader
            .variables
            .set_i32("target_pos_cx", camera.target.chunk.x);
        ui_loader
            .variables
            .set_i32("target_pos_cz", camera.target.chunk.z);
        ui_loader
            .variables
            .set_f32("target_pos_x", target_pos_render.x);
        ui_loader
            .variables
            .set_f32("target_pos_y", target_pos_render.y);
        ui_loader
            .variables
            .set_f32("target_pos_z", target_pos_render.z);
        self.terrain_renderer.update(
            &self.device,
            &self.queue,
            camera,
            aspect,
            settings,
            input_state,
            time,
        );
        self.ui_renderer.update(
            ui_loader,
            time,
            input_state,
            &self.queue,
            &PhysicalSize::new(self.config.width, self.config.height),
        );
        self.road_renderer.update(
            &self.terrain_renderer,
            &self.device,
            &self.queue,
            input_state,
            &self.terrain_renderer.last_picked,
            camera,
            &mut self.gizmo,
        );
        self.gizmo.update(
            &self.terrain_renderer,
            camera.target,
            time.total_game_time,
            &self.road_renderer.road_manager,
            settings,
        );
        self.gizmo.update_gizmo_vertices(
            camera.target,
            camera.orbit_radius,
            false,
            astronomy.sun_dir,
            self.terrain_renderer.chunk_size,
        );
    }

    fn execute_shadow_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        camera: &Camera,
        aspect: f32,
        _time: &TimeSystem,
        settings: &Settings,
    ) {
        for cascade in 0..CSM_CASCADES {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("CSM Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.pipelines.cascaded_shadow_map.layer_views[cascade],
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !settings.shadows_enabled {
                return;
            }
            let shadow_buf = &self.pipelines.cascaded_shadow_map.shadow_mat_buffers[cascade];

            //
            // if cascade != 0 && cascade != 1 {
            //     render_terrain_shadows(
            //         &mut pass,
            //         &mut self.render_manager,
            //         &self.terrain_renderer,
            //         &self.pipelines,
            //         settings,
            //         camera,
            //         aspect,
            //         shadow_buf,
            //     );
            // }

            render_roads_shadows(
                &mut pass,
                &mut self.render_manager,
                &self.road_renderer,
                &self.pipelines,
                settings,
                shadow_buf,
            );
        }
    }

    fn execute_main_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        config: &RenderPassConfig,
        camera: &Camera,
        aspect: f32,
        time: &TimeSystem,
        input_state: &InputState,
        ui_loader: &mut UiButtonLoader,
        settings: &Settings,
        show_world: bool,
    ) {
        // -------- Pass 1: Main world pass (writes depth) --------
        self.execute_world_pass(encoder, config, camera, settings, aspect);
        self.execute_ssao_pass(encoder, settings);
        // -------- Pass 2: Fog (samples depth, no depth attachment) --------
        self.execute_fog_pass(encoder, settings);

        // -------- Pass 3: UI (NOT fogged, because it draws after fog) Logic! --------
        self.execute_ui_pass(encoder, ui_loader, time, input_state);

        // -------- Pass 4: Debug preview (disabled by default) --------
        self.execute_debug_preview_pass(encoder, surface_view, settings);

        // Last Pass (but no, before debug so the image isn't tampered with! Actually, no)
        self.execute_tonemap_pass(encoder, surface_view, settings);
    }

    fn execute_world_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        config: &RenderPassConfig,
        camera: &Camera,
        settings: &Settings,
        aspect: f32,
    ) {
        // Prepare Terrain Uniforms (Needs queue)
        self.terrain_renderer.make_pick_uniforms(
            &self.queue,
            &self.pipelines.pick_uniforms.buffer,
            camera,
            self.terrain_renderer.chunk_size,
        );

        let mut pass = create_world_pass(encoder, &self.pipelines, config, self.msaa_samples);

        if !settings.show_world {
            return;
        }
        // NOTE: Timestamp indices:
        // Sky: 0-1, Terrain: 2-3, Water: 4-5, Roads: 6-7, Gizmo: 8-9
        // 1. Sky
        gpu_timestamp!(pass, &mut self.profiler, "Sky", {
            // All frame time names must be lowercase, I decided. (Doesn't matter anyway, cuz I .lowercase() anyway.)
            render_sky(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                settings,
                self.msaa_samples,
            );
        });

        // 2. Terrain
        gpu_timestamp!(pass, &mut self.profiler, "Terrain", {
            render_terrain(
                &mut pass,
                &mut self.render_manager,
                &self.terrain_renderer,
                &self.pipelines,
                settings,
                self.msaa_samples,
                camera,
                aspect,
            );
        });

        // 3. Water
        gpu_timestamp!(pass, &mut self.profiler, "Water", {
            render_water(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                settings,
                self.msaa_samples,
            );
        });

        // 4. Roads
        gpu_timestamp!(pass, &mut self.profiler, "Roads", {
            render_roads(
                &mut pass,
                &mut self.render_manager,
                &self.road_renderer,
                &self.pipelines,
                settings,
                self.msaa_samples,
            );
        });

        gpu_timestamp!(pass, &mut self.profiler, "Gizmo", {
            render_gizmo(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                settings,
                self.msaa_samples,
                &mut self.gizmo,
                camera,
                &self.device,
                &self.queue,
            );
        });
    }
    fn execute_ssao_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        if !settings.show_world {
            return;
        }

        // We LOAD the current HDR color and MULTIPLY it by AO using blending.
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_hdr_view,
            &self.pipelines.resolved_hdr_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("SSAO Pass (depth+normals)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Multiply blending: out = dst * srcColor
        // (we output AO in srcColor as vec3(ao))
        let multiply_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::Src,
                operation: BlendOperation::Add,
            },
            // keep alpha unchanged
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        };

        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: self.msaa_samples, // ok: we write into your MSAA HDR target
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::Ssao, // you need this pass type to bind depth+normals (see note below)
            targets: vec![Some(ColorTargetState {
                format: self.pipelines.msaa_hdr_view.texture().format(),
                blend: Some(multiply_blend),
                write_mask: ColorWrites::ALL,
            })],
        };
        let ssao_shader = if self.msaa_samples > 1 {
            "ssao_msaa.wgsl"
        } else {
            "ssao.wgsl"
        };
        // SSAO only needs camera matrices (inv_proj + view/proj) from your existing Uniforms.
        self.render_manager.render_fullscreen_pass(
            "SSAO",
            shader_dir().join(ssao_shader).as_path(),
            options,
            &[&self.pipelines.uniforms.buffer],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::Ssao,
        );
    }
    fn execute_fog_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_hdr_view,
            &self.pipelines.resolved_hdr_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Fog Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !settings.show_world {
            return;
        }
        let fog_shader = if self.msaa_samples > 1 {
            "fog_msaa.wgsl"
        } else {
            "fog.wgsl"
        };

        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: self.msaa_samples,
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::Fog, // (will be overwritten anyway)
            targets: vec![Some(ColorTargetState {
                format: self.pipelines.msaa_hdr_view.texture().format(),
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        };

        self.render_manager.render_fullscreen_pass(
            "Fog",
            shader_dir().join(fog_shader).as_path(),
            options,
            &[
                &self.pipelines.uniforms.buffer,
                &self.pipelines.fog_uniforms.buffer,
                &self.pipelines.pick_uniforms.buffer,
            ],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::Fog,
        );
    }

    fn execute_ui_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
    ) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_hdr_view,
            &self.pipelines.resolved_hdr_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass (UI)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.render_ui(&mut pass, ui_loader, time, input_state);
    }

    fn execute_debug_preview_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        settings: &Settings,
    ) {
        match settings.debug_view_state {
            DebugViewState::None => {
                return;
            }
            DebugViewState::Normals => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa_hdr_view,
                    &self.pipelines.resolved_hdr_view,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Normals Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug_view(
                    &self.pipelines.resolved_normal_view,
                    "Debug Normals Render",
                    self.pipelines.msaa_hdr_view.texture().format(), // target format (color)
                    self.msaa_samples,
                    &mut pass,
                );
            }
            DebugViewState::Depth => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa_hdr_view,
                    &self.pipelines.resolved_hdr_view,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Depth Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug_view(
                    &self.pipelines.depth_sample_view,
                    "Debug Depth Render",
                    self.pipelines.msaa_hdr_view.texture().format(), // target format (color)
                    self.msaa_samples,
                    &mut pass,
                );
            }
        }
    }

    fn execute_tonemap_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        settings: &Settings,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Tonemap Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::Normal,
            targets: vec![Some(ColorTargetState {
                format: surface_view.texture().format(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        };
        self.render_manager.render_fullscreen_pass(
            "Tonemap",
            shader_dir().join("tonemap.wgsl").as_path(),
            options,
            &[&self.pipelines.tonemapping_uniforms.buffer],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::Normal,
        );
    }

    fn render_ui(
        &mut self,
        pass: &mut RenderPass,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
    ) {
        let screen_uniform = ScreenUniform {
            size: [self.config.width as f32, self.config.height as f32],
            time: time.total_time as f32,
            enable_dither: 1,
            mouse: input_state.mouse.pos.to_array(),
        };

        self.queue.write_buffer(
            &self.ui_renderer.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&screen_uniform),
        );

        self.ui_renderer
            .render(&mut self.render_manager, pass, ui_loader, &self.pipelines);
    }

    pub(crate) fn cycle_msaa(&mut self) {
        let supported = get_supported_msaa_levels(&self.device, self.config.format);
        let current_idx = supported
            .iter()
            .position(|&s| s == self.msaa_samples)
            .unwrap_or(0);
        self.msaa_samples = supported[(current_idx + 1) % supported.len()];

        println!("MSAA changed to {}x", self.msaa_samples);

        self.pipelines.resize(&self.config, self.msaa_samples);
        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
        self.render_manager.invalidate_resize_bind_groups();
    }

    fn reload_all_shaders(&mut self) -> anyhow::Result<()> {
        self.ui_renderer.reload_shaders()?;
        Ok(())
    }

    fn check_shader_changes(&mut self, ui_loader: &mut UiButtonLoader) {
        let Some(watcher) = &self.shader_watcher else {
            return;
        };

        let changed = watcher.take_changed_wgsl_files();
        if changed.is_empty() {
            return;
        }

        let summary = changed
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect::<Vec<_>>()
            .join(", ");

        match self.reload_all_shaders() {
            Ok(()) => {
                let label = if summary.is_empty() {
                    "Shaders reloaded".to_string()
                } else {
                    format!("Shaders reloaded: {summary}")
                };
                println!("{}", label);
                ui_loader.log_console(format!("✅ {label}"));
            }
            Err(err) => ui_loader.log_console(format!("❌ Shader reload failed: {err}")),
        }
        self.render_manager.pipeline_manager.reload_shaders(changed);
    }
}

fn create_surface_and_adapter(
    window: Arc<Window>,
) -> (Surface<'static>, Adapter, PhysicalSize<u32>) {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
    });

    let size = window.inner_size();
    let surface = instance
        .create_surface(window)
        .expect("Surface creation failed");

    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapters found");

    println!("Backend: {:?}", adapter.get_info().backend);

    (surface, adapter, size)
}

fn create_surface_config(
    surface: &Surface,
    adapter: &Adapter,
    settings: &Settings,
    size: PhysicalSize<u32>,
) -> (SurfaceConfiguration, u32) {
    let surface_caps = surface.get_capabilities(adapter);

    let format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| *f == Rgba8UnormSrgb)
        .unwrap_or(surface_caps.formats[0]);
    // println!("{:?}", format);

    let alpha_mode = surface_caps
        .alpha_modes
        .iter()
        .copied()
        .find(|m| *m == CompositeAlphaMode::PostMultiplied)
        .unwrap_or(CompositeAlphaMode::Opaque);

    let present_mode = pick_present_mode(surface, adapter, settings.present_mode.clone().to_wgpu());

    let config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    let caps = adapter.get_texture_format_features(config.format);
    let msaa_samples = clamp_to_supported_msaa(caps, settings.msaa_samples);

    (config, msaa_samples)
}

fn create_device(adapter: &Adapter) -> (Device, Queue) {
    pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("Device"),
        required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | Features::TIMESTAMP_QUERY
            | Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        required_limits: Limits::default(),
        experimental_features: ExperimentalFeatures::disabled(),
        memory_hints: MemoryHints::default(),
        trace: Trace::Off,
    }))
    .expect("Device creation failed")
}

fn pick_present_mode(surface: &Surface, adapter: &Adapter, user_mode: PresentMode) -> PresentMode {
    let caps = surface.get_capabilities(adapter);
    let fallbacks = [
        user_mode,
        PresentMode::Mailbox,
        PresentMode::Immediate,
        PresentMode::Fifo,
    ];

    fallbacks
        .into_iter()
        .find(|mode| caps.present_modes.contains(mode))
        .unwrap_or(PresentMode::Fifo)
}

fn get_supported_msaa_levels(device: &Device, format: TextureFormat) -> Vec<u32> {
    let caps = device.features();
    let format_caps = device.create_texture(&TextureDescriptor {
        label: Some("MSAA probe"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    drop(format_caps);

    let mut levels = vec![1];
    for count in [2, 4, 8] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("MSAA probe"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: count,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        }));
        if result.is_ok() {
            levels.push(count);
        }
    }
    levels
}

fn clamp_to_supported_msaa(caps: TextureFormatFeatures, requested: u32) -> u32 {
    let can_render = caps
        .allowed_usages
        .contains(TextureUsages::RENDER_ATTACHMENT);

    if !can_render {
        println!("MSAA not usable: format cannot be a render attachment");
        return 1;
    }

    let supports_8 = caps
        .flags
        .contains(TextureFormatFeatureFlags::MULTISAMPLE_X8);
    let supports_4 = caps
        .flags
        .contains(TextureFormatFeatureFlags::MULTISAMPLE_X4);
    let supports_2 = caps
        .flags
        .contains(TextureFormatFeatureFlags::MULTISAMPLE_X2);

    if supports_8 {
        return if requested < 8 {
            println!("8x MSAA supported, but using {}x!", requested);
            requested
        } else {
            println!("8x MSAA supported, using {}x!", requested);
            8
        };
    }

    if supports_4 {
        if requested >= 8 {
            println!("8x requested but unsupported, falling back to 4x");
            return 4;
        }
        if requested >= 4 {
            println!("4x MSAA supported, using 4x");
            return 4;
        }
        println!("4x MSAA supported, but using {}x!", requested);
        return requested;
    }

    if supports_2 {
        if requested >= 4 {
            println!("Only 2x MSAA supported, falling back to 2x");
            return 2;
        }
        println!("2x MSAA supported, using {}x!", requested);
        return requested;
    }

    println!("MSAA not supported, falling back to 1x (WTF is this GPU huh?)");
    1
}

pub fn acquire_frame(
    surface: &Surface,
    device: &Device,
    config: &SurfaceConfiguration,
) -> Option<SurfaceTexture> {
    match surface.get_current_texture() {
        Ok(frame) => Some(frame),
        Err(SurfaceError::Outdated) => {
            // Surface is temporarily invalid
            None // skip this frame
        }
        Err(SurfaceError::Lost) => {
            surface.configure(device, config);
            None
        }
        Err(SurfaceError::OutOfMemory) => {
            panic!("OOM");
        }
        Err(e) => {
            eprintln!("Surface error: {:?}", e);
            None
        }
    }
}

fn road_material_keys() -> Vec<TextureCacheKey> {
    vec![
        TextureCacheKey {
            kind: MaterialKind::Concrete,
            params: Params {
                seed: 1,
                scale: 2.0,
                roughness: 1.0,
                color_primary: [0.32, 0.30, 0.28, 1.0],
                color_secondary: [0.15, 0.13, 0.10, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Goo,
            params: Params {
                seed: 0,
                scale: 3.0,
                roughness: 0.3,
                color_primary: [0.02, 0.02, 0.03, 1.0],
                color_secondary: [0.10, 0.10, 0.12, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 0,
                scale: 16.0,
                roughness: 0.5,
                color_primary: [0.004, 0.004, 0.004, 1.0],
                color_secondary: [0.015, 0.015, 0.015, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 1,
                scale: 16.0,
                roughness: 0.3,
                color_primary: [0.006, 0.006, 0.006, 1.0],
                color_secondary: [0.020, 0.020, 0.020, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 2,
                scale: 16.0,
                roughness: 0.5,
                color_primary: [0.04, 0.04, 0.006, 1.0],
                color_secondary: [0.120, 0.120, 0.120, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 3,
                scale: 16.0,
                roughness: 0.8,
                color_primary: [0.02, 0.02, 0.02, 1.0],
                color_secondary: [0.080, 0.080, 0.080, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
    ]
}

fn terrain_material_keys() -> Vec<TextureCacheKey> {
    vec![
        TextureCacheKey {
            kind: MaterialKind::Grass,
            params: Params {
                seed: 1337,
                scale: 40.0,
                roughness: 0.78, // Heavily reduced dry influence
                moisture: 0.65,  // Max lush bias
                color_primary: [0.02, 0.34, 0.01, 1.0], // Deep muted olive green—no neon
                color_secondary: [0.10, 0.60, 0.10, 1.0], // Dark neutral brown, zero yellow pop
                shadow_strength: 1.80, // Hard punchy shadows
                sheen_strength: 0.05, // Barely any highlight
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureCacheKey {
            kind: MaterialKind::Grass,
            params: Params {
                seed: 42,
                scale: 80.0,
                roughness: 0.65,                       // Way down—minimal dry yellow
                moisture: 0.70,                        // Balanced but not dead
                color_primary: [0.0, 0.33, 0.00, 1.0], // Ultra-dark muted base
                color_secondary: [0.02, 0.50, 0.00, 1.0], // Super dark brown shadow tone
                shadow_strength: 1.75,                 // Even deeper volume
                sheen_strength: 0.02,                  // None basically
                ..Default::default()
            },
            resolution: 1024,
        },
    ]
}
fn road_depth_stencil(bias: DepthBiasState) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::LessEqual,
        stencil: Default::default(),
        bias,
    }
}
fn create_world_pass<'a>(
    encoder: &'a mut CommandEncoder,
    pipelines: &'a Pipelines,
    config: &'a RenderPassConfig,
    msaa_samples: u32,
) -> RenderPass<'a> {
    let color_attachment = create_color_attachment(
        &pipelines.msaa_hdr_view,
        &pipelines.resolved_hdr_view,
        msaa_samples,
        config.background_color,
    );

    let normal_attachment = create_normal_attachment(
        &pipelines.msaa_normal_view,
        &pipelines.resolved_normal_view,
        msaa_samples,
    );

    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("World Pass"),
        color_attachments: &[Some(color_attachment), Some(normal_attachment)],
        depth_stencil_attachment: Some(create_depth_attachment(&pipelines.depth_view)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn render_gizmo(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
    gizmo: &mut Gizmo,
    camera: &Camera,
    device: &Device,
    queue: &Queue,
) {
    let targets = color_and_normals_targets(pipelines);
    render_manager.render(
        Vec::new(),
        "Gizmo",
        shader_dir().join("lines.wgsl").as_path(),
        PipelineOptions {
            topology: PrimitiveTopology::LineList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([LineVtxRender::layout()]),
            targets,
            ..Default::default()
        },
        &[&pipelines.uniforms.buffer],
        pass,
        pipelines,
        settings,
    );

    let vertex_count = gizmo.update_buffer(device, queue, camera.eye_world());
    pass.set_vertex_buffer(0, gizmo.gizmo_buffer.slice(..));
    pass.draw(0..vertex_count, 0..1);
    gizmo.clear();
}

fn render_water(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
) {
    let targets = color_and_normals_targets(pipelines);
    render_manager.render(
        Vec::new(),
        "Water",
        shader_dir().join("water.wgsl").as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: StencilState {
                    front: StencilFaceState {
                        compare: CompareFunction::Equal,
                        fail_op: StencilOperation::Keep,
                        depth_fail_op: StencilOperation::Keep,
                        pass_op: StencilOperation::Keep,
                    },
                    back: StencilFaceState {
                        compare: CompareFunction::Equal,
                        fail_op: StencilOperation::Keep,
                        depth_fail_op: StencilOperation::Keep,
                        pass_op: StencilOperation::Keep,
                    },
                    read_mask: 0xFF,
                    write_mask: 0x00,
                },
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([SimpleVertex::layout()]),
            targets,
            ..Default::default()
        },
        &[
            &pipelines.uniforms.buffer,
            &pipelines.water_uniforms.buffer,
            &pipelines.sky_uniforms.buffer,
        ],
        pass,
        pipelines,
        settings,
    );

    pass.set_stencil_reference(1);
    pass.set_vertex_buffer(0, pipelines.water_mesh_buffers.vertex.slice(..));
    pass.set_index_buffer(
        pipelines.water_mesh_buffers.index.slice(..),
        IndexFormat::Uint32,
    );
    pass.draw_indexed(0..pipelines.water_mesh_buffers.index_count, 0, 0..1);
}

fn render_sky(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
) {
    let sky_depth_stencil = Some(DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: Default::default(),
        bias: Default::default(),
    });
    let targets = color_and_normals_targets(pipelines);
    render_manager.render(
        Vec::new(),
        "Stars",
        shader_dir().join("stars.wgsl").as_path(),
        PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            depth_stencil: sky_depth_stencil.clone(),
            msaa_samples,
            vertex_layouts: Vec::from([STARS_VERTEX_LAYOUT]),
            targets: targets.clone(),
            ..Default::default()
        },
        &[&pipelines.uniforms.buffer, &pipelines.sky_uniforms.buffer],
        pass,
        pipelines,
        settings,
    );
    pass.set_vertex_buffer(0, pipelines.stars_mesh_buffers.vertex.slice(..));
    pass.draw(0..4, 0..STAR_COUNT);
    render_manager.render(
        Vec::new(),
        "Sky",
        shader_dir().join("sky.wgsl").as_path(),
        PipelineOptions {
            topology: Default::default(),
            depth_stencil: sky_depth_stencil,
            msaa_samples,
            vertex_layouts: Vec::new(),
            targets,
            ..Default::default()
        },
        &[&pipelines.uniforms.buffer, &pipelines.sky_uniforms.buffer],
        pass,
        pipelines,
        settings,
    );
    pass.draw(0..3, 0..1);
}

fn render_roads(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
) {
    let keys = road_material_keys();
    let shader_path = shader_dir().join("road.wgsl");

    let base_bias = DepthBiasState {
        constant: -3,
        slope_scale: -2.0,
        clamp: 0.0,
    };
    let targets = color_and_normals_targets(pipelines);
    render_manager.render(
        keys.clone(),
        "Roads",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(road_depth_stencil(base_bias)),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            targets: targets.clone(),
            ..Default::default()
        },
        &[
            &pipelines.uniforms.buffer,
            &road_renderer.road_appearance.normal_buffer,
        ],
        pass,
        pipelines,
        settings,
    );

    for chunk_id in &road_renderer.visible_draw_list {
        if let Some(gpu) = road_renderer.chunk_gpu.get(chunk_id) {
            pass.set_vertex_buffer(0, gpu.vertex.slice(..));
            pass.set_index_buffer(gpu.index.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }
    }

    if road_renderer.preview_gpu.is_empty() {
        return;
    }
    let (Some(vb), Some(ib)) = (&road_renderer.preview_gpu.vb, &road_renderer.preview_gpu.ib)
    else {
        return;
    };

    let preview_bias = DepthBiasState {
        constant: -4,
        slope_scale: -2.0,
        clamp: 0.0,
    };
    render_manager.render(
        keys,
        "Roads",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(road_depth_stencil(preview_bias)),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            targets,
            ..Default::default()
        },
        &[
            &pipelines.uniforms.buffer,
            &road_renderer.road_appearance.preview_buffer,
        ],
        pass,
        pipelines,
        settings,
    );

    pass.set_vertex_buffer(0, vb.slice(..));
    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
    pass.draw_indexed(0..road_renderer.preview_gpu.index_count, 0, 0..1);
}

fn render_terrain(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    terrain_renderer: &TerrainRenderer,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
    camera: &Camera,
    aspect: f32,
) {
    let keys = terrain_material_keys();
    let shader_path = shader_dir().join("terrain.wgsl");

    let make_stencil = |write_mask: u32| -> DepthStencilState {
        DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState {
                front: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Replace,
                },
                back: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Replace,
                },
                read_mask: 0xFF,
                write_mask,
            },
            bias: Default::default(),
        }
    };
    let targets = color_and_normals_targets(pipelines);
    pass.set_stencil_reference(0);
    render_manager.render(
        keys.clone(),
        "Terrain Pipeline (Above Water)",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0)),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            cull_mode: Some(Face::Front),
            targets: targets.clone(),
            ..Default::default()
        },
        &[&pipelines.uniforms.buffer, &pipelines.pick_uniforms.buffer],
        pass,
        pipelines,
        settings,
    );
    terrain_renderer.render(pass, camera, aspect, false);

    pass.set_stencil_reference(1);
    render_manager.render(
        keys,
        "Terrain Pipeline (Under Water)",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0xFF)),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            cull_mode: Some(Face::Front),
            targets,
            ..Default::default()
        },
        &[&pipelines.uniforms.buffer, &pipelines.pick_uniforms.buffer],
        pass,
        pipelines,
        settings,
    );
    terrain_renderer.render(pass, camera, aspect, true);
}

fn color_and_normals_targets(pipelines: &Pipelines) -> Vec<Option<ColorTargetState>> {
    vec![
        Some(ColorTargetState {
            format: pipelines.msaa_hdr_view.texture().format(),
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: pipelines.msaa_normal_view.texture().format(),
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
    ]
}

pub fn color_target(
    pipelines: &Pipelines,
    blend: Option<BlendState>,
) -> Vec<Option<ColorTargetState>> {
    vec![Some(ColorTargetState {
        format: pipelines.msaa_hdr_view.texture().format(),
        blend,
        write_mask: ColorWrites::ALL,
    })]
}
