use crate::components::camera::*;
use crate::data::Settings;
use crate::mouse_ray::*;
use crate::paths::shader_dir;
use crate::renderer::pipelines::{FogUniforms, Pipelines, make_new_uniforms};
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::ui::UiRenderer;
use crate::renderer::world_renderer::WorldRenderer;
use crate::resources::TimeSystem;
use crate::terrain::sky::{SkyRenderer, SkyUniform};
use crate::terrain::water::WaterUniform;
use crate::ui::input::MouseState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::vertex::LineVtx;
use crate::world::CameraBundle;
use std::f32::consts::TAU;
use std::sync::Arc;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;
// top-level (once)

pub struct RenderCore {
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,

    pub msaa_samples: u32,

    pub pipelines: Pipelines,
    ui_renderer: UiRenderer,
    pub world: WorldRenderer,

    size: PhysicalSize<u32>,

    shader_watcher: Option<ShaderWatcher>,
    sky: SkyRenderer,
}

impl RenderCore {
    pub fn new(window: Arc<Window>, settings: &Settings, camera: &Camera) -> Self {
        // --- Create instance and surface ---
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
        });

        let size = window.inner_size();
        let surface = instance
            .create_surface(window.clone())
            .expect("Surface creation failed");

        // --- Pick an adapter (blocking internally) ---
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapters found");

        println!("Backend: {:?}", adapter.get_info().backend);

        // --- Configure surface ---
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        let alpha_mode = surface_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::PostMultiplied)
            .unwrap_or(CompositeAlphaMode::Opaque);
        let user_mode = settings.present_mode.clone().to_wgpu();
        let present_mode = pick_fail_safe_present_mode(&surface, &adapter, user_mode);
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

        let mut msaa_samples = 4;
        let caps = adapter.get_texture_format_features(config.format);
        let supported = caps.flags.intersects(
            TextureFormatFeatureFlags::MULTISAMPLE_X8
                | TextureFormatFeatureFlags::MULTISAMPLE_X4
                | TextureFormatFeatureFlags::MULTISAMPLE_X2,
        );
        let can_render = caps
            .allowed_usages
            .contains(TextureUsages::RENDER_ATTACHMENT);
        if supported && can_render {
            if msaa_samples < 8 {
                println!("8x MSAA supported, but using {}x!", msaa_samples);
            } else {
                println!("8x MSAA supported, using {}x!", msaa_samples);
            }
        } else if msaa_samples == 8 {
            println!("Falling back to 4x");
            msaa_samples = 4;
        }
        // Request device + queue
        let features = Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let limits = Limits::default();
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("Device"),
            required_features: features,
            required_limits: limits,
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
        }))
        .expect("Device creation failed");

        // Configure surface
        surface.configure(&device, &config);

        let shader_dir = shader_dir();
        let shader_watcher = ShaderWatcher::new(&shader_dir).ok();

        let _aspect = config.width as f32 / config.height as f32;
        let pipelines = Pipelines::new(&device, &config, msaa_samples, &shader_dir, camera)
            .expect("Failed to create render pipelines");
        let mut ui_renderer =
            UiRenderer::new(&device, config.format, size, msaa_samples, &shader_dir)
                .expect("Failed to create UI pipelines");
        let font_ttf: &[u8] = include_bytes!("../../data/ui_data/ttf/JetBrainsMono-Regular.ttf");
        let _ = ui_renderer.build_text_atlas(&device, &queue, font_ttf, &[14, 18, 24], 1024, 1024);
        let world = WorldRenderer::new(&device);
        let sky = SkyRenderer::new();

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_samples,
            ui_renderer,
            world,
            size,
            shader_watcher,
            sky,
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
    }

    pub(crate) fn render(
        &mut self,
        camera_bundle: &mut CameraBundle,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        mouse: &MouseState,
        settings: &Settings,
    ) {
        let camera = &mut camera_bundle.camera;

        self.check_shader_changes(ui_loader);

        let aspect = self.config.width as f32 / self.config.height as f32;
        let cam_pos = camera.position();
        let orbit_radius = camera.orbit_radius;

        // ---------------------------------------------
        // TIME SCALES
        // ---------------------------------------------
        let day_length: f32 = 960.0;
        // let t_days = (time.total_time + day_length * 203.3) / day_length; // night
        let t_days: f32 = (time.total_game_time / day_length as f64) as f32;
        ui_loader.variables.set_f32("day_length", day_length);
        ui_loader.variables.set_f32("total_days", t_days);

        let day_phase = t_days % 1.0;
        let day_ang = day_phase * TAU;

        let year_phase = (t_days / 365.0) % 1.0;
        let _year_ang = year_phase * TAU;

        let base_year = 2026.0;
        let current_year = base_year + t_days / 365.0;
        ui_loader.variables.set_f32("base_year", base_year);
        ui_loader.variables.set_f32("current_year", current_year);

        // ---------------------------------------------
        // OBSERVER PARAMETERS
        // ---------------------------------------------
        let latitude = 48.0_f32.to_radians();
        let lat_rot = glam::Quat::from_rotation_x(latitude);

        let sidereal_factor = 1.0027379_f32;
        let sidereal_ang = day_ang * sidereal_factor;
        let earth_rot = glam::Quat::from_rotation_y(sidereal_ang);

        let obliquity = 23.439_f32.to_radians();
        ui_loader.variables.set_f32("earth_obliquity", obliquity);
        let obliq_rot = glam::Quat::from_rotation_x(-obliquity);

        // ---------------------------------------------
        // SUN (ECLIPTIC -> EQUATORIAL -> LOCAL SKY)
        // ---------------------------------------------
        let sun_ecliptic_lon = year_phase * TAU;
        let sun_ecl = glam::Vec3::new(sun_ecliptic_lon.cos(), 0.0, sun_ecliptic_lon.sin());
        let sun_eq = (obliq_rot * sun_ecl).normalize();
        let sun_decl = sun_eq.y.asin().to_degrees();
        ui_loader.variables.set_f32("sun_declination", sun_decl);

        let sun_dir = (lat_rot * earth_rot * sun_eq).normalize();

        // ---------------------------------------------
        // MOON ORBIT (LOW ACCURACY J2000 MODEL)
        // ---------------------------------------------
        let jd = 2451545.0 + t_days;
        let t = (jd - 2451545.0) / 36525.0;

        let n = (125.122 - 0.0529538083 * t).to_radians();
        let i = 5.145_f32.to_radians();
        let w = (318.063 + 0.1643573223 * t).to_radians();

        let a = 60.2666_f32;
        let e = 0.054900_f32;
        let m = (115.3654 + 13.06499295 * t_days).to_radians();

        let e_anom = m + e * m.sin() * (1.0 + e * m.cos());
        let xv = a * (e_anom.cos() - e);
        let yv = a * ((1.0 - e * e).sqrt() * e_anom.sin());
        let v = yv.atan2(xv);
        let r = (xv * xv + yv * yv).sqrt();

        let xh = r * (n.cos() * (v + w).cos() - n.sin() * (v + w).sin() * i.cos());
        let zh = r * ((v + w).sin() * i.sin());
        let yh = r * (n.sin() * (v + w).cos() + n.cos() * (v + w).sin() * i.cos());

        let moon_ecl = glam::Vec3::new(xh as f32, zh as f32, yh as f32).normalize();
        let moon_eq = (obliq_rot * moon_ecl).normalize();
        let moon_dir = (lat_rot * earth_rot * moon_eq).normalize();

        // ---------------------------------------------
        // MOON PHASE (0 new, 1 full)
        // ---------------------------------------------
        let phase_angle = sun_eq.dot(moon_eq).clamp(-1.0, 1.0).acos();
        let moon_phase = (1.0 - phase_angle.cos()) * 0.5;
        ui_loader.variables.set_f32("moon_phase", moon_phase);

        let (view, proj, view_proj) = camera.matrices(aspect);

        let ray = ray_from_mouse_pixels(
            glam::Vec2::new(mouse.pos.x, mouse.pos.y),
            self.config.width as f32,
            self.config.height as f32,
            view,
            proj,
        );

        self.world.pick_terrain_point(ray);

        let new_uniforms = make_new_uniforms(
            view,
            proj,
            view_proj,
            sun_dir,
            moon_dir,
            cam_pos,
            orbit_radius,
            time.total_time as f32,
        );
        self.queue.write_buffer(
            &self.pipelines.uniforms.buffer,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );

        // proj params for fog depth reconstruction
        let proj_params = [
            view.col(2).z, // proj[2][2]
            view.col(3).z, // proj[3][2]
        ];

        let fog_uniforms = FogUniforms {
            screen_size: [self.config.width as f32, self.config.height as f32],
            proj_params,
            fog_density: 0.0000,
            fog_height: 0.0,
            cam_height: camera.position().y,
            _pad0: 0.0,
            fog_color: [0.55, 0.55, 0.6],
            _pad1: 0.0,
            fog_sky_factor: 0.4,
            fog_height_falloff: 0.12,
            fog_start: 1000.0,
            fog_end: 10000.0,
        };

        self.queue.write_buffer(
            &self.pipelines.fog_uniforms.buffer,
            0,
            bytemuck::bytes_of(&fog_uniforms),
        );

        let sky_uniform = SkyUniform {
            exposure: 1.0,
            moon_phase,

            sun_size: 0.0465, // NDC radius (0.05 = kinda big fella)
            sun_intensity: 1.0,

            moon_size: 0.03,
            moon_intensity: 1.0,

            _pad1: 1.0,
            _pad2: 0.0,
        };

        self.queue.write_buffer(
            &self.pipelines.sky_uniforms.buffer,
            0,
            bytemuck::bytes_of(&sky_uniform),
        );

        self.world
            .update(&self.device, &self.queue, camera, aspect, settings);

        // get frame
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(SurfaceError::Outdated | SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                let frame = self.surface.get_current_texture().unwrap();

                // NOW recreate MSAA + depth using frame.texture.size()
                let size = frame.texture.size();
                self.config.width = size.width;
                self.config.height = size.height;
                self.pipelines.resize(&self.config, self.msaa_samples);

                frame
            }
            Err(e) => panic!("{e:?}"),
        };

        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // gizmo verts (unchanged)
        let t = camera.target;
        let s = camera.orbit_radius * 0.2;
        let axes = [
            // X
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x + s, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            // Y
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y + s, t.z],
                color: [0.2, 1.0, 0.2],
            },
            // Z
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 0.6, 1.0],
            },
            LineVtx {
                pos: [t.x, t.y, t.z + s],
                color: [0.2, 0.6, 1.0],
            },
        ];
        self.queue.write_buffer(
            &self.pipelines.gizmo_mesh_buffers.vertex,
            0,
            bytemuck::cast_slice(&axes),
        );

        // Update Water Uniform
        let wu = WaterUniform {
            sea_level: 0.0,
            _pad0: [0.0; 3],
            color: [0.05, 0.25, 0.35, 0.55],
            wave_tiling: 0.5,
            wave_strength: 0.1,
            _pad1: [0.0; 2],
        };

        self.queue.write_buffer(
            &self.pipelines.water_uniforms.buffer,
            0,
            bytemuck::bytes_of(&wu),
        );

        let background_color = Color {
            r: settings.background_color[0] as f64,
            g: settings.background_color[1] as f64,
            b: settings.background_color[2] as f64,
            a: settings.background_color[3] as f64,
        };

        let color_attachment = if self.pipelines.msaa_samples > 1 {
            // MSAA enabled
            Some(RenderPassColorAttachment {
                view: &self.pipelines.msaa_view,     // multisampled
                resolve_target: Some(&surface_view), // resolve into swapchain
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Clear(background_color),
                    store: StoreOp::Store,
                },
            })
        } else {
            // MSAA disabled
            Some(RenderPassColorAttachment {
                view: &surface_view,  // render directly to swapchain
                resolve_target: None, // resolve target must be None
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Clear(background_color),
                    store: StoreOp::Store,
                },
            })
        };

        // 1) MAIN 3D + FOG pass (MSAA -> resolve into surface_view)
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[color_attachment],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.pipelines.depth_view, // same sample_count as msaa_view
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: Some(Operations {
                        load: LoadOp::Clear(0),
                        store: StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Sky, Stars and Terrain
            self.sky.render(&mut pass, &self.pipelines);

            // Above Water Terrain
            self.world
                .make_pick_uniforms(&self.queue, &self.pipelines.pick_uniforms.buffer);
            pass.set_stencil_reference(0);
            self.world
                .render(&mut pass, &self.pipelines, camera, aspect, false);
            // Underwater Terrain
            pass.set_stencil_reference(1);

            self.world
                .render(&mut pass, &self.pipelines, camera, aspect, true);
            // Gizmo
            pass.set_pipeline(&self.pipelines.gizmo_pipeline.pipeline);
            pass.set_bind_group(0, &self.pipelines.uniforms.bind_group, &[]);
            pass.set_bind_group(1, &self.pipelines.fog_uniforms.bind_group, &[]);
            pass.set_vertex_buffer(0, self.pipelines.gizmo_mesh_buffers.vertex.slice(..));
            pass.draw(0..6, 0..1);

            pass.set_pipeline(&self.pipelines.water_pipeline.pipeline);
            pass.set_bind_group(0, &self.pipelines.uniforms.bind_group, &[]);

            // Render Water
            pass.set_stencil_reference(1); // Draw above underwater (1), not above water (0)
            pass.set_vertex_buffer(0, self.pipelines.water_mesh_buffers.vertex.slice(..));
            pass.set_index_buffer(
                self.pipelines.water_mesh_buffers.index.slice(..),
                IndexFormat::Uint32,
            );
            pass.set_bind_group(1, &self.pipelines.water_uniforms.bind_group, &[]);
            pass.draw_indexed(0..self.pipelines.water_mesh_buffers.index_count, 0, 0..1);

            // 2) UI pass on top of resolved image //
            let screen_uniform = crate::renderer::ui::ScreenUniform {
                size: [self.size.width as f32, self.size.height as f32],
                time: time.total_time as f32,
                enable_dither: 1,
                mouse: mouse.pos.to_array(),
            };

            self.queue.write_buffer(
                &self.ui_renderer.pipelines.uniform_buffer,
                0,
                bytemuck::bytes_of(&screen_uniform),
            );

            let size = (self.config.width as f32, self.config.height as f32);
            self.ui_renderer
                .render(&mut pass, ui_loader, &self.queue, time, size, mouse);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    pub(crate) fn cycle_msaa(&mut self) {
        // Pick next MSAA value
        self.msaa_samples = match self.msaa_samples {
            1 => 2,
            2 => 4,
            4 => 8,
            _ => 1,
        };

        println!("MSAA changed to {}x", self.msaa_samples);

        // Recreate pipelines with new sample count
        self.pipelines.msaa_samples = self.msaa_samples;
        self.pipelines.resize(&self.config, self.msaa_samples);
        self.pipelines.recreate_pipelines();

        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
    }

    fn reload_all_shaders(&mut self) -> anyhow::Result<()> {
        self.pipelines.reload_shaders()?;
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
                ui_loader.log_console(format!("✅ {label}"));
            }
            Err(err) => ui_loader.log_console(format!("❌ Shader reload failed: {err}")),
        }
    }
}

fn pick_fail_safe_present_mode(
    surface: &Surface,
    adapter: &Adapter,
    user_mode: PresentMode,
) -> PresentMode {
    let caps = surface.get_capabilities(adapter);

    // Fallback priority. Mailbox first, then Fifo (always supported), then Immediate.
    let fallback_chain = [
        user_mode,
        PresentMode::Mailbox,
        PresentMode::Immediate,
        PresentMode::Fifo, // required by spec...
    ];

    for mode in fallback_chain {
        if caps.present_modes.contains(&mode) {
            return mode;
        }
    }

    // Absolute guarantee
    PresentMode::Fifo
}
