use crate::data::Settings;
use crate::helpers::paths::data_dir;
use crate::renderer::render_core::{
    RenderCore, create_device, create_surface_and_adapter, create_surface_config,
};
use crate::renderer::shadows::CSM_CASCADES;
use crate::ui::actions::CommandQueue;
use crate::ui::ui_editor::UiButtonLoader;
use crate::world::world_core::WorldCore;
use std::sync::Arc;
use std::time::Instant;
use wgpu::Surface;
use winit::window::Window;

pub struct CommandQueues {
    pub(crate) ui_command_queue: CommandQueue,
}
impl CommandQueues {
    pub fn new() -> Self {
        Self {
            ui_command_queue: CommandQueue::new(),
        }
    }
}
pub struct Resources {
    pub settings: Settings,
    pub window: Arc<Window>,
    pub command_queues: CommandQueues,

    // The simulation & world core:
    pub world_core: WorldCore,

    // The GPU + render-only subsystems:
    pub render_core: RenderCore,

    pub ui_loader: UiButtonLoader,
    pub surface: Surface<'static>,
}

impl Resources {
    pub fn new(window: Arc<Window>) -> Self {
        let mut settings = Settings::load(data_dir("settings.toml"));
        let editor_mode = settings.editor_mode.clone();

        let (surface, adapter, size) = create_surface_and_adapter(window.clone());
        let (config, msaa_samples) = create_surface_config(&surface, &adapter, &mut settings, size);
        let (device, queue) = &create_device(&adapter);

        surface.configure(device, &config);

        let mut world_core = WorldCore::new(device, &settings);
        let camera_entity = world_core.world_state.main_camera();
        let camera = world_core.world_state.camera_mut(camera_entity).unwrap();
        camera.target = settings.player_pos;

        let render_core = RenderCore::new(device, queue, &config, size, adapter, &settings, camera);

        let mut ui_loader = UiButtonLoader::new(
            editor_mode,
            settings.override_mode,
            settings.show_gui,
            &settings.bend_mode.clone(),
            window.inner_size(),
        );
        ui_loader
            .variables
            .set_bool("editor_mode", settings.editor_mode);
        let mut command_queues = CommandQueues::new();
        ui_loader.set_starting_menu(&settings, &mut command_queues.ui_command_queue);
        world_core.time.total_game_time = settings.total_game_time;

        Self {
            surface,
            settings,
            ui_loader,
            window,
            command_queues,
            world_core,
            render_core,
        }
    }
}

pub struct TimeSystem {
    pub last_frame: Instant,
    pub start: Instant,

    // Render timing
    pub render_dt: f32,
    pub render_fps: f32,
    pub target_fps: f32,
    pub target_frametime: f32,

    // Fixed-step simulation timing
    pub sim_accumulator: f32,
    pub target_sim_dt: f32,
    pub prev_time_scale: f32,

    // Achieved speed measurement (windowed)
    pub achieved_speed: f32,
    achieved_speed_window_time: f32,
    achieved_speed_window_steps: u32,

    // Totals
    pub total_time: f64,
    pub total_game_time: f64,
    pub frame_count: u64,

    // Safety
    pub max_frame_dt: f32,

    // Speed-change detection
    pub speed_just_changed: bool,
    pub current_time_speed: f32,
}

impl TimeSystem {
    pub fn new() -> Self {
        let now = Instant::now();
        let target_fps = 100.0;
        let target_frametime = 1.0 / target_fps;

        let tps = 60.0;
        let target_sim_dt = 1.0 / tps;

        Self {
            last_frame: now,
            start: now,

            render_dt: 0.0,
            render_fps: 0.0,
            target_fps,
            target_frametime,

            sim_accumulator: 0.0,
            target_sim_dt,

            prev_time_scale: 1.0,

            achieved_speed: 1.0,
            achieved_speed_window_time: 0.0,
            achieved_speed_window_steps: 0,

            total_time: 0.0,
            total_game_time: 0.0,
            frame_count: 0,

            max_frame_dt: 0.25,

            speed_just_changed: false,
            current_time_speed: 1.0,
        }
    }

    pub fn set_tps(&mut self, tps: f32) {
        let tps = tps.max(1.0);
        self.target_sim_dt = 1.0 / tps;
        self.sim_accumulator = 0.0;
    }

    pub fn set_fps(&mut self, target_fps: f32) {
        self.target_fps = target_fps.max(1.0);
        self.target_frametime = 1.0 / self.target_fps;
    }

    /// Call once per render frame.
    pub fn begin_frame(&mut self, time_speed: f32) {
        let now = Instant::now();
        let mut dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        dt = dt.clamp(0.0, self.max_frame_dt);

        self.render_dt = dt;
        self.render_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };

        self.total_time += dt as f64;
        self.total_game_time += (dt as f64) * (time_speed as f64);
        self.frame_count += 1;

        // Detect speed change: flush accumulator immediately on ANY speed transition
        let speed_changed = (time_speed - self.current_time_speed).abs() > 1e-6;
        self.speed_just_changed = speed_changed;

        if speed_changed {
            self.sim_accumulator = 0.0;
            self.prev_time_scale = self.current_time_speed;
            self.current_time_speed = time_speed;
            // Reset measurement window on speed change
            self.achieved_speed = time_speed;
            self.achieved_speed_window_time = 0.0;
            self.achieved_speed_window_steps = 0;
        }

        // Accumulate wall-clock time for the measurement window
        self.achieved_speed_window_time += dt;

        // Accumulate sim time at current speed
        if time_speed >= 0.0 {
            self.sim_accumulator += dt * time_speed;
        }
    }

    /// Call after sim stepping is done for this frame, passing how many steps were taken.
    pub fn update_achieved_speed(&mut self, steps: u32) {
        self.achieved_speed_window_steps += steps;

        // Evaluate every 0.5 seconds of wall-clock time
        const WINDOW_DURATION: f32 = 0.5;

        if self.achieved_speed_window_time >= WINDOW_DURATION {
            let sim_time = self.achieved_speed_window_steps as f32 * self.target_sim_dt;
            self.achieved_speed = if self.achieved_speed_window_time > 0.0 {
                sim_time / self.achieved_speed_window_time
            } else {
                0.0
            };

            // Reset window
            self.achieved_speed_window_time = 0.0;
            self.achieved_speed_window_steps = 0;
        }
    }

    pub fn clear_sim_accumulator(&mut self) {
        self.sim_accumulator = 0.0;
    }

    pub fn clamp_sim_accumulator(&mut self, max_steps: usize) {
        let max = self.target_sim_dt * max_steps as f32;
        if self.sim_accumulator > max {
            self.sim_accumulator = max;
        }
    }

    #[inline]
    pub fn can_step_sim(&self) -> bool {
        self.target_sim_dt > 0.0 && self.sim_accumulator >= self.target_sim_dt
    }

    #[inline]
    pub fn consume_sim_step(&mut self) {
        self.sim_accumulator -= self.target_sim_dt;
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    // ── Current frame matrices ──────────────────────────────────
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],

    // ── Previous frame reprojection ─────────────────────────────
    pub prev_view_proj: [[f32; 4]; 4],

    // ── Shadow cascades ─────────────────────────────────────────
    pub lighting_view_proj: [[[f32; 4]; 4]; CSM_CASCADES],
    pub cascade_splits: [f32; 4],

    // ── Lighting ────────────────────────────────────────────────
    pub sun_direction: [f32; 3],
    pub time: f32,
    pub moon_direction: [f32; 3],
    pub orbit_radius: f32,

    // ── Current camera (chunk-relative) ─────────────────────────
    pub camera_local: [f32; 3], // vec3<f32> + 1 float pad
    pub chunk_size: f32,
    pub camera_chunk: [i32; 2], // vec2<i32>
    pub _pad_cam: [u32; 2],     // align to 16

    // ── Previous camera (chunk-relative) ────────────────────────
    pub prev_camera_local: [f32; 3], // vec3<f32> + 1 float pad
    pub frame_index: u32,
    pub prev_camera_chunk: [i32; 2], // vec2<i32>
    pub _pad_prev1: [i32; 2],        // align to 16

    // ── TAA jitter ──────────────────────────────────────────────
    pub curr_jitter: [f32; 2],
    pub prev_jitter: [f32; 2],

    // ── Misc settings ───────────────────────────────────────────
    pub reversed_depth_z: u32,
    pub csm_enabled: u32,
    pub near_far_depth: [f32; 2],
}
