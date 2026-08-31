use crate::app::GAME_VERSION;
use crate::data::FullScreenMode;
use crate::data::Settings;
use crate::helpers::paths::rusty_skylines_dir;
use crate::renderer::props::Props;
use crate::renderer::render_core::{
    Renderer, create_device, create_surface_and_adapter, create_surface_config,
};
use crate::renderer::shadows::CSM_CASCADES;
use crate::simulation::Simulation;
use crate::ui::actions::CommandQueue;
use crate::ui::ui_editor::Ui;
use crate::ui::variables::{Variables, load_colors};
use crate::world::astronomy::Astronomy;
use crate::world::game_state::GameState;
use crate::world::sound::sound::Sounds;
use crate::world::statisticals::demands::HOURS_PER_DAY;
use crate::world::world::World;
use gfxinfo::active_gpu;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use wgpu::Surface;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub struct CommandQueues {
    pub ui_command_queue: CommandQueue,
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
    pub window: Arc<Box<dyn Window>>,
    pub command_queues: CommandQueues,

    // The simulation & world core:
    pub world: World,
    pub simulation: Simulation,
    pub game_state: GameState,
    // The GPU + render-only subsystems:
    pub render_core: Renderer,

    pub ui: Ui,
    pub sounds: Sounds,
    pub surface: Surface<'static>,
    pub pending_fullscreen_change: Option<FullScreenMode>,
    pub pending_present_mode_change: bool,
}

fn ram_comment(sys: System) -> String {
    let total_ram = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    match total_ram {
        ..0.4 => format!(
            "{} GiB of RAM... You are in the deepest pits of hell with that low RAM! Go buy some! No DDR1 or DDR2 allowed!",
            total_ram
        ),
        0.4..1.8 => format!(
            "{:.4} GiB of RAM... Trying my game on your Windows 98 PC or what? I told you it wouldn't work!",
            total_ram
        ),
        1.8..4.5 => format!("{:.2} GiB of RAM... Sorry, you are poor!", total_ram),
        4.5..8.5 => format!("{:.2} GiB of RAM... Awkward Ram size!!", total_ram),
        8.5..16.9 => format!(
            "{:.2} GiB of RAM... You can game quite well with {:.2} GiB!!!",
            total_ram, total_ram
        ),
        16.9..33.6 => format!(
            "{:.2} GiB of RAM... Respect for the Ram!!!! Can I have it? No? Well, my game will have it!",
            total_ram
        ),
        33.6..69.1 => format!(
            "{:.2} GiB of RAM... Insane Ram size!!!!! Either you are rich or you bought it before AI fucked us over!",
            total_ram
        ),
        69.1..132.9 => format!(
            "{:.2} GiB of RAM... Holy Shit, go open some CAD software, why the fuck are you playing my game with {} GiB of RAM?!",
            total_ram, total_ram
        ),
        132.9..261.0 => format!(
            "{:.2} GiB of RAM... That's GALACTIC amounts! Did you forget to switch to your main PC instead of your server rack??",
            total_ram
        ),
        261.0..532.8 => format!(
            "{:.2} GiB of RAM... WTF? Did you steal my RAM? Is this normal??",
            total_ram
        ),
        532.8..1200.3 => format!(
            "{:.2} GiB of RAM... A fucking Terrabyte! There is no way in hell windows 18 needs that much RAM! So much spyware?",
            total_ram
        ),
        1200.3..4522.2 => format!(
            "{:.2} GiB of RAM... Goddamnit! You are literally playing on my Minecraft server! Or more like a fucking AI server...",
            total_ram
        ),
        4522.2..16200.4 => format!(
            "{:.2} GiB of RAM... I bet this is normal in your time! Just a casual stick of 4TiB DDR 11 Memory with 24000mhz... Yes, I future-proofed this message to past the year 2077!",
            total_ram
        ),
        16200.4.. => format!(
            "{:.2} GiB of RAM... You are literally a god. We must all bow down to you. You own ALL RAM... Fuck you and thank you god!",
            total_ram
        ),
        _ => panic!(
            "You have some goofy ass amount of RAM, wtf? {} GiB of Ram to be precise.",
            total_ram
        ),
    }
}
fn format_memory(bytes: u64) -> String {
    let mib = bytes as f64 / 1024.0 / 1024.0;
    let gib = mib / 1024.0;
    let tib = gib / 1024.0;

    if mib < 1024.0 {
        format!("{:.0} MiB", mib)
    } else if gib < 1024.0 {
        format!("{:.2} GiB", gib)
    } else {
        format!("{:.2} TiB", tib)
    }
}
impl Resources {
    pub fn new(window: Arc<Box<dyn Window>>, event_loop: &dyn ActiveEventLoop) -> Self {
        let mut settings = Settings::load(rusty_skylines_dir("settings.toml"));
        let mut variables = Variables::new();
        variables.set_string("GAME_VERSION", GAME_VERSION);
        let editor_mode = settings.editor_mode.clone();

        let (surface, adapter, size) = create_surface_and_adapter(window.clone(), event_loop);
        println!(" [app] surface + adapter created");
        let (config, msaa_samples) = create_surface_config(
            &surface,
            &adapter,
            &mut settings,
            &mut variables,
            size,
            false,
        );
        println!(
            "Surface size: {}x{}, Format: {:?}, Alpha Mode: {:?}",
            config.width, config.height, config.format, config.alpha_mode
        );

        let gpu_info = active_gpu();
        let adapter_info = adapter.get_info();
        if let Ok(gpu_info) = gpu_info {
            println!(
                "GPU: {} ({}), Driver: {} {}, Type: {:?}, VRAM: {}",
                gpu_info.model(),
                gpu_info.family(),
                adapter_info.driver,
                adapter_info.driver_info,
                adapter_info.device_type,
                format_memory(gpu_info.info().total_vram())
            );
        } else {
            println!(
                "GPU: {}, Driver: {} {}, Type: {:?}",
                adapter_info.name,
                adapter_info.driver,
                adapter_info.driver_info,
                adapter_info.device_type
            );
        }

        let mut sys = System::new_all();
        sys.refresh_all();
        let cpus = sys.cpus();

        if let Some(cpu) = cpus.first() {
            let physical_cores = System::physical_core_count().unwrap_or(0);
            //let cpu_name = cpu.name().split(" cpu").next().unwrap_or(cpu.name());
            println!(
                "CPU: {}, Cores: {}/{}, Architecture: {}",
                cpu.brand(),
                physical_cores,
                cpus.len(),
                std::env::consts::ARCH
            );
        }

        println!("RAM: {}", ram_comment(sys));
        println!(
            "OS: {}, OS kernel: {}, System name: {}, Average System load last 5 minutes: {}%",
            System::long_os_version().unwrap_or_default(),
            System::kernel_long_version(),
            System::name().unwrap_or_default(),
            System::load_average().five
        );
        println!(
            "Backend: {}, Present mode: {}, MSAA: {}x",
            adapter_info.backend, settings.present_mode, settings.msaa_samples
        );
        let (device, queue) = &create_device(&adapter);

        surface.configure(device, &config);
        println!("[app] surface configured");

        let game_state = GameState::new();
        let props = Props::new(device);
        let mut world_core = World::new(device, queue, &settings, &props);
        let camera = &mut world_core.world_state.camera;

        let render_core = Renderer::new(
            device, queue, &config, size, adapter, &settings, camera, props,
        );

        let mut ui_loader = Ui::new(&settings, variables, window.surface_size());
        ui_loader
            .variables
            .set_bool("editor_mode", settings.editor_mode);
        load_colors(
            rusty_skylines_dir("colors.toml"),
            &settings,
            &mut ui_loader.variables,
        );
        let mut command_queues = CommandQueues::new();
        ui_loader.set_starting_menu(&settings, &mut command_queues.ui_command_queue);
        world_core.time.update_hour();
        let pending_fullscreen_change = Some(settings.fullscreen_mode);
        Self {
            surface,
            settings,
            ui: ui_loader,
            window,
            command_queues,
            world: world_core,
            simulation: Simulation::new(),
            game_state,
            render_core,
            sounds: Sounds::new(),
            pending_fullscreen_change,
            pending_present_mode_change: false,
        }
    }

    pub fn reconfigure_surface(&mut self) {
        let (config, msaa_samples) = create_surface_config(
            &self.surface,
            &self.render_core.adapter,
            &mut self.settings,
            &mut self.ui.variables,
            self.window.surface_size(),
            true,
        );
        self.render_core.config = config;
        self.render_core.update_msaa(&self.settings);
        self.surface
            .configure(&self.render_core.device, &self.render_core.config);
    }
}

pub const DAYS_PER_YEAR: f64 = 20.0;
pub const HOURS_PER_YEAR: f64 = DAYS_PER_YEAR * HOURS_PER_DAY;
const SCHOOL_YEAR_START_DAY: u32 = (DAYS_PER_YEAR * 0.75) as u32; // ~September 1

pub struct Time {
    pub timer: Timer,
    pub last_frame: Instant,

    pub render_dt: f32,
    pub render_fps: f32,
    pub target_fps: f32,
    pub target_frametime: f32,

    pub sim_accumulator: f32,
    pub target_sim_dt: f32,
    pub prev_time_scale: f32,

    pub achieved_speed: f32,
    achieved_speed_window_time: f32,
    achieved_speed_window_steps: u32,

    pub total_time: f64,      // In seconds
    pub total_game_time: f64, // In sim seconds
    pub hour: f64,            // In hours
    pub total_hours: f64,
    pub total_days: f64,
    pub day_length: f64, // In sim seconds
    pub frame_count: u64,

    pub max_frame_dt: f32,

    pub speed_just_changed: bool,
    pub current_time_speed: f32,

    pub astronomy: Astronomy,
    frame_start: Instant,
    pub cpu_frame_duration: Duration,
    after_acquire_start: Instant,
}

impl Time {
    pub fn new() -> Self {
        let now = Instant::now();
        let target_fps = 100.0;
        let target_frametime = 1.0 / target_fps;

        let tps = 60.0;
        let target_sim_dt = 1.0 / tps;

        Self {
            timer: Default::default(),
            last_frame: now,

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
            hour: 0.0,
            total_hours: 0.0,
            total_days: 0.0,
            day_length: 1800.0,
            frame_count: 0,

            max_frame_dt: 0.25,

            speed_just_changed: false,
            current_time_speed: 1.0,
            astronomy: Astronomy::default(),
            frame_start: Instant::now(),
            cpu_frame_duration: Default::default(),
            after_acquire_start: Instant::now(),
        }
    }

    #[inline]
    pub fn sim_time(&self) -> f64 {
        self.total_game_time
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

    #[inline]
    pub fn is_rewinding(&self) -> bool {
        self.current_time_speed < 0.0
    }

    #[inline]
    pub fn time_direction(&self) -> f32 {
        if self.current_time_speed >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn begin_frame(&mut self, time_speed: f32) {
        let now = Instant::now();
        let raw_dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Pure wall-clock dt, no time_speed here
        let raw_dt = raw_dt.clamp(0.0, self.max_frame_dt);

        if self.render_dt == 0.0 {
            self.render_dt = raw_dt;
        } else {
            self.render_dt += (raw_dt - self.render_dt) * 0.05;
        }

        self.render_fps = if self.render_dt > 0.0 {
            1.0 / self.render_dt
        } else {
            0.0
        };

        self.total_time += self.render_dt as f64;

        let speed_changed = (time_speed - self.current_time_speed).abs() > 1e-6;
        self.speed_just_changed = speed_changed;

        if speed_changed {
            self.sim_accumulator = 0.0;
            self.prev_time_scale = self.current_time_speed;
            self.current_time_speed = time_speed;
            self.achieved_speed = time_speed;
            self.achieved_speed_window_time = 0.0;
            self.achieved_speed_window_steps = 0;
        }

        self.achieved_speed_window_time += self.render_dt;

        // time_speed scaling only applies to the sim accumulator
        self.sim_accumulator += self.render_dt * time_speed.abs();
    }
    pub fn end_frame(&mut self) {
        self.frame_count += 1;
    }
    pub fn update_achieved_speed(&mut self, steps: u32) {
        self.achieved_speed_window_steps += steps;

        const WINDOW_DURATION: f32 = 0.5;

        if self.achieved_speed_window_time >= WINDOW_DURATION {
            let sim_time = self.achieved_speed_window_steps as f32 * self.target_sim_dt;
            let raw_speed = if self.achieved_speed_window_time > 0.0 {
                sim_time / self.achieved_speed_window_time
            } else {
                0.0
            };

            self.achieved_speed = raw_speed * self.time_direction();

            self.achieved_speed_window_time = 0.0;
            self.achieved_speed_window_steps = 0;
        }
    }

    pub fn update_hour(&mut self) {
        self.total_days = self.total_game_time % self.day_length;
        self.total_hours = self.total_days * 24.0;
        let day_progress = self.total_days / self.day_length;
        self.hour = day_progress * 24.0;
    }
    #[inline]
    pub fn hour(&self) -> u32 {
        let hours = self.hour.floor() as u32;
        hours
    }
    #[inline]
    pub fn minute(&self) -> u32 {
        let minutes = ((self.hour.fract()) * 60.0) as u32;
        minutes
    }
    #[inline]
    pub fn is_new_day(&self) -> bool {
        (self.total_hours % HOURS_PER_DAY) as u64 == 0
    }
    #[inline]
    pub fn year(&self) -> u32 {
        let current_year = self.astronomy.current_year as u32;
        current_year
    }
    #[inline]
    pub fn school_year(&self) -> u32 {
        let current_year = self.astronomy.current_year as u32;
        let shifted_days = 365 - SCHOOL_YEAR_START_DAY;

        current_year + (shifted_days / 365)
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
        if self.target_sim_dt <= 0.0 || self.sim_accumulator < self.target_sim_dt {
            return false;
        }
        if self.is_rewinding() && self.total_game_time < 1e-9 {
            return false;
        }
        true
    }

    #[inline]
    pub fn consume_sim_step(&mut self) {
        self.sim_accumulator -= self.target_sim_dt;
        let dt = self.target_sim_dt as f64;
        //println!("{}", self.total_game_time);
        //println!("Yeah they are");
        if self.current_time_speed >= 0.0 {
            self.total_game_time += dt;
        } else {
            self.total_game_time = (self.total_game_time - dt).max(0.0);
        };
        self.update_hour();
    }

    #[inline]
    pub fn game_just_started(&self) -> bool {
        self.frame_count == 0
    }

    pub fn frame_checkpoint(&mut self, checkpoint_type: FrameTimeCheckpointType) -> Duration {
        use FrameTimeCheckpointType::*;
        match checkpoint_type {
            FrameStart => {
                self.frame_start = Instant::now();
            }
            BeforeAcquireFrame => {
                self.cpu_frame_duration = self.frame_start.elapsed();
            }
            AfterAcquireFrame => {
                self.after_acquire_start = Instant::now();
            }
            FrameEnd => {
                self.cpu_frame_duration += self.after_acquire_start.elapsed();
            }
        }
        self.cpu_frame_duration
    }
}

pub enum FrameTimeCheckpointType {
    FrameStart,
    BeforeAcquireFrame,
    AfterAcquireFrame,
    FrameEnd,
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
    pub cascade_splits: [f32; CSM_CASCADES],

    // ── Lighting ────────────────────────────────────────────────
    pub sun_direction: [f32; 3],
    pub time: f32,
    pub moon_direction: [f32; 3],
    pub orbit_radius: f32,

    // ── Current camera (chunk-relative) ─────────────────────────
    pub camera_local: [f32; 3], // vec3<f32> + 1 float pad
    pub chunk_size: f32,
    pub camera_chunk: [i32; 2], // vec2<i32>
    pub screen_size: [f32; 2],

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

#[derive(Debug, Default)]
pub struct Timer {
    starts: HashMap<String, Instant>,
    checkpoints: HashMap<String, Vec<Duration>>,
}

impl Timer {
    pub fn checkpoint(&mut self, name: &str, end: bool) {
        match end {
            false => {
                self.starts.insert(name.to_owned(), Instant::now());
            }

            true => {
                if let Some(start) = self.starts.remove(name) {
                    self.checkpoints
                        .entry(name.to_owned())
                        .or_default()
                        .push(start.elapsed());
                }
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<&[Duration]> {
        self.checkpoints.get(name).map(Vec::as_slice)
    }
    pub fn totals(&self) -> Vec<(String, Duration)> {
        self.checkpoints
            .iter()
            .map(|(name, durations)| {
                let total = durations.iter().copied().sum();
                (name.clone(), total)
            })
            .collect()
    }
    pub fn total(&self, name: &str) -> Duration {
        self.checkpoints
            .get(name)
            .map(|times| times.iter().copied().sum())
            .unwrap_or_default()
    }

    pub fn clear(&mut self) {
        self.starts.clear();
        self.checkpoints.clear();
    }
}
