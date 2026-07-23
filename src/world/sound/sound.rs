use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::resources::Resources;
use crate::world::sound::car_sounds::{CarAudioState, collect_car_audio};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    Device, Host, HostId, SampleFormat, SampleRate, Stream, StreamConfig, SupportedStreamConfig,
};

use crate::world::sound::{MAX_CARS_AUDIO, with_stderr_suppressed};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub fn run_sounds(resources: &mut Resources) {
    // every frame
    let sounds = &mut resources.sounds;

    // Health check and rebuild if needed
    sounds.check_and_rebuild();

    let Ok(mut state) = sounds.state.lock() else {
        return;
    };

    let camera = &resources.world.world_state.camera;
    let terrain = &resources.world.terrain;
    let car_storage = resources.world.cars.car_storage_mut();

    let current_eye = camera.eye_world();
    let prev_eye = camera.prev_eye_world();

    let vel = prev_eye.delta_to(current_eye);

    state.listener_pos = current_eye;
    state.listener_velocity = vel;
    state.listener_yaw = camera.yaw;
    state.listener_pitch = camera.pitch;
    state.cars.clear();

    collect_car_audio(&mut state, camera, terrain, car_storage);
    state.camera_height_above_ground =
        current_eye.local.y - terrain.get_height_at(camera.target, true);
}

pub struct AudioState {
    synths: [CarSynth; 10],
    pub cars: Vec<CarAudioState>,
    pub listener_pos: WorldPos,
    pub listener_velocity: glam::Vec3,
    pub listener_yaw: f32,
    pub listener_pitch: f32,
    pub chunk_size: ChunkSize,
    pub sample_rate: SampleRate,

    pub camera_height_above_ground: f32,
    pub wind_synth: WindSynth,
}

impl AudioState {
    pub fn new(sample_rate: SampleRate) -> Self {
        Self {
            cars: Vec::new(),
            synths: [CarSynth::new(sample_rate as f32); MAX_CARS_AUDIO],
            listener_pos: WorldPos::default(),
            listener_velocity: glam::Vec3::ZERO,
            listener_yaw: 0.0,
            listener_pitch: 0.0,
            chunk_size: ChunkSize::default(),
            sample_rate,

            camera_height_above_ground: 0.0,
            wind_synth: WindSynth::new(sample_rate as f32),
        }
    }
    pub fn clear(&mut self) {
        self.cars.clear();
    }
}

pub struct Sounds {
    stream: Option<Stream>,
    pub state: Arc<Mutex<AudioState>>,
    stream_error: Arc<AtomicBool>,
    sample_counter: Arc<AtomicU64>,
    last_sample_count: u64,
    last_health_check: Instant,
    rebuild_count: u64,
    consecutive_failures: u32,
    last_rebuild_attempt: Instant,
}

#[derive(Debug)]
pub enum AudioError {
    NoHostsAvailable,
    NoDevicesFound,
    NoConfigsSupported,
    StreamBuildFailed(String),
    StreamPlayFailed(String),
    ExhaustedAllOptions,
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoHostsAvailable => write!(f, "No audio hosts available"),
            Self::NoDevicesFound => write!(f, "No output devices found"),
            Self::NoConfigsSupported => write!(f, "No supported configurations"),
            Self::StreamBuildFailed(e) => write!(f, "Stream build failed: {}", e),
            Self::StreamPlayFailed(e) => write!(f, "Stream play failed: {}", e),
            Self::ExhaustedAllOptions => write!(f, "Exhausted all audio initialization options"),
        }
    }
}

impl std::error::Error for AudioError {}

#[derive(Debug, Clone)]
enum DeviceStrategy {
    Default,
    NameContains(String),
    NameExact(String),
    ByIndex(usize),
    First,
    Last,
}

#[derive(Debug, Clone)]
struct InitAttempt {
    host_id: Option<HostId>,
    strategy: DeviceStrategy,
    description: String,
}

impl Sounds {
    const MIN_REBUILD_INTERVAL: Duration = Duration::from_millis(200);
    const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(2);
    const MAX_CONSECUTIVE_FAILURES: u32 = 10;
    const BACKOFF_MAX: Duration = Duration::from_secs(30);

    pub fn new() -> Self {
        Self::new_robust().unwrap_or_else(|e| {
            eprintln!("Audio initialization failed: {}", e);
            eprintln!("Continuing without audio, will retry periodically...");
            Self::new_silent()
        })
    }

    fn new_silent() -> Self {
        let now = Instant::now();
        Self {
            stream: None,
            state: Arc::new(Mutex::new(AudioState::new(48000))),
            stream_error: Arc::new(AtomicBool::new(true)), // Trigger rebuild attempts
            sample_counter: Arc::new(AtomicU64::new(0)),
            last_sample_count: 0,
            last_health_check: now,
            rebuild_count: 0,
            consecutive_failures: 1,
            last_rebuild_attempt: now,
        }
    }

    pub fn check_and_rebuild(&mut self) {
        let now = Instant::now();

        // Check for explicit stream errors from callback
        let has_error = self.stream_error.load(Ordering::SeqCst);

        // Periodic health check - detect stalled/dead streams
        let mut stalled = false;
        if self.stream.is_some()
            && now.duration_since(self.last_health_check) >= Self::HEALTH_CHECK_INTERVAL
        {
            self.last_health_check = now;
            let current_samples = self.sample_counter.load(Ordering::SeqCst);

            if current_samples == self.last_sample_count && self.last_sample_count > 0 {
                stalled = true;
                eprintln!(
                    "Audio stream stalled (no samples processed in {:?})",
                    Self::HEALTH_CHECK_INTERVAL
                );
            }
            self.last_sample_count = current_samples;
        }

        // Check if stream was lost
        let stream_missing = self.stream.is_none();

        let needs_rebuild = has_error || stalled || stream_missing;

        if !needs_rebuild {
            // Reset failure counter on sustained success
            if self.consecutive_failures > 0
                && now.duration_since(self.last_rebuild_attempt) > Duration::from_secs(60)
            {
                self.consecutive_failures = 0;
            }
            return;
        }

        // Apply exponential backoff
        let backoff = self.calculate_backoff();
        if now.duration_since(self.last_rebuild_attempt) < backoff {
            return;
        }

        // Don't spam rebuilds forever
        if self.consecutive_failures >= Self::MAX_CONSECUTIVE_FAILURES {
            // Only try once per minute after giving up
            if now.duration_since(self.last_rebuild_attempt) < Duration::from_secs(60) {
                return;
            }
        }

        self.attempt_rebuild();
    }

    fn calculate_backoff(&self) -> Duration {
        if self.consecutive_failures == 0 {
            return Self::MIN_REBUILD_INTERVAL;
        }

        let factor = 2.0f32.powi((self.consecutive_failures - 1).min(10) as i32);
        let backoff_ms = (Self::MIN_REBUILD_INTERVAL.as_millis() as f32 * factor) as u64;
        Duration::from_millis(backoff_ms).min(Self::BACKOFF_MAX)
    }

    fn attempt_rebuild(&mut self) {
        let now = Instant::now();
        self.last_rebuild_attempt = now;
        self.stream_error.store(false, Ordering::SeqCst);

        // Drop old stream first
        if let Some(stream) = self.stream.take() {
            drop(stream);
            std::thread::sleep(Duration::from_millis(100));
        }

        let attempt_num = self.consecutive_failures + 1;
        println!(
            "Rebuilding audio (attempt #{}, backoff: {:?})...",
            attempt_num,
            self.calculate_backoff()
        );

        match Self::build_new_stream(
            Arc::clone(&self.state),
            Arc::clone(&self.stream_error),
            Arc::clone(&self.sample_counter),
        ) {
            Ok(stream) => {
                self.stream = Some(stream);
                self.rebuild_count += 1;
                self.consecutive_failures = 0;
                self.last_sample_count = 0;
                self.sample_counter.store(0, Ordering::SeqCst);
                self.last_health_check = now;
                println!(
                    "Audio rebuilt successfully! (total rebuilds: {})",
                    self.rebuild_count
                );
            }
            Err(e) => {
                self.consecutive_failures += 1;

                if self.consecutive_failures >= Self::MAX_CONSECUTIVE_FAILURES {
                    eprintln!(
                        "Audio rebuild failed {} times: {}",
                        self.consecutive_failures, e
                    );
                    eprintln!("Will continue retrying periodically...");
                } else {
                    eprintln!(
                        "Audio rebuild failed ({}/{}): {}",
                        self.consecutive_failures,
                        Self::MAX_CONSECUTIVE_FAILURES,
                        e
                    );
                }
                self.stream_error.store(true, Ordering::SeqCst);
            }
        }
    }

    pub fn force_rebuild(&mut self) {
        println!("Force audio rebuild requested");
        self.consecutive_failures = 0;
        self.stream_error.store(true, Ordering::SeqCst);
        self.last_rebuild_attempt = Instant::now() - Self::MIN_REBUILD_INTERVAL;
    }

    pub fn is_active(&self) -> bool {
        self.stream.is_some() && !self.stream_error.load(Ordering::SeqCst)
    }

    fn build_new_stream(
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
    ) -> Result<Stream, AudioError> {
        let available_hosts = with_stderr_suppressed(|| cpal::available_hosts());
        if available_hosts.is_empty() {
            return Err(AudioError::NoHostsAvailable);
        }

        let attempts = Self::build_all_attempts(&available_hosts);

        for attempt in &attempts {
            let result = with_stderr_suppressed(|| {
                Self::try_build_stream(
                    attempt,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                )
            });

            match result {
                Ok(stream) => {
                    println!(
                        "Audio initialized: {}, buffer size: {:?}",
                        attempt.description,
                        stream.buffer_size()
                    );
                    return Ok(stream);
                }
                Err(_) => continue,
            }
        }

        // ...
        Err(AudioError::ExhaustedAllOptions)
    }

    fn try_build_stream(
        attempt: &InitAttempt,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
    ) -> Result<Stream, AudioError> {
        let host = match attempt.host_id {
            Some(id) => cpal::host_from_id(id)
                .map_err(|e| AudioError::StreamBuildFailed(format!("Host error: {}", e)))?,
            None => cpal::default_host(),
        };

        let device = Self::get_device_by_strategy(&host, &attempt.strategy)?;
        let config = Self::get_working_config(&device)?;
        let sample_rate = config.sample_rate();

        {
            let mut s = state.lock().unwrap();
            s.sample_rate = sample_rate;
        }

        let stream = Self::build_stream_with_fallbacks(
            &device,
            &config,
            state,
            stream_error,
            sample_counter,
        )?;

        stream
            .play()
            .map_err(|e| AudioError::StreamPlayFailed(format!("{}", e)))?;

        Ok(stream)
    }

    pub fn new_robust() -> Result<Self, AudioError> {
        let available_hosts = cpal::available_hosts();
        if available_hosts.is_empty() {
            return Err(AudioError::NoHostsAvailable);
        }

        let state = Arc::new(Mutex::new(AudioState::new(48000)));
        let stream_error = Arc::new(AtomicBool::new(false));
        let sample_counter = Arc::new(AtomicU64::new(0));

        let stream = Self::build_new_stream(
            Arc::clone(&state),
            Arc::clone(&stream_error),
            Arc::clone(&sample_counter),
        )?;
        //println!("Audio Buffer Size: {:?}", stream.buffer_size());
        // if let Ok(state) = state.lock().as_mut().map(|s| s) {
        //     init_cars_audio(state);
        // }
        let now = Instant::now();
        Ok(Self {
            stream: Some(stream),
            state,
            stream_error,
            sample_counter,
            last_sample_count: 0,
            last_health_check: now,
            rebuild_count: 0,
            consecutive_failures: 0,
            last_rebuild_attempt: now,
        })
    }

    fn build_all_attempts(available_hosts: &[HostId]) -> Vec<InitAttempt> {
        let mut attempts = Vec::new();

        attempts.push(InitAttempt {
            host_id: None,
            strategy: DeviceStrategy::Default,
            description: "Default Host → Default Device".into(),
        });

        let platform_names = Self::get_platform_preferred_names();
        for name in platform_names {
            attempts.push(InitAttempt {
                host_id: None,
                strategy: DeviceStrategy::NameContains(name.clone()),
                description: format!("Default Host → Name contains '{}'", name),
            });
        }

        attempts.push(InitAttempt {
            host_id: None,
            strategy: DeviceStrategy::First,
            description: "Default Host → First Device".into(),
        });
        attempts.push(InitAttempt {
            host_id: None,
            strategy: DeviceStrategy::Last,
            description: "Default Host → Last Device".into(),
        });

        for &host_id in available_hosts {
            attempts.push(InitAttempt {
                host_id: Some(host_id),
                strategy: DeviceStrategy::Default,
                description: format!("{:?} → Default Device", host_id),
            });
        }

        for &host_id in available_hosts {
            attempts.push(InitAttempt {
                host_id: Some(host_id),
                strategy: DeviceStrategy::First,
                description: format!("{:?} → First Device", host_id),
            });
        }

        let common_names = vec![
            "pipewire",
            "PipeWire",
            "pulse",
            "PulseAudio",
            "Pulse",
            "jack",
            "JACK",
            "alsa",
            "ALSA",
            "default",
            "hw:",
            "sysdefault",
            "plughw",
            "dmix",
            "Speakers",
            "speakers",
            "Headphones",
            "headphones",
            "WASAPI",
            "Realtek",
            "realtek",
            "High Definition Audio",
            "Digital Audio",
            "HDMI",
            "DisplayPort",
            "Built-in",
            "built-in",
            "MacBook",
            "External",
            "AirPods",
            "airpods",
            "USB",
            "usb",
            "Audio",
            "audio",
            "Output",
            "output",
            "DAC",
            "dac",
            "Sound",
            "sound",
        ];

        for name in common_names {
            for &host_id in available_hosts {
                attempts.push(InitAttempt {
                    host_id: Some(host_id),
                    strategy: DeviceStrategy::NameContains(name.to_string()),
                    description: format!("{:?} → Name contains '{}'", host_id, name),
                });
            }
        }

        for &host_id in available_hosts {
            for idx in 0..10 {
                attempts.push(InitAttempt {
                    host_id: Some(host_id),
                    strategy: DeviceStrategy::ByIndex(idx),
                    description: format!("{:?} → Device Index {}", host_id, idx),
                });
            }
        }

        let mut seen = std::collections::HashSet::new();
        attempts.retain(|a| {
            let key = format!("{:?}-{:?}", a.host_id, a.strategy);
            seen.insert(key)
        });

        attempts
    }

    fn get_platform_preferred_names() -> Vec<String> {
        let mut names = Vec::new();

        #[cfg(target_os = "linux")]
        {
            names.extend(vec![
                "pipewire".into(),
                "PipeWire".into(),
                "pulse".into(),
                "PulseAudio".into(),
                "jack".into(),
                "JACK".into(),
                "alsa".into(),
                "ALSA".into(),
                "default".into(),
                "sysdefault".into(),
            ]);
        }

        #[cfg(target_os = "windows")]
        {
            names.extend(vec![
                "Speakers".into(),
                "speakers".into(),
                "Headphones".into(),
                "headphones".into(),
                "Realtek".into(),
                "High Definition".into(),
                "WASAPI".into(),
            ]);
        }

        #[cfg(target_os = "macos")]
        {
            names.extend(vec![
                "Built-in".into(),
                "built-in".into(),
                "MacBook".into(),
                "External".into(),
                "Output".into(),
            ]);
        }

        names.extend(vec![
            "default".into(),
            "Default".into(),
            "output".into(),
            "Output".into(),
        ]);

        names
    }

    fn get_device_by_strategy(
        host: &Host,
        strategy: &DeviceStrategy,
    ) -> Result<Device, AudioError> {
        match strategy {
            DeviceStrategy::Default => host
                .default_output_device()
                .ok_or(AudioError::NoDevicesFound),
            DeviceStrategy::NameContains(pattern) => {
                let pattern_lower = pattern.to_lowercase();
                host.output_devices()
                    .map_err(|_| AudioError::NoDevicesFound)?
                    .find(|d| {
                        d.description()
                            .ok()
                            .map(|d| d.name().to_string())
                            .unwrap_or("???".into())
                            .to_lowercase()
                            .contains(&pattern_lower)
                    })
                    .ok_or(AudioError::NoDevicesFound)
            }
            DeviceStrategy::NameExact(name) => host
                .output_devices()
                .map_err(|_| AudioError::NoDevicesFound)?
                .find(|d| {
                    d.description()
                        .ok()
                        .map(|d| d.name().to_string())
                        .unwrap_or("???".into())
                        == *name
                })
                .ok_or(AudioError::NoDevicesFound),
            DeviceStrategy::ByIndex(idx) => host
                .output_devices()
                .map_err(|_| AudioError::NoDevicesFound)?
                .nth(*idx)
                .ok_or(AudioError::NoDevicesFound),
            DeviceStrategy::First => host
                .output_devices()
                .map_err(|_| AudioError::NoDevicesFound)?
                .next()
                .ok_or(AudioError::NoDevicesFound),
            DeviceStrategy::Last => host
                .output_devices()
                .map_err(|_| AudioError::NoDevicesFound)?
                .last()
                .ok_or(AudioError::NoDevicesFound),
        }
    }

    fn get_working_config(device: &Device) -> Result<SupportedStreamConfig, AudioError> {
        if let Ok(config) = device.default_output_config() {
            return Ok(config);
        }

        let configs: Vec<_> = device
            .supported_output_configs()
            .map_err(|_| AudioError::NoConfigsSupported)?
            .collect();

        if configs.is_empty() {
            return Err(AudioError::NoConfigsSupported);
        }

        let preferred_rates = [
            48000u32, 48000, 96000, 22050, 88200, 192000, 16000, 8000, 32000,
        ];

        for &rate in &preferred_rates {
            for config in &configs {
                if config.channels() == 2
                    && config.min_sample_rate() <= rate
                    && config.max_sample_rate() >= rate
                {
                    return Ok(config.clone().with_sample_rate(rate));
                }
            }
        }

        for &rate in &preferred_rates {
            for config in &configs {
                if config.min_sample_rate() <= rate && config.max_sample_rate() >= rate {
                    return Ok(config.clone().with_sample_rate(rate));
                }
            }
        }

        Ok(configs[0].clone().with_max_sample_rate())
    }

    fn build_stream_with_fallbacks(
        device: &Device,
        config: &SupportedStreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
    ) -> Result<Stream, AudioError> {
        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.clone().into();
        let channels = stream_config.channels as usize;
        let sample_rate = stream_config.sample_rate;

        let formats_to_try = [
            sample_format,
            SampleFormat::F32,
            SampleFormat::I16,
            SampleFormat::I32,
            SampleFormat::U16,
            SampleFormat::F64,
        ];

        for format in formats_to_try {
            let result = match format {
                SampleFormat::F32 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| x,
                ),

                SampleFormat::F64 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| x as f64,
                ),

                SampleFormat::I8 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| (x * i8::MAX as f32) as i8,
                ),

                SampleFormat::I16 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| (x * i16::MAX as f32) as i16,
                ),

                SampleFormat::I32 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| (x * i32::MAX as f32) as i32,
                ),

                SampleFormat::I64 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| (x as f64 * i64::MAX as f64) as i64,
                ),

                SampleFormat::U8 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| ((x * 0.5 + 0.5) * u8::MAX as f32) as u8,
                ),

                SampleFormat::U16 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| ((x * 0.5 + 0.5) * u16::MAX as f32) as u16,
                ),

                SampleFormat::U32 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| ((x as f64 * 0.5 + 0.5) * u32::MAX as f64) as u32,
                ),

                SampleFormat::U64 => Self::build_stream(
                    device,
                    stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                    |x| ((x as f64 * 0.5 + 0.5) * u64::MAX as f64) as u64,
                ),

                _ => continue,
            };

            if result.is_ok() {
                return result;
            }
        }

        Err(AudioError::StreamBuildFailed("All formats failed".into()))
    }

    fn build_stream<T>(
        device: &Device,
        config: StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
        convert: fn(f32) -> T,
    ) -> Result<Stream, AudioError>
    where
        T: cpal::SizedSample + Send + 'static,
    {
        let error_flag = Arc::clone(&stream_error);

        device
            .build_output_stream(
                config,
                move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];

                    fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);

                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = convert(*sample);
                    }

                    sample_counter.fetch_add(data.len() as u64, Ordering::SeqCst);
                },
                move |err| {
                    eprintln!("Audio stream error: {}", err);
                    error_flag.store(true, Ordering::SeqCst);
                },
                None,
            )
            .map_err(|e| AudioError::StreamBuildFailed(e.to_string()))
    }

    fn nuclear_fallback(
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
    ) -> Result<Stream, AudioError> {
        for host_id in cpal::available_hosts() {
            let Ok(host) = cpal::host_from_id(host_id) else {
                continue;
            };
            let Ok(devices) = host.output_devices() else {
                continue;
            };

            for device in devices {
                let Ok(configs) = device.supported_output_configs() else {
                    continue;
                };

                for config_range in configs {
                    let config = config_range.with_max_sample_rate();
                    let stream_config: StreamConfig = config.clone().into();
                    let sample_rate = stream_config.sample_rate;

                    {
                        let mut s = state.lock().unwrap();
                        s.sample_rate = sample_rate;
                    }

                    if let Ok(stream) = Self::build_stream_with_fallbacks(
                        &device,
                        &config,
                        Arc::clone(&state),
                        Arc::clone(&stream_error),
                        Arc::clone(&sample_counter),
                    ) {
                        if stream.play().is_ok() {
                            let name = device
                                .description()
                                .ok()
                                .map(|d| d.name().to_string())
                                .unwrap_or("???".into());
                            println!("Audio recovered via nuclear fallback: {}", name);
                            return Ok(stream);
                        }
                    }
                }
            }
        }

        Err(AudioError::ExhaustedAllOptions)
    }
}

fn fill_audio_buffer(
    data: &mut [f32],
    state_arc: &Arc<Mutex<AudioState>>,
    channels: usize,
    _sample_rate: SampleRate,
) {
    let Ok(mut state) = state_arc.try_lock() else {
        data.fill(0.0);
        return;
    };

    let yaw = state.listener_yaw;
    let pitch = state.listener_pitch;

    let forward = glam::Vec3::new(
        yaw.cos() * pitch.cos(),
        pitch.sin(),
        yaw.sin() * pitch.cos(),
    )
    .normalize();

    let mut right_vec = glam::Vec3::Y.cross(forward);
    if right_vec.length_squared() < 1.0e-6 {
        right_vec = glam::Vec3::X.cross(forward);
    }
    right_vec = right_vec.normalize();

    let listener_pos = state.listener_pos;
    let listener_velocity = state.listener_velocity;

    let mut effective_rpm = [0.0f32; MAX_CARS_AUDIO];
    let mut throttle = [0.0f32; MAX_CARS_AUDIO];
    let mut gain = [0.0f32; MAX_CARS_AUDIO];
    let mut pan_l = [0.0f32; MAX_CARS_AUDIO];
    let mut pan_r = [0.0f32; MAX_CARS_AUDIO];

    let car_count = state.cars.len().min(MAX_CARS_AUDIO);
    const SPEED_OF_SOUND: f32 = 343.0;

    // Process only closest 10 cars (position and every field stays the same, we do not need to recompute all this crap for all 512 samples or so!)
    for idx in 0..car_count {
        let car = &state.cars[idx];
        let to_car = listener_pos.direction_to(car.position);
        let distance = to_car.length().max(0.01);
        let dir = to_car / distance;

        let attenuation = 1.0 / (1.0 + 0.015 * distance * distance);
        let pan = dir.dot(right_vec).clamp(-1.0, 1.0);

        pan_l[idx] = ((1.0 - pan) * 0.5).sqrt();
        pan_r[idx] = ((1.0 + pan) * 0.5).sqrt();

        let rel_radial = car.velocity.dot(dir) - listener_velocity.dot(dir);
        let doppler = (SPEED_OF_SOUND / (SPEED_OF_SOUND + rel_radial)).clamp(0.5, 2.0);

        effective_rpm[idx] = car.rpm * doppler;
        throttle[idx] = car.throttle;
        gain[idx] = attenuation * 0.7;
    }

    let cam_height_above_ground = state.camera_height_above_ground;

    for frame in data.chunks_mut(channels) {
        let mut left = 0.0f32;
        let mut right = 0.0f32;

        let (wl, wr) = state.wind_synth.generate(cam_height_above_ground);

        left += wl * 0.6;
        right += wr * 0.6;

        for idx in 0..car_count {
            let sample = state.synths[idx].generate(effective_rpm[idx], throttle[idx]) * gain[idx];
            left += sample * pan_l[idx];
            right += sample * pan_r[idx];
        }

        if channels >= 2 {
            frame[0] = left.clamp(-1.0, 1.0);
            frame[1] = right.clamp(-1.0, 1.0);
        } else if channels == 1 {
            frame[0] = ((left + right) * 0.5).clamp(-1.0, 1.0);
        }
    }
}
#[derive(Copy, Clone)]
struct CarSynth {
    harmonic_phase: [f32; 4],
    pulse_phase: f32,
    noise_phase: f32,
    sample_rate: f32,
    last_rpm: f32,
    throttle: f32,
    filter_state: [f32; 2],
}

impl CarSynth {
    fn new(sample_rate: f32) -> Self {
        Self {
            harmonic_phase: [0.0; 4],
            pulse_phase: 0.0,
            noise_phase: 0.0,
            sample_rate,
            last_rpm: 0.0,
            throttle: 0.0,
            filter_state: [0.0; 2],
        }
    }

    #[inline]
    fn generate(&mut self, rpm: f32, throttle: f32) -> f32 {
        self.last_rpm = self.last_rpm * 0.92 + rpm.max(0.0) * 0.08;
        self.throttle = throttle.clamp(0.0, 1.0);

        let rpm = self.last_rpm.max(0.0);
        let rpm_norm = (rpm / 7000.0).clamp(0.0, 1.0);
        let crank_hz = rpm / 60.0;
        let firing_rate = crank_hz * 2.0;

        let harmonic = self.osc(0, crank_hz * 1.0) * (0.22 + rpm_norm * 0.30)
            + self.osc(1, crank_hz * 2.0) * (0.18 + rpm_norm * 0.22)
            + self.osc(2, crank_hz * 3.0) * (0.12 + rpm_norm * 0.16)
            + self.osc(3, crank_hz * 4.0) * (0.08 + rpm_norm * 0.10);

        let pulse = self.combustion_pulse(firing_rate);

        let noise = self.filtered_noise(0, 0.96) * 0.03 + self.filtered_noise(1, 0.85) * 0.02;

        let mut engine = harmonic * 0.8 + pulse * (0.25 + self.throttle * 0.35) + noise;

        engine = engine / (1.0 + engine.abs() * 0.6);
        engine *= (0.15 + self.throttle * 0.85).sqrt();
        engine *= 0.4 + rpm_norm * 0.9;

        engine.clamp(-1.0, 1.0)
    }

    #[inline]
    fn osc(&mut self, idx: usize, freq: f32) -> f32 {
        let phase = &mut self.harmonic_phase[idx];
        *phase += freq / self.sample_rate;
        *phase = phase.fract();
        (*phase * std::f32::consts::TAU).sin()
    }

    #[inline]
    fn combustion_pulse(&mut self, freq: f32) -> f32 {
        self.pulse_phase += freq / self.sample_rate;
        self.pulse_phase = self.pulse_phase.fract();

        let x = (self.pulse_phase * std::f32::consts::TAU).sin();
        x.max(0.0).powi(4)
    }

    #[inline]
    fn filtered_noise(&mut self, band: usize, alpha: f32) -> f32 {
        self.noise_phase += 1.0 / self.sample_rate;

        let t = self.noise_phase;
        let x = (t * 12.9898 + band as f32 * 78.233).sin() * 43758.5453;
        let noise = (x - x.floor()) * 2.0 - 1.0;

        self.filter_state[band] = self.filter_state[band] * alpha + noise * (1.0 - alpha);
        self.filter_state[band]
    }
}

use std::f32::consts::PI;

struct WindSynth {
    sample_rate: f32,
    rng_l: u32,
    rng_r: u32,
    rng_w1: u32,
    rng_w2: u32,
    rng_w3: u32,
    walk1: f32,
    walk2: f32,
    walk3: f32,
    k1: f32,
    k2: f32,
    k3: f32,
    lfo_phase: f32,
    lfo_inc: f32,
    svf_low_l: f32,
    svf_band_l: f32,
    svf_low_r: f32,
    svf_band_r: f32,
    prev_l: f32,
    prev_r: f32,
}

impl WindSynth {
    fn new(sample_rate: f32) -> Self {
        let k1 = 1.0 - (-2.0 * PI * 0.025 / sample_rate).exp();
        let k2 = 1.0 - (-2.0 * PI * 0.18 / sample_rate).exp();
        let k3 = 1.0 - (-2.0 * PI * 1.1 / sample_rate).exp();
        Self {
            sample_rate,
            rng_l: 0x1234_5678,
            rng_r: 0x8765_4321,
            rng_w1: 0xABCD_1234,
            rng_w2: 0xDEAD_BEEF,
            rng_w3: 0xF00D_CAFE,
            walk1: 0.0,
            walk2: 0.0,
            walk3: 0.0,
            k1,
            k2,
            k3,
            lfo_phase: 0.0,
            lfo_inc: 2.0 * PI * 0.032 / sample_rate,
            svf_low_l: 0.0,
            svf_band_l: 0.0,
            svf_low_r: 0.0,
            svf_band_r: 0.0,
            prev_l: 0.0,
            prev_r: 0.0,
        }
    }

    #[inline]
    fn next_u32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    #[inline]
    fn white(state: &mut u32) -> f32 {
        (Self::next_u32(state) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    #[inline]
    fn svf(input: f32, f: f32, q: f32, low: &mut f32, band: &mut f32) -> (f32, f32, f32) {
        *low += f * *band;
        let high = input - *low - q * *band;
        *band += f * high;
        (*low, *band, high)
    }

    #[inline]
    fn generate(&mut self, height: f32) -> (f32, f32) {
        let h = (height / 1000.0).clamp(0.0, 1.0);
        let intensity = h * h;

        let n1 = Self::white(&mut self.rng_w1);
        self.walk1 += (n1 - self.walk1) * self.k1;
        let n2 = Self::white(&mut self.rng_w2);
        self.walk2 += (n2 - self.walk2) * self.k2;
        let n3 = Self::white(&mut self.rng_w3);
        self.walk3 += (n3 - self.walk3) * self.k3;

        self.lfo_phase += self.lfo_inc;
        if self.lfo_phase > 2.0 * PI {
            self.lfo_phase -= 2.0 * PI;
        }
        let swell = self.lfo_phase.sin() * 0.5 + 0.5;

        let raw = self.walk1 * 0.5 + self.walk2 * 0.32 + self.walk3 * 0.18;
        let g = ((raw * 0.5 + 0.5) * 0.82 + swell * 0.18).clamp(0.0, 1.0);
        let gust = g.powf(1.7);

        let cutoff_hz = (140.0 + intensity * 700.0 + gust * (500.0 + intensity * 1400.0))
            .clamp(60.0, self.sample_rate * 0.16);
        let f = 2.0 * (PI * cutoff_hz / self.sample_rate).sin();
        let q = (1.9 - gust * 1.1).clamp(0.55, 1.9);

        let nl = Self::white(&mut self.rng_l);
        let nr = Self::white(&mut self.rng_r);

        let (low_l, band_l, high_l) =
            Self::svf(nl, f, q, &mut self.svf_low_l, &mut self.svf_band_l);
        let (low_r, band_r, high_r) =
            Self::svf(nr, f, q, &mut self.svf_low_r, &mut self.svf_band_r);

        let amp = intensity * (0.12 + gust * 0.9);

        let sig_l = (low_l * 0.85 + band_l * 1.35 + high_l * 0.35) * amp;
        let sig_r = (low_r * 0.85 + band_r * 1.35 + high_r * 0.35) * amp;

        self.prev_l += (sig_l - self.prev_l) * 0.35;
        self.prev_r += (sig_r - self.prev_r) * 0.35;

        (self.prev_l.tanh() * 0.95, self.prev_r.tanh() * 0.95)
    }
}
