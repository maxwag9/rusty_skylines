use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::resources::Resources;
use crate::world::sound::car_sounds::{CarAudioState, collect_car_audio};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    Device, Host, HostId, SampleFormat, SampleRate, Stream, StreamConfig, SupportedStreamConfig,
};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod car_sounds;

pub fn run_sounds(resources: &mut Resources) {
    let sounds = &mut resources.sounds;

    // Health check and rebuild if needed
    sounds.check_and_rebuild();

    let mut state = sounds.state.lock().unwrap();

    let camera = &resources.world.world_state.camera;
    let terrain = &resources.world.terrain;
    let car_storage = resources.world.cars.car_storage();

    let current_eye = camera.eye_world();
    let prev_eye = camera.prev_eye_world();

    let vel = current_eye.delta_to(prev_eye);

    state.listener_pos = current_eye;
    state.listener_velocity = vel;
    state.listener_yaw = camera.yaw;
    state.listener_pitch = camera.pitch;
    state.cars.clear();

    collect_car_audio(&mut state, camera, terrain, car_storage);
}

#[derive(Default)]
pub struct AudioState {
    pub cars: Vec<CarAudioState>,
    pub listener_pos: WorldPos,
    pub listener_velocity: glam::Vec3,
    pub listener_yaw: f32,
    pub listener_pitch: f32,
    pub chunk_size: ChunkSize,
    pub sample_rate: SampleRate,
}

impl AudioState {
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
            state: Arc::new(Mutex::new(AudioState::default())),
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
        let available_hosts = cpal::available_hosts();
        if available_hosts.is_empty() {
            return Err(AudioError::NoHostsAvailable);
        }

        let attempts = Self::build_all_attempts(&available_hosts);

        for attempt in &attempts {
            match Self::try_build_stream(
                attempt,
                Arc::clone(&state),
                Arc::clone(&stream_error),
                Arc::clone(&sample_counter),
            ) {
                Ok(stream) => {
                    println!("Audio initialized: {}", attempt.description);
                    return Ok(stream);
                }
                Err(_) => continue,
            }
        }

        // Nuclear fallback with retries
        for retry in 0..3 {
            if retry > 0 {
                std::thread::sleep(Duration::from_millis(200));
            }
            if let Ok(stream) = Self::nuclear_fallback(
                Arc::clone(&state),
                Arc::clone(&stream_error),
                Arc::clone(&sample_counter),
            ) {
                return Ok(stream);
            }
        }

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

        let state = Arc::new(Mutex::new(AudioState::default()));
        let stream_error = Arc::new(AtomicBool::new(false));
        let sample_counter = Arc::new(AtomicU64::new(0));

        let stream = Self::build_new_stream(
            Arc::clone(&state),
            Arc::clone(&stream_error),
            Arc::clone(&sample_counter),
        )?;

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
            44100u32, 48000, 96000, 22050, 88200, 192000, 16000, 8000, 32000,
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
                SampleFormat::F32 => Self::build_stream_f32(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::I16 => Self::build_stream_i16(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::U16 => Self::build_stream_u16(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::I8 => Self::build_stream_i8(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::I32 => Self::build_stream_i32(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::I64 => Self::build_stream_i64(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::U8 => Self::build_stream_u8(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::U32 => Self::build_stream_u32(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::U64 => Self::build_stream_u64(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                SampleFormat::F64 => Self::build_stream_f64(
                    device,
                    &stream_config,
                    Arc::clone(&state),
                    Arc::clone(&stream_error),
                    Arc::clone(&sample_counter),
                    channels,
                    sample_rate,
                ),
                _ => continue,
            };

            if result.is_ok() {
                return result;
            }
        }

        Err(AudioError::StreamBuildFailed("All formats failed".into()))
    }

    fn build_stream_f32(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    Self::fill_audio_buffer(data, &state, channels, sample_rate);
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

    fn build_stream_i16(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = (*sample * i16::MAX as f32) as i16;
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

    fn build_stream_i32(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [i32], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = (*sample * i32::MAX as f32) as i32;
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

    fn build_stream_u16(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = ((*sample * 0.5 + 0.5) * u16::MAX as f32) as u16;
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

    fn build_stream_i8(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [i8], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = (*sample * i8::MAX as f32) as i8;
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

    fn build_stream_u8(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [u8], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = ((*sample * 0.5 + 0.5) * u8::MAX as f32) as u8;
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

    fn build_stream_i64(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [i64], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = (*sample as f64 * i64::MAX as f64) as i64;
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

    fn build_stream_u32(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [u32], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = ((*sample as f64 * 0.5 + 0.5) * u32::MAX as f64) as u32;
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

    fn build_stream_u64(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [u64], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = ((*sample as f64 * 0.5 + 0.5) * u64::MAX as f64) as u64;
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

    fn build_stream_f64(
        device: &Device,
        config: &StreamConfig,
        state: Arc<Mutex<AudioState>>,
        stream_error: Arc<AtomicBool>,
        sample_counter: Arc<AtomicU64>,
        channels: usize,
        sample_rate: SampleRate,
    ) -> Result<Stream, AudioError> {
        let error_flag = Arc::clone(&stream_error);
        device
            .build_output_stream(
                config,
                move |data: &mut [f64], _: &cpal::OutputCallbackInfo| {
                    let mut float_buf = vec![0.0f32; data.len()];
                    Self::fill_audio_buffer(&mut float_buf, &state, channels, sample_rate);
                    for (out, sample) in data.iter_mut().zip(float_buf.iter()) {
                        *out = *sample as f64;
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

    fn fill_audio_buffer(
        data: &mut [f32],
        state_arc: &Arc<Mutex<AudioState>>,
        channels: usize,
        sample_rate: SampleRate,
    ) {
        const SPEED_OF_SOUND: f32 = 343.0;

        let Ok(mut state) = state_arc.try_lock() else {
            // Can't get lock, output silence
            data.fill(0.0);
            return;
        };

        for frame in data.chunks_mut(channels) {
            let mut left = 0.0;
            let mut right = 0.0;

            let yaw = state.listener_yaw;
            let pitch = state.listener_pitch;

            let forward = glam::Vec3::new(
                yaw.cos() * pitch.cos(),
                pitch.sin(),
                yaw.sin() * pitch.cos(),
            )
            .normalize();

            let right_vec = forward.cross(glam::Vec3::Y).normalize();
            let listener_pos = state.listener_pos;
            let listener_velocity = state.listener_velocity;
            let chunk_size = state.chunk_size;

            for car in &mut state.cars {
                let to_car = car.position.delta_to(listener_pos);
                let distance = to_car.length().max(0.01);
                let dir = to_car.normalize();

                let attenuation = 1.0 / (1.0 + 0.015 * distance * distance);

                let pan = dir.dot(right_vec).clamp(-1.0, 1.0);
                let pan_l = ((1.0 - pan) * 0.5).sqrt();
                let pan_r = ((1.0 + pan) * 0.5).sqrt();

                let rel_vel = car.velocity.dot(dir) - listener_velocity.dot(dir);
                let doppler = (SPEED_OF_SOUND / (SPEED_OF_SOUND - rel_vel)).clamp(0.5, 2.0);

                let cylinders = 4.0;
                let engine_freq = car.rpm / 60.0 * (cylinders * 0.5) * doppler;

                car.phase += engine_freq / sample_rate as f32;
                if car.phase >= 1.0 {
                    car.phase -= 1.0;
                }

                let t = car.phase * 2.0 * PI;

                let fundamental = t.sin();
                let second = (2.0 * t).sin() * 0.5;
                let third = (3.0 * t).sin() * 0.25;
                let rough = (t * 13.0).sin() * 0.05;

                let rpm_norm = (car.rpm / 7000.0).clamp(0.1, 1.0);

                let mut engine =
                    (fundamental * 0.5 + second * 0.5 + third * 0.3 + rough) * rpm_norm;

                engine = engine.tanh();

                let shadow = 1.0 - pan.abs() * 0.4;
                engine *= shadow;

                let sample = engine * attenuation * 0.7;

                left += sample * pan_l;
                right += sample * pan_r;
            }

            if channels >= 2 {
                frame[0] = left;
                frame[1] = right;
            } else {
                frame[0] = (left + right) * 0.5;
            }
        }
    }
}
