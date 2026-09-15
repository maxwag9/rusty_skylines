use serde::Deserialize;
use std::time::Duration;

#[derive(Default, Debug, Clone, Copy, Deserialize)]
pub enum Waveform {
    #[default]
    Sine,
    Triangle,
    Square,
    Saw,
    Noise,
    Pulse,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SfxLayer {
    pub waveform: Waveform,

    // Pitch
    pub start_freq: f32,
    pub end_freq: f32,
    #[serde(default = "one")]
    pub glide_curve: f32,

    // FM
    #[serde(default)]
    pub fm_ratio: f32,
    #[serde(default)]
    pub fm_amount: f32,

    // Pulse width (used by Square + Pulse)
    #[serde(default = "half")]
    pub pulse_width: f32, // 0.01 … 0.99

    // Temporal
    #[serde(default)]
    pub delay: f32,
    #[serde(default = "zeropointtwo")]
    pub duration: f32,

    // Per-layer ADSR
    #[serde(default = "tiny")]
    pub attack: f32,
    #[serde(default = "zeropointone")]
    pub decay: f32,
    #[serde(default)]
    pub sustain: f32,
    #[serde(default = "zeropointone")]
    pub release: f32,

    // Filtering (0 = disabled / bypass)
    #[serde(default)]
    pub lp_freq: f32, // low-pass cutoff in Hz (0 = off)
    #[serde(default)]
    pub hp_freq: f32, // high-pass cutoff in Hz (0 = off)

    // Mixing
    #[serde(default = "one")]
    pub gain: f32,
    #[serde(default)]
    pub pan: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SfxConfig {
    pub name: String,
    pub layers: Vec<SfxLayer>,

    // Global master
    #[serde(default = "one")]
    pub gain: f32,
    #[serde(default)]
    pub pan: f32,

    // NEW global settings
    #[serde(default = "one")]
    pub pitch_scale: f32, // multiplies every frequency
    #[serde(default = "one")]
    pub time_scale: f32, // speeds up / slows down the whole SFX

    #[serde(default = "zeropointone")]
    pub deduplication_time: f32,
}

fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn tiny() -> f32 {
    0.001
}
fn zeropointone() -> f32 {
    0.1
}
fn zeropointtwo() -> f32 {
    0.2
}
// impl SfxKind {
//     pub fn config(self) -> SfxConfig {
//         match self {
//             SfxKind::ButtonPress => SfxConfig {
//                 layers: vec![SfxLayer {
//                     waveform: Waveform::Sine,
//                     start_freq: 640.0,
//                     end_freq: 320.0,
//                     fm_ratio: 1.5,
//                     fm_amount: 0.15,
//                     gain: 0.5,
//                 }],
//                 duration: Duration::from_millis(40),
//                 attack: 0.001,
//                 decay: 0.02,
//                 sustain: 0.0,
//                 release: 0.01,
//                 glide_curve: 1.0,
//                 gain: 0.20,
//                 pan: 0.0,
//             },
//             SfxKind::ButtonHover => SfxConfig {
//                 layers: vec![SfxLayer {
//                     waveform: Waveform::Sine,
//                     start_freq: 1000.0,
//                     end_freq: 1200.0,
//                     fm_ratio: 1.0,
//                     fm_amount: 0.3,
//                     gain: 0.5
//                 }],
//                 duration: Duration::from_millis(10),
//                 attack: 0.001,
//                 decay: 0.002,
//                 sustain: 0.0,
//                 release: 0.05,
//                 glide_curve: 1.0,
//                 gain: 0.18,
//                 pan: 0.0
//             },
//             //             },
//             //             SfxKind::ButtonHover => SfxConfig { // TODO: Perfect for Key presses, sounds like apple iphone keyboard
//             //                 layers: vec![SfxLayer {
//             //                     waveform: Waveform::Sine,
//             //                     start_freq: 1000.0,
//             //                     end_freq: 1200.0,
//             //                     fm_ratio: 1.0,
//             //                     fm_amount: 0.6,
//             //                     gain: 0.5
//             //                 }],
//             //                 duration: Duration::from_millis(50),
//             //                 attack: 0.001,
//             //                 decay: 0.002,
//             //                 sustain: 0.0,
//             //                 release: 0.05,
//             //                 glide_curve: 1.0,
//             //                 gain: 0.18,
//             //                 pan: 0.0
//             //             },
//             SfxKind::RoadPlaced => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Noise,
//                         start_freq: 0.0,
//                         end_freq: 0.0,
//                         fm_ratio: 0.0,
//                         fm_amount: 0.0,
//                         gain: 0.5,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 180.0,
//                         end_freq: 90.0,
//                         fm_ratio: 1.0,
//                         fm_amount: 0.0,
//                         gain: 0.8,
//                     },
//                 ],
//                 duration: Duration::from_millis(90),
//                 attack: 0.002,
//                 decay: 0.08,
//                 sustain: 0.0,
//                 release: 0.02,
//                 glide_curve: 2.0,
//                 gain: 0.5,
//                 pan: 0.0,
//             },
//             SfxKind::RoadRemoved => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Noise,
//                         start_freq: 0.0,
//                         end_freq: 0.0,
//                         fm_ratio: 0.0,
//                         fm_amount: 0.0,
//                         gain: 0.6,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Triangle,
//                         start_freq: 140.0,
//                         end_freq: 60.0,
//                         fm_ratio: 1.0,
//                         fm_amount: 0.0,
//                         gain: 0.5,
//                     },
//                 ],
//                 duration: Duration::from_millis(110),
//                 attack: 0.002,
//                 decay: 0.09,
//                 sustain: 0.0,
//                 release: 0.03,
//                 glide_curve: 1.5,
//                 gain: 0.45,
//                 pan: 0.0,
//             },
//             SfxKind::BuildingPlaced => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 260.0,
//                         end_freq: 520.0,
//                         fm_ratio: 2.0,
//                         fm_amount: 0.4,
//                         gain: 0.7,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Triangle,
//                         start_freq: 520.0,
//                         end_freq: 780.0,
//                         fm_ratio: 1.0,
//                         fm_amount: 0.0,
//                         gain: 0.3,
//                     },
//                 ],
//                 duration: Duration::from_millis(160),
//                 attack: 0.005,
//                 decay: 0.1,
//                 sustain: 0.2,
//                 release: 0.06,
//                 glide_curve: 0.6,
//                 gain: 0.5,
//                 pan: 0.0,
//             },
//             SfxKind::BuildingUpgraded => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 660.0,
//                         end_freq: 1320.0,
//                         fm_ratio: 3.0,
//                         fm_amount: 0.6,
//                         gain: 0.6,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 990.0,
//                         end_freq: 1980.0,
//                         fm_ratio: 2.0,
//                         fm_amount: 0.3,
//                         gain: 0.3,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Triangle,
//                         start_freq: 1320.0,
//                         end_freq: 2640.0,
//                         fm_ratio: 1.0,
//                         fm_amount: 0.0,
//                         gain: 0.15,
//                     },
//                 ],
//                 duration: Duration::from_millis(280),
//                 attack: 0.003,
//                 decay: 0.05,
//                 sustain: 0.6,
//                 release: 0.22,
//                 glide_curve: 0.4,
//                 gain: 0.55,
//                 pan: 0.0,
//             },
//             SfxKind::BuildingDemolished => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Noise,
//                         start_freq: 0.0,
//                         end_freq: 0.0,
//                         fm_ratio: 0.0,
//                         fm_amount: 0.0,
//                         gain: 0.7,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 200.0,
//                         end_freq: 50.0,
//                         fm_ratio: 1.0,
//                         fm_amount: 0.0,
//                         gain: 0.6,
//                     },
//                 ],
//                 duration: Duration::from_millis(220),
//                 attack: 0.002,
//                 decay: 0.15,
//                 sustain: 0.0,
//                 release: 0.05,
//                 glide_curve: 1.8,
//                 gain: 0.55,
//                 pan: 0.0,
//             },
//             SfxKind::Error => SfxConfig {
//                 layers: vec![SfxLayer {
//                     waveform: Waveform::Square,
//                     start_freq: 220.0,
//                     end_freq: 180.0,
//                     fm_ratio: 1.0,
//                     fm_amount: 0.0,
//                     gain: 0.6,
//                 }],
//                 duration: Duration::from_millis(150),
//                 attack: 0.001,
//                 decay: 0.03,
//                 sustain: 0.5,
//                 release: 0.05,
//                 glide_curve: 1.0,
//                 gain: 0.4,
//                 pan: 0.0,
//             },
//             SfxKind::Coins => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 1500.0,
//                         end_freq: 2200.0,
//                         fm_ratio: 4.0,
//                         fm_amount: 0.5,
//                         gain: 0.5,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 2200.0,
//                         end_freq: 3100.0,
//                         fm_ratio: 3.0,
//                         fm_amount: 0.3,
//                         gain: 0.3,
//                     },
//                 ],
//                 duration: Duration::from_millis(140),
//                 attack: 0.001,
//                 decay: 0.04,
//                 sustain: 0.1,
//                 release: 0.09,
//                 glide_curve: 0.5,
//                 gain: 0.4,
//                 pan: 0.0,
//             },
//             SfxKind::Notification => SfxConfig {
//                 layers: vec![
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 880.0,
//                         end_freq: 880.0,
//                         fm_ratio: 2.0,
//                         fm_amount: 0.2,
//                         gain: 0.6,
//                     },
//                     SfxLayer {
//                         waveform: Waveform::Sine,
//                         start_freq: 1320.0,
//                         end_freq: 1320.0,
//                         fm_ratio: 1.0,
//                         fm_amount: 0.0,
//                         gain: 0.3,
//                     },
//                 ],
//                 duration: Duration::from_millis(200),
//                 attack: 0.005,
//                 decay: 0.05,
//                 sustain: 0.5,
//                 release: 0.1,
//                 glide_curve: 1.0,
//                 gain: 0.45,
//                 pan: 0.0,
//             },

#[derive(Debug)]
struct SfxLayerState {
    phase: f32,
    mod_phase: f32,
    noise_state: u32,
    started: bool,
    // filter memory
    lp_state: f32,
    hp_state: f32,
}

#[derive(Debug)]
pub struct SfxVoice {
    pub config: SfxConfig,
    layer_states: Vec<SfxLayerState>,
    pub elapsed: f32,
    sample_rate: f32,
    total_duration: f32,
}

impl SfxVoice {
    pub fn new(config: SfxConfig, sample_rate: f32) -> Self {
        let scale = config.time_scale.max(0.001);
        let total_duration = config
            .layers
            .iter()
            .map(|l| (l.delay + l.duration + l.release) / scale)
            .fold(0.0f32, f32::max);

        let layer_states = config
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| SfxLayerState {
                phase: 0.0,
                mod_phase: 0.0,
                noise_state: (0x9E37_79B9u32
                    ^ layer.start_freq.to_bits()
                    ^ (i as u32).wrapping_mul(0x85EB_CA6B))
                    | 1,
                started: false,
                lp_state: 0.0,
                hp_state: 0.0,
            })
            .collect();

        Self {
            config,
            layer_states,
            elapsed: 0.0,
            sample_rate,
            total_duration,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.elapsed >= self.total_duration
    }

    #[inline]
    fn layer_envelope(layer: &SfxLayer, local_t: f32) -> f32 {
        let a = layer.attack.max(0.0001);
        let d = layer.decay.max(0.0001);
        let s = layer.sustain;
        let r = layer.release.max(0.0001);
        let sustain_end = layer.duration;

        if local_t < 0.0 {
            0.0
        } else if local_t < a {
            local_t / a
        } else if local_t < a + d {
            let dt = (local_t - a) / d;
            1.0 + (s - 1.0) * dt
        } else if local_t < sustain_end {
            s
        } else {
            let rt = (local_t - sustain_end) / r;
            (s * (1.0 - rt.clamp(0.0, 1.0))).max(0.0)
        }
    }

    #[inline]
    fn one_pole_lp(state: &mut f32, x: f32, cutoff: f32, sr: f32) -> f32 {
        if cutoff <= 0.0 || cutoff >= sr * 0.49 {
            *state = x;
            return x;
        }
        let rc = 1.0 / (std::f32::consts::TAU * cutoff);
        let dt = 1.0 / sr;
        let alpha = dt / (rc + dt);
        *state += alpha * (x - *state);
        *state
    }

    #[inline]
    fn one_pole_hp(state: &mut f32, x: f32, cutoff: f32, sr: f32) -> f32 {
        if cutoff <= 0.0 {
            *state = x;
            return x;
        }
        if cutoff >= sr * 0.49 {
            return 0.0;
        }
        let rc = 1.0 / (std::f32::consts::TAU * cutoff);
        let dt = 1.0 / sr;
        let alpha = rc / (rc + dt);
        let y = alpha * (*state + x - *state); // classic one-pole HP
        *state = x;
        y
    }

    pub(crate) fn next_sample(&mut self) -> (f32, f32) {
        let dt = 1.0 / self.sample_rate;
        let time_scale = self.config.time_scale.max(0.001);
        let pitch_scale = self.config.pitch_scale.max(0.001);

        let mut left = 0.0f32;
        let mut right = 0.0f32;

        for (layer, state) in self.config.layers.iter().zip(self.layer_states.iter_mut()) {
            // time is scaled globally
            let local_t = (self.elapsed * time_scale) - layer.delay;

            if local_t < 0.0 {
                continue;
            }

            if !state.started {
                state.started = true;
            }

            let env = Self::layer_envelope(layer, local_t);
            if env <= 0.00001 {
                continue;
            }

            // Pitch glide (also scaled)
            let glide_t = (local_t / layer.duration.max(0.0001))
                .clamp(0.0, 1.0)
                .powf(layer.glide_curve.max(0.0001));

            let freq =
                (layer.start_freq + (layer.end_freq - layer.start_freq) * glide_t) * pitch_scale;

            // FM
            let mod_freq = freq * layer.fm_ratio;
            state.mod_phase = (state.mod_phase + mod_freq * dt).fract();
            let modulator = (state.mod_phase * std::f32::consts::TAU).sin() * layer.fm_amount;

            state.phase = (state.phase + (freq + modulator * freq) * dt).fract();

            // Oscillator with pulse width support
            let pw = layer.pulse_width.clamp(0.01, 0.99);

            let osc = match layer.waveform {
                Waveform::Sine => (state.phase * std::f32::consts::TAU).sin(),
                Waveform::Triangle => 4.0 * (state.phase - (state.phase + 0.5).floor()).abs() - 1.0,
                Waveform::Square | Waveform::Pulse => {
                    if state.phase < pw {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Waveform::Saw => 2.0 * (state.phase - (state.phase + 0.5).floor()),
                Waveform::Noise => {
                    let mut x = state.noise_state;
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    state.noise_state = x;
                    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
                }
            };

            // Filters (applied in series: HP then LP)
            let mut filtered = osc;
            if layer.hp_freq > 0.0 {
                filtered = Self::one_pole_hp(
                    &mut state.hp_state,
                    filtered,
                    layer.hp_freq,
                    self.sample_rate,
                );
            }
            if layer.lp_freq > 0.0 {
                filtered = Self::one_pole_lp(
                    &mut state.lp_state,
                    filtered,
                    layer.lp_freq,
                    self.sample_rate,
                );
            }

            let sample = filtered * env * layer.gain;

            // constant-power pan
            let pan = (layer.pan + self.config.pan).clamp(-1.0, 1.0);
            let angle = (pan + 1.0) * 0.25 * std::f32::consts::PI;
            left += sample * angle.cos();
            right += sample * angle.sin();
        }

        self.elapsed += dt;

        let master = self.config.gain;
        (left * master, right * master)
    }
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let seconds = f64::deserialize(deserializer)?;

    if !seconds.is_finite() || seconds < 0.0 {
        return Err(serde::de::Error::custom(
            "duration must be a finite, non-negative number of seconds",
        ));
    }

    Ok(Duration::from_secs_f64(seconds))
}
