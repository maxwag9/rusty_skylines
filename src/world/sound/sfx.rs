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
}

#[derive(Default, Debug, Clone, Copy, Deserialize)]
pub struct SfxLayer {
    pub waveform: Waveform,
    pub start_freq: f32,
    pub end_freq: f32,
    pub fm_ratio: f32,
    pub fm_amount: f32,
    pub gain: f32,
}

#[derive(Default, Debug, Clone, Deserialize)]
pub struct SfxConfig {
    pub name: String,
    pub layers: Vec<SfxLayer>,
    #[serde(deserialize_with = "deserialize_duration")]
    pub duration: Duration,
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
    pub glide_curve: f32,
    pub gain: f32,
    pub pan: f32,
    #[serde(default = "zeropointone")]
    pub deduplication_time: f32,
}
fn zeropointone() -> f32 {
    0.07
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
}

#[derive(Debug)]
pub struct SfxVoice {
    pub config: SfxConfig,
    layer_states: Vec<SfxLayerState>,
    pub elapsed: f32,
    sample_rate: f32,
}

impl SfxVoice {
    pub fn new(config: SfxConfig, sample_rate: f32) -> Self {
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
            })
            .collect();

        Self {
            config,
            layer_states,
            elapsed: 0.0,
            sample_rate,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.elapsed >= self.config.duration.as_secs_f32() + self.config.release
    }

    fn envelope(&self) -> f32 {
        let t = self.elapsed;
        let a = self.config.attack.max(0.0001);
        let d = self.config.decay.max(0.0001);
        let s = self.config.sustain;
        let sustain_end = self.config.duration.as_secs_f32();
        let r = self.config.release.max(0.0001);

        if t < a {
            t / a
        } else if t < a + d {
            let dt = (t - a) / d;
            1.0 + (s - 1.0) * dt
        } else if t < sustain_end {
            s
        } else {
            let rt = (t - sustain_end) / r;
            (s * (1.0 - rt.clamp(0.0, 1.0))).max(0.0)
        }
    }

    pub(crate) fn next_sample(&mut self) -> f32 {
        let dt = 1.0 / self.sample_rate;
        let dur = self.config.duration.as_secs_f32().max(0.0001);
        let glide_t = (self.elapsed / dur)
            .clamp(0.0, 1.0)
            .powf(self.config.glide_curve.max(0.0001));
        let env = self.envelope();
        let mut mixed = 0.0f32;

        for (layer, layer_state) in self.config.layers.iter().zip(self.layer_states.iter_mut()) {
            let freq = layer.start_freq + (layer.end_freq - layer.start_freq) * glide_t;
            let mod_freq = freq * layer.fm_ratio;
            layer_state.mod_phase += mod_freq * dt;
            layer_state.mod_phase = layer_state.mod_phase.fract();
            let modulator = (layer_state.mod_phase * std::f32::consts::TAU).sin() * layer.fm_amount;

            layer_state.phase += (freq + modulator * freq) * dt;
            layer_state.phase = layer_state.phase.fract();

            let osc = match layer.waveform {
                Waveform::Sine => (layer_state.phase * std::f32::consts::TAU).sin(),
                Waveform::Triangle => {
                    4.0 * (layer_state.phase - (layer_state.phase + 0.5).floor()).abs() - 1.0
                }
                Waveform::Square => {
                    if layer_state.phase < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Waveform::Saw => 2.0 * (layer_state.phase - (layer_state.phase + 0.5).floor()),
                Waveform::Noise => {
                    let mut x = layer_state.noise_state;
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    layer_state.noise_state = x;
                    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
                }
            };

            mixed += osc * layer.gain;
        }

        self.elapsed += dt;
        mixed * env * self.config.gain
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
