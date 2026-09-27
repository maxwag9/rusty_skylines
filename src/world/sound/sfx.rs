use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, Deserialize, Serialize)]
pub enum Waveform {
    #[default]
    Sine,
    Triangle,
    Square,
    Saw,
    Noise,
    Pulse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SfxLayer {
    pub waveform: Waveform,

    /// Note sequence. Whitespace separated tokens:
    ///
    ///   NOTE[:LEN][@VEL]     e.g.  C5   C5:2   E4:1/2   G4:0.25@0.6   R:1.5
    ///   |                    bar line (only used for validation, makes no sound)
    ///
    /// NOTE is a pitch (C-B, optional #/b, octave: `C#5`, `Bb3`) or `R` / `_` for a rest.
    /// LEN is in beats: a number (`0.25`) or a fraction (`1/4`). Omitted = `note_len`.
    /// VEL is a 0..1 volume multiplier for that one note. Omitted = 1.
    #[serde(default)]
    pub notes: String,

    /// Default length in beats for tokens without an explicit `:LEN`.
    #[serde(default = "one")]
    pub note_len: f32,

    /// Fraction (0..=1) of each note's length that is held before its release begins.
    #[serde(default = "point_nine")]
    pub gate: f32,

    #[serde(default)]
    pub start_freq: f32,
    #[serde(default)]
    pub end_freq: f32,

    #[serde(default = "one")]
    pub glide_curve: f32,

    #[serde(default)]
    pub fm_ratio: f32,
    #[serde(default)]
    pub fm_amount: f32,

    #[serde(default = "half")]
    pub pulse_width: f32,

    /// Seconds (SFX time, i.e. before `time_scale`) before this layer starts.
    #[serde(default)]
    pub delay: f32,
    /// Length in seconds of a NON-note layer (glide layers, noise bursts...).
    /// Ignored when `notes` is used: note lengths come from `note_len` / `:LEN`.
    #[serde(default = "zeropointtwo")]
    pub duration: f32,

    #[serde(default = "tiny")]
    pub attack: f32,
    #[serde(default = "zeropointone")]
    pub decay: f32,
    #[serde(default)]
    pub sustain: f32,
    #[serde(default = "zeropointone")]
    pub release: f32,

    #[serde(default)]
    pub lp_freq: f32,
    #[serde(default)]
    pub hp_freq: f32,

    #[serde(default = "one")]
    pub gain: f32,
    #[serde(default)]
    pub pan: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SfxConfig {
    pub name: String,
    pub layers: Vec<SfxLayer>,

    // Global master
    #[serde(default = "one")]
    pub gain: f32,
    #[serde(default)]
    pub pan: f32,

    #[serde(default = "one")]
    pub pitch_scale: f32, // multiplies every frequency
    #[serde(default = "one")]
    pub time_scale: f32, // speeds up / slows down the whole SFX

    /// Tempo. One beat = 60 / bpm seconds. Every layer shares this grid,
    /// which is what keeps layers locked together.
    #[serde(default = "default_bpm")]
    pub bpm: f32,

    /// If > 0, every bar (text between `|` tokens) in every note layer must be
    /// exactly this many beats long, otherwise a warning is printed.
    #[serde(default)]
    pub bar_beats: f32,

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
fn point_nine() -> f32 {
    0.9
}
fn default_bpm() -> f32 {
    120.0
}

// ---------------------------------------------------------------------------
// Note parsing / scheduling
// ---------------------------------------------------------------------------

/// One scheduled note. Times are in SFX seconds (before `time_scale`),
/// computed once from cumulative beat counts, so layers can never drift.
#[derive(Debug, Clone, Copy)]
struct NoteEvent {
    start: f64,
    len: f64,
    freq: Option<f32>, // None = rest
    vel: f32,
}

/// Returns Ok(None) for a rest.
fn parse_pitch(s: &str) -> Result<Option<f32>, String> {
    if s.eq_ignore_ascii_case("R") || s == "_" {
        return Ok(None);
    }

    let bytes = s.as_bytes();

    let mut semitone: i32 = match bytes.first().map(|b| b.to_ascii_uppercase()) {
        Some(b'C') => 0,
        Some(b'D') => 2,
        Some(b'E') => 4,
        Some(b'F') => 5,
        Some(b'G') => 7,
        Some(b'A') => 9,
        Some(b'B') => 11,
        _ => return Err(format!("bad note name '{s}'")),
    };

    let mut index = 1;
    match bytes.get(index) {
        Some(b'#') => {
            semitone += 1;
            index += 1;
        }
        Some(b'b') | Some(b'B') => {
            semitone -= 1;
            index += 1;
        }
        _ => {}
    }

    let octave: i32 = s
        .get(index..)
        .and_then(|o| o.parse().ok())
        .ok_or_else(|| format!("bad octave in '{s}'"))?;

    let midi = (octave + 1) * 12 + semitone;
    Ok(Some(440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0)))
}

/// "0.25" or "1/4" -> beats. Must be finite and > 0.
fn parse_beats(s: &str) -> Result<f64, String> {
    let bad = || format!("bad length '{s}'");

    let v = match s.split_once('/') {
        Some((n, d)) => {
            let n: f64 = n.parse().map_err(|_| bad())?;
            let d: f64 = d.parse().map_err(|_| bad())?;
            n / d
        }
        None => s.parse::<f64>().map_err(|_| bad())?,
    };

    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(bad())
    }
}

/// NOTE[:LEN][@VEL] -> (freq, beats, velocity)
fn parse_token(tok: &str, default_len: f64) -> Result<(Option<f32>, f64, f32), String> {
    let (rest, vel) = match tok.split_once('@') {
        Some((r, v)) => (
            r,
            v.parse::<f32>()
                .map_err(|_| format!("bad velocity '{v}'"))?,
        ),
        None => (tok, 1.0),
    };

    let (pitch, len) = match rest.split_once(':') {
        Some((p, l)) => (p, parse_beats(l)?),
        None => (rest, default_len),
    };

    Ok((parse_pitch(pitch)?, len, vel.max(0.0)))
}

fn check_bar(bar_no: usize, len: f64, expected: f64, warnings: &mut Vec<String>) {
    if expected > 0.0 && (len - expected).abs() > 1e-6 {
        warnings.push(format!(
            "bar {bar_no} is {len} beats long, expected {expected}"
        ));
    }
}

/// Turns a layer's note string into absolute-time events.
/// Bad tokens are reported and replaced by a rest of the default length,
/// so a typo can never shift the rest of the layer.
fn compile_notes(
    layer: &SfxLayer,
    beat_secs: f64,
    bar_beats: f64,
) -> (Vec<NoteEvent>, Vec<String>) {
    let mut events = Vec::new();
    let mut warnings = Vec::new();

    let default_len = layer.note_len.max(0.0001) as f64;

    let mut cursor = 0.0f64; // in beats
    let mut bar_start = 0.0f64;
    let mut bar_no = 1usize;

    for tok in layer.notes.split_whitespace() {
        if tok == "|" {
            check_bar(bar_no, cursor - bar_start, bar_beats, &mut warnings);
            bar_start = cursor;
            bar_no += 1;
            continue;
        }

        let (freq, len, vel) = match parse_token(tok, default_len) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(format!("token '{tok}': {e}"));
                (None, default_len, 1.0)
            }
        };

        events.push(NoteEvent {
            start: cursor * beat_secs,
            len: len * beat_secs,
            freq,
            vel,
        });

        cursor += len;
    }

    // final bar without a trailing `|`
    if cursor - bar_start > 1e-9 {
        check_bar(bar_no, cursor - bar_start, bar_beats, &mut warnings);
    }

    (events, warnings)
}

impl SfxConfig {
    /// Human-readable problems (bad tokens, bars of the wrong length). Empty = fine.
    pub fn validate(&self) -> Vec<String> {
        let beat_secs = 60.0 / self.bpm.max(1.0) as f64;
        let bar_beats = self.bar_beats.max(0.0) as f64;

        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let (_, warnings) = compile_notes(layer, beat_secs, bar_beats);
            out.extend(warnings.into_iter().map(|w| format!("layer {i}: {w}")));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Voice
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SfxLayerState {
    phase: f32,
    mod_phase: f32,
    noise_state: u32,
    lp_state: f32,
    hp_state: f32,
    events: Vec<NoteEvent>, // empty => plain glide layer
    next_event: usize,      // next event whose start time hasn't been reached yet
    active: Option<usize>,  // last pitched event that has started (keeps its release tail)
}

#[derive(Debug)]
pub struct SfxVoice {
    pub config: SfxConfig,
    layer_states: Vec<SfxLayerState>,
    pub elapsed: f32,
    sample_index: u64, // the real clock: integer, so it can't accumulate rounding error
    sample_rate: f32,
    total_duration: f64,
}

impl SfxVoice {
    pub fn new(config: SfxConfig, sample_rate: f32) -> Self {
        let scale = config.time_scale.max(0.001) as f64;
        let beat_secs = 60.0 / config.bpm.max(1.0) as f64;
        let bar_beats = config.bar_beats.max(0.0) as f64;

        let mut total_duration = 0.0f64;
        let mut layer_states = Vec::with_capacity(config.layers.len());

        for (i, layer) in config.layers.iter().enumerate() {
            let (events, warnings) = compile_notes(layer, beat_secs, bar_beats);
            for w in &warnings {
                eprintln!("sfx '{}' layer {}: {}", config.name, i, w);
            }

            let body = match events.last() {
                Some(last) => last.start + last.len,
                None => layer.duration as f64,
            };
            let layer_total = (layer.delay as f64 + body + layer.release as f64) / scale;
            total_duration = total_duration.max(layer_total);

            layer_states.push(SfxLayerState {
                phase: 0.0,
                mod_phase: 0.0,
                noise_state: (0x9E37_79B9u32
                    ^ layer.start_freq.to_bits()
                    ^ (i as u32).wrapping_mul(0x85EB_CA6B))
                    | 1,
                lp_state: 0.0,
                hp_state: 0.0,
                events,
                next_event: 0,
                active: None,
            });
        }

        Self {
            config,
            layer_states,
            elapsed: 0.0,
            sample_index: 0,
            sample_rate,
            total_duration,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.sample_index as f64 / self.sample_rate as f64 >= self.total_duration
    }

    /// `t` = time since note start, `gate` = time at which release begins.
    /// Release starts from whatever level the envelope has reached at `gate`,
    /// so a short gate never causes a jump.
    #[inline]
    fn layer_envelope(layer: &SfxLayer, t: f32, gate: f32) -> f32 {
        let a = layer.attack.max(0.0001);
        let d = layer.decay.max(0.0001);
        let s = layer.sustain;
        let r = layer.release.max(0.0001);

        let level = |t: f32| -> f32 {
            if t < a {
                t / a
            } else if t < a + d {
                1.0 + (s - 1.0) * ((t - a) / d)
            } else {
                s
            }
        };

        if t < 0.0 {
            0.0
        } else if t < gate {
            level(t)
        } else {
            (level(gate) * (1.0 - (t - gate) / r).clamp(0.0, 1.0)).max(0.0)
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
        let y = alpha * (*state + x - *state);

        *state = x;
        y
    }

    pub(crate) fn next_sample(&mut self) -> (f32, f32) {
        let dt = 1.0 / self.sample_rate;
        let pitch_scale = self.config.pitch_scale.max(0.001);

        // One shared clock for every layer, derived from an integer counter.
        let sfx_t = (self.sample_index as f64 / self.sample_rate as f64)
            * self.config.time_scale.max(0.001) as f64;

        let mut left = 0.0f32;
        let mut right = 0.0f32;

        for (layer, state) in self.config.layers.iter().zip(self.layer_states.iter_mut()) {
            let local_t = sfx_t - layer.delay as f64;

            if local_t < 0.0 {
                continue;
            }

            let (freq, note_t, gate, vel) = if !state.events.is_empty() {
                // Enter every event whose start time has been reached.
                while state.next_event < state.events.len()
                    && local_t >= state.events[state.next_event].start
                {
                    if state.events[state.next_event].freq.is_some() {
                        state.active = Some(state.next_event);
                        state.phase = 0.0;
                        state.mod_phase = 0.0;
                    }
                    state.next_event += 1;
                }

                // A rest leaves the previous note alone, so its release tail can ring out.
                let Some(active) = state.active else { continue };
                let ev = state.events[active];
                let Some(base) = ev.freq else { continue };

                let gate = (ev.len * layer.gate.clamp(0.01, 1.0) as f64) as f32;

                (
                    base * pitch_scale,
                    (local_t - ev.start) as f32,
                    gate,
                    ev.vel,
                )
            } else {
                let glide_t = (local_t as f32 / layer.duration.max(0.0001))
                    .clamp(0.0, 1.0)
                    .powf(layer.glide_curve.max(0.0001));

                let freq = (layer.start_freq + (layer.end_freq - layer.start_freq) * glide_t)
                    * pitch_scale;

                (freq, local_t as f32, layer.duration, 1.0)
            };

            let env = Self::layer_envelope(layer, note_t, gate) * vel;

            if env <= 0.00001 {
                continue;
            }

            let mod_freq = freq * layer.fm_ratio;

            state.mod_phase = (state.mod_phase + mod_freq * dt).fract();

            let modulator = (state.mod_phase * std::f32::consts::TAU).sin() * layer.fm_amount;

            state.phase = (state.phase + (freq + modulator * freq) * dt).fract();

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

            let pan = (layer.pan + self.config.pan).clamp(-1.0, 1.0);
            let angle = (pan + 1.0) * 0.25 * std::f32::consts::PI;

            left += sample * angle.cos();
            right += sample * angle.sin();
        }

        self.sample_index += 1;
        self.elapsed = (self.sample_index as f64 / self.sample_rate as f64) as f32;

        let master = self.config.gain;

        (left * master, right * master)
    }
}
