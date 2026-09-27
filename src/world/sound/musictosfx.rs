// use std::error::Error;
// use std::f32::consts::PI;
// use std::path::Path;
// use rustfft::{num_complex::Complex, FftPlanner};
// use symphonia::core::codecs::audio::AudioDecoderOptions;
// use symphonia::core::errors::Error as SymphoniaError;
// use symphonia::core::formats::{FormatOptions, TrackType};
// use symphonia::core::formats::probe::Hint;
// use symphonia::core::io::MediaSourceStream;
// use symphonia::core::meta::MetadataOptions;
// use crate::world::sound::sfx::{SfxConfig, SfxLayer, Waveform};
//
// const TRANSCRIBE_RATE: u32 = 22_050;
// const FFT_SIZE: usize = 4096;
// const HOP_SIZE: usize = 1024;
// const MIN_MIDI: i32 = 28;
// const MAX_MIDI: i32 = 108;
// const GRID_BEATS: f64 = 0.125;
// const STEPS_PER_BAR: usize = 32;
// const STICK_RATIO: f32 = 0.58;
// const ONSET_WEIGHT: f32 = 1.8;
//
// #[derive(Clone, Copy)]
// struct PitchCandidate {
//     midi: i32,
//     score: f32,
// }
//
// fn midi_frequency(midi: i32) -> f32 {
//     440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0)
// }
//
// fn midi_name(midi: i32) -> String {
//     const NAMES: [&str; 12] = [
//         "C", "C#", "D", "D#", "E", "F",
//         "F#", "G", "G#", "A", "A#", "B",
//     ];
//     let octave = midi.div_euclid(12) - 1;
//     let note = midi.rem_euclid(12) as usize;
//     format!("{}{}", NAMES[note], octave)
// }
//
// fn beat_length_string(beats: usize) -> String {
//     match beats {
//         1 => "0.125".to_string(),
//         2 => "0.25".to_string(),
//         4 => "0.5".to_string(),
//         8 => "1".to_string(),
//         16 => "2".to_string(),
//         32 => "4".to_string(),
//         _ => format!("{:.6}", beats as f32 * GRID_BEATS as f32),
//     }
// }
//
// fn decode_audio_file(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
//     let file = std::fs::File::open(path)?;
//     let mss = MediaSourceStream::new(Box::new(file), Default::default());
//     let mut hint = Hint::new();
//     if let Some(ext) = path.extension().and_then(|x| x.to_str()) {
//         hint.with_extension(ext);
//     }
//     let mut format = symphonia::default::get_probe().probe(
//         &hint,
//         mss,
//         FormatOptions::default(),
//         MetadataOptions::default(),
//     )?;
//     let track = format
//         .default_track(TrackType::Audio)
//         .ok_or("audio file contains no audio track")?;
//     let sample_rate = match track.codec_params.as_ref() {
//         None => return Err("Audio file Codec Params couldn't be determined".into()),
//         Some(c) => match c.audio() {
//             None => return Err("Audio file Audio Codec Params couldn't be determined".into()),
//             Some(a) => a.sample_rate.ok_or("Audio track has no sample rate")?,
//         },
//     };
//     let track_id = track.id;
//     let mut decoder = symphonia::default::get_codecs().make_audio_decoder(
//         track.codec_params.as_ref().unwrap().audio().unwrap(),
//         &AudioDecoderOptions::default(),
//     )?;
//     let mut mono = Vec::<f32>::new();
//     loop {
//         let packet = match format.next_packet() {
//             Ok(Some(packet)) => packet,
//             Ok(None) => break,
//             Err(SymphoniaError::ResetRequired) => break,
//             Err(SymphoniaError::IoError(_)) => continue,
//             Err(SymphoniaError::DecodeError(_)) => continue,
//             Err(e) => return Err(Box::new(e)),
//         };
//         if packet.track_id != track_id {
//             continue;
//         }
//         let decoded = match decoder.decode(&packet) {
//             Ok(decoded) => decoded,
//             Err(SymphoniaError::IoError(_)) => continue,
//             Err(SymphoniaError::DecodeError(_)) => continue,
//             Err(e) => return Err(Box::new(e)),
//         };
//         let channels = decoded.spec().channels().count();
//         if channels == 0 {
//             continue;
//         }
//         let mut interleaved = vec![0.0f32; decoded.frames() * channels];
//         decoded.copy_to_slice_interleaved(&mut interleaved);
//         for frame in interleaved.chunks_exact(channels) {
//             let mut sum = 0.0f32;
//             for sample in frame {
//                 sum += *sample;
//             }
//             mono.push(sum / channels as f32);
//         }
//     }
//     if mono.is_empty() {
//         return Err("decoded audio contains no samples".into());
//     }
//     let mean = mono.iter().copied().sum::<f32>() / mono.len() as f32;
//     for sample in &mut mono {
//         *sample -= mean;
//     }
//     let peak = mono.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
//     if peak > 0.0 {
//         let gain = 1.0 / peak.max(1.0);
//         for sample in &mut mono {
//             *sample *= gain;
//         }
//     }
//     Ok((mono, sample_rate))
// }
//
// fn resample_linear(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
//     if input.is_empty() || source_rate == target_rate {
//         return input.to_vec();
//     }
//     let ratio = target_rate as f64 / source_rate as f64;
//     let output_len = (((input.len() - 1) as f64) * ratio).ceil() as usize + 1;
//     let mut output = Vec::with_capacity(output_len);
//     for i in 0..output_len {
//         let source_pos = i as f64 / ratio;
//         let index = source_pos.floor() as usize;
//         let frac = source_pos - index as f64;
//         if index + 1 >= input.len() {
//             output.push(input[input.len() - 1]);
//             continue;
//         }
//         let a = input[index] as f64;
//         let b = input[index + 1] as f64;
//         output.push((a + (b - a) * frac) as f32);
//     }
//     output
// }
//
// fn local_magnitude(magnitude: &[f32], index: usize) -> f32 {
//     let start = index.saturating_sub(1);
//     let end = (index + 1).min(magnitude.len() - 1);
//     magnitude[start..=end]
//         .iter()
//         .copied()
//         .fold(0.0f32, f32::max)
// }
//
// fn harmonic_pitch_score(magnitude: &[f32], frequency: f32, sample_rate: f32) -> f32 {
//     let nyquist = sample_rate * 0.5;
//     let mut score = 0.0f32;
//     let mut weight_sum = 0.0f32;
//     for harmonic in 1..=8 {
//         let harmonic_freq = frequency * harmonic as f32;
//         if harmonic_freq >= nyquist {
//             break;
//         }
//         let bin = harmonic_freq * FFT_SIZE as f32 / sample_rate;
//         let index = bin.round() as usize;
//         if index >= magnitude.len() {
//             break;
//         }
//         let weight = if harmonic == 1 {
//             2.1
//         } else if harmonic == 2 {
//             0.85
//         } else {
//             0.48 / (harmonic as f32).sqrt()
//         };
//         score += local_magnitude(magnitude, index) * weight;
//         weight_sum += weight;
//     }
//     if weight_sum > 0.0 {
//         score / weight_sum
//     } else {
//         0.0
//     }
// }
//
// fn estimate_bpm(flux: &[f32], frame_rate: f32) -> f32 {
//     if flux.len() < 16 {
//         return 120.0;
//     }
//     let min_bpm = 55.0;
//     let max_bpm = 190.0;
//     let min_lag = ((frame_rate * 60.0 / max_bpm).round() as usize).max(2);
//     let max_lag = (frame_rate * 60.0 / min_bpm).round() as usize;
//     let limit = flux.len().min((frame_rate * 100.0) as usize);
//     let energy_a = flux[..limit].iter().map(|x| x * x).sum::<f32>();
//     if energy_a <= 1e-10 {
//         return 120.0;
//     }
//     let mut best_lag = 0usize;
//     let mut best_score = -1.0f32;
//     for lag in min_lag..=max_lag.min(limit / 2) {
//         let mut correlation = 0.0f32;
//         let mut energy_b = 0.0f32;
//         for i in lag..limit {
//             correlation += flux[i] * flux[i - lag];
//             energy_b += flux[i - lag] * flux[i - lag];
//         }
//         let denom = (energy_a * energy_b).sqrt();
//         if denom <= 1e-10 {
//             continue;
//         }
//         let normalized = correlation / denom;
//         if normalized > best_score {
//             best_score = normalized;
//             best_lag = lag;
//         }
//     }
//     if best_lag == 0 {
//         return 120.0;
//     }
//     let bpm = 60.0 * frame_rate / best_lag as f32;
//     if bpm < 68.0 {
//         bpm * 2.0
//     } else if bpm > 165.0 {
//         bpm * 0.5
//     } else {
//         bpm
//     }
// }
//
// fn analyze_audio(samples: &[f32], sample_rate: u32) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
//     let mut planner = FftPlanner::<f32>::new();
//     let fft = planner.plan_fft_forward(FFT_SIZE);
//     let window: Vec<f32> = (0..FFT_SIZE)
//         .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos())
//         .collect();
//     let pitch_count = (MAX_MIDI - MIN_MIDI + 1) as usize;
//     let mut pitch_frames = Vec::<Vec<f32>>::new();
//     let mut flux = Vec::<f32>::new();
//     let mut frame_energy = Vec::<f32>::new();
//     let mut previous_magnitude = vec![0.0f32; FFT_SIZE / 2 + 1];
//     let mut fft_buffer = vec![Complex::<f32>::new(0.0, 0.0); FFT_SIZE];
//     let mut position = 0usize;
//     while position < samples.len() {
//         for i in 0..FFT_SIZE {
//             let sample = samples.get(position + i).copied().unwrap_or(0.0);
//             fft_buffer[i] = Complex::new(sample * window[i], 0.0);
//         }
//         fft.process(&mut fft_buffer);
//         let mut magnitude = vec![0.0f32; FFT_SIZE / 2 + 1];
//         for i in 0..=FFT_SIZE / 2 {
//             magnitude[i] = fft_buffer[i].norm();
//         }
//         let mut current_flux = 0.0f32;
//         for i in 1..magnitude.len() {
//             let delta = magnitude[i] - previous_magnitude[i];
//             if delta > 0.0 {
//                 current_flux += delta;
//             }
//         }
//         flux.push(current_flux);
//         previous_magnitude.copy_from_slice(&magnitude);
//         let spectral_norm = magnitude.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
//         frame_energy.push(spectral_norm);
//         let mut frame_scores = vec![0.0f32; pitch_count];
//         for (offset, midi) in (MIN_MIDI..=MAX_MIDI).enumerate() {
//             let frequency = midi_frequency(midi);
//             let score = harmonic_pitch_score(&magnitude, frequency, sample_rate as f32);
//             frame_scores[offset] = score / spectral_norm;
//         }
//         pitch_frames.push(frame_scores);
//         position += HOP_SIZE;
//     }
//     (pitch_frames, flux, frame_energy)
// }
//
// fn choose_pitches(
//     scores: &[f32],
//     previous: &[Option<i32>],
//     is_onset: bool,
//     max_pitches: usize,
// ) -> Vec<PitchCandidate> {
//     if scores.is_empty() {
//         return Vec::new();
//     }
//     let maximum = scores.iter().copied().fold(0.0f32, f32::max);
//     if maximum <= 1e-8 {
//         return Vec::new();
//     }
//
//     let mut candidates = Vec::<PitchCandidate>::new();
//     for (index, raw) in scores.iter().copied().enumerate() {
//         if raw < maximum * 0.14 {
//             continue;
//         }
//         let midi = MIN_MIDI + index as i32;
//         let mut score = raw;
//
//         for &prev in previous {
//             if let Some(prev_midi) = prev {
//                 let dist = (midi - prev_midi).abs();
//                 let prev_idx = (prev_midi - MIN_MIDI) as usize;
//                 let prev_raw = if prev_idx < scores.len() {
//                     scores[prev_idx]
//                 } else {
//                     0.0
//                 };
//
//                 if prev_raw >= maximum * STICK_RATIO {
//                     if dist == 0 {
//                         score += if is_onset { 0.9 } else { 1.6 };
//                     } else if dist == 1 {
//                         score += 0.35;
//                     }
//                 } else if dist == 0 {
//                     score += 0.15;
//                 }
//             }
//         }
//
//         if is_onset {
//             score *= ONSET_WEIGHT;
//         }
//
//         candidates.push(PitchCandidate { midi, score });
//     }
//
//     candidates.sort_by(|a, b| {
//         b.score
//             .partial_cmp(&a.score)
//             .unwrap_or(std::cmp::Ordering::Equal)
//     });
//
//     let mut selected = Vec::<PitchCandidate>::new();
//     for candidate in candidates {
//         let conflicts = selected.iter().any(|other| {
//             let d = (candidate.midi - other.midi).abs();
//             d < 2 || (d == 2 && (candidate.score - other.score).abs() < 0.07)
//         });
//         if !conflicts {
//             selected.push(candidate);
//             if selected.len() >= max_pitches {
//                 break;
//             }
//         }
//     }
//     selected.sort_by_key(|x| x.midi);
//     selected
// }
//
// fn build_note_string(steps: &[Option<(i32, f32)>], step_count: usize) -> String {
//     let mut output = String::new();
//     for bar_start in (0..step_count).step_by(STEPS_PER_BAR) {
//         let bar_end = (bar_start + STEPS_PER_BAR).min(step_count);
//         let mut cursor = bar_start;
//         while cursor < bar_end {
//             let current = steps[cursor];
//             let mut length = 1usize;
//             while cursor + length < bar_end {
//                 let next = steps[cursor + length];
//                 let same = match (current, next) {
//                     (None, None) => true,
//                     (Some((a, av)), Some((b, bv))) => {
//                         a == b && (av - bv).abs() < 0.11
//                     }
//                     _ => false,
//                 };
//                 if !same {
//                     break;
//                 }
//                 length += 1;
//             }
//             if !output.is_empty() {
//                 output.push(' ');
//             }
//             let length_string = beat_length_string(length);
//             match current {
//                 Some((midi, velocity)) => {
//                     output.push_str(&format!(
//                         "{}:{}@{:.2}",
//                         midi_name(midi),
//                         length_string,
//                         velocity.clamp(0.08, 1.0)
//                     ));
//                 }
//                 None => {
//                     output.push_str(&format!("R:{}", length_string));
//                 }
//             }
//             cursor += length;
//         }
//         if bar_end == bar_start + STEPS_PER_BAR {
//             output.push_str(" |");
//         }
//     }
//     output.trim_end().to_string()
// }
//
// fn build_noise_string(onset_strength: &[f32], step_count: usize) -> String {
//     if onset_strength.is_empty() {
//         return "R:4".to_string();
//     }
//     let mean = onset_strength.iter().copied().sum::<f32>() / onset_strength.len() as f32;
//     let variance = onset_strength
//         .iter()
//         .map(|x| {
//             let d = *x - mean;
//             d * d
//         })
//         .sum::<f32>()
//         / onset_strength.len() as f32;
//     let threshold = mean + variance.sqrt() * 0.65;
//     let mut steps = vec![false; step_count];
//     for i in 0..step_count {
//         if onset_strength[i] < threshold {
//             continue;
//         }
//         let left = if i > 0 { onset_strength[i - 1] } else { -1.0 };
//         let right = if i + 1 < step_count {
//             onset_strength[i + 1]
//         } else {
//             -1.0
//         };
//         if onset_strength[i] >= left && onset_strength[i] >= right {
//             steps[i] = true;
//         }
//     }
//     let mut output = String::new();
//     for bar_start in (0..step_count).step_by(STEPS_PER_BAR) {
//         let bar_end = (bar_start + STEPS_PER_BAR).min(step_count);
//         let mut cursor = bar_start;
//         while cursor < bar_end {
//             if !output.is_empty() {
//                 output.push(' ');
//             }
//             if steps[cursor] {
//                 output.push_str("C4:0.125@0.82");
//                 cursor += 1;
//             } else {
//                 let mut length = 1usize;
//                 while cursor + length < bar_end && !steps[cursor + length] {
//                     length += 1;
//                 }
//                 output.push_str(&format!("R:{}", beat_length_string(length)));
//                 cursor += length;
//             }
//         }
//         if bar_end == bar_start + STEPS_PER_BAR {
//             output.push_str(" |");
//         }
//     }
//     output.trim_end().to_string()
// }
//
// pub fn song_to_sfx_yaml(
//     path: &Path,
//     bpm_hint: Option<f32>,
//     max_layers: usize,
// ) -> Result<(String, SfxConfig), Box<dyn Error>> {
//     let (decoded, decoded_rate) = decode_audio_file(path)?;
//     let samples = resample_linear(&decoded, decoded_rate, TRANSCRIBE_RATE);
//     let (pitch_frames, flux, frame_energy) = analyze_audio(&samples, TRANSCRIBE_RATE);
//     let analysis_frame_rate = TRANSCRIBE_RATE as f32 / HOP_SIZE as f32;
//     let bpm = bpm_hint
//         .unwrap_or_else(|| estimate_bpm(&flux, analysis_frame_rate))
//         .clamp(40.0, 220.0);
//     let duration_seconds = samples.len() as f64 / TRANSCRIBE_RATE as f64;
//     let beats = duration_seconds * bpm as f64 / 60.0;
//     let raw_steps = (beats / GRID_BEATS).ceil() as usize;
//     let step_count = ((raw_steps + STEPS_PER_BAR - 1) / STEPS_PER_BAR) * STEPS_PER_BAR;
//     let total_layers = max_layers.clamp(1, 12);
//     let add_noise = total_layers > 1;
//     let pitched_layers = if add_noise {
//         total_layers - 1
//     } else {
//         1
//     };
//     let step_seconds = 60.0 / bpm as f64 * GRID_BEATS;
//     let mut voice_steps = vec![vec![None; step_count]; pitched_layers];
//     let mut onset_strength = vec![0.0f32; step_count];
//     let mut previous_midis: Vec<Option<i32>> = vec![None; pitched_layers];
//
//     let flux_mean = if flux.is_empty() {
//         0.0
//     } else {
//         flux.iter().copied().sum::<f32>() / flux.len() as f32
//     };
//     let flux_std = if flux.is_empty() {
//         0.0
//     } else {
//         let var = flux.iter().map(|x| {
//             let d = *x - flux_mean;
//             d * d
//         }).sum::<f32>() / flux.len() as f32;
//         var.sqrt()
//     };
//     let onset_thresh = flux_mean + flux_std * 0.9;
//
//     for step in 0..step_count {
//         let start_time = step as f64 * step_seconds;
//         let end_time = (step + 1) as f64 * step_seconds;
//         let start_frame = (start_time * analysis_frame_rate as f64).floor() as usize;
//         let end_frame = (end_time * analysis_frame_rate as f64).ceil() as usize;
//         let start_frame = start_frame.min(pitch_frames.len());
//         let end_frame = end_frame.min(pitch_frames.len());
//         if start_frame >= end_frame {
//             previous_midis = vec![None; pitched_layers];
//             continue;
//         }
//         let pitch_count = pitch_frames[0].len();
//         let mut averaged = vec![0.0f32; pitch_count];
//         let mut frames_used = 0usize;
//         let mut energy_sum = 0.0f32;
//         let mut step_flux = 0.0f32;
//         for frame in start_frame..end_frame {
//             for pitch in 0..pitch_count {
//                 averaged[pitch] += pitch_frames[frame][pitch];
//             }
//             frames_used += 1;
//             if frame < flux.len() {
//                 step_flux = step_flux.max(flux[frame]);
//                 onset_strength[step] = onset_strength[step].max(flux[frame]);
//             }
//             if frame < frame_energy.len() {
//                 energy_sum += frame_energy[frame];
//             }
//         }
//         if frames_used > 0 {
//             let divisor = frames_used as f32;
//             for value in &mut averaged {
//                 *value /= divisor;
//             }
//             energy_sum /= divisor;
//         }
//
//         let max_score = averaged.iter().copied().fold(0.0f32, f32::max);
//         if max_score < 0.105 {
//             previous_midis = vec![None; pitched_layers];
//             continue;
//         }
//
//         let is_onset = step_flux > onset_thresh;
//         let selected = choose_pitches(&averaged, &previous_midis, is_onset, pitched_layers);
//         let maximum = selected.iter().map(|x| x.score).fold(0.0f32, f32::max);
//         let energy_norm = (energy_sum / (energy_sum + 0.18)).clamp(0.0, 1.0);
//
//         let mut new_previous = vec![None; pitched_layers];
//         for (layer_index, candidate) in selected.iter().enumerate() {
//             if layer_index >= pitched_layers {
//                 break;
//             }
//             let pitch_vel = (candidate.score / maximum.max(1e-9)).sqrt().clamp(0.12, 1.0);
//             let velocity = (pitch_vel * 0.78 + energy_norm * 0.22).clamp(0.09, 1.0);
//             voice_steps[layer_index][step] = Some((candidate.midi, velocity));
//             new_previous[layer_index] = Some(candidate.midi);
//         }
//         previous_midis = new_previous;
//     }
//
//     let name = path
//         .file_stem()
//         .and_then(|x| x.to_str())
//         .unwrap_or("ConvertedSong")
//         .to_string();
//     let mut layers = Vec::with_capacity(total_layers);
//     for (layer_index, steps) in voice_steps.iter().enumerate() {
//         let notes = build_note_string(steps, step_count);
//         let gain = 0.42 / (1.0 + layer_index as f32 * 0.10);
//         layers.push(SfxLayer {
//             waveform: Waveform::Sine,
//             notes,
//             note_len: 0.125,
//             gate: 0.90,
//             start_freq: 0.0,
//             end_freq: 0.0,
//             glide_curve: 1.0,
//             fm_ratio: 0.0,
//             fm_amount: 0.0,
//             pulse_width: 0.5,
//             delay: 0.0,
//             duration: 0.2,
//             attack: 0.004,
//             decay: 0.07,
//             sustain: 0.78,
//             release: 0.07,
//             lp_freq: 0.0,
//             hp_freq: 0.0,
//             gain,
//             pan: 0.0,
//         });
//     }
//     if add_noise {
//         let noise_notes = build_noise_string(&onset_strength, step_count);
//         layers.push(SfxLayer {
//             waveform: Waveform::Noise,
//             notes: noise_notes,
//             note_len: 0.125,
//             gate: 0.04,
//             start_freq: 0.0,
//             end_freq: 0.0,
//             glide_curve: 1.0,
//             fm_ratio: 0.0,
//             fm_amount: 0.0,
//             pulse_width: 0.5,
//             delay: 0.0,
//             duration: 0.2,
//             attack: 0.001,
//             decay: 0.018,
//             sustain: 0.0,
//             release: 0.06,
//             lp_freq: 0.0,
//             hp_freq: 750.0,
//             gain: 0.15,
//             pan: 0.0,
//         });
//     }
//     let config = SfxConfig {
//         name,
//         layers,
//         gain: 0.55,
//         pan: 0.0,
//         pitch_scale: 1.0,
//         time_scale: 1.0,
//         bpm,
//         bar_beats: 4.0,
//         deduplication_time: 0.1,
//     };
//     let yaml = serde_yaml::to_string(&config)?;
//     Ok((yaml, config))
// }
