//! Percussion post-stage for SpectralPrism.
//!
//! This is a **third pipeline step** that runs *after* fusion: it takes the
//! stereo loop buffer any `FusionMode` produced (`render::LoopBufferData`)
//! and re-synthesizes it as a one-shot percussive hit. The fused output
//! supplies the modal character (its strongest spectral peaks become a bank
//! of decaying sine partials, so the hit still sounds like the loaded
//! sample/Fusion result), while the percussion parameters supply the
//! transient, noise, pitch and amplitude envelopes that make it percussive.
//!
//! Unlike the freeze/re-synthesis path this output is deliberately **not
//! loopable**: it decays to silence within `length_seconds`. The app's
//! one-shot playback path (FEAT-9) consumes it.
//!
//! The design is modal-body + noise + transient because the freeze/resynth
//! pipeline is tonal and sustained and cannot make a convincing kick/snare
//! transient by itself.

use crate::analysis::analyze_frame;
use crate::fft::FreezeFft;
use crate::render::LoopBufferData;
use crate::stereo::{decorrelation_spread, width_multiplier};
use crate::window::sine_window;
use std::f32::consts::TAU;

/// Absolute sanity bounds for the one-shot length, mirroring the loop-length
/// clamp convention in `render.rs`.
pub const MIN_PERCUSSION_SECONDS: f32 = 0.03;
pub const MAX_PERCUSSION_SECONDS: f32 = 2.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NoiseColor {
    White,
    Pink,
    /// One-pole high-pass then low-pass, a cheap "bright band" for hats.
    BandLimited,
}

/// Percussion parameters for the post-fusion stage. All percentages are
/// user-facing 0-100 values, matching the other SpectralPrism controls.
#[derive(Clone, Debug)]
pub struct PercussionParams {
    pub noise_amount_pct: f32,
    pub noise_color: NoiseColor,
    pub noise_decay_seconds: f32,
    /// Short decaying tone/click added on top of the body.
    pub transient_amount_pct: f32,
    pub transient_decay_seconds: f32,
    pub transient_frequency_hz: f32,
    /// Global pitch envelope applied to every modal partial. Start low/end
    /// high gives an upward sweep; a kick scoops from high to low.
    pub pitch_start_semitones: f32,
    pub pitch_end_semitones: f32,
    pub pitch_decay_seconds: f32,
    pub amp_decay_seconds: f32,
    /// Modal body level.
    pub body_amount_pct: f32,
    pub partial_count: usize,
    /// Base decay for the strongest partial; higher partials decay faster.
    pub partial_decay_seconds: f32,
    /// Ring-modulates the body against an inharmonic ratio derived from the
    /// fused content, for metallic/digital flavours.
    pub digital_amount_pct: f32,
    /// Saturation/overdrive 0-100 applied to the summed body+noise+transient
    /// before normalisation. 0 leaves the signal untouched.
    pub drive_amount_pct: f32,
    /// Feed-forward peak compression 0-100. Squashes the hit's peak so the
    /// following normalisation lifts the body; 0 disables it.
    pub compress_amount_pct: f32,
    pub stereo_width_pct: f32,
    pub length_seconds: f32,
}

impl Default for PercussionParams {
    fn default() -> Self {
        kick_preset()
    }
}

pub fn kick_preset() -> PercussionParams {
    PercussionParams {
        noise_amount_pct: 8.0,
        noise_color: NoiseColor::White,
        noise_decay_seconds: 0.04,
        transient_amount_pct: 60.0,
        transient_decay_seconds: 0.005,
        transient_frequency_hz: 2000.0,
        pitch_start_semitones: 30.0,
        pitch_end_semitones: -28.0,
        pitch_decay_seconds: 0.04,
        amp_decay_seconds: 0.3,
        body_amount_pct: 100.0,
        partial_count: 12,
        partial_decay_seconds: 0.35,
        digital_amount_pct: 0.0,
        drive_amount_pct: 0.0,
        compress_amount_pct: 0.0,
        stereo_width_pct: 20.0,
        length_seconds: 0.6,
    }
}

pub fn snare_preset() -> PercussionParams {
    PercussionParams {
        noise_amount_pct: 75.0,
        noise_color: NoiseColor::White,
        noise_decay_seconds: 0.2,
        transient_amount_pct: 70.0,
        transient_decay_seconds: 0.005,
        transient_frequency_hz: 2400.0,
        pitch_start_semitones: 12.0,
        pitch_end_semitones: -12.0,
        pitch_decay_seconds: 0.02,
        amp_decay_seconds: 0.24,
        body_amount_pct: 60.0,
        partial_count: 16,
        partial_decay_seconds: 0.18,
        digital_amount_pct: 10.0,
        drive_amount_pct: 0.0,
        compress_amount_pct: 0.0,
        stereo_width_pct: 60.0,
        length_seconds: 0.4,
    }
}

pub fn metal_preset() -> PercussionParams {
    PercussionParams {
        noise_amount_pct: 20.0,
        noise_color: NoiseColor::BandLimited,
        noise_decay_seconds: 0.18,
        transient_amount_pct: 45.0,
        transient_decay_seconds: 0.007,
        transient_frequency_hz: 3600.0,
        pitch_start_semitones: 14.0,
        pitch_end_semitones: -20.0,
        pitch_decay_seconds: 0.045,
        amp_decay_seconds: 0.5,
        body_amount_pct: 90.0,
        partial_count: 24,
        partial_decay_seconds: 0.4,
        digital_amount_pct: 45.0,
        drive_amount_pct: 0.0,
        compress_amount_pct: 0.0,
        stereo_width_pct: 80.0,
        length_seconds: 0.9,
    }
}

pub fn hat_preset() -> PercussionParams {
    PercussionParams {
        noise_amount_pct: 92.0,
        noise_color: NoiseColor::BandLimited,
        noise_decay_seconds: 0.07,
        transient_amount_pct: 55.0,
        transient_decay_seconds: 0.0025,
        transient_frequency_hz: 8500.0,
        pitch_start_semitones: 0.0,
        pitch_end_semitones: 0.0,
        pitch_decay_seconds: 0.05,
        amp_decay_seconds: 0.09,
        body_amount_pct: 20.0,
        partial_count: 8,
        partial_decay_seconds: 0.05,
        digital_amount_pct: 0.0,
        drive_amount_pct: 0.0,
        compress_amount_pct: 0.0,
        stereo_width_pct: 50.0,
        length_seconds: 0.18,
    }
}

struct Partial {
    freq: f32,
    amp: f32,
    /// Per-partial decay multiplier (higher partials decay faster).
    decay_scale: f32,
    bin: usize,
}

/// Extracts the `count` strongest separated spectral peaks from the fused
/// output's first frame. These become the modal body, so the percussion
/// inherits the character of whatever Fusion mode produced `input`.
fn extract_partials(input: &LoopBufferData, count: usize) -> Vec<Partial> {
    let fft = FreezeFft::new();
    let window = sine_window(fft.size());
    let frame = match input.channels.first() {
        Some(channel) if !channel.is_empty() => analyze_frame(channel, 0, &fft, &window),
        _ => return Vec::new(),
    };
    let sample_rate = input.sample_rate.max(1.0);
    let bins = frame.mag.len();
    let mut candidates: Vec<(f32, usize)> = Vec::new();
    for bin in 2..bins.saturating_sub(1) {
        let mag = frame.mag[bin];
        if mag > 1e-9 && mag > frame.mag[bin - 1] && mag >= frame.mag[bin + 1] {
            candidates.push((mag, bin));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut chosen: Vec<(f32, usize)> = Vec::new();
    for (mag, bin) in candidates {
        if chosen.iter().any(|(_, other)| bin.abs_diff(*other) < 3) {
            continue;
        }
        chosen.push((mag, bin));
        if chosen.len() >= count.max(1) {
            break;
        }
    }
    let max_mag = chosen.iter().map(|(m, _)| *m).fold(0.0f32, f32::max).max(1e-9);
    let base_freq = chosen
        .first()
        .map(|(_, bin)| *bin as f32 * sample_rate / fft.size() as f32)
        .unwrap_or(70.0);

    chosen
        .into_iter()
        .map(|(mag, bin)| {
            let freq = (bin as f32 * sample_rate / fft.size() as f32).max(20.0);
            Partial {
                freq,
                amp: mag / max_mag,
                // A physical body's high modes ring out sooner; reference the
                // strongest (usually lowest) partial so a kick keeps its thump.
                decay_scale: (base_freq / freq).clamp(0.08, 4.0),
                bin,
            }
        })
        .collect()
}

/// Deterministic xorshift PRNG so percussion renders are reproducible.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn bipolar(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

struct NoiseGen {
    rng: Rng,
    color: NoiseColor,
    lp: f32,
    hp_prev_in: f32,
    hp_prev_out: f32,
}

impl NoiseGen {
    fn new(color: NoiseColor, seed: u32) -> Self {
        NoiseGen {
            rng: Rng::new(seed),
            color,
            lp: 0.0,
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,
        }
    }

    fn sample(&mut self) -> f32 {
        let white = self.rng.bipolar();
        match self.color {
            NoiseColor::White => white,
            NoiseColor::Pink => {
                // One-pole low-pass + make-up gain: a rough pink-ish tilt.
                self.lp = self.lp * 0.95 + white * 0.05;
                (self.lp * 4.0).clamp(-1.0, 1.0)
            }
            NoiseColor::BandLimited => {
                // One-pole high-pass (removes rumble) then a fixed low-pass
                // floor, leaving a bright band for hats.
                let hp = white - self.hp_prev_in + 0.9 * self.hp_prev_out;
                self.hp_prev_in = white;
                self.hp_prev_out = hp;
                (hp * 0.5).clamp(-1.0, 1.0)
            }
        }
    }
}

/// Soft-clip waveshaper. `amount` is 0-1: at 0 the sample passes through
/// untouched, higher values push it into `tanh` saturation (adding odd
/// harmonics) and crossfade to the saturated signal.
fn apply_drive(sample: f32, amount: f32) -> f32 {
    let mix = amount.clamp(0.0, 1.0);
    if mix <= 0.0 {
        return sample;
    }
    let gain = 1.0 + mix * 12.0;
    let wet = (sample * gain).tanh();
    sample * (1.0 - mix) + wet * mix
}

/// Linked-stereo feed-forward peak compressor. `amount` is 0-1; the threshold
/// is derived from the buffer's own peak so the detector is independent of the
/// render's overall loudness. Instant attack means the hit's transient is
/// caught; the gain reduction is held and released over ~50 ms. A following
/// `peak_normalize` lifts the squashed body back up, which is what makes the
/// result read as compressed.
fn compress_channels(channels: &mut [Vec<f32>], sample_rate: f32, amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    let peak = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak <= 1e-9 {
        return;
    }
    let threshold = peak * (1.0 - 0.6 * amount);
    let ratio = 1.0 + amount * 7.0;
    let release = (-1.0 / (0.05 * sample_rate.max(1.0))).exp();
    let frames = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    let mut envelope = 0.0f32;
    for i in 0..frames {
        let level = channels.iter().map(|c| c[i].abs()).fold(0.0f32, f32::max);
        // Peak-hold with release: catches transients instantly, then lets the
        // gain recover smoothly.
        envelope = level.max(envelope * release);
        let gain = if envelope > threshold {
            (threshold + (envelope - threshold) / ratio) / envelope
        } else {
            1.0
        };
        for channel in channels.iter_mut() {
            channel[i] *= gain;
        }
    }
}

fn peak_normalize(channels: &mut [Vec<f32>], target_peak: f32) {
    let current = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
    if current > 1e-9 {
        let scale = target_peak / current;
        channels.iter_mut().flatten().for_each(|s| *s *= scale);
    }
}

/// Renders a short, non-looping percussive one-shot from the fused loop
/// `input` (the output of any `FusionMode`). See the module docs for the
/// pipeline position and rationale.
pub fn render_percussion(input: &LoopBufferData, params: &PercussionParams) -> LoopBufferData {
    let sample_rate = input.sample_rate.max(1.0);
    let length = params
        .length_seconds
        .clamp(MIN_PERCUSSION_SECONDS, MAX_PERCUSSION_SECONDS);
    let frames = ((length * sample_rate).round() as usize).max(1);

    let partials = extract_partials(input, params.partial_count);
    let inv_sr = 1.0 / sample_rate;
    let width = width_multiplier(params.stereo_width_pct);
    let body = (params.body_amount_pct / 100.0).max(0.0);
    let noise_amount = (params.noise_amount_pct / 100.0).max(0.0);
    let transient_amount = (params.transient_amount_pct / 100.0).max(0.0);
    let digital = (params.digital_amount_pct / 100.0).clamp(0.0, 1.0);

    let pitch_decay = params.pitch_decay_seconds.max(1e-4);
    let amp_decay = params.amp_decay_seconds.max(1e-4);
    let partial_decay = params.partial_decay_seconds.max(1e-4);
    let noise_decay = params.noise_decay_seconds.max(1e-4);
    let transient_decay = params.transient_decay_seconds.max(1e-4);
    let transient_freq = params.transient_frequency_hz.max(1.0);
    let ring_freq = partials.first().map(|p| p.freq * 2.7).unwrap_or(300.0);

    let mut left = vec![0.0f32; frames];
    let mut right = vec![0.0f32; frames];
    let mut phases = vec![0.0f32; partials.len()];
    let mut noise_l = NoiseGen::new(params.noise_color, 0x9E37_79B9);
    let mut noise_r = NoiseGen::new(params.noise_color, 0x85EB_CA6B);

    for i in 0..frames {
        let t = i as f32 * inv_sr;
        // Exponential pitch envelope: value are semitone offsets from the
        // partial's own frequency, so 0 is the fused pitch.
        let semis = params.pitch_end_semitones
            + (params.pitch_start_semitones - params.pitch_end_semitones)
                * (-t / pitch_decay).exp();
        let pitch_ratio = (semis / 12.0).exp2();
        let amp_env = (1.0 - (-t / 0.0008).exp()) * (-t / amp_decay).exp();

        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for (k, partial) in partials.iter().enumerate() {
            phases[k] += TAU * partial.freq * pitch_ratio * inv_sr;
            let decay = (-t / (partial_decay * partial.decay_scale)).exp();
            let amp = partial.amp * decay * body;
            let spread = decorrelation_spread(partial.bin) * width;
            l += amp * phases[k].sin();
            r += amp * (phases[k] + spread).sin();
        }

        if digital > 0.0 {
            let ring = (TAU * ring_freq * t).sin();
            let ring_mod = (1.0 - digital) + digital * ring;
            l *= ring_mod;
            r *= ring_mod;
        }

        let transient =
            transient_amount * (-t / transient_decay).exp() * (TAU * transient_freq * t).sin();
        l += transient;
        r += transient;

        let noise_env = (-t / noise_decay).exp();
        l += noise_l.sample() * noise_amount * noise_env;
        r += noise_r.sample() * noise_amount * noise_env;

        left[i] = l * amp_env;
        right[i] = r * amp_env;
    }

    let mut channels = vec![left, right];
    let drive = (params.drive_amount_pct / 100.0).clamp(0.0, 1.0);
    if drive > 0.0 {
        for sample in channels.iter_mut().flatten() {
            *sample = apply_drive(*sample, drive);
        }
    }
    compress_channels(
        &mut channels,
        sample_rate,
        (params.compress_amount_pct / 100.0).clamp(0.0, 1.0),
    );
    peak_normalize(&mut channels, 1.0);

    LoopBufferData {
        channels,
        sample_rate,
        root_note: input.root_note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fused_test_input() -> LoopBufferData {
        let sample_rate = 48_000.0;
        let frames = (sample_rate * 0.5) as usize;
        let left: Vec<f32> = (0..frames)
            .map(|i| (TAU * 70.0 * i as f32 / sample_rate).sin() * 0.8)
            .collect();
        let right = left.clone();
        LoopBufferData {
            channels: vec![left, right],
            sample_rate,
            root_note: 60,
        }
    }

    /// Fraction of spectral energy below `cutoff_hz` in the first frame.
    fn low_energy_ratio(buffer: &LoopBufferData, cutoff_hz: f32) -> f32 {
        let fft = FreezeFft::new();
        let window = sine_window(fft.size());
        let analysis = analyze_frame(&buffer.channels[0], 0, &fft, &window);
        let cutoff_bin = ((cutoff_hz * fft.size() as f32 / buffer.sample_rate).round() as usize)
            .min(analysis.mag.len());
        let total: f32 = analysis.mag.iter().map(|m| m * m).sum();
        let low: f32 = analysis.mag[..cutoff_bin].iter().map(|m| m * m).sum();
        low / total.max(1e-12)
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn output_has_requested_length_and_root_note() {
        let input = fused_test_input();
        let params = PercussionParams {
            length_seconds: 0.25,
            ..kick_preset()
        };
        let out = render_percussion(&input, &params);
        assert_eq!(out.channels.len(), 2);
        assert_eq!(out.root_note, input.root_note);
        assert_eq!(out.channels[0].len(), (0.25 * input.sample_rate) as usize);
        assert_eq!(out.channels[0].len(), out.channels[1].len());
    }

    #[test]
    fn kick_has_strong_low_frequency_energy() {
        let input = fused_test_input();
        let out = render_percussion(&input, &kick_preset());
        assert!(
            low_energy_ratio(&out, 150.0) > 0.4,
            "kick should be dominated by low frequencies, got {}",
            low_energy_ratio(&out, 150.0)
        );
    }

    #[test]
    fn kick_decays_to_near_silence_within_its_length() {
        let input = fused_test_input();
        let out = render_percussion(&input, &kick_preset());
        let channel = &out.channels[0];
        let window = (channel.len() / 20).max(1);
        let head = rms(&channel[..window]);
        let tail = rms(&channel[channel.len() - window..]);
        assert!(head > 0.0);
        assert!(
            tail < head * 0.05,
            "tail RMS {} should be far below head {}",
            tail,
            head
        );
    }

    #[test]
    fn snare_is_more_broadband_than_kick() {
        let input = fused_test_input();
        let kick = render_percussion(&input, &kick_preset());
        let snare = render_percussion(&input, &snare_preset());
        let kick_low = low_energy_ratio(&kick, 150.0);
        let snare_low = low_energy_ratio(&snare, 150.0);
        assert!(
            snare_low < kick_low,
            "snare low-band ratio {} should be below kick's {}",
            snare_low,
            kick_low
        );
    }

    #[test]
    fn every_preset_is_finite_bounded_and_reproducible() {
        let input = fused_test_input();
        for params in [
            kick_preset(),
            snare_preset(),
            metal_preset(),
            hat_preset(),
        ] {
            let a = render_percussion(&input, &params);
            let b = render_percussion(&input, &params);
            assert_eq!(a.channels, b.channels, "renders must be deterministic");
            for sample in a.channels.iter().flatten() {
                assert!(sample.is_finite(), "non-finite sample {}", sample);
                assert!(sample.abs() <= 1.0 + 1e-4, "sample {} out of range", sample);
            }
            let peak = a.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
            assert!(peak > 0.5, "peak-normalized output should be loud, got {}", peak);
        }
    }

    #[test]
    fn empty_input_still_produces_finite_output() {
        let input = LoopBufferData {
            channels: vec![Vec::new(), Vec::new()],
            sample_rate: 44_100.0,
            root_note: 60,
        };
        let out = render_percussion(&input, &hat_preset());
        assert!(!out.channels[0].is_empty());
        assert!(out.channels.iter().flatten().all(|s| s.is_finite()));
    }

    /// Fraction of spectral energy above `cutoff_hz` in the first frame.
    fn high_energy_ratio(buffer: &[f32], cutoff_hz: f32, sample_rate: f32) -> f32 {
        let fft = FreezeFft::new();
        let window = sine_window(fft.size());
        let analysis = analyze_frame(buffer, 0, &fft, &window);
        let cutoff_bin =
            ((cutoff_hz * fft.size() as f32 / sample_rate).round() as usize).min(analysis.mag.len());
        let total: f32 = analysis.mag.iter().map(|m| m * m).sum();
        let high: f32 = analysis.mag[cutoff_bin..].iter().map(|m| m * m).sum();
        high / total.max(1e-12)
    }

    fn peak_to_rms(channels: &[Vec<f32>]) -> f32 {
        let peak = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
        let rms = rms(&channels.concat());
        peak / rms.max(1e-12)
    }

    #[test]
    fn drive_adds_harmonics_and_stays_bounded() {
        let sample_rate = 48_000.0;
        let n = FreezeFft::new().size() * 2;
        let clean: Vec<f32> = (0..n)
            .map(|i| (TAU * 220.0 * i as f32 / sample_rate).sin() * 0.7)
            .collect();
        let driven: Vec<f32> = clean.iter().map(|s| apply_drive(*s, 0.9)).collect();
        for s in &driven {
            assert!(s.is_finite(), "driven sample must be finite");
            assert!(s.abs() <= 1.0 + 1e-4, "driven sample {} out of range", s);
        }
        let clean_high = high_energy_ratio(&clean, 400.0, sample_rate);
        let driven_high = high_energy_ratio(&driven, 400.0, sample_rate);
        assert!(
            driven_high > clean_high * 5.0,
            "drive should add harmonics: clean {} vs driven {}",
            clean_high,
            driven_high
        );
    }

    #[test]
    fn compression_reduces_peak_to_rms_ratio() {
        // A loud, sparse transient over a quiet body: the compressor should
        // squash the peaks relative to the body.
        let mut body = vec![0.02f32; 48_000];
        body[0] = 1.0;
        body[16] = 0.9;
        body[32] = 0.8;
        let mut channels = vec![body.clone(), body];
        let before = peak_to_rms(&channels);
        compress_channels(&mut channels, 48_000.0, 0.9);
        let after = peak_to_rms(&channels);
        assert!(
            after < before,
            "compression should reduce peak-to-RMS: before {} after {}",
            before,
            after
        );
        assert!(channels.iter().flatten().all(|s| s.is_finite()));
    }

    #[test]
    fn drive_and_compression_are_off_by_default() {
        for params in [
            kick_preset(),
            snare_preset(),
            metal_preset(),
            hat_preset(),
        ] {
            assert_eq!(params.drive_amount_pct, 0.0);
            assert_eq!(params.compress_amount_pct, 0.0);
        }
    }
}
