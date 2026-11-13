//! A 4-operator FM synth used as an alternative Source A: its rendered
//! buffer stands in for a loaded sample and runs through the rest of the
//! chain unchanged.
//!
//! Routing is a free matrix: every operator can modulate every operator
//! (including itself - feedback - and in cycles). To make any routing
//! computable, each operator reads the others' outputs from the *previous*
//! sample; feedback reads the average of its last two outputs, the usual
//! trick that keeps strong feedback from hunting. Operator settings are
//! static for the length of the render - movement comes from later stages.

use std::f64::consts::TAU;

/// Number of operators.
pub const OPS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpWave {
    Sine,
    Square,
    Triangle,
    Saw,
    /// White noise; ignores incoming modulation (it has no phase).
    Noise,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FmOperator {
    pub wave: OpWave,
    /// Frequency ratio coarse step: 0 = x0.5, 1..=16 = x1..x16.
    pub coarse: u8,
    /// Fine ratio offset in percent of the coarse ratio (0..99).
    pub fine_pct: f32,
    /// Detune in cents (-50..50).
    pub detune_cents: f32,
    /// How much of this operator is heard (0..100 %).
    pub output_pct: f32,
    /// How strongly each operator modulates this one (0..100 %);
    /// `mod_from[self]` is feedback.
    pub mod_from: [f32; OPS],
}

impl FmOperator {
    pub const fn sine(coarse: u8, output_pct: f32) -> Self {
        Self { wave: OpWave::Sine, coarse, fine_pct: 0.0, detune_cents: 0.0, output_pct, mod_from: [0.0; OPS] }
    }

    /// Effective frequency ratio to the fundamental (detune excluded).
    pub fn ratio(&self) -> f64 {
        let coarse = if self.coarse == 0 { 0.5 } else { self.coarse.min(16) as f64 };
        coarse * (1.0 + self.fine_pct.clamp(0.0, 99.0) as f64 / 100.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FmSynthParams {
    pub ops: [FmOperator; OPS],
    /// Rendered length (rounded to whole periods of the fundamental).
    pub length_seconds: f32,
    /// L/R detune spread in cents (L down, R up by half each).
    pub stereo_spread_cents: f32,
}

impl Default for FmSynthParams {
    /// A simple 2-operator patch: op 2 (x1) modulating op 1 (x1) at 25 %.
    fn default() -> Self {
        let mut op1 = FmOperator::sine(1, 100.0);
        op1.mod_from[1] = 25.0;
        Self { ops: [op1, FmOperator::sine(1, 0.0), FmOperator::sine(2, 0.0), FmOperator::sine(3, 0.0)], length_seconds: 2.0, stereo_spread_cents: 0.0 }
    }
}

impl FmSynthParams {
    /// Operators that are heard or modulate a heard path at all (non-zero
    /// output or modulation into another operator).
    pub fn active_ops(&self) -> usize {
        (0..OPS).filter(|&j| self.ops[j].output_pct > 0.0 || (0..OPS).any(|i| i != j && self.ops[i].mod_from[j] > 0.0)).count()
    }
}

/// Matrix percentage -> modulation index in radians.
pub fn mod_index(pct: f32) -> f64 {
    let x = (pct.clamp(0.0, 100.0) / 100.0) as f64;
    x * x * 10.0
}

/// Frequency of `root_note` in Hz.
pub fn note_hz(root_note: u8) -> f64 {
    440.0 * 2f64.powf((root_note as f64 - 69.0) / 12.0)
}

/// PolyBLEP residual for a discontinuity at phase 0, `dt` = phase step.
fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

struct Xorshift(u32);

impl Xorshift {
    fn next_bipolar(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f64 / u32::MAX as f64) * 2.0 - 1.0
    }
}

/// One channel of `frames` samples with every operator detuned by an extra
/// `side_cents`. Not normalised.
fn render_channel(params: &FmSynthParams, sample_rate: f64, f0: f64, side_cents: f64, frames: usize) -> Vec<f32> {
    let mut inc = [0.0f64; OPS];
    let mut index = [[0.0f64; OPS]; OPS];
    let mut level = [0.0f64; OPS];
    for (i, op) in params.ops.iter().enumerate() {
        let cents = op.detune_cents.clamp(-50.0, 50.0) as f64 + side_cents;
        inc[i] = (f0 * op.ratio() * 2f64.powf(cents / 1200.0) / sample_rate).min(0.5);
        for (slot, &pct) in index[i].iter_mut().zip(&op.mod_from) {
            *slot = mod_index(pct);
        }
        level[i] = (op.output_pct.clamp(0.0, 100.0) / 100.0) as f64;
    }
    let mut phase = [0.0f64; OPS];
    let mut prev_t = [0.0f64; OPS];
    let mut out1 = [0.0f64; OPS]; // previous sample's outputs
    let mut out2 = [0.0f64; OPS]; // the one before
    let mut rng: [Xorshift; OPS] = std::array::from_fn(|i| Xorshift(0x9E37_79B9 ^ (i as u32 + 1).wrapping_mul(0x85EB_CA6B)));
    let mut buf = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut now = [0.0f64; OPS];
        for i in 0..OPS {
            let mut pm = 0.0;
            for j in 0..OPS {
                if index[i][j] != 0.0 {
                    let src = if i == j { 0.5 * (out1[j] + out2[j]) } else { out1[j] };
                    pm += index[i][j] * src;
                }
            }
            let t = (phase[i] + pm / TAU).rem_euclid(1.0);
            // Effective phase step this sample (modulation included), for PolyBLEP.
            let step = (t - prev_t[i]).rem_euclid(1.0);
            let dt = step.min(1.0 - step).clamp(1e-6, 0.5);
            prev_t[i] = t;
            now[i] = match params.ops[i].wave {
                OpWave::Sine => (TAU * t).sin(),
                OpWave::Triangle => 1.0 - 4.0 * (t - 0.5).abs(),
                OpWave::Saw => 2.0 * t - 1.0 - poly_blep(t, dt),
                OpWave::Square => {
                    let naive = if t < 0.5 { 1.0 } else { -1.0 };
                    naive + poly_blep(t, dt) - poly_blep((t + 0.5).rem_euclid(1.0), dt)
                }
                OpWave::Noise => rng[i].next_bipolar(),
            };
            phase[i] = (phase[i] + inc[i]).rem_euclid(1.0);
        }
        out2 = out1;
        out1 = now;
        buf.push((0..OPS).map(|i| level[i] * now[i]).sum::<f64>() as f32);
    }
    buf
}

/// Loudness the rendered synth is brought to (RMS), about where a typical
/// peak-normalised sample sits. A sustained FM tone normalised to full-scale
/// *peak* is roughly twice as loud as a sample, which drove the voice
/// engine's safety limiter into audible distortion on even a single note.
pub const TARGET_RMS: f32 = 0.25;

/// Scales to `TARGET_RMS`, never letting the peak exceed full scale.
fn normalize_loudness(channels: &mut [Vec<f32>]) {
    let peak = channels.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    let count = channels.iter().map(Vec::len).sum::<usize>().max(1);
    let rms = (channels.iter().flatten().map(|v| v * v).sum::<f32>() / count as f32).sqrt();
    if peak > 1e-6 && peak.is_finite() && rms > 1e-9 {
        let scale = (TARGET_RMS / rms).min(1.0 / peak);
        for v in channels.iter_mut().flatten() {
            *v *= scale;
        }
    }
}

/// Peak-normalises (for the GUI preview, where only the shape matters).
fn normalize_peak(channels: &mut [Vec<f32>]) {
    let peak = channels.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > 1e-6 && peak.is_finite() {
        for v in channels.iter_mut().flatten() {
            *v /= peak;
        }
    }
}

/// Frames for `cycles` whole periods of `f0`.
fn frames_for_cycles(cycles: f64, f0: f64, sample_rate: f64) -> usize {
    ((cycles * sample_rate / f0).round() as usize).max(1)
}

/// Renders the synth as a stereo, loudness-normalised (`TARGET_RMS`) buffer pitched so the
/// fundamental is `root_note`.
pub fn render_fm_synth(params: &FmSynthParams, sample_rate: f32, root_note: u8) -> Vec<Vec<f32>> {
    let sr = sample_rate.max(1.0) as f64;
    let f0 = note_hz(root_note);
    let cycles = (params.length_seconds.clamp(0.05, 60.0) as f64 * f0).round().max(1.0);
    let frames = frames_for_cycles(cycles, f0, sr);
    let half = params.stereo_spread_cents.clamp(0.0, 100.0) as f64 / 2.0;
    let mut out = vec![render_channel(params, sr, f0, -half, frames), render_channel(params, sr, f0, half, frames)];
    normalize_loudness(&mut out);
    out
}

/// A few periods (mono, normalised) for a GUI waveform preview.
pub fn render_fm_cycles(params: &FmSynthParams, sample_rate: f32, root_note: u8, cycles: u32) -> Vec<f32> {
    let sr = sample_rate.max(1.0) as f64;
    let f0 = note_hz(root_note);
    let frames = frames_for_cycles(cycles.max(1) as f64, f0, sr);
    // Settle feedback paths first so the preview shows the steady state.
    let warmup = frames_for_cycles(8.0, f0, sr);
    let mut buf = render_channel(params, sr, f0, 0.0, warmup + frames).split_off(warmup);
    normalize_peak(std::slice::from_mut(&mut buf));
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn zero_crossings(x: &[f32]) -> usize {
        x.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count()
    }

    fn solo(op: FmOperator) -> FmSynthParams {
        FmSynthParams { ops: [op, FmOperator::sine(1, 0.0), FmOperator::sine(1, 0.0), FmOperator::sine(1, 0.0)], length_seconds: 1.0, stereo_spread_cents: 0.0 }
    }

    /// Rough "brightness": mean absolute first difference.
    fn roughness(x: &[f32]) -> f32 {
        x.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / x.len() as f32
    }

    #[test]
    fn single_sine_operator_plays_the_root_fundamental() {
        let out = render_fm_synth(&solo(FmOperator::sine(1, 100.0)), SR, 60);
        let secs = out[0].len() as f64 / SR as f64;
        let hz = zero_crossings(&out[0]) as f64 / secs;
        assert!((hz - note_hz(60)).abs() < 2.0, "measured {hz} Hz");
        assert_eq!(out[0], out[1], "no spread -> identical channels");
    }

    #[test]
    fn length_is_whole_cycles_of_the_fundamental() {
        let out = render_fm_synth(&FmSynthParams::default(), SR, 60);
        let cycles = out[0].len() as f64 * note_hz(60) / SR as f64;
        assert!((cycles - cycles.round()).abs() < 0.02, "{cycles} cycles");
    }

    /// Magnitude of the DFT bin at `hz`, normalised by length.
    fn bin(x: &[f32], hz: f64) -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (n, &v) in x.iter().enumerate() {
            let w = TAU * hz * n as f64 / SR as f64;
            re += v as f64 * w.cos();
            im += v as f64 * w.sin();
        }
        (re * re + im * im).sqrt() / x.len() as f64
    }

    #[test]
    fn modulation_adds_harmonics() {
        let f0 = note_hz(60);
        let plain = render_fm_synth(&solo(FmOperator::sine(1, 100.0)), SR, 60);
        assert!(bin(&plain[0], 2.0 * f0) < 0.01);
        // Even the gentle default patch puts energy in the 2nd harmonic.
        let gentle = render_fm_synth(&FmSynthParams { length_seconds: 1.0, ..FmSynthParams::default() }, SR, 60);
        assert!(bin(&gentle[0], 2.0 * f0) > 0.05, "{}", bin(&gentle[0], 2.0 * f0));
        let mut strong = FmSynthParams { length_seconds: 1.0, ..FmSynthParams::default() };
        strong.ops[0].mod_from[1] = 75.0;
        let strong = render_fm_synth(&strong, SR, 60);
        assert!(roughness(&strong[0]) > roughness(&plain[0]) * 1.5);
    }

    #[test]
    fn strong_feedback_and_cycles_stay_finite_and_normalised() {
        let mut p = FmSynthParams::default();
        for i in 0..OPS {
            p.ops[i].mod_from = [100.0; OPS];
            p.ops[i].output_pct = 100.0;
        }
        let out = render_fm_synth(&p, SR, 60);
        let peak = out.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(out.iter().flatten().all(|v| v.is_finite()));
        assert!(peak <= 1.0 + 1e-6 && peak > 0.1, "peak {peak}");
    }

    #[test]
    fn every_waveform_renders() {
        for wave in [OpWave::Sine, OpWave::Square, OpWave::Triangle, OpWave::Saw, OpWave::Noise] {
            let out = render_fm_synth(&solo(FmOperator { wave, ..FmOperator::sine(2, 100.0) }), SR, 60);
            assert!(out[0].iter().all(|v| v.is_finite()), "{wave:?}");
            assert!(out[0].iter().any(|v| v.abs() > 0.2), "{wave:?} silent");
        }
    }

    #[test]
    fn stereo_spread_separates_the_channels() {
        let p = FmSynthParams { stereo_spread_cents: 20.0, ..FmSynthParams::default() };
        let out = render_fm_synth(&p, SR, 60);
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn output_sits_at_sample_like_loudness_with_peak_headroom() {
        for p in [FmSynthParams::default(), solo(FmOperator { wave: OpWave::Square, ..FmOperator::sine(1, 100.0) })] {
            let out = render_fm_synth(&p, SR, 60);
            let all: Vec<f32> = out.concat();
            let rms = (all.iter().map(|v| v * v).sum::<f32>() / all.len() as f32).sqrt();
            let peak = all.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!((rms - TARGET_RMS).abs() < 0.01, "rms {rms}");
            assert!(peak < 0.75, "peak {peak}");
        }
    }

    #[test]
    fn all_outputs_off_is_silent() {
        let out = render_fm_synth(&solo(FmOperator::sine(1, 0.0)), SR, 60);
        assert!(out.iter().flatten().all(|&v| v == 0.0));
    }

    #[test]
    fn ratio_and_active_op_count() {
        assert_eq!(FmOperator::sine(0, 0.0).ratio(), 0.5);
        assert_eq!(FmOperator { fine_pct: 50.0, ..FmOperator::sine(2, 0.0) }.ratio(), 3.0);
        assert_eq!(FmSynthParams::default().active_ops(), 2);
    }

    #[test]
    fn preview_is_the_requested_number_of_cycles() {
        let buf = render_fm_cycles(&FmSynthParams::default(), SR, 60, 3);
        assert_eq!(buf.len(), (3.0 * SR as f64 / note_hz(60)).round() as usize);
    }
}
