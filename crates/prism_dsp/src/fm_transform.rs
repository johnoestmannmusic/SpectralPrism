//! FM transform (tracker-only): the source sample becomes an FM *operator*.
//!
//! A sine carrier at the render root (MIDI 60 x `carrier_ratio`) is phase-
//! modulated by the (peak-normalised) sample itself, optionally through a
//! second sine operator - the ShrineGen 0016 three-operator layout
//! (`fm.rs`: Stack 3->2->1 / Pair 2->1<-3) with the sample standing in for
//! operator 3. The result is pitched (it follows the tracker note like any
//! instrument) but carries the sample's own motion and grit as sidebands:
//! metallic blips, bells and growls made from your own recordings.
//!
//! - `follow` makes the carrier's amplitude track the sample's envelope, so a
//!   mechanical hit becomes a pitched blip with the hit's own shape.
//! - `index_sway_*` slowly breathes the modulation depth (0016 sways each
//!   drone voice's index the same way).
//! - `stereo_detune_cents` spreads L/R carriers apart for width.
//! - `hold` / `levels` are the 0016 Blip/Crush lo-fi: sample-and-hold every
//!   `hold` samples, then quantise to `levels` steps.
//!
//! Deterministic; output is always stereo and the same length as the input.

use std::f64::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmAlgorithm {
    /// sample -> carrier.
    Direct,
    /// sample -> op2 -> carrier.
    Stack,
    /// op2 -> carrier <- sample.
    Pair,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FmTransformParams {
    pub algorithm: FmAlgorithm,
    /// Carrier frequency as a multiple of the root note (0.25-8).
    pub carrier_ratio: f32,
    /// How hard the sample modulates (radians of phase per unit sample).
    pub index: f32,
    /// Op2 frequency as a multiple of the root note (0.25-8).
    pub op2_ratio: f32,
    /// Op2's modulation depth into the carrier (radians).
    pub op2_index: f32,
    /// Op2 self-feedback (radians).
    pub op2_feedback: f32,
    /// Index sway LFO rate (Hz, 0 = off) and depth (0-1).
    pub index_sway_hz: f32,
    pub index_sway_depth: f32,
    /// 0 = steady carrier, 1 = carrier amplitude follows the sample.
    pub follow: f32,
    /// L/R carrier detune spread in cents (L down, R up by half each).
    pub stereo_detune_cents: f32,
    /// Dry sample vs FM (0-100 %).
    pub mix_pct: f32,
    /// Sample-and-hold length in samples (1 = off).
    pub hold: u32,
    /// Quantise levels (0/1 = off, else 2-64).
    pub levels: u32,
}

impl Default for FmTransformParams {
    fn default() -> Self {
        Self {
            algorithm: FmAlgorithm::Stack,
            carrier_ratio: 1.0,
            index: 2.0,
            op2_ratio: 2.0,
            op2_index: 1.0,
            op2_feedback: 0.0,
            index_sway_hz: 0.0,
            index_sway_depth: 0.3,
            follow: 1.0,
            stereo_detune_cents: 0.0,
            mix_pct: 100.0,
            hold: 1,
            levels: 0,
        }
    }
}

pub const MAX_RATIO: f32 = 8.0;
pub const MIN_RATIO: f32 = 0.25;
pub const MAX_INDEX: f32 = 12.0;
pub const MAX_OP2_INDEX: f32 = 6.0;
pub const MAX_FEEDBACK: f32 = 1.5;
pub const MAX_HOLD: u32 = 48;
pub const MAX_LEVELS: u32 = 64;

fn midi_hz(note: f64) -> f64 {
    440.0 * 2f64.powf((note - 69.0) / 12.0)
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Applies the FM transform to `channels` (one Vec per channel, mono or
/// stereo) and returns exactly two output channels.
pub fn render_fm_transform(channels: &[Vec<f32>], sample_rate: f32, root_note: u8, params: &FmTransformParams) -> Vec<Vec<f32>> {
    let left_in: &[f32] = channels.first().map(Vec::as_slice).unwrap_or(&[]);
    let right_in: &[f32] = channels.get(1).map(Vec::as_slice).unwrap_or(left_in);
    let len = left_in.len().min(right_in.len());
    if len == 0 || !(sample_rate > 0.0) {
        return vec![Vec::new(), Vec::new()];
    }
    let sr = sample_rate as f64;

    let carrier_ratio = finite_or(params.carrier_ratio, 1.0).clamp(MIN_RATIO, MAX_RATIO) as f64;
    let op2_ratio = finite_or(params.op2_ratio, 2.0).clamp(MIN_RATIO, MAX_RATIO) as f64;
    let index = finite_or(params.index, 0.0).clamp(0.0, MAX_INDEX) as f64;
    let op2_index = finite_or(params.op2_index, 0.0).clamp(0.0, MAX_OP2_INDEX) as f64;
    let feedback = finite_or(params.op2_feedback, 0.0).clamp(0.0, MAX_FEEDBACK) as f64;
    let sway_hz = finite_or(params.index_sway_hz, 0.0).clamp(0.0, 20.0) as f64;
    let sway_depth = finite_or(params.index_sway_depth, 0.0).clamp(0.0, 1.0) as f64;
    let follow = finite_or(params.follow, 1.0).clamp(0.0, 1.0);
    let detune = finite_or(params.stereo_detune_cents, 0.0).clamp(0.0, 100.0) as f64;
    let mix = (finite_or(params.mix_pct, 100.0) / 100.0).clamp(0.0, 1.0);
    let hold = params.hold.clamp(1, MAX_HOLD) as usize;
    let levels = if params.levels >= 2 { params.levels.min(MAX_LEVELS) } else { 0 };

    // Peak-normalise the modulator so `index` means the same thing for any
    // recording level.
    let peak = left_in[..len].iter().chain(right_in[..len].iter()).fold(0.0f32, |m, &s| m.max(s.abs()));
    let norm = if peak > 1e-9 { 1.0 / peak } else { 0.0 };

    // Shared (mono) envelope so both channels breathe together.
    let attack = (-1.0 / (0.002 * sr)).exp() as f32;
    let release = (-1.0 / (0.030 * sr)).exp() as f32;
    let mut env = vec![0.0f32; len];
    let mut e = 0.0f32;
    for i in 0..len {
        let x = left_in[i].abs().max(right_in[i].abs()) * norm;
        let c = if x > e { attack } else { release };
        e = c * e + (1.0 - c) * x;
        env[i] = e;
    }
    let env_peak = env.iter().fold(0.0f32, |m, &v| m.max(v));
    if env_peak > 1e-9 {
        env.iter_mut().for_each(|v| *v /= env_peak);
    }

    // 5 ms edge fades on the FM part, so a steady carrier never clicks at the
    // ends of the slice.
    let fade = ((0.005 * sr) as usize).clamp(1, (len / 2).max(1));

    let base_hz = midi_hz(root_note as f64);
    let render_channel = |input: &[f32], cents: f64| -> Vec<f32> {
        let carrier_hz = base_hz * carrier_ratio * 2f64.powf(cents / 1200.0);
        let op2_hz = base_hz * op2_ratio * 2f64.powf(cents / 1200.0);
        let inc1 = carrier_hz / sr;
        let inc2 = op2_hz / sr;
        let mut phase1 = 0.0f64;
        let mut phase2 = 0.0f64;
        let mut last2 = 0.0f64;
        let mut out = vec![0.0f32; len];
        for i in 0..len {
            let m = (input[i] * norm) as f64;
            let t = i as f64 / sr;
            let sway = if sway_hz > 0.0 { 1.0 - sway_depth * (0.5 - 0.5 * (TAU * sway_hz * t).sin()) } else { 1.0 };
            let carrier_pm = match params.algorithm {
                FmAlgorithm::Direct => index * sway * m,
                FmAlgorithm::Stack => {
                    let o2 = (TAU * phase2 + index * sway * m + feedback * last2).sin();
                    last2 = o2;
                    op2_index * sway * o2
                }
                FmAlgorithm::Pair => {
                    let o2 = (TAU * phase2 + feedback * last2).sin();
                    last2 = o2;
                    index * sway * m + op2_index * sway * o2
                }
            };
            let fm = (TAU * phase1 + carrier_pm).sin() as f32;
            phase1 = (phase1 + inc1).fract();
            phase2 = (phase2 + inc2).fract();

            let amp = (1.0 - follow) + follow * env[i];
            let edge = (i.min(len - 1 - i) as f32 / fade as f32).min(1.0);
            let wet = fm * amp * edge;
            out[i] = (1.0 - mix) * (input[i] * norm) + mix * wet;
        }
        lofi(&mut out, hold, levels);
        out
    };

    vec![render_channel(&left_in[..len], -detune / 2.0), render_channel(&right_in[..len], detune / 2.0)]
}

/// Sample-and-hold every `hold` samples, then quantise to `levels` steps.
fn lofi(signal: &mut [f32], hold: usize, levels: u32) {
    if hold > 1 {
        let mut held = 0.0f32;
        for (i, s) in signal.iter_mut().enumerate() {
            if i % hold == 0 {
                held = *s;
            }
            *s = held;
        }
    }
    if levels >= 2 {
        let l = levels as f32;
        signal.iter_mut().for_each(|s| *s = (*s * l).round() / l);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44_100.0;

    fn noise(len: usize) -> Vec<f32> {
        let mut state = 12345u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Dominant frequency by zero-crossing count over the middle of `x`.
    fn zero_cross_hz(x: &[f32]) -> f32 {
        let mid = &x[x.len() / 4..3 * x.len() / 4];
        let crossings = mid.windows(2).filter(|w| (w[0] <= 0.0) != (w[1] <= 0.0)).count();
        crossings as f32 / 2.0 / (mid.len() as f32 / SR)
    }

    #[test]
    fn zero_index_is_a_pure_carrier_at_root_times_ratio() {
        let params = FmTransformParams { algorithm: FmAlgorithm::Direct, index: 0.0, follow: 0.0, carrier_ratio: 2.0, ..Default::default() };
        let out = render_fm_transform(&[noise(SR as usize)], SR, 60, &params);
        assert_eq!(out.len(), 2);
        let hz = zero_cross_hz(&out[0]);
        let expected = 261.6256 * 2.0;
        assert!((hz - expected).abs() < 3.0, "carrier {} Hz, expected {}", hz, expected);
        assert_eq!(out[0], out[1], "no detune: L == R");
    }

    #[test]
    fn modulation_changes_the_signal_and_stays_bounded() {
        let source = vec![noise(SR as usize / 2)];
        for algorithm in [FmAlgorithm::Direct, FmAlgorithm::Stack, FmAlgorithm::Pair] {
            let plain = render_fm_transform(&source, SR, 60, &FmTransformParams { algorithm, index: 0.0, op2_index: 0.0, ..Default::default() });
            let fm = render_fm_transform(&source, SR, 60, &FmTransformParams { algorithm, index: 4.0, op2_index: 2.0, op2_feedback: 0.7, index_sway_hz: 0.3, ..Default::default() });
            assert_ne!(plain[0], fm[0], "{:?}", algorithm);
            assert!(fm[0].iter().all(|s| s.is_finite() && s.abs() <= 1.0 + 1e-6), "{:?}", algorithm);
        }
    }

    #[test]
    fn follow_tracks_the_source_envelope() {
        // A burst then silence: with follow = 1 the tail must be ~silent.
        let mut source = noise(SR as usize);
        for s in source.iter_mut().skip(SR as usize / 4) {
            *s = 0.0;
        }
        let out = render_fm_transform(&[source], SR, 60, &FmTransformParams::default());
        let head = out[0][1000..5000].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let tail = out[0][(SR as usize * 3) / 4..].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(head > 0.3, "head {}", head);
        assert!(tail < 1e-3, "tail {}", tail);
    }

    #[test]
    fn mix_zero_returns_the_normalised_dry_sample() {
        let source = noise(4096).into_iter().map(|s| s * 0.25).collect::<Vec<_>>();
        let out = render_fm_transform(&[source.clone()], SR, 60, &FmTransformParams { mix_pct: 0.0, ..Default::default() });
        let peak = source.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        for (o, s) in out[0].iter().zip(source.iter()) {
            assert!((o - s / peak).abs() < 1e-5);
        }
    }

    #[test]
    fn hold_and_levels_quantise() {
        let source = noise(4096);
        let out = render_fm_transform(&[source], SR, 60, &FmTransformParams { mix_pct: 0.0, hold: 6, levels: 3, ..Default::default() });
        let l = &out[0];
        for chunk in l.chunks(6) {
            assert!(chunk.iter().all(|&s| s == chunk[0]), "held runs of 6");
        }
        for &s in l {
            let steps = s * 3.0;
            assert!((steps - steps.round()).abs() < 1e-5, "{} is not on a 1/3 grid", s);
        }
    }

    #[test]
    fn detune_spreads_the_channels() {
        let out = render_fm_transform(&[noise(8192)], SR, 60, &FmTransformParams { stereo_detune_cents: 14.0, ..Default::default() });
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn empty_input_is_empty_stereo() {
        let out = render_fm_transform(&[Vec::new()], SR, 60, &FmTransformParams::default());
        assert_eq!(out, vec![Vec::<f32>::new(), Vec::new()]);
    }
}
