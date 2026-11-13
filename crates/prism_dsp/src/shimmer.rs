//! Shimmer freeze (tracker-only): the ShrineGen 0016 "Glitch Ambient" drone
//! freeze, as an alternative to the phase-locked vocoder freeze.
//!
//! Where the classic freeze captures ONE frame and keeps every bin's phase
//! advancing at its measured rate (a steady, glassy hold), Shimmer:
//! - **averages** the power spectrum over a span of the source
//!   (`span_pct` of the sample, centred on the freeze point; 100% = the
//!   whole sample), so the frozen colour is the sample's overall timbre
//!   rather than one instant of it, and zeroes bins below `low_cut_hz`;
//! - **resynthesises** with a fresh uniform random phase for every bin on
//!   every hop (independently per channel, blended by Stereo Width), with a
//!   Hann synthesis window at 4x overlap. The magnitude never changes but the
//!   fine structure is always new, which is what makes the pad shimmer and
//!   never sound static or phasey.
//!
//! Output is circularly overlap-added over the loop, so the buffer loops
//! seamlessly (every frame is independent; the last frame's tail simply
//! wraps onto the start). Fully deterministic for a given `seed`.

use realfft::num_complex::Complex32;

use crate::analysis::analyze_frame;
use crate::fft::FreezeFft;
use crate::resynth::ola_accumulate_circular;
use crate::window::sine_window;
use std::f32::consts::{PI, TAU};

/// Default low cut, from 0016's drone (`drone.rs`: bins below 110 Hz zeroed).
pub const DEFAULT_LOW_CUT_HZ: f32 = 110.0;
pub const MAX_LOW_CUT_HZ: f32 = 1000.0;

/// Shimmer settings carried alongside the other freeze parameters
/// (`fusion::FusionRenderParams::shimmer`). `enabled == false` (the default)
/// leaves every render bit-identical to the classic freeze.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShimmerParams {
    pub enabled: bool,
    /// Part of the usable sample averaged, centred on the freeze point
    /// (0-100). 0 = the single freeze-point frame, 100 = the whole sample.
    pub span_pct: f32,
    /// Bins below this frequency are zeroed (0-1000 Hz).
    pub low_cut_hz: f32,
    /// Random-phase seed.
    pub seed: u32,
}

impl Default for ShimmerParams {
    fn default() -> Self {
        Self { enabled: false, span_pct: 100.0, low_cut_hz: DEFAULT_LOW_CUT_HZ, seed: 0 }
    }
}

/// Synthesis hop: a quarter window (4x overlap, as in 0016).
pub fn shimmer_hop(fft: &FreezeFft) -> usize {
    fft.size() / 4
}

/// Average magnitude of `signal` over `span_pct` of its usable length centred
/// on `center_pct` (both 0-100), using the same sine analysis window as the
/// classic freeze so levels are comparable. Power (|X|^2) is averaged, then
/// square-rooted - the RMS magnitude per bin. `gain_pct` (0-100, never a
/// boost) scales the result, matching `freeze::analyze_freeze_point`.
pub fn averaged_magnitude(signal: &[f32], center_pct: f32, span_pct: f32, gain_pct: f32, fft: &FreezeFft) -> Vec<f32> {
    let n = fft.size();
    let window = sine_window(n);
    let max_pos = signal.len().saturating_sub(n);
    let (start, end) = span_range(max_pos, center_pct, span_pct);
    let step = shimmer_hop(fft).max(1);

    let bins = fft.num_bins();
    let mut power = vec![0.0f64; bins];
    let mut frames = 0usize;
    let mut pos = start;
    loop {
        let frame = analyze_frame(signal, pos, fft, &window);
        for (p, &m) in power.iter_mut().zip(frame.mag.iter()) {
            *p += (m as f64) * (m as f64);
        }
        frames += 1;
        if pos >= end {
            break;
        }
        pos = (pos + step).min(end);
    }

    let gain = (gain_pct / 100.0).clamp(0.0, 1.0) as f64;
    power.iter().map(|&p| ((p / frames as f64).sqrt() * gain) as f32).collect()
}

/// `[start, end]` frame positions for a span of `span_pct` around
/// `center_pct`, clamped inside `0..=max_pos`.
fn span_range(max_pos: usize, center_pct: f32, span_pct: f32) -> (usize, usize) {
    let center = (center_pct.clamp(0.0, 100.0) / 100.0) * max_pos as f32;
    let half = (span_pct.clamp(0.0, 100.0) / 100.0) * max_pos as f32 / 2.0;
    // Keep the span's full width when it hits an edge, by sliding it inward.
    let mut start = center - half;
    let mut end = center + half;
    if start < 0.0 {
        end -= start;
        start = 0.0;
    }
    if end > max_pos as f32 {
        start -= end - max_pos as f32;
        end = max_pos as f32;
    }
    let start = start.max(0.0).round() as usize;
    let end = (end.round() as usize).min(max_pos).max(start);
    (start, end)
}

/// Zeroes every bin below `low_cut_hz`, plus DC and Nyquist.
pub fn apply_low_cut(mag: &mut [f32], low_cut_hz: f32, sample_rate: f32, fft_size: usize) {
    let cut_bin = ((low_cut_hz.clamp(0.0, MAX_LOW_CUT_HZ) * fft_size as f32 / sample_rate).ceil() as usize).max(1);
    let last = mag.len().saturating_sub(1);
    for (k, m) in mag.iter_mut().enumerate() {
        if k < cut_bin || k == last {
            *m = 0.0;
        }
    }
}

/// SplitMix64: tiny, deterministic, no dependency.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Periodic Hann window.
fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 - 0.5 * (TAU * i as f32 / n as f32).cos()).collect()
}

/// Random-phase overlap-add renderer. `frame_mags(frame, frames, mag_l,
/// mag_r)` fills the (already low-cut) magnitudes for synthesis frame
/// `frame` of `frames`; `frame_width(frame, frames)` returns the 0-1 stereo
/// decorrelation for that frame (0 = the right channel reuses the left's
/// random phases exactly, 1 = fully independent phases). `out_len` must be a
/// whole number of `shimmer_hop`s.
pub fn render_shimmer<M, W>(fft: &FreezeFft, out_len: usize, seed: u32, mut frame_mags: M, frame_width: W) -> (Vec<f32>, Vec<f32>)
where
    M: FnMut(usize, usize, &mut [f32], &mut [f32]),
    W: Fn(usize, usize) -> f32,
{
    let n = fft.size();
    let hop = shimmer_hop(fft);
    let bins = fft.num_bins();
    let window = hann(n);
    // Random-phase frames preserve energy (Parseval): a frame analysed with
    // the sine window carries half the source power per sample, and Hann^2 at
    // 4x overlap sums to 1.5, so the output power is 0.75x the source's.
    // sqrt(4/3) restores the source's own RMS.
    let gain = (4.0f32 / 3.0).sqrt();

    let frames = (out_len / hop).max(1);
    let mut rng_l = Rng::new(seed as u64 * 2 + 1);
    let mut rng_r = Rng::new(seed as u64 * 2 + 2);
    let mut mag_l = vec![0.0f32; bins];
    let mut mag_r = vec![0.0f32; bins];
    let mut spectrum = fft.make_spectrum_buffer();
    let mut time = fft.make_time_buffer();
    let mut accum_l = vec![0.0f32; out_len.max(1)];
    let mut accum_r = vec![0.0f32; out_len.max(1)];
    let mut phases = vec![0.0f32; bins];
    let last = bins - 1;

    for f in 0..frames {
        frame_mags(f, frames, &mut mag_l, &mut mag_r);
        let width = frame_width(f, frames).clamp(0.0, 1.0);

        for (k, p) in phases.iter_mut().enumerate() {
            *p = rng_l.unit() * TAU;
            spectrum[k] = bin_value(k, last, mag_l[k], *p);
        }
        fft.inverse(&mut spectrum, &mut time);
        window_and_scale(&mut time, &window, gain);
        ola_accumulate_circular(&mut accum_l, f * hop, &time);

        for (k, &p) in phases.iter().enumerate() {
            // Always draw, so L's sequence is independent of width.
            let offset = (rng_r.unit() - 0.5) * TAU * width;
            spectrum[k] = bin_value(k, last, mag_r[k], p + offset);
        }
        fft.inverse(&mut spectrum, &mut time);
        window_and_scale(&mut time, &window, gain);
        ola_accumulate_circular(&mut accum_r, f * hop, &time);
    }

    (accum_l, accum_r)
}

#[inline]
fn bin_value(k: usize, last: usize, mag: f32, phase: f32) -> Complex32 {
    if k == 0 || k == last {
        // DC / Nyquist must be purely real; low cut zeroes them anyway.
        Complex32::new(if phase < PI { mag } else { -mag }, 0.0)
    } else {
        Complex32::new(mag * phase.cos(), mag * phase.sin())
    }
}

fn window_and_scale(frame: &mut [f32], window: &[f32], gain: f32) {
    for (s, &w) in frame.iter_mut().zip(window.iter()) {
        *s *= w * gain;
    }
}

/// Convenience: render a static shimmer (one fixed magnitude pair) - the
/// non-modulated paths.
pub fn render_static_shimmer(fft: &FreezeFft, out_len: usize, seed: u32, mag_l: &[f32], mag_r: &[f32], width: f32) -> (Vec<f32>, Vec<f32>) {
    render_shimmer(
        fft,
        out_len,
        seed,
        |_, _, l, r| {
            l.copy_from_slice(mag_l);
            r.copy_from_slice(mag_r);
        },
        |_, _| width,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44_100.0;

    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.unit() * 2.0 - 1.0).collect()
    }

    fn tone(len: usize, hz: f32) -> Vec<f32> {
        (0..len).map(|i| (TAU * hz * i as f32 / SR).sin() * 0.5).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn span_range_is_clamped_and_keeps_width() {
        assert_eq!(span_range(1000, 50.0, 100.0), (0, 1000));
        assert_eq!(span_range(1000, 50.0, 0.0), (500, 500));
        // Pushed against the start edge, the span slides inward.
        assert_eq!(span_range(1000, 0.0, 20.0), (0, 200));
        assert_eq!(span_range(1000, 100.0, 20.0), (800, 1000));
        assert_eq!(span_range(0, 50.0, 100.0), (0, 0));
    }

    #[test]
    fn zero_span_matches_the_single_freeze_frame() {
        let fft = FreezeFft::with_size(2048);
        let signal = tone(SR as usize, 440.0);
        let avg = averaged_magnitude(&signal, 30.0, 0.0, 100.0, &fft);
        let single = crate::freeze::analyze_freeze_point(&signal, 30.0, 100.0, &fft);
        for (a, b) in avg.iter().zip(single.mag.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn low_cut_zeroes_low_bins() {
        let mut mag = vec![1.0f32; 1025];
        apply_low_cut(&mut mag, 110.0, SR, 2048);
        let cut_bin = (110.0f32 * 2048.0 / SR).ceil() as usize;
        assert!(mag[..cut_bin].iter().all(|&m| m == 0.0));
        assert_eq!(mag[cut_bin], 1.0);
        assert_eq!(mag[1024], 0.0);
    }

    #[test]
    fn deterministic_seamless_and_steady() {
        let fft = FreezeFft::with_size(2048);
        let signal = noise(SR as usize * 2, 7);
        let mut mag = averaged_magnitude(&signal, 50.0, 100.0, 100.0, &fft);
        apply_low_cut(&mut mag, 0.0, SR, 2048);
        let out_len = 2048 * 40;
        let (l1, r1) = render_static_shimmer(&fft, out_len, 3, &mag, &mag, 1.0);
        let (l2, _) = render_static_shimmer(&fft, out_len, 3, &mag, &mag, 1.0);
        let (l3, _) = render_static_shimmer(&fft, out_len, 4, &mag, &mag, 1.0);
        assert_eq!(l1, l2, "same seed must render identically");
        assert_ne!(l1, l3, "a different seed must render differently");
        assert_ne!(l1, r1, "width 1 must decorrelate L/R");

        // No attack: the first and last quarters are as loud as the middle.
        let q = out_len / 4;
        let head = rms(&l1[..q]);
        let mid = rms(&l1[q..3 * q]);
        let tail = rms(&l1[3 * q..]);
        assert!((head / mid - 1.0).abs() < 0.2, "head {} vs mid {}", head, mid);
        assert!((tail / mid - 1.0).abs() < 0.2, "tail {} vs mid {}", tail, mid);

        // Seamless wrap.
        let max_step = l1.windows(2).fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()));
        let seam = (l1[0] - l1[out_len - 1]).abs();
        assert!(seam <= max_step * 1.01 + 1e-6);

        // Level tracks the source's RMS (white noise is stationary).
        let ratio = mid / rms(&signal);
        assert!((ratio - 1.0).abs() < 0.25, "output/source rms ratio {}", ratio);
    }

    #[test]
    fn width_zero_is_mono() {
        let fft = FreezeFft::with_size(2048);
        let signal = tone(SR as usize, 330.0);
        let mag = averaged_magnitude(&signal, 50.0, 100.0, 100.0, &fft);
        let (l, r) = render_static_shimmer(&fft, 2048 * 8, 1, &mag, &mag, 0.0);
        assert_eq!(l, r);
    }

    #[test]
    fn averaging_keeps_a_tone_on_its_bin() {
        let fft = FreezeFft::with_size(2048);
        let bin = 40usize;
        let hz = bin as f32 * SR / 2048.0;
        let signal = tone(SR as usize, hz);
        let mag = averaged_magnitude(&signal, 50.0, 100.0, 100.0, &fft);
        let (l, _) = render_static_shimmer(&fft, 2048 * 16, 9, &mag, &mag, 0.0);
        let check = analyze_frame(&l, 2048 * 4, &fft, &sine_window(2048));
        let peak = check.mag.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(peak, bin);
    }
}
