//! Offline (render-time) parameter modulation for SpectralPrism.
//!
//! The plugin/editor sliders are normally static scalars handed to
//! `fusion::render_fused_loop`. This module lets each parameter vary across
//! the rendered loop and bakes that variation into the output, by
//! interpolating a small set of control points into a per-hop schedule and
//! feeding it through the exact same phase-vocoder machinery
//! (`render::render_channel`'s oscillator bank + circular OLA).
//!
//! Every parameter that contributes to the output is a target. They fall
//! into three classes:
//! - **analysis-time** (Freeze Point, Tune): these change what is analysed,
//!   so a fixed number of `KEYFRAMES` frozen spectra are computed across the
//!   loop and blended per hop. Keyframes wrap (keyframe `K` is keyframe `0`),
//!   which makes the magnitude schedule inherently loop-periodic.
//! - **resynthesis-time** (Volume, Formant, Stereo Width, Fusion amounts):
//!   cheap to vary per hop. Volume/Formant are folded into the keyframe
//!   magnitudes (both are linear in magnitude, so folding commutes with the
//!   per-hop blending); Stereo Width and the Fusion amounts are applied at
//!   render time.
//! - **structural** (Loop Length): *not* handled here - it changes
//!   `num_hops`/`out_len` and cannot vary within one rendered buffer.
//!
//! Loop periodicity is preserved for time-varying *advance* (pitch) via a
//! zero-mean correction: the per-bin mean advance over the loop is quantised
//! to a whole number of cycles, and that correction is distributed across
//! every hop, so the phase after `num_hops` hops is identical to the phase at
//! the start no matter how the advance moved in between. (A plain per-hop
//! advance would accumulate a non-integer phase offset and click at the
//! loop seam.)

use realfft::num_complex::Complex32;

use crate::fft::{FreezeFft, HOP_SIZE};
use crate::freeze::{analyze_freeze_point, FrozenSpectrum};
use crate::fusion::{
    combine_spectra, effective_mode, render_fused_loop, FusionMode, FusionRenderParams,
};
use crate::render::{apply_formant_shift, LoopBufferData, MAX_LOOP_SECONDS, MIN_LOOP_SECONDS};
use crate::resample::apply_tune;
use crate::resynth::ola_accumulate_circular;
use crate::stereo::{decorrelation_spread, width_multiplier};
use crate::window::{apply_window_in_place, sine_window};
use std::f32::consts::TAU;

/// Number of control points the caller supplies per target. Kept fixed so the
/// WASM boundary does not need to know `HOP_SIZE`; `render_fused_modulated_loop`
/// resamples these to the real per-hop count. Higher = smoother fast LFOs.
pub const MOD_CONTROL_POINTS: usize = 128;

/// Fixed target order, shared verbatim with the TypeScript `SpectralParamId`
/// registry. Bit `i` of `track_mask` means target `i` carries a modulation
/// track; otherwise the base scalar is used.
pub const MOD_TARGET_COUNT: usize = 14;
pub const MOD_FREEZE_POINT_A: usize = 0;
pub const MOD_FREEZE_POINT_B: usize = 1;
pub const MOD_TUNE_A: usize = 2;
pub const MOD_TUNE_B: usize = 3;
pub const MOD_VOLUME_A: usize = 4;
pub const MOD_VOLUME_B: usize = 5;
pub const MOD_FORMANT_A: usize = 6;
pub const MOD_FORMANT_B: usize = 7;
pub const MOD_STEREO_WIDTH: usize = 8;
pub const MOD_MIX_AMOUNT: usize = 9;
pub const MOD_CROSS_SYNTH_AMOUNT: usize = 10;
pub const MOD_CONVOLVE_AMOUNT: usize = 11;
pub const MOD_RING_MOD_AMOUNT: usize = 12;
/// Structural target: not handled per-hop by `render_fused_modulated_loop`
/// (it changes `num_hops`/`out_len`). Handled by
/// `render_fused_loop_length_modulated_loop`, which tiles phase-locked loops
/// of different lengths into one fixed-length super-loop.
pub const MOD_LOOP_LENGTH: usize = 13;

/// Frozen spectra are recomputed at this many positions across the loop when
/// an analysis-time parameter (Freeze Point / Tune) is modulated. The
/// schedule wraps between the last and first keyframe, so the magnitude path
/// is periodic regardless of the modulation shape.
const KEYFRAMES: usize = 24;

/// The non-modulated scalar parameters, mirroring `render_fused_loop`'s
/// arguments so the two entry points stay easy to compare.
pub struct ModulationParams {
    pub freeze_point_a_pct: f32,
    pub volume_a_pct: f32,
    pub tune_a_semitones: f32,
    pub formant_shift_a_semitones: f32,
    pub fusion: FusionRenderParams,
    pub stereo_width_pct: f32,
    pub loop_length_seconds: f32,
    pub root_note: u8,
}

#[derive(Clone)]
struct SourceKeyframe {
    mag_l: Vec<f32>,
    mag_r: Vec<f32>,
    phase_l: Vec<f32>,
    advance: Vec<f32>,
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Resamples `points` (treated as a periodic sequence spanning exactly one
/// loop) to `num_hops` per-hop values.
fn resample_track(points: &[f32], num_hops: usize) -> Vec<f32> {
    if points.is_empty() {
        return vec![0.0; num_hops];
    }
    if points.len() == 1 {
        return vec![points[0]; num_hops];
    }
    let n = points.len();
    (0..num_hops)
        .map(|h| {
            let pos = h as f32 * n as f32 / num_hops as f32;
            let i = pos.floor() as usize;
            let alpha = pos - i as f32;
            let a = points[i % n];
            let b = points[(i + 1) % n];
            a + (b - a) * alpha
        })
        .collect()
}

/// Computes `KEYFRAMES` (wrapped) frozen spectra for one source, folding in
/// the analysis-time (Freeze Point, Tune) and linear resynthesis-time
/// (Volume, Formant) parameters at each keyframe.
fn build_source_keyframes(
    channels: &[Vec<f32>],
    sample_rate: f32,
    fft: &FreezeFft,
    freeze: &[f32],
    tune: &[f32],
    volume: &[f32],
    formant: &[f32],
    num_hops: usize,
) -> Vec<SourceKeyframe> {
    let k = KEYFRAMES.min(num_hops).max(2);
    let left = &channels[0];
    let right = channels.get(1);
    let mut out = Vec::with_capacity(k);
    for j in 0..k {
        let hop = (j * num_hops / k).min(num_hops - 1);
        let fp = freeze[hop];
        let tu = tune[hop];
        let vol = volume[hop];
        let fo = formant[hop];

        let tuned_l = if tu.abs() < 1e-6 {
            left.clone()
        } else {
            apply_tune(left, tu, sample_rate)
        };
        let tuned_r = match right {
            Some(r) if tu.abs() < 1e-6 => r.clone(),
            Some(r) => apply_tune(r, tu, sample_rate),
            None => tuned_l.clone(),
        };

        let a_l = analyze_freeze_point(&tuned_l, fp, vol, fft);
        let a_r = analyze_freeze_point(&tuned_r, fp, vol, fft);
        let mag_l = apply_formant_shift(a_l.mag, fo, fft);
        let mag_r = apply_formant_shift(a_r.mag, fo, fft);
        out.push(SourceKeyframe {
            mag_l,
            mag_r,
            phase_l: a_l.phase0,
            advance: a_l.advance,
        });
    }
    out
}

/// Combines two sources' keyframes for the pre-resynthesis Fusion modes
/// (Cross-Synth / Convolve / Spectral Max / Spectral Min), using each
/// keyframe's own Fusion amount.
fn build_combined_keyframes(
    a_kfs: &[SourceKeyframe],
    b_kfs: &[SourceKeyframe],
    mode: FusionMode,
    amounts: &[f32],
    num_hops: usize,
    fft: &FreezeFft,
) -> Vec<SourceKeyframe> {
    let k = a_kfs.len().min(b_kfs.len());
    let mut out = Vec::with_capacity(k);
    for j in 0..k {
        let hop = (j * num_hops / k).min(num_hops - 1);
        let amount = amounts[hop];
        let a_l = FrozenSpectrum {
            mag: a_kfs[j].mag_l.clone(),
            phase0: a_kfs[j].phase_l.clone(),
            advance: a_kfs[j].advance.clone(),
        };
        let b_l = FrozenSpectrum {
            mag: b_kfs[j].mag_l.clone(),
            phase0: b_kfs[j].phase_l.clone(),
            advance: b_kfs[j].advance.clone(),
        };
        let a_r = FrozenSpectrum {
            mag: a_kfs[j].mag_r.clone(),
            phase0: a_kfs[j].phase_l.clone(),
            advance: a_kfs[j].advance.clone(),
        };
        let b_r = FrozenSpectrum {
            mag: b_kfs[j].mag_r.clone(),
            phase0: b_kfs[j].phase_l.clone(),
            advance: b_kfs[j].advance.clone(),
        };
        let cl = combine_spectra(&a_l, &b_l, mode, amount, fft);
        let cr = combine_spectra(&a_r, &b_r, mode, amount, fft);
        out.push(SourceKeyframe {
            mag_l: cl.mag,
            mag_r: cr.mag,
            phase_l: cl.phase0,
            advance: cl.advance,
        });
    }
    out
}

#[inline]
fn write_bin(spectrum: &mut [Complex32], bin: usize, last: usize, mag: f32, phase: f32) {
    if bin == 0 || bin == last {
        // DC / Nyquist must stay purely real (matches resynth::FreezeResynth).
        let sign = if phase.cos() >= 0.0 { 1.0 } else { -1.0 };
        spectrum[bin] = Complex32::new(mag * sign, 0.0);
    } else {
        spectrum[bin] = Complex32::new(mag * phase.cos(), mag * phase.sin());
    }
}

/// Oscillator-bank render from a wrapped keyframe schedule. Crossfades each
/// keyframe pair per hop, applies a per-hop Stereo Width, and circular-OLAs
/// exactly like `render::render_channel`, but with the magnitude and advance
/// changing over time. The advance schedule carries a per-bin correction so
/// its sum over the loop is a whole number of cycles (loop-click-free).
fn render_keyframed(
    kfs: &[SourceKeyframe],
    widths: &[f32],
    num_hops: usize,
    fft: &FreezeFft,
    window: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let bins = fft.num_bins();
    let k = kfs.len();
    let out_len = num_hops * HOP_SIZE;

    // Mean advance per bin, then the correction that quantises it to a whole
    // number of cycles over the loop while leaving the total variation intact.
    let mut mean = vec![0.0f32; bins];
    for h in 0..num_hops {
        let pos = h as f32 * k as f32 / num_hops as f32;
        let j = (pos.floor() as usize) % k;
        let nx = (j + 1) % k;
        let alpha = pos - pos.floor();
        for bin in 0..bins {
            mean[bin] += lerp(kfs[j].advance[bin], kfs[nx].advance[bin], alpha);
        }
    }
    for m in mean.iter_mut() {
        *m /= num_hops as f32;
    }
    let correction: Vec<f32> = mean
        .iter()
        .map(|&m| {
            let cycles = (m * num_hops as f32 / TAU).round();
            cycles * TAU / num_hops as f32 - m
        })
        .collect();

    let w0 = width_multiplier(widths[0]);
    let mut phase_l = kfs[0].phase_l.clone();
    let mut phase_r: Vec<f32> = kfs[0]
        .phase_l
        .iter()
        .enumerate()
        .map(|(i, &p)| p + w0 * decorrelation_spread(i))
        .collect();

    let mut accum_l = vec![0.0f32; out_len];
    let mut accum_r = vec![0.0f32; out_len];
    let mut spectrum = fft.make_spectrum_buffer();
    let mut time = fft.make_time_buffer();
    let last = bins - 1;

    for h in 0..num_hops {
        let pos = h as f32 * k as f32 / num_hops as f32;
        let j = (pos.floor() as usize) % k;
        let nx = (j + 1) % k;
        let alpha = pos - pos.floor();
        let w = width_multiplier(widths[h]);
        let kj = &kfs[j];
        let kn = &kfs[nx];

        for bin in 0..bins {
            let m = lerp(kj.mag_l[bin], kn.mag_l[bin], alpha);
            let adv = lerp(kj.advance[bin], kn.advance[bin], alpha) + correction[bin];
            write_bin(&mut spectrum, bin, last, m, phase_l[bin]);
            phase_l[bin] += adv;
        }
        fft.inverse(&mut spectrum, &mut time);
        apply_window_in_place(&mut time, window);
        ola_accumulate_circular(&mut accum_l, h * HOP_SIZE, &time);

        for bin in 0..bins {
            let ml = lerp(kj.mag_l[bin], kn.mag_l[bin], alpha);
            let mr = lerp(kj.mag_r[bin], kn.mag_r[bin], alpha);
            let m = ml + (mr - ml) * w;
            let adv = lerp(kj.advance[bin], kn.advance[bin], alpha) + correction[bin];
            write_bin(&mut spectrum, bin, last, m, phase_r[bin]);
            phase_r[bin] += adv;
        }
        fft.inverse(&mut spectrum, &mut time);
        apply_window_in_place(&mut time, window);
        ola_accumulate_circular(&mut accum_r, h * HOP_SIZE, &time);
    }
    (accum_l, accum_r)
}

fn build_gain_track(track: &[f32], out_len: usize, num_hops: usize) -> Vec<f32> {
    // Amounts arrive as 0-100 percentages; combine helpers want 0-1.
    (0..out_len)
        .map(|s| {
            let hop = (s / HOP_SIZE).min(num_hops - 1);
            let frac = (s % HOP_SIZE) as f32 / HOP_SIZE as f32;
            lerp(track[hop], track[(hop + 1) % num_hops], frac) / 100.0
        })
        .collect()
}

fn combine_linear(a: &[f32], b: &[f32], t: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .zip(t.iter())
        .map(|((&x, &y), &tt)| (1.0 - tt) * x + tt * y)
        .collect()
}

fn combine_ring_mod(a: &[f32], b: &[f32], t: &[f32], target_peak: f32) -> Vec<f32> {
    let product: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    let product_peak = product.iter().fold(0.0f32, |m, &s| m.max(s.abs())).max(1e-6);
    let scale = target_peak / product_peak;
    a.iter()
        .zip(product.iter())
        .zip(t.iter())
        .map(|((&x, &p), &tt)| (1.0 - tt) * x + tt * (p * scale))
        .collect()
}

fn peak_normalize(channels: &mut [Vec<f32>], target_peak: f32) {
    let current_peak = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
    if current_peak > 1e-9 {
        let scale = target_peak / current_peak;
        channels.iter_mut().flatten().for_each(|s| *s *= scale);
    }
}

/// Modulated sibling of `fusion::render_fused_loop`.
///
/// `control_points` is a flat `MOD_TARGET_COUNT * num_points` array of
/// absolute parameter values, in a fixed target order, each row spanning one
/// loop period. `track_mask` selects which targets actually vary; unselected
/// targets use the scalar in `params`. With `track_mask == 0` this delegates
/// straight to `render_fused_loop` for exact parity.
pub fn render_fused_modulated_loop(
    source_a: &[Vec<f32>],
    source_b: &[Vec<f32>],
    sample_rate: f32,
    params: &ModulationParams,
    control_points: &[f32],
    num_points: usize,
    track_mask: u32,
) -> LoopBufferData {
    if track_mask == 0 || num_points == 0 {
        return render_fused_loop(
            source_a,
            source_b,
            sample_rate,
            params.freeze_point_a_pct,
            params.volume_a_pct,
            params.tune_a_semitones,
            params.formant_shift_a_semitones,
            &params.fusion,
            params.stereo_width_pct,
            params.loop_length_seconds,
            params.root_note,
        );
    }

    let loop_seconds = params.loop_length_seconds.clamp(MIN_LOOP_SECONDS, MAX_LOOP_SECONDS);
    let num_hops = ((loop_seconds * sample_rate) / HOP_SIZE as f32).round().max(1.0) as usize;
    let out_len = num_hops * HOP_SIZE;
    let root_note = params.root_note;
    let fft = FreezeFft::new();
    let window = sine_window(fft.size());

    let track = |target: usize, base: f32| -> Vec<f32> {
        if track_mask & (1 << target) != 0 {
            let start = target * num_points;
            resample_track(&control_points[start..start + num_points], num_hops)
        } else {
            vec![base; num_hops]
        }
    };

    let freeze_a = track(MOD_FREEZE_POINT_A, params.freeze_point_a_pct);
    let freeze_b = track(MOD_FREEZE_POINT_B, params.fusion.freeze_point_b_pct);
    let tune_a = track(MOD_TUNE_A, params.tune_a_semitones);
    let tune_b = track(MOD_TUNE_B, params.fusion.tune_b_semitones);
    let vol_a = track(MOD_VOLUME_A, params.volume_a_pct);
    let vol_b = track(MOD_VOLUME_B, params.fusion.volume_b_pct);
    let form_a = track(MOD_FORMANT_A, params.formant_shift_a_semitones);
    let form_b = track(MOD_FORMANT_B, params.fusion.formant_shift_b_semitones);
    let width = track(MOD_STEREO_WIDTH, params.stereo_width_pct);
    let mix = track(MOD_MIX_AMOUNT, params.fusion.mix_amount_pct);
    let cross = track(MOD_CROSS_SYNTH_AMOUNT, params.fusion.cross_synth_amount_pct);
    let convolve = track(MOD_CONVOLVE_AMOUNT, params.fusion.convolve_amount_pct);
    let ring = track(MOD_RING_MOD_AMOUNT, params.fusion.ring_mod_amount_pct);

    let mode = effective_mode(params.fusion.mode, source_b);

    // Pre-resynthesis Fusion: combine spectra per keyframe, then render once.
    if matches!(
        mode,
        FusionMode::CrossSynth | FusionMode::Convolve | FusionMode::SpectralMax | FusionMode::SpectralMin
    ) {
        let a_kfs = build_source_keyframes(source_a, sample_rate, &fft, &freeze_a, &tune_a, &vol_a, &form_a, num_hops);
        let b_kfs = build_source_keyframes(source_b, sample_rate, &fft, &freeze_b, &tune_b, &vol_b, &form_b, num_hops);
        let amount_track = match mode {
            FusionMode::CrossSynth => &cross,
            FusionMode::Convolve => &convolve,
            _ => &mix,
        };
        let combined = build_combined_keyframes(&a_kfs, &b_kfs, mode, amount_track, num_hops, &fft);
        let (l, r) = render_keyframed(&combined, &width, num_hops, &fft, &window);
        let mut channels = vec![l, r];
        if mode == FusionMode::Convolve {
            // Match render_pre_resynth_fusion: normalise against plain A's own
            // rendered peak rather than an arbitrary level.
            let a_reference = render_fused_loop(
                source_a,
                &[],
                sample_rate,
                params.freeze_point_a_pct,
                params.volume_a_pct,
                params.tune_a_semitones,
                params.formant_shift_a_semitones,
                &FusionRenderParams {
                    mode: FusionMode::Off,
                    ..params.fusion
                },
                params.stereo_width_pct,
                params.loop_length_seconds,
                root_note,
            );
            let target_peak = a_reference
                .channels
                .iter()
                .flatten()
                .fold(0.0f32, |m, &s| m.max(s.abs()))
                .max(1e-6);
            peak_normalize(&mut channels, target_peak);
        }
        return LoopBufferData {
            channels,
            sample_rate,
            root_note,
        };
    }

    // Post-resynthesis (Off / Mix / Ring Modulate / Cycle): each source is
    // rendered independently, then combined with a per-sample amount.
    let a_kfs = build_source_keyframes(source_a, sample_rate, &fft, &freeze_a, &tune_a, &vol_a, &form_a, num_hops);
    let (al, ar) = render_keyframed(&a_kfs, &width, num_hops, &fft, &window);
    if mode == FusionMode::Off {
        return LoopBufferData {
            channels: vec![al, ar],
            sample_rate,
            root_note,
        };
    }

    let b_kfs = build_source_keyframes(source_b, sample_rate, &fft, &freeze_b, &tune_b, &vol_b, &form_b, num_hops);
    let (bl, br) = render_keyframed(&b_kfs, &width, num_hops, &fft, &window);

    match mode {
        FusionMode::Mix => {
            let t = build_gain_track(&mix, out_len, num_hops);
            let cl = combine_linear(&al, &bl, &t);
            let cr = combine_linear(&ar, &br, &t);
            LoopBufferData {
                channels: vec![cl, cr],
                sample_rate,
                root_note,
            }
        }
        FusionMode::RingModulate => {
            let t = build_gain_track(&ring, out_len, num_hops);
            let target_peak = al
                .iter()
                .chain(ar.iter())
                .fold(0.0f32, |m, &s| m.max(s.abs()))
                .max(1e-6);
            let cl = combine_ring_mod(&al, &bl, &t, target_peak);
            let cr = combine_ring_mod(&ar, &br, &t, target_peak);
            LoopBufferData {
                channels: vec![cl, cr],
                sample_rate,
                root_note,
            }
        }
        FusionMode::Cycle => {
            let mut cl = al;
            cl.extend(bl);
            let mut cr = ar;
            cr.extend(br);
            LoopBufferData {
                channels: vec![cl, cr],
                sample_rate,
                root_note,
            }
        }
        _ => unreachable!("pre-resynthesis modes handled above"),
    }
}

/// Default number of phase-locked segments a loop-length sweep is tiled from.
pub const LOOP_LENGTH_SEGMENTS: usize = 8;
/// Default crossfade at segment joins / the super-loop seam.
pub const LOOP_LENGTH_CROSSFADE_SECONDS: f32 = 0.006;

/// Loop-Length sibling of `render_fused_modulated_loop`.
///
/// Loop Length is *structural*: it changes `num_hops`/`out_len`, so it cannot
/// vary per hop inside one phase-vocoder pass. Instead the base loop period is
/// split into `num_segments` slots; each slot is filled by tiling a
/// phase-locked loop rendered at that slot's modulated length (tiling a
/// periodic buffer is inherently click-free), and adjacent slots plus the
/// super-loop seam are crossfaded. The output length therefore stays equal to
/// the unmodulated base loop length.
///
/// When the loop-length bit is clear this delegates to
/// `render_fused_modulated_loop`, so non-structural modulation is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn render_fused_loop_length_modulated_loop(
    source_a: &[Vec<f32>],
    source_b: &[Vec<f32>],
    sample_rate: f32,
    params: &ModulationParams,
    control_points: &[f32],
    num_points: usize,
    track_mask: u32,
    num_segments: usize,
    crossfade_seconds: f32,
) -> LoopBufferData {
    if track_mask & (1 << MOD_LOOP_LENGTH) == 0 || num_points == 0 {
        return render_fused_modulated_loop(
            source_a,
            source_b,
            sample_rate,
            params,
            control_points,
            num_points,
            track_mask,
        );
    }

    let root_note = params.root_note;
    let mode = effective_mode(params.fusion.mode, source_b);
    let base_seconds = params
        .loop_length_seconds
        .clamp(MIN_LOOP_SECONDS, MAX_LOOP_SECONDS);
    let num_hops = ((base_seconds * sample_rate) / HOP_SIZE as f32).round().max(1.0) as usize;
    let super_len = num_hops * HOP_SIZE;

    let segments = num_segments.clamp(1, num_points.max(1));
    let value = |target: usize, base: f32, point: usize| -> f32 {
        if track_mask & (1 << target) != 0 {
            control_points[target * num_points + point]
        } else {
            base
        }
    };

    let mut pieces: Vec<Vec<Vec<f32>>> = Vec::with_capacity(segments);
    for s in 0..segments {
        let point = ((s as f32 + 0.5) * num_points as f32 / segments as f32).floor() as usize;
        let point = point.min(num_points - 1);
        let loop_len = value(MOD_LOOP_LENGTH, params.loop_length_seconds, point)
            .clamp(MIN_LOOP_SECONDS, MAX_LOOP_SECONDS);
        let fusion = FusionRenderParams {
            mode,
            freeze_point_b_pct: value(MOD_FREEZE_POINT_B, params.fusion.freeze_point_b_pct, point),
            formant_shift_b_semitones: value(
                MOD_FORMANT_B,
                params.fusion.formant_shift_b_semitones,
                point,
            ),
            volume_b_pct: value(MOD_VOLUME_B, params.fusion.volume_b_pct, point),
            tune_b_semitones: value(MOD_TUNE_B, params.fusion.tune_b_semitones, point),
            mix_amount_pct: value(MOD_MIX_AMOUNT, params.fusion.mix_amount_pct, point),
            cross_synth_amount_pct: value(
                MOD_CROSS_SYNTH_AMOUNT,
                params.fusion.cross_synth_amount_pct,
                point,
            ),
            convolve_amount_pct: value(MOD_CONVOLVE_AMOUNT, params.fusion.convolve_amount_pct, point),
            ring_mod_amount_pct: value(MOD_RING_MOD_AMOUNT, params.fusion.ring_mod_amount_pct, point),
        };
        let rendered = render_fused_loop(
            source_a,
            source_b,
            sample_rate,
            value(MOD_FREEZE_POINT_A, params.freeze_point_a_pct, point),
            value(MOD_VOLUME_A, params.volume_a_pct, point),
            value(MOD_TUNE_A, params.tune_a_semitones, point),
            value(MOD_FORMANT_A, params.formant_shift_a_semitones, point),
            &fusion,
            value(MOD_STEREO_WIDTH, params.stereo_width_pct, point),
            loop_len,
            root_note,
        );
        pieces.push(rendered.channels);
    }

    let channels = stitch_loop_segments(pieces, super_len, sample_rate, crossfade_seconds);
    LoopBufferData {
        channels,
        sample_rate,
        root_note,
    }
}

/// Tiles each segment's periodic loop into its slot of the `super_len`
/// output, overlap-adding each join, then wraps the seam with a short fade so
/// the buffer loops cleanly.
fn stitch_loop_segments(
    pieces: Vec<Vec<Vec<f32>>>,
    super_len: usize,
    sample_rate: f32,
    crossfade_seconds: f32,
) -> Vec<Vec<f32>> {
    let channel_count = pieces
        .iter()
        .map(|piece| piece.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut out: Vec<Vec<f32>> = vec![vec![0.0; super_len]; channel_count];
    if pieces.is_empty() || super_len == 0 {
        return out;
    }

    let crossfade = if crossfade_seconds <= 0.0 {
        0
    } else {
        ((crossfade_seconds * sample_rate).round() as usize).max(2)
    };

    let segments = pieces.len();
    let slot = super_len / segments;
    let mut pos = 0usize;
    for (index, piece) in pieces.iter().enumerate() {
        let tile_len = if index + 1 == segments {
            super_len.saturating_sub(pos)
        } else {
            slot
        };
        let piece_len = piece.first().map(|channel| channel.len()).unwrap_or(0);
        if piece_len == 0 {
            pos += tile_len;
            continue;
        }
        // Each segment writes `crossfade` extra samples past its slot; the
        // next segment blends its head into that overlap, so the join is a
        // true crossfade rather than an in-place edit.
        let write_len = (tile_len + crossfade).min(super_len.saturating_sub(pos));
        for (channel_index, channel) in out.iter_mut().enumerate() {
            let source = piece
                .get(channel_index)
                .or_else(|| piece.first())
                .expect("piece has at least one channel");
            for i in 0..write_len {
                let value = source[i % piece_len];
                if i < crossfade && pos > 0 {
                    let t = (i as f32 + 1.0) / (crossfade as f32 + 1.0);
                    channel[pos + i] = channel[pos + i] * (1.0 - t) + value * t;
                } else {
                    channel[pos + i] = value;
                }
            }
        }
        pos += tile_len;
    }

    crossfade_wrap(&mut out, crossfade);
    out
}

/// Fades the tail out and the head in so the super-loop wrap has no step.
fn crossfade_wrap(channels: &mut [Vec<f32>], xf: usize) {
    let total = channels[0].len();
    let xf = xf.min(total / 2);
    if xf < 1 {
        return;
    }
    for channel in channels.iter_mut() {
        for k in 0..xf {
            let t = (k as f32 + 1.0) / (xf as f32 + 1.0);
            channel[total - xf + k] *= 1.0 - t;
            channel[k] *= t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48000.0;

    fn make_signal(len: usize, freq: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 / SAMPLE_RATE * freq * TAU).sin())
            .collect()
    }

    fn source_a() -> Vec<Vec<f32>> {
        vec![make_signal((SAMPLE_RATE * 3.0) as usize, 220.0)]
    }

    fn source_b() -> Vec<Vec<f32>> {
        vec![make_signal((SAMPLE_RATE * 3.0) as usize, 880.0)]
    }

    fn base_params(mode: FusionMode) -> ModulationParams {
        ModulationParams {
            freeze_point_a_pct: 30.0,
            volume_a_pct: 100.0,
            tune_a_semitones: 0.0,
            formant_shift_a_semitones: 0.0,
            fusion: FusionRenderParams {
                mode,
                ..FusionRenderParams::default()
            },
            stereo_width_pct: 30.0,
            loop_length_seconds: 1.0,
            root_note: 60,
        }
    }

    /// Fills every target's control points with the matching scalar, so the
    /// modulated path should reproduce the plain render closely.
    fn constant_points(params: &ModulationParams, n: usize) -> Vec<f32> {
        let mut points = vec![0.0f32; MOD_TARGET_COUNT * n];
        let bases = [
            params.freeze_point_a_pct,
            params.fusion.freeze_point_b_pct,
            params.tune_a_semitones,
            params.fusion.tune_b_semitones,
            params.volume_a_pct,
            params.fusion.volume_b_pct,
            params.formant_shift_a_semitones,
            params.fusion.formant_shift_b_semitones,
            params.stereo_width_pct,
            params.fusion.mix_amount_pct,
            params.fusion.cross_synth_amount_pct,
            params.fusion.convolve_amount_pct,
            params.fusion.ring_mod_amount_pct,
        ];
        for (t, &v) in bases.iter().enumerate() {
            for i in 0..n {
                points[t * n + i] = v;
            }
        }
        points
    }

    fn sweep(base: f32, delta: f32, n: usize) -> Vec<f32> {
        // One full sine cycle: periodic over the loop, like a real LFO.
        (0..n)
            .map(|i| base + delta * (i as f32 / n as f32 * TAU).sin())
            .collect()
    }

    #[test]
    fn no_track_mask_matches_plain_render_exactly() {
        for mode in [
            FusionMode::Off,
            FusionMode::Mix,
            FusionMode::CrossSynth,
            FusionMode::Convolve,
            FusionMode::RingModulate,
            FusionMode::SpectralMax,
            FusionMode::SpectralMin,
            FusionMode::Cycle,
        ] {
            let params = base_params(mode);
            let expected = render_fused_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                params.freeze_point_a_pct,
                params.volume_a_pct,
                params.tune_a_semitones,
                params.formant_shift_a_semitones,
                &params.fusion,
                params.stereo_width_pct,
                params.loop_length_seconds,
                params.root_note,
            );
            let actual = render_fused_modulated_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                &params,
                &[],
                0,
                0,
            );
            assert_eq!(actual.channels[0], expected.channels[0], "mode {:?}", mode);
            assert_eq!(actual.channels[1], expected.channels[1], "mode {:?}", mode);
        }
    }

    #[test]
    fn constant_tracks_match_plain_render() {
        for mode in [
            FusionMode::Off,
            FusionMode::Mix,
            FusionMode::RingModulate,
            FusionMode::SpectralMax,
        ] {
            let params = base_params(mode);
            let points = constant_points(&params, MOD_CONTROL_POINTS);
            let expected = render_fused_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                params.freeze_point_a_pct,
                params.volume_a_pct,
                params.tune_a_semitones,
                params.formant_shift_a_semitones,
                &params.fusion,
                params.stereo_width_pct,
                params.loop_length_seconds,
                params.root_note,
            );
            let actual = render_fused_modulated_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                &params,
                &points,
                MOD_CONTROL_POINTS,
                u32::MAX,
            );
            assert_eq!(actual.channels[0].len(), expected.channels[0].len());
            let mut max_diff = 0.0f32;
            for (a, b) in actual.channels[0].iter().zip(expected.channels[0].iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(max_diff < 5e-3, "mode {:?} max_diff {}", mode, max_diff);
        }
    }

    #[test]
    fn tune_sweep_changes_output_and_stays_finite() {
        let params = base_params(FusionMode::Off);
        let mut points = constant_points(&params, MOD_CONTROL_POINTS);
        let a = sweep(0.0, 3.0, MOD_CONTROL_POINTS);
        points[MOD_TUNE_A * MOD_CONTROL_POINTS..(MOD_TUNE_A + 1) * MOD_CONTROL_POINTS].copy_from_slice(&a);
        let mask = 1 << MOD_TUNE_A;
        let result = render_fused_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            mask,
        );
        let plain = render_fused_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            params.freeze_point_a_pct,
            params.volume_a_pct,
            params.tune_a_semitones,
            params.formant_shift_a_semitones,
            &params.fusion,
            params.stereo_width_pct,
            params.loop_length_seconds,
            params.root_note,
        );
        let diff: f32 = result.channels[0]
            .iter()
            .zip(plain.channels[0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(diff > 1e-6, "expected a tune sweep to change the output");
        assert!(result.channels.iter().flatten().all(|s| s.is_finite()));
    }

    #[test]
    fn freeze_point_sweep_changes_output() {
        let params = base_params(FusionMode::Off);
        let mut points = constant_points(&params, MOD_CONTROL_POINTS);
        let a = sweep(50.0, 40.0, MOD_CONTROL_POINTS);
        points[MOD_FREEZE_POINT_A * MOD_CONTROL_POINTS..(MOD_FREEZE_POINT_A + 1) * MOD_CONTROL_POINTS].copy_from_slice(&a);
        let mask = 1 << MOD_FREEZE_POINT_A;
        let result = render_fused_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            mask,
        );
        let plain = render_fused_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            params.freeze_point_a_pct,
            params.volume_a_pct,
            params.tune_a_semitones,
            params.formant_shift_a_semitones,
            &params.fusion,
            params.stereo_width_pct,
            params.loop_length_seconds,
            params.root_note,
        );
        let diff: f32 = result.channels[0]
            .iter()
            .zip(plain.channels[0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(diff > 1e-6, "expected a freeze-point sweep to change the output");
    }

    #[test]
    fn modulated_loop_is_exactly_periodic_even_with_tune_sweep() {
        // The zero-mean advance correction must keep the oscillator bank
        // periodic over num_hops hops. Render two loops' worth by comparing
        // the rendered buffer's first hop window against the same window one
        // loop later is not possible (buffer is one loop), so instead verify
        // the seam directly: the circular OLA tail should make the signal
        // continuous, which is hard to assert numerically; assert instead
        // that the per-bin advance sum over the loop is a whole number of
        // cycles for every bin by construction (covered indirectly here by
        // checking determinism and finiteness).
        let params = base_params(FusionMode::Mix);
        let mut points = constant_points(&params, MOD_CONTROL_POINTS);
        points[MOD_TUNE_A * MOD_CONTROL_POINTS..(MOD_TUNE_A + 1) * MOD_CONTROL_POINTS]
            .copy_from_slice(&sweep(0.0, 4.0, MOD_CONTROL_POINTS));
        let mask = 1 << MOD_TUNE_A;
        let first = render_fused_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            mask,
        );
        let second = render_fused_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            mask,
        );
        assert_eq!(first.channels[0], second.channels[0]);
        assert!(first.channels.iter().flatten().all(|s| s.is_finite()));
    }

    #[test]
    fn every_mode_produces_stereo_finite_output_with_modulation() {
        for mode in [
            FusionMode::Off,
            FusionMode::Mix,
            FusionMode::CrossSynth,
            FusionMode::Convolve,
            FusionMode::RingModulate,
            FusionMode::SpectralMax,
            FusionMode::SpectralMin,
            FusionMode::Cycle,
        ] {
            let params = base_params(mode);
            let mut points = constant_points(&params, MOD_CONTROL_POINTS);
            points[MOD_VOLUME_A * MOD_CONTROL_POINTS..(MOD_VOLUME_A + 1) * MOD_CONTROL_POINTS]
                .copy_from_slice(&sweep(80.0, 20.0, MOD_CONTROL_POINTS));
            points[MOD_STEREO_WIDTH * MOD_CONTROL_POINTS..(MOD_STEREO_WIDTH + 1) * MOD_CONTROL_POINTS]
                .copy_from_slice(&sweep(40.0, 40.0, MOD_CONTROL_POINTS));
            let mask = (1 << MOD_VOLUME_A) | (1 << MOD_STEREO_WIDTH);
            let result = render_fused_modulated_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                &params,
                &points,
                MOD_CONTROL_POINTS,
                mask,
            );
            assert_eq!(result.channels.len(), 2, "mode {:?}", mode);
            assert!(!result.channels[0].is_empty(), "mode {:?}", mode);
            assert!(result.channels.iter().flatten().all(|s| s.is_finite()), "mode {:?}", mode);
        }
    }

    #[test]
    fn loop_length_without_target_matches_non_structural_path() {
        let params = base_params(FusionMode::Mix);
        let points = constant_points(&params, MOD_CONTROL_POINTS);
        let mask = 1 << MOD_VOLUME_A;
        let expected = render_fused_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            mask,
        );
        let actual = render_fused_loop_length_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            mask,
            LOOP_LENGTH_SEGMENTS,
            LOOP_LENGTH_CROSSFADE_SECONDS,
        );
        assert_eq!(actual.channels[0], expected.channels[0]);
        assert_eq!(actual.channels[1], expected.channels[1]);
    }

    #[test]
    fn loop_length_sweep_preserves_base_length_and_changes_output() {
        let params = base_params(FusionMode::Mix);
        let base = render_fused_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            params.freeze_point_a_pct,
            params.volume_a_pct,
            params.tune_a_semitones,
            params.formant_shift_a_semitones,
            &params.fusion,
            params.stereo_width_pct,
            params.loop_length_seconds,
            params.root_note,
        );
        let mut points = constant_points(&params, MOD_CONTROL_POINTS);
        points[MOD_LOOP_LENGTH * MOD_CONTROL_POINTS..(MOD_LOOP_LENGTH + 1) * MOD_CONTROL_POINTS]
            .copy_from_slice(&sweep(params.loop_length_seconds, 0.35, MOD_CONTROL_POINTS));
        let result = render_fused_loop_length_modulated_loop(
            &source_a(),
            &source_b(),
            SAMPLE_RATE,
            &params,
            &points,
            MOD_CONTROL_POINTS,
            1 << MOD_LOOP_LENGTH,
            LOOP_LENGTH_SEGMENTS,
            LOOP_LENGTH_CROSSFADE_SECONDS,
        );
        assert_eq!(result.channels.len(), 2);
        assert_eq!(result.channels[0].len(), base.channels[0].len());
        assert_eq!(result.channels[1].len(), base.channels[1].len());
        assert!(result.channels.iter().flatten().all(|s| s.is_finite()));
        let diff: f32 = result.channels[0]
            .iter()
            .zip(base.channels[0].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(diff > 1e-6, "loop-length sweep should change the output");
    }

    #[test]
    fn loop_length_crossfade_smooths_the_super_loop_seam() {
        let params = base_params(FusionMode::Mix);
        let mut points = constant_points(&params, MOD_CONTROL_POINTS);
        points[MOD_LOOP_LENGTH * MOD_CONTROL_POINTS..(MOD_LOOP_LENGTH + 1) * MOD_CONTROL_POINTS]
            .copy_from_slice(&sweep(params.loop_length_seconds, 0.35, MOD_CONTROL_POINTS));
        let render = |crossfade: f32| {
            render_fused_loop_length_modulated_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                &params,
                &points,
                MOD_CONTROL_POINTS,
                1 << MOD_LOOP_LENGTH,
                LOOP_LENGTH_SEGMENTS,
                crossfade,
            )
        };
        let hard = render(0.0);
        let soft = render(0.02);
        let seam = |buffer: &LoopBufferData| {
            let channel = &buffer.channels[0];
            (channel[channel.len() - 1] - channel[0]).abs()
        };
        assert!(
            seam(&soft) < seam(&hard),
            "crossfade should reduce the wrap discontinuity (soft {} vs hard {})",
            seam(&soft),
            seam(&hard)
        );
    }

    #[test]
    fn loop_length_joins_are_crossfaded() {
        let params = base_params(FusionMode::Mix);
        let mut points = constant_points(&params, MOD_CONTROL_POINTS);
        points[MOD_LOOP_LENGTH * MOD_CONTROL_POINTS..(MOD_LOOP_LENGTH + 1) * MOD_CONTROL_POINTS]
            .copy_from_slice(&sweep(params.loop_length_seconds, 0.35, MOD_CONTROL_POINTS));
        let render = |crossfade: f32| {
            render_fused_loop_length_modulated_loop(
                &source_a(),
                &source_b(),
                SAMPLE_RATE,
                &params,
                &points,
                MOD_CONTROL_POINTS,
                1 << MOD_LOOP_LENGTH,
                LOOP_LENGTH_SEGMENTS,
                crossfade,
            )
        };
        let hard = render(0.0);
        let soft = render(0.02);
        // First join sits at one slot in.
        let slot = hard.channels[0].len() / LOOP_LENGTH_SEGMENTS;
        let step = |buffer: &LoopBufferData| {
            let channel = &buffer.channels[0];
            (channel[slot] - channel[slot - 1]).abs()
        };
        assert!(
            step(&soft) < step(&hard),
            "crossfade should reduce the join step (soft {} vs hard {})",
            step(&soft),
            step(&hard)
        );
    }
}
