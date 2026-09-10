//! Spectral Fusion: combines Sample A's and Sample B's independently frozen
//! spectra into one output, several different ways. Mirrors (does not
//! literally port, given SpectralPrism's different phase-vocoder
//! architecture) the sibling "Lantern" project's Spectral Fusion algorithm
//! family referenced in this crate's own module doc comment (`lib.rs`).
//!
//! Every mode is a *one-shot* combine, in one of two places:
//! - **Pre-resynthesis** (Cross-Synth, Convolve, Spectral Max, Spectral
//!   Min): A's and B's `FrozenSpectrum`s are combined into a single frozen
//!   spectrum, which then flows through the exact same
//!   quantize/width-blend/formant/resynth/OLA pipeline
//!   `render::render_frozen_loop` already uses for a single source.
//! - **Post-resynthesis** (Mix, Ring Modulate, Cycle) plus the trivial
//!   single-source selects (Off, Audition): A and B are each fully,
//!   independently resynthesized via the existing, unmodified
//!   `render::render_frozen_loop`, then combined as audio buffers.
//!
//! FM ("waveforms act as operators") is deliberately not implemented here:
//! it needs one signal to continuously modulate the other's instantaneous
//! phase *during* resynthesis, which is a different kind of change (to
//! `resynth::FreezeResynth` itself) than the one-shot combines below - see
//! `manage/KANBAN.md` for the tracked idea.

use crate::fft::{FreezeFft, FFT_SIZE, HOP_SIZE};
use crate::formant::compute_spectral_envelope;
use crate::freeze::{analyze_freeze_point, FrozenSpectrum};
use crate::render::{apply_formant_shift, quantize_advance_for_loop, render_channel, render_frozen_loop, LoopBufferData, MAX_LOOP_SECONDS, MIN_LOOP_SECONDS};
use crate::stereo::{decorrelation_spread, width_multiplier};
use crate::window::sine_window;

/// Which combination of Sample A's and Sample B's frozen spectra becomes
/// the plugin's output, and how. Plain enum with no nih_plug dependency,
/// matching this crate's "pure DSP core" design - the plugin-facing,
/// automatable `EnumParam` version lives in `prism_plugin` and converts to
/// this one at the render-request boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionMode {
    /// Sample A's freeze only - the plugin's original, pre-Fusion behavior.
    Off,
    /// Sample B's freeze only - A is ignored entirely.
    Audition,
    /// Linear crossfade between A's and B's independently frozen loops.
    Mix,
    /// B's overall spectral shape (formants) imposed onto A's fine
    /// structure and phase.
    CrossSynth,
    /// Per-bin complex multiply of A's and B's frozen spectra (frequency-
    /// domain multiplication = time-domain convolution).
    Convolve,
    /// Time-domain sample-by-sample multiply of A's and B's resynthesized
    /// loops.
    RingModulate,
    /// Per-bin: the louder of A/B wins (a spectral collage).
    SpectralMax,
    /// Per-bin: the quieter of A/B wins (a spectral gate).
    SpectralMin,
    /// One full loop of A, then one full loop of B, then repeat.
    Cycle,
}

impl FusionMode {
    /// Every mode except `Off` needs Sample B loaded to produce anything
    /// different from plain Sample A.
    pub fn needs_sample_b(self) -> bool {
        self != FusionMode::Off
    }

    /// Whether this mode combines A's and B's spectra once *before*
    /// resynthesis, or combines two fully resynthesized audio buffers
    /// *after*.
    fn is_pre_resynth(self) -> bool {
        matches!(self, FusionMode::CrossSynth | FusionMode::Convolve | FusionMode::SpectralMax | FusionMode::SpectralMin)
    }
}

/// Every Fusion-related render parameter bundled together, mirroring
/// `render_worker::RenderRequest`'s flat style one level down.
#[derive(Clone, Copy, PartialEq)]
pub struct FusionRenderParams {
    pub mode: FusionMode,
    pub freeze_point_b_pct: f32,
    pub formant_shift_b_semitones: f32,
    pub mix_amount_pct: f32,
    pub cross_synth_amount_pct: f32,
    pub convolve_amount_pct: f32,
    pub ring_mod_amount_pct: f32,
}

/// Degrades `mode` to `Off` if it needs Sample B (`FusionMode::needs_sample_b`)
/// but `source_b` has no audio in it yet - the render-time half of the
/// graceful-degrade story for "a Fusion mode is selected but Sample B
/// hasn't been loaded" (the other half is the plugin's own editor warning
/// label, which reads the real, un-degraded param value so it can tell the
/// user why nothing changed). Shared by every call site that renders from
/// live plugin state (the background render worker, and the plugin's
/// synchronous first render in `initialize()`), so this check only needs
/// to be gotten right once.
pub fn effective_mode(mode: FusionMode, source_b: &[Vec<f32>]) -> FusionMode {
    if mode.needs_sample_b() && source_b.iter().all(|c| c.is_empty()) {
        FusionMode::Off
    } else {
        mode
    }
}

impl Default for FusionRenderParams {
    fn default() -> Self {
        Self {
            mode: FusionMode::Off,
            freeze_point_b_pct: 50.0,
            formant_shift_b_semitones: 0.0,
            mix_amount_pct: 50.0,
            cross_synth_amount_pct: 100.0,
            convolve_amount_pct: 100.0,
            ring_mod_amount_pct: 100.0,
        }
    }
}

/// Top-level Spectral Fusion entry point - the fusion-aware sibling of
/// `render::render_frozen_loop`. `source_a`/`source_b` are each one
/// `Vec<f32>` per channel, the same shape `render_frozen_loop` expects.
#[allow(clippy::too_many_arguments)]
pub fn render_fused_loop(
    source_a: &[Vec<f32>],
    source_b: &[Vec<f32>],
    sample_rate: f32,
    freeze_point_a_pct: f32,
    formant_shift_a_semitones: f32,
    fusion: &FusionRenderParams,
    stereo_width_pct: f32,
    loop_length_seconds: f32,
    root_note: u8,
) -> LoopBufferData {
    if fusion.mode.is_pre_resynth() {
        render_pre_resynth_fusion(source_a, source_b, sample_rate, freeze_point_a_pct, formant_shift_a_semitones, fusion, stereo_width_pct, loop_length_seconds, root_note)
    } else {
        render_post_resynth_fusion(source_a, source_b, sample_rate, freeze_point_a_pct, formant_shift_a_semitones, fusion, stereo_width_pct, loop_length_seconds, root_note)
    }
}

/// Cross-Synth / Convolve / Spectral Max / Spectral Min: combine A's and
/// B's `FrozenSpectrum`s into one, then run the result through exactly the
/// same quantize/width-blend/resynth pipeline `render_frozen_loop` uses for
/// a single source (see that function's doc comment for why phase-locking
/// + circular OLA is what makes the loop click-free).
#[allow(clippy::too_many_arguments)]
fn render_pre_resynth_fusion(
    source_a: &[Vec<f32>],
    source_b: &[Vec<f32>],
    sample_rate: f32,
    freeze_point_a_pct: f32,
    formant_shift_a_semitones: f32,
    fusion: &FusionRenderParams,
    stereo_width_pct: f32,
    loop_length_seconds: f32,
    root_note: u8,
) -> LoopBufferData {
    let fft = FreezeFft::new();
    let window = sine_window(FFT_SIZE);

    let a_left = source_a.first().expect("render_fused_loop requires at least one channel in Sample A");
    let a_right = source_a.get(1).unwrap_or(a_left);
    let b_left = source_b.first().expect("render_fused_loop requires at least one channel in Sample B");
    let b_right = source_b.get(1).unwrap_or(b_left);

    let loop_seconds = loop_length_seconds.clamp(MIN_LOOP_SECONDS, MAX_LOOP_SECONDS);
    let num_hops = ((loop_seconds * sample_rate) / HOP_SIZE as f32).round().max(1.0) as usize;
    let out_len = num_hops * HOP_SIZE;
    let width = width_multiplier(stereo_width_pct);

    let mut a_l = analyze_freeze_point(a_left, freeze_point_a_pct, &fft);
    let mut a_r = analyze_freeze_point(a_right, freeze_point_a_pct, &fft);
    let mut b_l = analyze_freeze_point(b_left, fusion.freeze_point_b_pct, &fft);
    let mut b_r = analyze_freeze_point(b_right, fusion.freeze_point_b_pct, &fft);
    a_l.mag = apply_formant_shift(a_l.mag, formant_shift_a_semitones, &fft);
    a_r.mag = apply_formant_shift(a_r.mag, formant_shift_a_semitones, &fft);
    b_l.mag = apply_formant_shift(b_l.mag, fusion.formant_shift_b_semitones, &fft);
    b_r.mag = apply_formant_shift(b_r.mag, fusion.formant_shift_b_semitones, &fft);

    let amount = match fusion.mode {
        FusionMode::CrossSynth => fusion.cross_synth_amount_pct,
        FusionMode::Convolve => fusion.convolve_amount_pct,
        _ => 0.0,
    };
    let combined_left = combine_spectra(&a_l, &b_l, fusion.mode, amount, &fft);
    let combined_right_natural = combine_spectra(&a_r, &b_r, fusion.mode, amount, &fft);

    // Stereo width/decorrelation applied on top of the *combined* result,
    // exactly `render::render_frozen_loop`'s own logic, just fed the fused
    // left/right spectra instead of a single source's.
    let quantized_advance = quantize_advance_for_loop(&combined_left.advance, num_hops);
    let right_mag: Vec<f32> =
        combined_left.mag.iter().zip(combined_right_natural.mag.iter()).map(|(&l, &r)| l + (r - l) * width).collect();
    let right_phase0: Vec<f32> =
        combined_left.phase0.iter().enumerate().map(|(k, &p)| p + width * decorrelation_spread(k)).collect();

    let left_out = render_channel(combined_left.mag, combined_left.phase0, quantized_advance.clone(), &window, &fft, out_len);
    let right_out = render_channel(right_mag, right_phase0, quantized_advance, &window, &fft, out_len);
    let mut channels = vec![left_out, right_out];

    if fusion.mode == FusionMode::Convolve {
        // mag_a * mag_b isn't level-normalized (multiplying two magnitude
        // spectra together, unlike blending them, has no natural ceiling),
        // so bring the result back toward Sample A's own peak rather than
        // let it swing arbitrarily loud or quiet as Amount changes. The
        // reference is A's own *rendered* peak (via the same
        // analyze/resynth pipeline), not its raw time-domain peak - phase-
        // vocoder reconstruction doesn't preserve peak amplitude exactly,
        // so normalizing against the raw source would apply a small,
        // spurious scale even at 0% Amount (where the combined spectrum is
        // otherwise bit-identical to plain A).
        let a_reference =
            render_frozen_loop(source_a, sample_rate, freeze_point_a_pct, formant_shift_a_semitones, stereo_width_pct, loop_length_seconds, root_note);
        let target_peak = a_reference.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs())).max(1e-6);
        peak_normalize(&mut channels, target_peak);
    }

    LoopBufferData { channels, sample_rate, root_note }
}

fn combine_spectra(a: &FrozenSpectrum, b: &FrozenSpectrum, mode: FusionMode, amount_pct: f32, fft: &FreezeFft) -> FrozenSpectrum {
    match mode {
        FusionMode::CrossSynth => cross_synth(a, b, amount_pct, fft),
        FusionMode::Convolve => convolve(a, b, amount_pct),
        FusionMode::SpectralMax => spectral_select(a, b, true),
        FusionMode::SpectralMin => spectral_select(a, b, false),
        _ => unreachable!("combine_spectra is only called for pre-resynthesis fusion modes"),
    }
}

/// Imposes B's overall spectral shape (formants) onto A's fine structure
/// and phase - `amount_pct` blends the *envelope* continuously from A's own
/// shape (0%) to B's (100%); A's fine structure and phase are kept at every
/// amount. Adapted from the sibling "Lantern" project's cross-synth (which
/// operates on whole-clip complex spectra via a separate FFT type); here it
/// operates directly on one already-frozen `FrozenSpectrum` per source,
/// reusing `formant::compute_spectral_envelope` - the same cepstral-
/// liftering envelope extraction already used for the Formant Shift
/// control, which is exactly the technique the sibling project's
/// `spectral_envelope()` reimplements with a second, full-complex FFT.
fn cross_synth(a: &FrozenSpectrum, b: &FrozenSpectrum, amount_pct: f32, fft: &FreezeFft) -> FrozenSpectrum {
    let t = (amount_pct / 100.0).clamp(0.0, 1.0);
    let env_a = compute_spectral_envelope(&a.mag, fft);
    let env_b = compute_spectral_envelope(&b.mag, fft);
    let mag = a
        .mag
        .iter()
        .zip(env_a.iter())
        .zip(env_b.iter())
        .map(|((&am, &ea), &eb)| {
            let ea = ea.max(1e-9);
            (am / ea) * eb.max(1e-9).powf(t) * ea.powf(1.0 - t)
        })
        .collect();
    FrozenSpectrum { mag, phase0: a.phase0.clone(), advance: a.advance.clone() }
}

/// Per-bin complex multiply of A's and B's frozen spectra - the frequency-
/// domain operation that corresponds to time-domain convolution.
/// `amount_pct` blends continuously from A alone (0%, bit-identical) to the
/// full complex product (100%). Phase0 stays A's throughout (lerping two
/// arbitrary absolute phases isn't well-defined and can cancel); `advance`
/// is a rotation *rate*, which blends cleanly as a scalar since multiplying
/// two rotating phasors sums their rates.
fn convolve(a: &FrozenSpectrum, b: &FrozenSpectrum, amount_pct: f32) -> FrozenSpectrum {
    let t = (amount_pct / 100.0).clamp(0.0, 1.0);
    let mag = a.mag.iter().zip(b.mag.iter()).map(|(&am, &bm)| am + t * (am * bm - am)).collect();
    let advance = a.advance.iter().zip(b.advance.iter()).map(|(&aa, &ab)| aa + t * ab).collect();
    FrozenSpectrum { mag, phase0: a.phase0.clone(), advance }
}

/// Per-bin magnitude comparison: `want_max` keeps whichever of A/B is
/// louder at each frequency bin (a spectral collage - brighter/more
/// aggressive than either alone); otherwise keeps whichever is quieter (a
/// spectral gate, keeping only what both sources share). The winning
/// source's phase0/advance travel with its magnitude at each bin.
fn spectral_select(a: &FrozenSpectrum, b: &FrozenSpectrum, want_max: bool) -> FrozenSpectrum {
    let n = a.mag.len();
    let mut mag = Vec::with_capacity(n);
    let mut phase0 = Vec::with_capacity(n);
    let mut advance = Vec::with_capacity(n);
    for k in 0..n {
        let a_wins = if want_max { a.mag[k] >= b.mag[k] } else { a.mag[k] <= b.mag[k] };
        if a_wins {
            mag.push(a.mag[k]);
            phase0.push(a.phase0[k]);
            advance.push(a.advance[k]);
        } else {
            mag.push(b.mag[k]);
            phase0.push(b.phase0[k]);
            advance.push(b.advance[k]);
        }
    }
    FrozenSpectrum { mag, phase0, advance }
}

/// Off / Audition / Mix / Ring Modulate / Cycle: each of A and B is fully,
/// independently resynthesized via the existing, unmodified
/// `render_frozen_loop` (each already applies its own formant shift and
/// stereo width), then combined as plain audio buffers.
#[allow(clippy::too_many_arguments)]
fn render_post_resynth_fusion(
    source_a: &[Vec<f32>],
    source_b: &[Vec<f32>],
    sample_rate: f32,
    freeze_point_a_pct: f32,
    formant_shift_a_semitones: f32,
    fusion: &FusionRenderParams,
    stereo_width_pct: f32,
    loop_length_seconds: f32,
    root_note: u8,
) -> LoopBufferData {
    match fusion.mode {
        FusionMode::Off => {
            render_frozen_loop(source_a, sample_rate, freeze_point_a_pct, formant_shift_a_semitones, stereo_width_pct, loop_length_seconds, root_note)
        }
        FusionMode::Audition => render_frozen_loop(
            source_b,
            sample_rate,
            fusion.freeze_point_b_pct,
            fusion.formant_shift_b_semitones,
            stereo_width_pct,
            loop_length_seconds,
            root_note,
        ),
        FusionMode::Mix | FusionMode::RingModulate | FusionMode::Cycle => {
            let a = render_frozen_loop(source_a, sample_rate, freeze_point_a_pct, formant_shift_a_semitones, stereo_width_pct, loop_length_seconds, root_note);
            let b = render_frozen_loop(
                source_b,
                sample_rate,
                fusion.freeze_point_b_pct,
                fusion.formant_shift_b_semitones,
                stereo_width_pct,
                loop_length_seconds,
                root_note,
            );
            match fusion.mode {
                FusionMode::Mix => mix_buffers(&a, &b, fusion.mix_amount_pct),
                FusionMode::RingModulate => ring_modulate_buffers(&a, &b, fusion.ring_mod_amount_pct),
                FusionMode::Cycle => cycle_buffers(a, b),
                _ => unreachable!(),
            }
        }
        FusionMode::CrossSynth | FusionMode::Convolve | FusionMode::SpectralMax | FusionMode::SpectralMin => {
            unreachable!("pre-resynthesis fusion modes are handled by render_pre_resynth_fusion")
        }
    }
}

/// Linear crossfade between A's and B's independently frozen loops -
/// `mix_pct` 0% is plain A, 100% is plain B, in between a simple volume
/// blend of both.
fn mix_buffers(a: &LoopBufferData, b: &LoopBufferData, mix_pct: f32) -> LoopBufferData {
    let t = (mix_pct / 100.0).clamp(0.0, 1.0);
    let channels =
        a.channels.iter().zip(b.channels.iter()).map(|(ac, bc)| ac.iter().zip(bc.iter()).map(|(&x, &y)| (1.0 - t) * x + t * y).collect()).collect();
    LoopBufferData { channels, sample_rate: a.sample_rate, root_note: a.root_note }
}

/// Time-domain sample-by-sample multiply of A's and B's resynthesized
/// loops - classic ring modulation. The raw product is peak-normalized
/// against A's own peak (multiplying two arbitrary-level signals isn't
/// level-preserving) before being crossfaded against dry A by
/// `amount_pct`.
fn ring_modulate_buffers(a: &LoopBufferData, b: &LoopBufferData, amount_pct: f32) -> LoopBufferData {
    let t = (amount_pct / 100.0).clamp(0.0, 1.0);
    let target_peak = a.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs())).max(1e-6);
    let channels = a
        .channels
        .iter()
        .zip(b.channels.iter())
        .map(|(ac, bc)| {
            let product: Vec<f32> = ac.iter().zip(bc.iter()).map(|(&x, &y)| x * y).collect();
            let product_peak = product.iter().fold(0.0f32, |m, &s| m.max(s.abs())).max(1e-6);
            let scale = target_peak / product_peak;
            ac.iter().zip(product.iter()).map(|(&x, &p)| (1.0 - t) * x + t * (p * scale)).collect()
        })
        .collect();
    LoopBufferData { channels, sample_rate: a.sample_rate, root_note: a.root_note }
}

/// Concatenates one full loop of A then one full loop of B per channel - an
/// alternating pattern rather than a blend. The result is 2x the length of
/// a single `render_frozen_loop` call, which is safe: playback
/// (`resample::PlaybackReader`) wraps against whatever length the current
/// loop buffer actually is, not an assumed `loop_length_seconds *
/// sample_rate`, so a longer buffer just plays and loops correctly with no
/// downstream changes needed. "Loop Length" continues to mean "length of
/// one A or B cycle" - the actual looped buffer is twice that, by design.
fn cycle_buffers(a: LoopBufferData, b: LoopBufferData) -> LoopBufferData {
    let sample_rate = a.sample_rate;
    let root_note = a.root_note;
    let channels = a
        .channels
        .into_iter()
        .zip(b.channels)
        .map(|(mut ac, bc)| {
            ac.extend(bc);
            ac
        })
        .collect();
    LoopBufferData { channels, sample_rate, root_note }
}

fn peak_normalize(channels: &mut [Vec<f32>], target_peak: f32) {
    let current_peak = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
    if current_peak > 1e-9 {
        let scale = target_peak / current_peak;
        channels.iter_mut().flatten().for_each(|s| *s *= scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_signal(len: usize, freq_hz: f32, sample_rate: f32) -> Vec<f32> {
        (0..len).map(|i| (i as f32 / sample_rate * freq_hz * std::f32::consts::TAU).sin()).collect()
    }

    const SAMPLE_RATE: f32 = 48000.0;

    fn source_a() -> Vec<Vec<f32>> {
        vec![make_test_signal((SAMPLE_RATE * 3.0) as usize, 220.0, SAMPLE_RATE)]
    }

    fn source_b() -> Vec<Vec<f32>> {
        vec![make_test_signal((SAMPLE_RATE * 3.0) as usize, 880.0, SAMPLE_RATE)]
    }

    fn off_params() -> FusionRenderParams {
        FusionRenderParams::default()
    }

    #[test]
    fn off_mode_matches_plain_render_frozen_loop() {
        let a = source_a();
        let expected = render_frozen_loop(&a, SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let fusion = FusionRenderParams { mode: FusionMode::Off, ..off_params() };
        let actual = render_fused_loop(&a, &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        assert_eq!(actual.channels[0], expected.channels[0]);
        assert_eq!(actual.channels[1], expected.channels[1]);
    }

    #[test]
    fn audition_mode_matches_b_alone() {
        let b = source_b();
        let fusion = FusionRenderParams { mode: FusionMode::Audition, freeze_point_b_pct: 40.0, ..off_params() };
        let expected = render_frozen_loop(&b, SAMPLE_RATE, 40.0, 0.0, 0.0, 1.0, 60);
        let actual = render_fused_loop(&source_a(), &b, SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        assert_eq!(actual.channels[0], expected.channels[0]);
    }

    #[test]
    fn mix_at_zero_is_plain_a() {
        let fusion = FusionRenderParams { mode: FusionMode::Mix, mix_amount_pct: 0.0, ..off_params() };
        let expected = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let actual = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        for (x, y) in actual.channels[0].iter().zip(expected.channels[0].iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn mix_at_hundred_is_plain_b() {
        let fusion = FusionRenderParams { mode: FusionMode::Mix, mix_amount_pct: 100.0, ..off_params() };
        let expected = render_frozen_loop(&source_b(), SAMPLE_RATE, 50.0, 0.0, 0.0, 1.0, 60);
        let actual = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        for (x, y) in actual.channels[0].iter().zip(expected.channels[0].iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn mix_at_fifty_is_between_a_and_b() {
        let fusion = FusionRenderParams { mode: FusionMode::Mix, mix_amount_pct: 50.0, ..off_params() };
        let a_alone = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let b_alone = render_frozen_loop(&source_b(), SAMPLE_RATE, 50.0, 0.0, 0.0, 1.0, 60);
        let mixed = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        for i in 0..mixed.channels[0].len() {
            let expected = 0.5 * a_alone.channels[0][i] + 0.5 * b_alone.channels[0][i];
            assert!((mixed.channels[0][i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn cross_synth_at_zero_amount_is_close_to_plain_a() {
        let fusion = FusionRenderParams { mode: FusionMode::CrossSynth, cross_synth_amount_pct: 0.0, ..off_params() };
        let expected = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let actual = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        // Cepstral envelope round-trip is a smoothed approximation, not a
        // perfect identity, so allow a small tolerance rather than exact
        // equality.
        let mut max_diff = 0.0f32;
        for (x, y) in actual.channels[0].iter().zip(expected.channels[0].iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 0.05, "expected 0% Cross-Synth to stay close to plain A, max diff {}", max_diff);
    }

    #[test]
    fn cross_synth_at_full_amount_differs_from_plain_a() {
        let fusion = FusionRenderParams { mode: FusionMode::CrossSynth, cross_synth_amount_pct: 100.0, ..off_params() };
        let plain_a = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let fused = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        let diff_energy: f32 = fused.channels[0].iter().zip(plain_a.channels[0].iter()).map(|(x, y)| (x - y).powi(2)).sum();
        assert!(diff_energy > 1e-6, "expected full Cross-Synth to audibly differ from plain A");
    }

    #[test]
    fn convolve_at_zero_amount_is_plain_a() {
        let fusion = FusionRenderParams { mode: FusionMode::Convolve, convolve_amount_pct: 0.0, ..off_params() };
        let expected = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let actual = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        for (x, y) in actual.channels[0].iter().zip(expected.channels[0].iter()) {
            assert!((x - y).abs() < 1e-4);
        }
    }

    #[test]
    fn convolve_output_stays_peak_normalized_near_as_own_peak() {
        let fusion = FusionRenderParams { mode: FusionMode::Convolve, convolve_amount_pct: 100.0, ..off_params() };
        let a_alone = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let a_peak = a_alone.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
        let fused = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        let fused_peak = fused.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!((fused_peak - a_peak).abs() < 1e-3, "expected Convolve output peak-normalized near A's own peak ({}), got {}", a_peak, fused_peak);
    }

    #[test]
    fn spectral_max_picks_larger_magnitude_per_bin() {
        let a = FrozenSpectrum { mag: vec![1.0, 5.0, 2.0], phase0: vec![0.1, 0.2, 0.3], advance: vec![0.01, 0.02, 0.03] };
        let b = FrozenSpectrum { mag: vec![4.0, 2.0, 2.0], phase0: vec![0.4, 0.5, 0.6], advance: vec![0.04, 0.05, 0.06] };
        let result = spectral_select(&a, &b, true);
        assert_eq!(result.mag, vec![4.0, 5.0, 2.0]);
        assert_eq!(result.phase0, vec![0.4, 0.2, 0.3]);
    }

    #[test]
    fn spectral_min_picks_smaller_magnitude_per_bin() {
        let a = FrozenSpectrum { mag: vec![1.0, 5.0, 2.0], phase0: vec![0.1, 0.2, 0.3], advance: vec![0.01, 0.02, 0.03] };
        let b = FrozenSpectrum { mag: vec![4.0, 2.0, 2.0], phase0: vec![0.4, 0.5, 0.6], advance: vec![0.04, 0.05, 0.06] };
        let result = spectral_select(&a, &b, false);
        assert_eq!(result.mag, vec![1.0, 2.0, 2.0]);
        assert_eq!(result.phase0, vec![0.1, 0.5, 0.3]);
    }

    #[test]
    fn ring_modulate_differs_from_plain_a_and_stays_peak_normalized() {
        let fusion = FusionRenderParams { mode: FusionMode::RingModulate, ring_mod_amount_pct: 100.0, ..off_params() };
        let a_alone = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let a_peak = a_alone.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
        let fused = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);
        let fused_peak = fused.channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!((fused_peak - a_peak).abs() < 1e-3, "expected Ring Modulate output peak-normalized near A's own peak");

        let diff_energy: f32 = fused.channels[0].iter().zip(a_alone.channels[0].iter()).map(|(x, y)| (x - y).powi(2)).sum();
        assert!(diff_energy > 1e-6, "expected full Ring Modulate to audibly differ from plain A");
    }

    #[test]
    fn cycle_buffer_is_twice_loop_length_and_each_half_matches_independent_renders() {
        let fusion = FusionRenderParams { mode: FusionMode::Cycle, ..off_params() };
        let a_alone = render_frozen_loop(&source_a(), SAMPLE_RATE, 30.0, 0.0, 0.0, 1.0, 60);
        let b_alone = render_frozen_loop(&source_b(), SAMPLE_RATE, 50.0, 0.0, 0.0, 1.0, 60);
        let cycled = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 0.0, 1.0, 60);

        let one_cycle_len = a_alone.channels[0].len();
        assert_eq!(cycled.channels[0].len(), one_cycle_len * 2);
        assert_eq!(cycled.channels[0][..one_cycle_len], a_alone.channels[0][..]);
        assert_eq!(cycled.channels[0][one_cycle_len..], b_alone.channels[0][..]);
    }

    #[test]
    fn render_fused_loop_never_panics_and_always_outputs_stereo_for_every_mode() {
        let modes = [
            FusionMode::Off,
            FusionMode::Audition,
            FusionMode::Mix,
            FusionMode::CrossSynth,
            FusionMode::Convolve,
            FusionMode::RingModulate,
            FusionMode::SpectralMax,
            FusionMode::SpectralMin,
            FusionMode::Cycle,
        ];
        for mode in modes {
            let fusion = FusionRenderParams { mode, ..off_params() };
            let result = render_fused_loop(&source_a(), &source_b(), SAMPLE_RATE, 30.0, 0.0, &fusion, 30.0, 1.0, 60);
            assert_eq!(result.channels.len(), 2, "mode {:?} did not output stereo", mode);
            assert!(!result.channels[0].is_empty(), "mode {:?} produced an empty buffer", mode);
            for &s in result.channels.iter().flatten() {
                assert!(s.is_finite(), "mode {:?} produced a non-finite sample", mode);
            }
        }
    }
}
