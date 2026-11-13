//! The full render-time instrument chain:
//!
//! ```text
//! Source A ─▶ FM transform ─▶ Spectral freeze/fusion ─▶ Percussion ─▶ MicroTextures ─▶ loop buffer
//!                 (pre)          (+ Source B, mod routes)    (post)          (post)
//! ```
//!
//! Mirrors SpectralPrism Tracker's `makeSpectralRenderer`
//! (`src/wasm/prism.ts`) so a patch renders identically in both:
//! - FM transforms (a slice of) Sample A before anything else.
//! - FM with the Spectral freeze off returns the FM output as-is (a blip
//!   keeps its transient instead of being frozen into a loop).
//! - Otherwise the freeze/fusion runs - loop-length-modulated, modulated or
//!   plain depending on which routes are active - whenever *any* stage is on.
//! - Percussion, then MicroTextures, post-process that output.
//! - With every stage off the chain is a plain sampler: the (trimmed) source
//!   is passed through untouched.

use crate::fm_synth::{render_fm_synth, FmSynthParams};
use crate::fm_transform::{render_fm_transform, FmTransformParams};
use crate::fusion::{render_fused_loop, FusionRenderParams};
use crate::microtextures::{render_microtextures, MicroTextureParams};
use crate::modroute::{sample_mod_tracks, ModRoute, MOD_SLOTS};
use crate::modulate::{
    render_fused_loop_length_modulated_loop, render_fused_modulated_loop, ModulationParams, LOOP_LENGTH_CROSSFADE_SECONDS, LOOP_LENGTH_SEGMENTS,
    MOD_CONVOLVE_AMOUNT, MOD_CROSS_SYNTH_AMOUNT, MOD_FORMANT_A, MOD_FORMANT_B, MOD_FREEZE_POINT_A, MOD_FREEZE_POINT_B, MOD_LOOP_LENGTH, MOD_MIX_AMOUNT,
    MOD_RING_MOD_AMOUNT, MOD_STEREO_WIDTH, MOD_TARGET_COUNT, MOD_TUNE_A, MOD_TUNE_B, MOD_VOLUME_A, MOD_VOLUME_B,
};
use crate::percussion::{render_percussion, PercussionParams};
use crate::render::{LoopBufferData, DEFAULT_LOOP_SECONDS, DEFAULT_ROOT_NOTE};

/// Where Source A's audio comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SourceKind {
    /// The loaded sample (or the built-in placeholder tone).
    #[default]
    Sample,
    /// The 4-operator FM synth (`fm_synth`), rendered into a buffer that
    /// stands in for the sample.
    FmSynth,
}

/// Every render-time parameter of the instrument. `Copy + PartialEq` so the
/// plugin's render worker can collapse identical back-to-back requests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainParams {
    // --- Source ---
    pub source_kind: SourceKind,
    /// Used when `source_kind == FmSynth`.
    pub fm_synth: FmSynthParams,
    // Trim: plain-sampler mode with a sample source only.
    /// Trim start in seconds.
    pub trim_start_sec: f32,
    /// Trim end in seconds; 0 (or <= start) = to the end of the sample.
    pub trim_end_sec: f32,

    // --- FM pre-stage ---
    pub fm_enabled: bool,
    pub fm: FmTransformParams,
    pub fm_slice_start_sec: f32,
    /// 0 = to the end.
    pub fm_slice_length_sec: f32,

    // --- Spectral freeze / fusion ---
    pub spectral_enabled: bool,
    pub freeze_point_a_pct: f32,
    pub volume_a_pct: f32,
    pub tune_a_semitones: f32,
    pub formant_shift_a_semitones: f32,
    pub stereo_width_pct: f32,
    pub loop_length_seconds: f32,
    pub fusion: FusionRenderParams,
    pub mod_routes: [ModRoute; MOD_SLOTS],

    // --- Post-stages ---
    pub percussion_enabled: bool,
    pub percussion: PercussionParams,
    pub micro_enabled: bool,
    pub micro: MicroTextureParams,
}

impl Default for ChainParams {
    fn default() -> Self {
        Self {
            source_kind: SourceKind::Sample,
            fm_synth: FmSynthParams::default(),
            trim_start_sec: 0.0,
            trim_end_sec: 0.0,
            fm_enabled: false,
            fm: default_fm_params(),
            fm_slice_start_sec: 0.0,
            fm_slice_length_sec: 0.0,
            spectral_enabled: true,
            freeze_point_a_pct: 50.0,
            volume_a_pct: 100.0,
            tune_a_semitones: 0.0,
            formant_shift_a_semitones: 0.0,
            stereo_width_pct: 30.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams::default(),
            mod_routes: [ModRoute::default(); MOD_SLOTS],
            percussion_enabled: false,
            percussion: PercussionParams::default(),
            micro_enabled: false,
            micro: default_micro_params(),
        }
    }
}

/// The tracker's `defaultFmSettings()`.
pub fn default_fm_params() -> FmTransformParams {
    FmTransformParams {
        algorithm: crate::fm_transform::FmAlgorithm::Stack,
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

/// The tracker's `defaultMicroTextureSettings()` (differs from
/// `MicroTextureParams::default()` in retrigger rate and formant mix).
pub fn default_micro_params() -> MicroTextureParams {
    MicroTextureParams { retrigger_hz: 8.0, formant_mix: 0.5, ..MicroTextureParams::default() }
}

impl ChainParams {
    /// True when any DSP stage is on; false = plain sampler passthrough.
    pub fn renders(&self) -> bool {
        self.spectral_enabled || self.fm_enabled || self.percussion_enabled || self.micro_enabled
    }

    /// True when the result should play once rather than loop: the
    /// percussion stage always produces a hit, and FM without the freeze
    /// transforms a hit too (the tracker's auto one-shot rules).
    pub fn suggests_one_shot(&self) -> bool {
        self.percussion_enabled || (self.fm_enabled && !self.spectral_enabled)
    }

    /// Unmodulated value of every modulation target, in `MOD_*` order.
    pub fn mod_base_values(&self) -> [f32; MOD_TARGET_COUNT] {
        let mut b = [0.0; MOD_TARGET_COUNT];
        b[MOD_FREEZE_POINT_A] = self.freeze_point_a_pct;
        b[MOD_FREEZE_POINT_B] = self.fusion.freeze_point_b_pct;
        b[MOD_TUNE_A] = self.tune_a_semitones;
        b[MOD_TUNE_B] = self.fusion.tune_b_semitones;
        b[MOD_VOLUME_A] = self.volume_a_pct;
        b[MOD_VOLUME_B] = self.fusion.volume_b_pct;
        b[MOD_FORMANT_A] = self.formant_shift_a_semitones;
        b[MOD_FORMANT_B] = self.fusion.formant_shift_b_semitones;
        b[MOD_STEREO_WIDTH] = self.stereo_width_pct;
        b[MOD_MIX_AMOUNT] = self.fusion.mix_amount_pct;
        b[MOD_CROSS_SYNTH_AMOUNT] = self.fusion.cross_synth_amount_pct;
        b[MOD_CONVOLVE_AMOUNT] = self.fusion.convolve_amount_pct;
        b[MOD_RING_MOD_AMOUNT] = self.fusion.ring_mod_amount_pct;
        b[MOD_LOOP_LENGTH] = self.loop_length_seconds;
        b
    }
}

fn shortest_len(channels: &[Vec<f32>]) -> usize {
    channels.iter().map(Vec::len).min().unwrap_or(0)
}

fn rms(channels: &[Vec<f32>]) -> f32 {
    let count = channels.iter().map(Vec::len).sum::<usize>().max(1);
    (channels.iter().flatten().map(|v| v * v).sum::<f32>() / count as f32).sqrt()
}

/// Scales `channels` to `target_rms`, capped so the peak stays <= 1.0.
fn match_loudness(channels: &mut [Vec<f32>], target_rms: f32) {
    let current = rms(channels);
    let peak = channels.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    if current > 1e-9 && peak.is_finite() && peak > 0.0 {
        let scale = (target_rms / current).min(1.0 / peak);
        channels.iter_mut().flatten().for_each(|v| *v *= scale);
    }
}

/// `[start, end)` sample range for a start/length-or-end pair in seconds.
fn slice_range(total: usize, sample_rate: f32, start_sec: f32, end: Option<f32>) -> (usize, usize) {
    let start = ((start_sec.max(0.0) * sample_rate).round() as usize).min(total);
    let end = match end {
        Some(e) if e > 0.0 => ((e * sample_rate).round() as usize).clamp(start, total),
        _ => total,
    };
    (start, end)
}

fn slice_channels(channels: &[Vec<f32>], start: usize, end: usize) -> Vec<Vec<f32>> {
    channels.iter().map(|c| c[start.min(c.len())..end.min(c.len())].to_vec()).collect()
}

/// Always exactly two channels (mono is duplicated, extras are dropped).
fn to_stereo(mut channels: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    match channels.len() {
        0 => vec![Vec::new(), Vec::new()],
        1 => {
            let l = channels.remove(0);
            vec![l.clone(), l]
        }
        _ => {
            channels.truncate(2);
            channels
        }
    }
}

/// The trimmed source as-is: the plain-sampler output.
pub fn passthrough(source_a: &[Vec<f32>], sample_rate: f32, trim_start_sec: f32, trim_end_sec: f32) -> LoopBufferData {
    let total = shortest_len(source_a);
    let end = if trim_end_sec > trim_start_sec { Some(trim_end_sec) } else { None };
    let (start, end) = slice_range(total, sample_rate, trim_start_sec, end);
    LoopBufferData { channels: to_stereo(slice_channels(source_a, start, end)), sample_rate, root_note: DEFAULT_ROOT_NOTE }
}

/// Renders the whole chain. `source_a`/`source_b` are one `Vec<f32>` per
/// channel; an empty `source_b` degrades B-requiring fusion modes to Off
/// (see `fusion::effective_mode`, applied by the caller).
pub fn render_chain(params: &ChainParams, source_a: &[Vec<f32>], source_b: &[Vec<f32>], sample_rate: f32) -> LoopBufferData {
    // Resolve Source A: the FM synth's buffer stands in for the sample
    // everywhere below (the loaded sample itself is left untouched).
    let synth;
    let source_a: &[Vec<f32>] = match params.source_kind {
        SourceKind::Sample => source_a,
        SourceKind::FmSynth => {
            synth = render_fm_synth(&params.fm_synth, sample_rate, DEFAULT_ROOT_NOTE);
            &synth
        }
    };
    if !params.renders() || shortest_len(source_a) == 0 {
        // Trim only applies to a loaded sample; the synth buffer is generated
        // at the length asked for.
        let (start, end) = match params.source_kind {
            SourceKind::Sample => (params.trim_start_sec, params.trim_end_sec),
            SourceKind::FmSynth => (0.0, 0.0),
        };
        return passthrough(source_a, sample_rate, start, end);
    }

    // FM pre-stage on (a slice of) Sample A.
    let mut fm_output: Option<Vec<Vec<f32>>> = None;
    if params.fm_enabled {
        let total = shortest_len(source_a);
        let length_end = if params.fm_slice_length_sec > 0.0 { Some(params.fm_slice_start_sec.max(0.0) + params.fm_slice_length_sec) } else { None };
        let (start, end) = slice_range(total, sample_rate, params.fm_slice_start_sec, length_end);
        if end > start {
            let slice = slice_channels(source_a, start, end);
            let mut out = render_fm_transform(&slice, sample_rate, DEFAULT_ROOT_NOTE, &params.fm);
            // The transform's carrier runs at full amplitude whatever the
            // input level, roughly doubling the loudness and driving the
            // voice engine's limiter into distortion - match it back to the
            // input's loudness (never above full-scale peak).
            match_loudness(&mut out, rms(&slice));
            fm_output = Some(out);
        }
    }
    let a: &[Vec<f32>] = fm_output.as_deref().unwrap_or(source_a);

    let mut result = if let (Some(fm), false) = (&fm_output, params.spectral_enabled) {
        LoopBufferData { channels: fm.clone(), sample_rate, root_note: DEFAULT_ROOT_NOTE }
    } else {
        let tracks = sample_mod_tracks(&params.mod_base_values(), &params.mod_routes, params.loop_length_seconds);
        if tracks.mask == 0 {
            render_fused_loop(
                a,
                source_b,
                sample_rate,
                params.freeze_point_a_pct,
                params.volume_a_pct,
                params.tune_a_semitones,
                params.formant_shift_a_semitones,
                &params.fusion,
                params.stereo_width_pct,
                params.loop_length_seconds,
                DEFAULT_ROOT_NOTE,
            )
        } else {
            let mp = ModulationParams {
                freeze_point_a_pct: params.freeze_point_a_pct,
                volume_a_pct: params.volume_a_pct,
                tune_a_semitones: params.tune_a_semitones,
                formant_shift_a_semitones: params.formant_shift_a_semitones,
                fusion: params.fusion,
                stereo_width_pct: params.stereo_width_pct,
                loop_length_seconds: params.loop_length_seconds,
                root_note: DEFAULT_ROOT_NOTE,
            };
            if tracks.mask & (1 << MOD_LOOP_LENGTH) != 0 {
                render_fused_loop_length_modulated_loop(
                    a,
                    source_b,
                    sample_rate,
                    &mp,
                    &tracks.points,
                    tracks.num_points,
                    tracks.mask,
                    LOOP_LENGTH_SEGMENTS,
                    LOOP_LENGTH_CROSSFADE_SECONDS,
                )
            } else {
                render_fused_modulated_loop(a, source_b, sample_rate, &mp, &tracks.points, tracks.num_points, tracks.mask)
            }
        }
    };

    if params.percussion_enabled {
        result = render_percussion(&result, &params.percussion);
    }
    if params.micro_enabled {
        result = render_microtextures(&result, &params.micro);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modroute::ModShape;
    use crate::modulate::MOD_TUNE_A;

    const SR: f32 = 44_100.0;

    fn source(seconds: f32) -> Vec<Vec<f32>> {
        let n = (SR * seconds) as usize;
        let l: Vec<f32> = (0..n).map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / SR).sin() * 0.8).collect();
        vec![l.clone(), l]
    }

    fn assert_audible(buf: &LoopBufferData) {
        assert_eq!(buf.channels.len(), 2);
        assert!(!buf.channels[0].is_empty());
        assert!(buf.channels.iter().flatten().all(|v| v.is_finite()));
        assert!(buf.channels.iter().flatten().any(|v| v.abs() > 1e-4));
    }

    fn all_off() -> ChainParams {
        ChainParams { spectral_enabled: false, ..ChainParams::default() }
    }

    #[test]
    fn all_stages_off_passes_the_trimmed_source_through() {
        let src = source(2.0);
        let p = ChainParams { trim_start_sec: 0.5, trim_end_sec: 1.5, ..all_off() };
        let out = render_chain(&p, &src, &[], SR);
        assert_eq!(out.channels[0].len(), SR as usize);
        assert_eq!(out.channels[0][0], src[0][(0.5 * SR) as usize]);
    }

    #[test]
    fn trim_end_zero_means_to_the_end() {
        let src = source(1.0);
        let out = render_chain(&ChainParams { trim_start_sec: 0.25, ..all_off() }, &src, &[], SR);
        assert_eq!(out.channels[0].len(), src[0].len() - (0.25 * SR) as usize);
    }

    #[test]
    fn mono_passthrough_becomes_stereo() {
        let src = vec![source(0.5).remove(0)];
        let out = render_chain(&all_off(), &src, &[], SR);
        assert_eq!(out.channels.len(), 2);
        assert_eq!(out.channels[0], out.channels[1]);
    }

    #[test]
    fn spectral_only_renders_a_loop_of_the_requested_length() {
        let p = ChainParams { loop_length_seconds: 2.0, ..ChainParams::default() };
        let out = render_chain(&p, &source(2.0), &[], SR);
        assert_audible(&out);
        let secs = out.channels[0].len() as f32 / SR;
        assert!((secs - 2.0).abs() < 0.1, "loop was {secs}s");
    }

    #[test]
    fn fm_without_spectral_keeps_the_source_length() {
        let src = source(1.0);
        let p = ChainParams { fm_enabled: true, ..all_off() };
        let out = render_chain(&p, &src, &[], SR);
        assert_audible(&out);
        assert_eq!(out.channels[0].len(), src[0].len());
        assert!(p.suggests_one_shot());
    }

    #[test]
    fn fm_slice_limits_the_fm_output() {
        let p = ChainParams { fm_enabled: true, fm_slice_start_sec: 0.25, fm_slice_length_sec: 0.5, ..all_off() };
        let out = render_chain(&p, &source(1.0), &[], SR);
        assert_eq!(out.channels[0].len(), (0.5 * SR) as usize);
    }

    #[test]
    fn fm_into_spectral_renders() {
        let p = ChainParams { fm_enabled: true, loop_length_seconds: 1.0, ..ChainParams::default() };
        assert_audible(&render_chain(&p, &source(1.0), &[], SR));
    }

    #[test]
    fn percussion_produces_a_short_hit() {
        let p = ChainParams { percussion_enabled: true, ..ChainParams::default() };
        let out = render_chain(&p, &source(1.0), &[], SR);
        assert_audible(&out);
        assert!(out.channels[0].len() as f32 / SR <= p.percussion.length_seconds + 0.05);
    }

    #[test]
    fn microtextures_post_stage_renders() {
        let p = ChainParams { micro_enabled: true, loop_length_seconds: 1.0, ..ChainParams::default() };
        assert_audible(&render_chain(&p, &source(1.0), &[], SR));
    }

    #[test]
    fn modulated_and_loop_length_modulated_renders_are_audible() {
        let mut p = ChainParams { loop_length_seconds: 1.0, ..ChainParams::default() };
        p.mod_routes[0] = ModRoute { target: Some(MOD_TUNE_A), shape: ModShape::Lfo, depth: 3.0, rate_hz: 1.0, phase: 0.0, bipolar: true };
        assert_audible(&render_chain(&p, &source(1.0), &[], SR));
        p.mod_routes[1] = ModRoute { target: Some(MOD_LOOP_LENGTH), shape: ModShape::Ramp, depth: 0.5, rate_hz: 0.0, phase: 0.0, bipolar: true };
        assert_audible(&render_chain(&p, &source(1.0), &[], SR));
    }

    #[test]
    fn every_stage_at_once_renders() {
        let p = ChainParams { fm_enabled: true, percussion_enabled: true, micro_enabled: true, loop_length_seconds: 1.0, ..ChainParams::default() };
        assert_audible(&render_chain(&p, &source(1.0), &[], SR));
    }

    #[test]
    fn empty_source_never_panics() {
        let p = ChainParams { fm_enabled: true, percussion_enabled: true, micro_enabled: true, ..ChainParams::default() };
        let out = render_chain(&p, &[Vec::new()], &[], SR);
        assert_eq!(out.channels.len(), 2);
    }

    #[test]
    fn fm_source_with_every_stage_off_plays_the_synth_buffer() {
        let p = ChainParams { source_kind: SourceKind::FmSynth, trim_start_sec: 0.5, ..all_off() };
        let out = render_chain(&p, &source(0.1), &[], SR);
        let synth = render_fm_synth(&p.fm_synth, SR, DEFAULT_ROOT_NOTE);
        assert_eq!(out.channels, synth, "trim is ignored and the sample is not used");
    }

    #[test]
    fn fm_source_through_spectral_renders_a_loop() {
        let p = ChainParams { source_kind: SourceKind::FmSynth, loop_length_seconds: 1.0, ..ChainParams::default() };
        // An empty Sample A must not matter in FM mode.
        assert_audible(&render_chain(&p, &[Vec::new()], &[], SR));
    }

    #[test]
    fn fm_transform_output_is_level_matched_to_its_input() {
        let src = source(1.0); // a 0.8-peak sine, rms ~0.57
        let quiet: Vec<Vec<f32>> = src.iter().map(|c| c.iter().map(|v| v * 0.25).collect()).collect();
        let p = ChainParams { fm_enabled: true, ..all_off() };
        let out = render_chain(&p, &quiet, &[], SR);
        let (rin, rout) = (rms(&quiet), rms(&out.channels));
        assert!((rout - rin).abs() < 0.02, "in {rin} out {rout}");
        assert!(out.channels.iter().flatten().all(|v| v.abs() <= 1.0));
    }
}
