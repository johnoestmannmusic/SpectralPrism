//! Parameters for the instrument stages ported from SpectralPrism Tracker
//! (Source, FM, Spectral extras, Modulation, Percussion, MicroTextures,
//! Voice, Filter, Chord, Chorus). Every id is stage-prefixed and stable -
//! they're the automation/session keys, and the keys of a preset's
//! `stage_params` map (see `preset.rs`). Defaults match the tracker's
//! `default*Settings()` functions.
//!
//! Unit-interval tracker settings (0..1) are exposed here as 0-100 % for
//! readability and divided back down where the DSP settings are built.

use nih_plug::prelude::*;
use prism_dsp::chord::MAX_CHORD_VOICES;
use prism_dsp::fft::FFT_SIZES;
use prism_dsp::filter::cutoff_to_hz;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------- formatting ---
// Every slider shows at most one decimal place.

type V2s = Arc<dyn Fn(f32) -> String + Send + Sync>;
type S2v = Arc<dyn Fn(&str) -> Option<f32> + Send + Sync>;

/// Rounds to one decimal place (and never shows "-0.0").
pub(crate) fn one_dp(v: f32) -> String {
    format!("{:.1}", (v * 10.0).round() / 10.0 + 0.0)
}

/// One decimal place; the unit is added by the param (`with_unit`).
pub(crate) fn v2s_1dp() -> V2s {
    Arc::new(one_dp)
}

/// Whole numbers, for params that step in units (st, ct, Hz).
pub(crate) fn v2s_int() -> V2s {
    Arc::new(|v| format!("{:.0}", v.round() + 0.0))
}

/// A time in seconds: "450.0 ms" below one second, "1.3 s" above - so
/// short decays keep their precision at one decimal place. Use with no
/// `with_unit` (the unit is part of the text).
pub(crate) fn format_seconds(v: f32) -> String {
    if v.abs() < 1.0 {
        format!("{} ms", one_dp(v * 1000.0))
    } else {
        format!("{} s", one_dp(v))
    }
}

pub(crate) fn v2s_seconds() -> V2s {
    Arc::new(format_seconds)
}

/// Parses typed-in times: "450 ms", "0.45 s" or a bare number of seconds.
pub(crate) fn s2v_seconds() -> S2v {
    Arc::new(|text| {
        let t = text.trim().to_lowercase();
        if let Some(ms) = t.strip_suffix("ms") {
            ms.trim().parse::<f32>().ok().map(|v| v / 1000.0)
        } else {
            t.trim_end_matches('s').trim().parse().ok()
        }
    })
}

fn pct(name: &str, default: f32) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Linear { min: 0.0, max: 100.0 }).with_unit(" %").with_step_size(0.1).with_value_to_string(v2s_1dp())
}

/// A param in seconds gets the ms/s formatter; anything else one decimal.
fn formatted(param: FloatParam, unit: &'static str) -> FloatParam {
    if unit == " s" {
        param.with_value_to_string(v2s_seconds()).with_string_to_value(s2v_seconds())
    } else {
        param.with_unit(unit).with_value_to_string(v2s_1dp())
    }
}

fn lin(name: &str, default: f32, min: f32, max: f32, unit: &'static str) -> FloatParam {
    formatted(FloatParam::new(name, default, FloatRange::Linear { min, max }), unit)
}

fn skew(name: &str, default: f32, min: f32, max: f32, unit: &'static str) -> FloatParam {
    formatted(FloatParam::new(name, default, FloatRange::Skewed { min, max, factor: FloatRange::skew_factor(-1.5) }), unit)
}

fn semis(name: &str, default: f32, range: f32) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Linear { min: -range, max: range }).with_unit(" st").with_step_size(1.0).with_value_to_string(v2s_int())
}

fn seconds_or(zero_label: &'static str) -> V2s {
    Arc::new(move |v| if v <= 0.0 { zero_label.to_string() } else { format_seconds(v) })
}

// ---------------------------------------------------------------- enums ---

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlaybackModeParam {
    /// One-shot for hits (Percussion, FM alone, plain samples), loop for frozen pads.
    #[id = "auto"]
    #[name = "Auto"]
    Auto,
    #[id = "loop"]
    #[name = "Loop"]
    Loop,
    #[id = "ping-pong"]
    #[name = "Ping-pong"]
    PingPong,
    #[id = "one-shot"]
    #[name = "One-shot"]
    OneShot,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FmAlgorithmParam {
    #[id = "direct"]
    #[name = "Direct (sample > carrier)"]
    Direct,
    #[id = "stack"]
    #[name = "Stack (sample > op2 > carrier)"]
    Stack,
    #[id = "pair"]
    #[name = "Pair (op2 + sample > carrier)"]
    Pair,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FftSizeParam {
    #[id = "2048"]
    #[name = "Tight (2048)"]
    S2048,
    #[id = "8192"]
    #[name = "Smooth (8192)"]
    S8192,
    #[id = "32768"]
    #[name = "Smear (32768)"]
    S32768,
}

impl FftSizeParam {
    pub(crate) fn size(self) -> usize {
        FFT_SIZES[self as usize]
    }
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FreezeStyleParam {
    #[id = "lock"]
    #[name = "Phase-lock"]
    Lock,
    #[id = "shimmer"]
    #[name = "Shimmer"]
    Shimmer,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoiseColorParam {
    #[id = "white"]
    #[name = "White"]
    White,
    #[id = "pink"]
    #[name = "Pink"]
    Pink,
    #[id = "band-limited"]
    #[name = "Band-limited"]
    BandLimited,
}

/// `None` plus the 14 `prism_dsp::modulate::MOD_*` targets, in that order.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModTargetParam {
    #[id = "none"]
    #[name = "(none)"]
    None,
    #[id = "freezePointA"]
    #[name = "Freeze Point A"]
    FreezePointA,
    #[id = "freezePointB"]
    #[name = "Freeze Point B"]
    FreezePointB,
    #[id = "tuneA"]
    #[name = "Tune A"]
    TuneA,
    #[id = "tuneB"]
    #[name = "Tune B"]
    TuneB,
    #[id = "volumeA"]
    #[name = "Volume A"]
    VolumeA,
    #[id = "volumeB"]
    #[name = "Volume B"]
    VolumeB,
    #[id = "formantA"]
    #[name = "Formant A"]
    FormantA,
    #[id = "formantB"]
    #[name = "Formant B"]
    FormantB,
    #[id = "stereoWidth"]
    #[name = "Stereo Width"]
    StereoWidth,
    #[id = "mixAmount"]
    #[name = "Mix Amount"]
    MixAmount,
    #[id = "crossSynthAmount"]
    #[name = "Cross-Synth Amount"]
    CrossSynthAmount,
    #[id = "convolveAmount"]
    #[name = "Convolve Amount"]
    ConvolveAmount,
    #[id = "ringModAmount"]
    #[name = "Ring Mod Amount"]
    RingModAmount,
    #[id = "loopLength"]
    #[name = "Loop Length"]
    LoopLength,
}

impl ModTargetParam {
    /// The `MOD_*` index, or `None` for an empty slot.
    pub(crate) fn dsp_index(self) -> Option<usize> {
        (self as usize).checked_sub(1)
    }
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModShapeParam {
    #[id = "lfo"]
    #[name = "LFO"]
    Lfo,
    #[id = "ramp"]
    #[name = "Ramp"]
    Ramp,
    #[id = "random"]
    #[name = "Random"]
    Random,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FilterModeParam {
    #[id = "off"]
    #[name = "Off"]
    Off,
    #[id = "lowpass"]
    #[name = "Low-pass"]
    Lowpass,
    #[id = "highpass"]
    #[name = "High-pass"]
    Highpass,
    #[id = "bandpass"]
    #[name = "Band-pass"]
    Bandpass,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChordShapeParam {
    #[id = "major"]
    #[name = "Major"]
    Major,
    #[id = "minor"]
    #[name = "Minor"]
    Minor,
    #[id = "sus2"]
    #[name = "Sus2"]
    Sus2,
    #[id = "sus4"]
    #[name = "Sus4"]
    Sus4,
    #[id = "major7"]
    #[name = "Major 7"]
    Major7,
    #[id = "minor7"]
    #[name = "Minor 7"]
    Minor7,
    #[id = "dom7"]
    #[name = "Dominant 7"]
    Dom7,
    #[id = "add9"]
    #[name = "Add 9"]
    Add9,
    #[id = "power"]
    #[name = "Power"]
    Power,
    #[id = "open"]
    #[name = "Open"]
    Open,
    #[id = "custom"]
    #[name = "Custom"]
    Custom,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceModeParam {
    #[id = "sample"]
    #[name = "Sample"]
    Sample,
    #[id = "fm-synth"]
    #[name = "FM Synth"]
    FmSynth,
}

#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpWaveParam {
    #[id = "sine"]
    #[name = "Sine"]
    Sine,
    #[id = "square"]
    #[name = "Square"]
    Square,
    #[id = "triangle"]
    #[name = "Triangle"]
    Triangle,
    #[id = "saw"]
    #[name = "Saw"]
    Saw,
    #[id = "noise"]
    #[name = "Noise"]
    Noise,
}

// ---------------------------------------------------------- stage params ---

#[derive(Params)]
pub(crate) struct SourceParams {
    /// Sample A, or the FM synth rendered into a buffer in its place.
    #[id = "src_mode"]
    pub(crate) mode: EnumParam<SourceModeParam>,
    /// Trim applies to the plain sampler (Spectral/FM/Percussion/Texture all
    /// off); the spectral stages read the whole sample, as in the tracker.
    #[id = "src_trim_start"]
    pub(crate) trim_start: FloatParam,
    #[id = "src_trim_end"]
    pub(crate) trim_end: FloatParam,
    #[id = "src_playback"]
    pub(crate) playback: EnumParam<PlaybackModeParam>,
}

impl Default for SourceParams {
    fn default() -> Self {
        Self {
            mode: EnumParam::new("Source", SourceModeParam::Sample),
            trim_start: lin("Trim Start", 0.0, 0.0, 30.0, " s").with_step_size(0.01),
            trim_end: lin("Trim End", 0.0, 0.0, 30.0, "").with_step_size(0.01).with_value_to_string(seconds_or("end")).with_string_to_value(s2v_seconds()),
            playback: EnumParam::new("Playback", PlaybackModeParam::Auto),
        }
    }
}

#[derive(Params)]
pub(crate) struct FmParams {
    #[id = "fm_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "fm_algorithm"]
    pub(crate) algorithm: EnumParam<FmAlgorithmParam>,
    #[id = "fm_slice_start"]
    pub(crate) slice_start: FloatParam,
    #[id = "fm_slice_length"]
    pub(crate) slice_length: FloatParam,
    #[id = "fm_carrier_ratio"]
    pub(crate) carrier_ratio: FloatParam,
    #[id = "fm_index"]
    pub(crate) index: FloatParam,
    #[id = "fm_op2_ratio"]
    pub(crate) op2_ratio: FloatParam,
    #[id = "fm_op2_index"]
    pub(crate) op2_index: FloatParam,
    #[id = "fm_op2_feedback"]
    pub(crate) op2_feedback: FloatParam,
    #[id = "fm_follow"]
    pub(crate) follow: FloatParam,
    #[id = "fm_sway_rate"]
    pub(crate) sway_rate: FloatParam,
    #[id = "fm_sway_depth"]
    pub(crate) sway_depth: FloatParam,
    #[id = "fm_detune"]
    pub(crate) detune: FloatParam,
    #[id = "fm_mix"]
    pub(crate) mix: FloatParam,
    #[id = "fm_hold"]
    pub(crate) hold: IntParam,
    #[id = "fm_levels"]
    pub(crate) levels: IntParam,
}

impl Default for FmParams {
    fn default() -> Self {
        let ratio = |name: &str, default: f32| {
            FloatParam::new(name, default, FloatRange::Linear { min: 0.25, max: 8.0 })
                .with_step_size(0.25)
                .with_value_to_string(Arc::new(|v| format!("×{}", format!("{v:.3}").trim_end_matches('0').trim_end_matches('.'))))
        };
        Self {
            enabled: BoolParam::new("FM", false),
            algorithm: EnumParam::new("FM Algorithm", FmAlgorithmParam::Stack),
            slice_start: lin("FM Slice Start", 0.0, 0.0, 30.0, " s").with_step_size(0.01),
            slice_length: lin("FM Slice Length", 0.0, 0.0, 30.0, "").with_step_size(0.01).with_value_to_string(seconds_or("to end")).with_string_to_value(s2v_seconds()),
            carrier_ratio: ratio("Carrier Ratio", 1.0),
            index: lin("FM Index", 2.0, 0.0, 12.0, "").with_step_size(0.1),
            op2_ratio: ratio("Op2 Ratio", 2.0),
            op2_index: lin("Op2 Index", 1.0, 0.0, 6.0, "").with_step_size(0.1),
            op2_feedback: lin("Op2 Feedback", 0.0, 0.0, 1.5, "").with_step_size(0.05),
            follow: pct("Follow Envelope", 100.0),
            sway_rate: lin("Index Sway Rate", 0.0, 0.0, 2.0, "")
                .with_step_size(0.01)
                .with_value_to_string(Arc::new(|v| if v <= 0.0 { "off".into() } else { format!("{} Hz", one_dp(v)) })),
            sway_depth: pct("Index Sway Depth", 30.0),
            detune: lin("Stereo Detune", 0.0, 0.0, 25.0, " ct").with_step_size(1.0).with_value_to_string(v2s_int()),
            mix: pct("FM Mix", 100.0),
            hold: IntParam::new("Sample Hold", 1, IntRange::Linear { min: 1, max: 48 })
                .with_value_to_string(Arc::new(|v| if v <= 1 { "off".into() } else { format!("{v} smp") })),
            levels: IntParam::new("Quantise Levels", 0, IntRange::Linear { min: 0, max: 64 })
                .with_value_to_string(Arc::new(|v| if v < 2 { "off".into() } else { v.to_string() })),
        }
    }
}

/// Additions to the original Spectral params (which stay flat on
/// `PrismPluginParams` so their ids are untouched).
#[derive(Params)]
pub(crate) struct SpectralExtraParams {
    /// Default on: the plugin has always frozen, so old sessions/presets
    /// keep sounding the same.
    #[id = "spectral_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "spectral_fft_size"]
    pub(crate) fft_size: EnumParam<FftSizeParam>,
    #[id = "spectral_freeze_style"]
    pub(crate) freeze_style: EnumParam<FreezeStyleParam>,
    #[id = "shimmer_span"]
    pub(crate) shimmer_span: FloatParam,
    #[id = "shimmer_low_cut"]
    pub(crate) shimmer_low_cut: FloatParam,
    #[id = "shimmer_seed"]
    pub(crate) shimmer_seed: IntParam,
}

impl Default for SpectralExtraParams {
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("Spectral", true),
            fft_size: EnumParam::new("Smear (FFT)", FftSizeParam::S2048),
            freeze_style: EnumParam::new("Freeze Style", FreezeStyleParam::Lock),
            shimmer_span: pct("Shimmer Span", 100.0),
            shimmer_low_cut: lin("Shimmer Low Cut", 110.0, 0.0, 1000.0, " Hz").with_step_size(10.0).with_value_to_string(v2s_int()),
            shimmer_seed: IntParam::new("Shimmer Seed", 0, IntRange::Linear { min: 0, max: 9999 }),
        }
    }
}

/// One modulation route; four of these are nested as an array (ids get a
/// `_1`..`_4` suffix).
#[derive(Params)]
pub(crate) struct ModSlotParams {
    #[id = "mod_target"]
    pub(crate) target: EnumParam<ModTargetParam>,
    #[id = "mod_shape"]
    pub(crate) shape: EnumParam<ModShapeParam>,
    /// In the target's own units (st / % / s).
    #[id = "mod_depth"]
    pub(crate) depth: FloatParam,
    #[id = "mod_rate"]
    pub(crate) rate: FloatParam,
    #[id = "mod_phase"]
    pub(crate) phase: FloatParam,
    #[id = "mod_bipolar"]
    pub(crate) bipolar: BoolParam,
}

impl Default for ModSlotParams {
    fn default() -> Self {
        Self {
            target: EnumParam::new("Mod Target", ModTargetParam::None),
            shape: EnumParam::new("Mod Shape", ModShapeParam::Lfo),
            depth: lin("Mod Depth", 0.0, -100.0, 100.0, "").with_step_size(0.1),
            rate: skew("Mod Rate", 0.25, 0.01, 20.0, " Hz").with_step_size(0.01),
            phase: pct("Mod Phase", 0.0),
            bipolar: BoolParam::new("Mod Bipolar", true),
        }
    }
}

#[derive(Params)]
pub(crate) struct PercussionParams {
    #[id = "perc_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "perc_noise_amount"]
    pub(crate) noise_amount: FloatParam,
    #[id = "perc_noise_color"]
    pub(crate) noise_color: EnumParam<NoiseColorParam>,
    #[id = "perc_noise_decay"]
    pub(crate) noise_decay: FloatParam,
    #[id = "perc_transient_amount"]
    pub(crate) transient_amount: FloatParam,
    #[id = "perc_transient_decay"]
    pub(crate) transient_decay: FloatParam,
    #[id = "perc_transient_freq"]
    pub(crate) transient_freq: FloatParam,
    #[id = "perc_pitch_start"]
    pub(crate) pitch_start: FloatParam,
    #[id = "perc_pitch_end"]
    pub(crate) pitch_end: FloatParam,
    #[id = "perc_pitch_decay"]
    pub(crate) pitch_decay: FloatParam,
    #[id = "perc_amp_decay"]
    pub(crate) amp_decay: FloatParam,
    #[id = "perc_body_amount"]
    pub(crate) body_amount: FloatParam,
    #[id = "perc_partials"]
    pub(crate) partials: IntParam,
    #[id = "perc_partial_decay"]
    pub(crate) partial_decay: FloatParam,
    #[id = "perc_digital"]
    pub(crate) digital: FloatParam,
    #[id = "perc_drive"]
    pub(crate) drive: FloatParam,
    #[id = "perc_compress"]
    pub(crate) compress: FloatParam,
    #[id = "perc_width"]
    pub(crate) width: FloatParam,
    #[id = "perc_length"]
    pub(crate) length: FloatParam,
}

impl Default for PercussionParams {
    /// Defaults are the tracker's kick preset.
    fn default() -> Self {
        let k = prism_dsp::percussion::kick_preset();
        Self {
            enabled: BoolParam::new("Percussion", false),
            noise_amount: pct("Noise Amount", k.noise_amount_pct),
            noise_color: EnumParam::new("Noise Colour", NoiseColorParam::White),
            noise_decay: lin("Noise Decay", k.noise_decay_seconds, 0.0, 1.0, " s").with_step_size(0.005),
            transient_amount: pct("Transient Amount", k.transient_amount_pct),
            transient_decay: lin("Transient Decay", k.transient_decay_seconds, 0.0, 0.2, " s").with_step_size(0.001),
            transient_freq: skew("Transient Freq", k.transient_frequency_hz, 100.0, 12000.0, " Hz").with_step_size(10.0).with_value_to_string(v2s_int()),
            pitch_start: semis("Pitch Start", k.pitch_start_semitones, 48.0),
            pitch_end: semis("Pitch End", k.pitch_end_semitones, 48.0),
            pitch_decay: lin("Pitch Decay", k.pitch_decay_seconds, 0.0, 0.5, " s").with_step_size(0.002),
            amp_decay: lin("Amp Decay", k.amp_decay_seconds, 0.0, 2.0, " s").with_step_size(0.005),
            body_amount: pct("Body Amount", k.body_amount_pct),
            partials: IntParam::new("Partials", k.partial_count as i32, IntRange::Linear { min: 1, max: 32 }),
            partial_decay: lin("Partial Decay", k.partial_decay_seconds, 0.0, 2.0, " s").with_step_size(0.005),
            digital: pct("Digital", k.digital_amount_pct),
            drive: pct("Drive", k.drive_amount_pct),
            compress: pct("Compression", k.compress_amount_pct),
            width: pct("Hit Width", k.stereo_width_pct),
            length: lin("Hit Length", k.length_seconds, 0.05, 2.0, " s").with_step_size(0.01),
        }
    }
}

#[derive(Params)]
pub(crate) struct MicroTextureParams {
    #[id = "mt_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "mt_grain"]
    pub(crate) grain: FloatParam,
    #[id = "mt_density"]
    pub(crate) density: FloatParam,
    #[id = "mt_jitter"]
    pub(crate) jitter: FloatParam,
    #[id = "mt_reverse"]
    pub(crate) reverse: FloatParam,
    #[id = "mt_pitch_scatter"]
    pub(crate) pitch_scatter: FloatParam,
    #[id = "mt_pan_scatter"]
    pub(crate) pan_scatter: FloatParam,
    #[id = "mt_volume_variance"]
    pub(crate) volume_variance: FloatParam,
    #[id = "mt_density_mod_rate"]
    pub(crate) density_mod_rate: FloatParam,
    #[id = "mt_density_mod_depth"]
    pub(crate) density_mod_depth: FloatParam,
    #[id = "mt_chaos"]
    pub(crate) chaos: FloatParam,
    #[id = "mt_retrigger_rate"]
    pub(crate) retrigger_rate: FloatParam,
    #[id = "mt_retrigger_amount"]
    pub(crate) retrigger_amount: FloatParam,
    #[id = "mt_bit_depth"]
    pub(crate) bit_depth: IntParam,
    #[id = "mt_downsample"]
    pub(crate) downsample: IntParam,
    #[id = "mt_formant_shift"]
    pub(crate) formant_shift: FloatParam,
    #[id = "mt_formant_resonance"]
    pub(crate) formant_resonance: FloatParam,
    #[id = "mt_formant_mix"]
    pub(crate) formant_mix: FloatParam,
}

impl Default for MicroTextureParams {
    /// The tracker's `defaultMicroTextureSettings()`.
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("MicroTextures", false),
            grain: lin("Grain Size", 0.08, 0.005, 0.5, " s").with_step_size(0.005),
            density: lin("Density", 12.0, 0.5, 60.0, " Hz").with_step_size(0.5),
            jitter: pct("Jitter", 25.0),
            reverse: pct("Reverse", 20.0),
            pitch_scatter: lin("Pitch Scatter", 0.0, 0.0, 24.0, " st").with_step_size(0.5),
            pan_scatter: pct("Pan Scatter", 40.0),
            volume_variance: pct("Volume Variance", 30.0),
            density_mod_rate: lin("Density Mod Rate", 0.5, 0.05, 10.0, " Hz").with_step_size(0.05),
            density_mod_depth: pct("Density Mod Depth", 0.0),
            chaos: pct("Grain Chaos", 0.0),
            retrigger_rate: lin("Retrigger Rate", 8.0, 0.5, 30.0, " Hz").with_step_size(0.5),
            retrigger_amount: pct("Retrigger Amount", 0.0),
            bit_depth: IntParam::new("Bit Depth", 16, IntRange::Linear { min: 2, max: 16 }),
            downsample: IntParam::new("Downsample", 1, IntRange::Linear { min: 1, max: 32 }).with_unit("x"),
            formant_shift: semis("Texture Formant", 0.0, 24.0),
            formant_resonance: pct("Formant Resonance", 40.0),
            formant_mix: pct("Formant Mix", 50.0),
        }
    }
}

#[derive(Params)]
pub(crate) struct VoiceParams {
    #[id = "voice_transpose"]
    pub(crate) transpose: FloatParam,
    #[id = "voice_fine"]
    pub(crate) fine: FloatParam,
    #[id = "voice_vibrato_speed"]
    pub(crate) vibrato_speed: FloatParam,
    #[id = "voice_vibrato_depth"]
    pub(crate) vibrato_depth: FloatParam,
    /// Default on (the plugin was always polyphonic).
    #[id = "voice_polyphonic"]
    pub(crate) polyphonic: BoolParam,
    #[id = "voice_cap"]
    pub(crate) voice_cap: IntParam,
    #[id = "voice_choke"]
    pub(crate) choke: BoolParam,
}

impl Default for VoiceParams {
    fn default() -> Self {
        Self {
            transpose: semis("Transpose", 0.0, 48.0),
            fine: lin("Fine Tune", 0.0, -100.0, 100.0, " ct").with_step_size(1.0).with_value_to_string(v2s_int()),
            vibrato_speed: lin("Vibrato Speed", 6.0, 0.0, 20.0, " Hz").with_step_size(0.1),
            vibrato_depth: lin("Vibrato Depth", 0.0, 0.0, 12.0, " st").with_step_size(0.05),
            polyphonic: BoolParam::new("Polyphonic", true),
            voice_cap: IntParam::new("Voice Cap", prism_dsp::voice::MAX_VOICES as i32, IntRange::Linear { min: 1, max: prism_dsp::voice::MAX_VOICES as i32 }),
            choke: BoolParam::new("Choke", false),
        }
    }
}

#[derive(Params)]
pub(crate) struct FilterParams {
    #[id = "flt_mode"]
    pub(crate) mode: EnumParam<FilterModeParam>,
    #[id = "flt_cutoff"]
    pub(crate) cutoff: FloatParam,
    #[id = "flt_resonance"]
    pub(crate) resonance: FloatParam,
    #[id = "flt_env"]
    pub(crate) env_amount: FloatParam,
    #[id = "flt_attack"]
    pub(crate) attack: FloatParam,
    #[id = "flt_decay"]
    pub(crate) decay: FloatParam,
    #[id = "flt_sustain"]
    pub(crate) sustain: FloatParam,
    #[id = "flt_release"]
    pub(crate) release: FloatParam,
    #[id = "flt_key_follow"]
    pub(crate) key_follow: FloatParam,
}

impl Default for FilterParams {
    /// The tracker's `defaultLayerFilter()` (cutoff 0.75 normalised ≈ 3.6 kHz).
    fn default() -> Self {
        Self {
            mode: EnumParam::new("Filter", FilterModeParam::Off),
            cutoff: FloatParam::new("Cutoff", cutoff_to_hz(0.75), FloatRange::Skewed { min: 20.0, max: 20_000.0, factor: FloatRange::skew_factor(-2.0) })
                .with_value_to_string(Arc::new(|hz| if hz >= 1000.0 { format!("{} kHz", one_dp(hz / 1000.0)) } else { format!("{} Hz", one_dp(hz)) })),
            resonance: pct("Resonance", 20.0),
            env_amount: lin("Filter Env Amount", 0.0, -6.0, 6.0, " oct").with_step_size(0.1),
            attack: skew("Filter Attack", 0.005, 0.001, 5.0, " s"),
            decay: skew("Filter Decay", 0.3, 0.001, 10.0, " s"),
            sustain: pct("Filter Sustain", 0.0),
            release: skew("Filter Release", 0.3, 0.001, 10.0, " s"),
            key_follow: pct("Key Follow", 0.0),
        }
    }
}

#[derive(Params)]
pub(crate) struct ChordParams {
    #[id = "chord_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "chord_shape"]
    pub(crate) shape: EnumParam<ChordShapeParam>,
    #[id = "chord_inversion"]
    pub(crate) inversion: IntParam,
    #[id = "chord_octaves"]
    pub(crate) octaves: IntParam,
    #[id = "chord_detune"]
    pub(crate) detune: FloatParam,
    #[id = "chord_strum"]
    pub(crate) strum: FloatParam,
    #[id = "chord_pan_spread"]
    pub(crate) pan_spread: FloatParam,
    #[id = "chord_voices"]
    pub(crate) voices: IntParam,
    /// Root-relative semitones for the Custom shape, e.g. "0, 4, 7, 11".
    /// Persisted text rather than a param (it's a list, not a number).
    #[persist = "chord-custom-intervals"]
    pub(crate) custom_intervals: Mutex<String>,
}

impl Default for ChordParams {
    /// The tracker's `defaultChordSettings()`.
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("Chord", false),
            shape: EnumParam::new("Chord Shape", ChordShapeParam::Major),
            inversion: IntParam::new("Inversion", 0, IntRange::Linear { min: -2, max: 2 }),
            octaves: IntParam::new("Octaves", 1, IntRange::Linear { min: 1, max: 3 }),
            detune: lin("Chord Detune", 0.0, 0.0, 50.0, " ct").with_step_size(1.0).with_value_to_string(v2s_int()),
            strum: lin("Strum", 0.0, 0.0, 0.2, "").with_step_size(0.005).with_value_to_string(Arc::new(|v| format!("{} ms", one_dp(v * 1000.0)))),
            pan_spread: pct("Chord Pan Spread", 30.0),
            voices: IntParam::new("Chord Voices", 12, IntRange::Linear { min: 1, max: MAX_CHORD_VOICES as i32 }),
            custom_intervals: Mutex::new("0, 4, 7".to_string()),
        }
    }
}

#[derive(Params)]
pub(crate) struct ChorusParams {
    #[id = "cho_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "cho_rate"]
    pub(crate) rate: FloatParam,
    #[id = "cho_depth"]
    pub(crate) depth: FloatParam,
    #[id = "cho_delay"]
    pub(crate) delay: FloatParam,
    #[id = "cho_mix"]
    pub(crate) mix: FloatParam,
    #[id = "cho_spread"]
    pub(crate) spread: FloatParam,
}

impl Default for ChorusParams {
    /// The tracker's `defaultChorus()`.
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("Chorus", false),
            rate: lin("Chorus Rate", 0.6, 0.05, 8.0, " Hz").with_step_size(0.05),
            depth: lin("Chorus Depth", 2.5, 0.0, 10.0, " ms").with_step_size(0.1),
            delay: lin("Chorus Delay", 12.0, 1.0, 40.0, " ms").with_step_size(0.5),
            mix: pct("Chorus Mix", 50.0),
            spread: pct("Chorus Spread", 100.0),
        }
    }
}

/// Post-FX reverb (`prism_dsp::reverb`), after the chorus.
#[derive(Params)]
pub(crate) struct ReverbParams {
    #[id = "rev_enabled"]
    pub(crate) enabled: BoolParam,
    #[id = "rev_size"]
    pub(crate) size: FloatParam,
    #[id = "rev_decay"]
    pub(crate) decay: FloatParam,
    #[id = "rev_damping"]
    pub(crate) damping: FloatParam,
    #[id = "rev_predelay"]
    pub(crate) predelay: FloatParam,
    #[id = "rev_width"]
    pub(crate) width: FloatParam,
    #[id = "rev_mix"]
    pub(crate) mix: FloatParam,
}

impl Default for ReverbParams {
    fn default() -> Self {
        let d = prism_dsp::reverb::ReverbSettings::default();
        Self {
            enabled: BoolParam::new("Reverb", false),
            size: pct("Reverb Size", d.size_pct),
            decay: skew("Reverb Decay", d.decay_seconds, 0.3, 60.0, " s"),
            damping: FloatParam::new("Reverb Damping", d.damping_hz, FloatRange::Skewed { min: 500.0, max: 20_000.0, factor: FloatRange::skew_factor(-1.5) })
                .with_unit(" Hz")
                .with_step_size(10.0)
                .with_value_to_string(v2s_int()),
            predelay: lin("Reverb Pre-delay", d.predelay_seconds, 0.0, 0.5, " s").with_step_size(0.001),
            width: pct("Reverb Width", d.width_pct),
            mix: pct("Reverb Mix", d.mix_pct),
        }
    }
}

#[derive(Params)]
pub(crate) struct OutputParams {
    /// Master volume at the end of the chain (after the voices and chorus,
    /// *before* the safety soft limiter - so turning it down removes limiter
    /// distortion), in dB. The bottom of the range is silence.
    #[id = "output_volume"]
    pub(crate) volume: FloatParam,
}

/// Bottom of the Output Volume range, treated as silence.
pub(crate) const OUTPUT_MIN_DB: f32 = -60.0;
pub(crate) const OUTPUT_MAX_DB: f32 = 6.0;

impl Default for OutputParams {
    fn default() -> Self {
        Self {
            volume: FloatParam::new("Output Volume", 0.0, FloatRange::Linear { min: OUTPUT_MIN_DB, max: OUTPUT_MAX_DB })
                .with_step_size(0.1)
                .with_value_to_string(Arc::new(|db| if db <= OUTPUT_MIN_DB { "-inf dB".into() } else { format!("{db:+.1} dB") })),
        }
    }
}

/// dB (as set on `OutputParams::volume`) to linear gain.
pub(crate) fn output_gain(db: f32) -> f32 {
    if db <= OUTPUT_MIN_DB {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use crate::params::PrismPluginParams;

    /// Most digits after any decimal point in `text`.
    fn max_decimals(text: &str) -> usize {
        let chars: Vec<char> = text.chars().collect();
        let mut most = 0;
        for (i, c) in chars.iter().enumerate() {
            if *c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
                most = most.max(chars[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count());
            }
        }
        most
    }

    #[test]
    fn every_slider_shows_at_most_one_decimal_place() {
        let params = PrismPluginParams::default();
        // FM ratios move in quarter steps (x1.25, x0.75), so they keep two.
        let exceptions = ["fm_carrier_ratio", "fm_op2_ratio"];
        for (id, ptr, _) in params.param_map() {
            if exceptions.contains(&id.as_str()) {
                continue;
            }
            for i in 0..=40 {
                let n = i as f32 / 40.0;
                // SAFETY: `params` outlives the pointer.
                let text = unsafe { ptr.normalized_value_to_string(n, true) };
                assert!(max_decimals(&text) <= 1, "{id} shows {text:?}");
            }
        }
    }

    #[test]
    fn seconds_format_and_parse_round_trip() {
        assert_eq!(format_seconds(0.005), "5.0 ms");
        assert_eq!(format_seconds(0.3), "300.0 ms");
        assert_eq!(format_seconds(1.25), "1.3 s");
        let parse = s2v_seconds();
        assert_eq!(parse("300 ms"), Some(0.3));
        assert_eq!(parse("1.5 s"), Some(1.5));
        assert_eq!(parse("2"), Some(2.0));
        assert_eq!(one_dp(-0.04), "0.0");
    }
}

/// The FM Synth source's global settings.
#[derive(Params)]
pub(crate) struct FmSourceParams {
    #[id = "fms_length"]
    pub(crate) length: FloatParam,
    #[id = "fms_stereo"]
    pub(crate) stereo: FloatParam,
}

impl Default for FmSourceParams {
    fn default() -> Self {
        Self {
            length: lin("FM Source Length", 2.0, 0.5, 10.0, " s").with_step_size(0.1),
            stereo: lin("FM Source Stereo Spread", 0.0, 0.0, 50.0, " ct").with_step_size(1.0).with_value_to_string(v2s_int()),
        }
    }
}

/// One FM Synth source operator; four are nested as an array (ids get a
/// `_1`..`_4` suffix). `from1`..`from4` are this operator's modulation
/// inputs - `fms_from2_1` is "op 2 modulates op 1"; `fromN_N` is feedback.
#[derive(Params)]
pub(crate) struct FmOperatorParams {
    #[id = "fms_wave"]
    pub(crate) wave: EnumParam<OpWaveParam>,
    #[id = "fms_coarse"]
    pub(crate) coarse: IntParam,
    #[id = "fms_fine"]
    pub(crate) fine: FloatParam,
    #[id = "fms_detune"]
    pub(crate) detune: FloatParam,
    #[id = "fms_out"]
    pub(crate) output: FloatParam,
    #[id = "fms_from1"]
    pub(crate) from1: FloatParam,
    #[id = "fms_from2"]
    pub(crate) from2: FloatParam,
    #[id = "fms_from3"]
    pub(crate) from3: FloatParam,
    #[id = "fms_from4"]
    pub(crate) from4: FloatParam,
}

impl FmOperatorParams {
    /// Defaults for operator `index` (0-based), matching
    /// `prism_dsp::fm_synth::FmSynthParams::default()`: op 1 is heard and
    /// modulated by op 2; ops 3/4 sit at x2/x3, silent until routed.
    pub(crate) fn for_op(index: usize) -> Self {
        let d = prism_dsp::fm_synth::FmSynthParams::default().ops[index];
        let n = index + 1;
        let amount = |from: usize| pct(&format!("Op {from} > Op {n}"), d.mod_from[from - 1]);
        Self {
            wave: EnumParam::new(format!("Op {n} Wave"), OpWaveParam::Sine),
            coarse: IntParam::new(format!("Op {n} Ratio"), d.coarse as i32, IntRange::Linear { min: 0, max: 16 })
                .with_value_to_string(Arc::new(|v| if v <= 0 { "x0.5".into() } else { format!("x{v}") }))
                .with_string_to_value(Arc::new(|t| {
                    let v: f32 = t.trim().trim_start_matches(['x', 'X', '×']).parse().ok()?;
                    Some(if v < 1.0 { 0 } else { v.round() as i32 })
                })),
            fine: lin(&format!("Op {n} Fine"), d.fine_pct, 0.0, 99.0, " %").with_step_size(1.0).with_value_to_string(v2s_int()),
            detune: lin(&format!("Op {n} Detune"), d.detune_cents, -50.0, 50.0, " ct").with_step_size(1.0).with_value_to_string(v2s_int()),
            output: pct(&format!("Op {n} Output"), d.output_pct),
            from1: amount(1),
            from2: amount(2),
            from3: amount(3),
            from4: amount(4),
        }
    }

    pub(crate) fn from(&self, source: usize) -> &FloatParam {
        [&self.from1, &self.from2, &self.from3, &self.from4][source]
    }
}

#[cfg(test)]
mod fm_source_tests {
    use crate::params::{current_fm_synth, PrismPluginParams};

    #[test]
    fn plugin_fm_source_defaults_match_the_dsp_defaults() {
        assert_eq!(current_fm_synth(&PrismPluginParams::default()), prism_dsp::fm_synth::FmSynthParams::default());
    }
}
