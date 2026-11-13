//! Every host-visible parameter, plus the conversions from parameter values
//! to the DSP crate's render-time (`ChainParams`) and real-time settings.

use crate::render_worker::RenderRequest;
pub(crate) use crate::stage_params::*;
use prism_dsp::chain::ChainParams;
use prism_dsp::chord::{parse_intervals_into, ChordSettings, ChordShape};
use prism_dsp::chorus::ChorusSettings;
use prism_dsp::filter::{FilterMode, FilterSettings};
use prism_dsp::fm_transform::{FmAlgorithm, FmTransformParams};
use prism_dsp::microtextures::MicroTextureParams as DspMicroTextureParams;
use prism_dsp::modroute::{ModRoute, ModShape, MOD_SLOTS};
use prism_dsp::percussion::{NoiseColor, PercussionParams as DspPercussionParams};
use prism_dsp::shimmer::ShimmerParams;
use prism_dsp::voice::{AdsrSettings, PlaybackMode, VoiceManager, VoiceSettings};
use crate::theme::{BASE_EDITOR_HEIGHT, BASE_EDITOR_WIDTH, DEFAULT_SCALE};
use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use prism_dsp::fusion::{FusionMode as DspFusionMode, FusionRenderParams};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Which combination of Sample A's and Sample B's frozen spectra becomes
/// the plugin's output - see `prism_dsp::fusion` for what each mode
/// actually does under the hood. This is the automatable, host-visible
/// mirror of `prism_dsp::fusion::FusionMode` (that one has no nih_plug
/// dependency by design - see that crate's module doc comment); `to_dsp()`
/// converts at the render-request boundary (`RenderWorker::spawn`).
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FusionMode {
    #[id = "off"]
    #[name = "Off"]
    Off,
    #[id = "mix"]
    #[name = "Mix"]
    Mix,
    #[id = "cross-synth"]
    #[name = "Cross-Synth"]
    CrossSynth,
    #[id = "convolve"]
    #[name = "Convolve"]
    Convolve,
    #[id = "ring-modulate"]
    #[name = "Ring Modulate"]
    RingModulate,
    #[id = "spectral-max"]
    #[name = "Spectral Max"]
    SpectralMax,
    #[id = "spectral-min"]
    #[name = "Spectral Min"]
    SpectralMin,
    #[id = "cycle"]
    #[name = "Cycle"]
    Cycle,
}

impl Default for FusionMode {
    fn default() -> Self {
        FusionMode::Off
    }
}

impl FusionMode {
    pub(crate) fn to_dsp(self) -> DspFusionMode {
        match self {
            FusionMode::Off => DspFusionMode::Off,
            FusionMode::Mix => DspFusionMode::Mix,
            FusionMode::CrossSynth => DspFusionMode::CrossSynth,
            FusionMode::Convolve => DspFusionMode::Convolve,
            FusionMode::RingModulate => DspFusionMode::RingModulate,
            FusionMode::SpectralMax => DspFusionMode::SpectralMax,
            FusionMode::SpectralMin => DspFusionMode::SpectralMin,
            FusionMode::Cycle => DspFusionMode::Cycle,
        }
    }

    /// Every mode except `Off` needs Sample B loaded to sound different from
    /// plain Sample A - used to show the "load Sample B" warning in the
    /// editor (the render-worker side of this same guard lives in
    /// `RenderWorker::spawn`).
    pub(crate) fn needs_sample_b(self) -> bool {
        self != FusionMode::Off
    }

    /// The stable id this mode is saved/restored as in the plugin's own
    /// preset `.spjson` files - a string, not the enum itself, so a future
    /// variant reordering can't silently reinterpret an old preset (mirrors
    /// nih_plug's own `#[id]`-based automation-safety story for the real
    /// param). An unrecognized id (a preset from a newer plugin version, or
    /// a hand-edited file) falls back to `Off` rather than failing to load.
    pub(crate) fn preset_id(self) -> &'static str {
        match self {
            FusionMode::Off => "off",
            FusionMode::Mix => "mix",
            FusionMode::CrossSynth => "cross-synth",
            FusionMode::Convolve => "convolve",
            FusionMode::RingModulate => "ring-modulate",
            FusionMode::SpectralMax => "spectral-max",
            FusionMode::SpectralMin => "spectral-min",
            FusionMode::Cycle => "cycle",
        }
    }

    pub(crate) fn from_preset_id(id: &str) -> Self {
        match id {
            "mix" => FusionMode::Mix,
            "cross-synth" => FusionMode::CrossSynth,
            "convolve" => FusionMode::Convolve,
            "ring-modulate" => FusionMode::RingModulate,
            "spectral-max" => FusionMode::SpectralMax,
            "spectral-min" => FusionMode::SpectralMin,
            "cycle" => FusionMode::Cycle,
            _ => FusionMode::Off,
        }
    }

    /// Two teaching-oriented sentences describing the currently selected
    /// algorithm, shown in the editor's info box underneath Sample B -
    /// prefixed with "Single | <Name>: " for the one mode that only ever
    /// plays one sample unmodified (Off - Audition was removed as
    /// redundant with Mix at 100%, which does the same thing), or
    /// "Fusion | <Name>: " for every mode that actually combines A and B,
    /// so the box always names what's currently selected before explaining
    /// it.
    pub(crate) fn info_text(self) -> &'static str {
        match self {
            FusionMode::Off => "Single | Freeze: Only Sample A's frozen snapshot plays. A Freeze Point captures a single spectral instant of a sample and loops it forever, which is the whole idea behind SpectralPrism.",
            FusionMode::Mix => "Fusion | Mix: A's and B's independently frozen loops are crossfaded together using the Mix Blend slider. At 0% you hear pure A, at 100% pure B, and in between a simple volume blend of both.",
            FusionMode::CrossSynth => "Fusion | Cross-Synth: B's overall spectral shape (formants) is imposed onto A's fine detail and phase, so A keeps its texture but takes on B's tonal color. The Amount slider fades this reshaping in from A's own shape (0%) to B's shape (100%).",
            FusionMode::Convolve => "Fusion | Convolve: A's and B's frozen spectra are multiplied together bin by bin, which is how audio convolution works in the frequency domain. This tends to produce dense, resonant, often unpredictable new timbres, dialed in with the Amount slider.",
            FusionMode::RingModulate => "Fusion | Ring Modulation: A's and B's resynthesized loops are multiplied together sample by sample, the classic ring-modulation technique. This creates metallic, bell-like inharmonic tones, blended against plain A with the Amount slider.",
            FusionMode::SpectralMax => "Fusion | Spectral Max: At every frequency bin, whichever of A or B is louder there wins and is used in the output. The result favors each source's strongest frequencies, often sounding brighter or more aggressive than either alone.",
            FusionMode::SpectralMin => "Fusion | Spectral Min: At every frequency bin, whichever of A or B is quieter there wins and is used in the output. The result keeps only what both sources have in common, often sounding darker or thinner than either alone.",
            FusionMode::Cycle => "Fusion | Cycle: One full loop of A's frozen sound plays, then one full loop of B's, then it repeats - an alternating pattern rather than a blend. Good for rhythmic back-and-forth textures instead of a simultaneous combination.",
        }
    }
}

#[derive(Params)]
pub(crate) struct PrismPluginParams {
    #[persist = "editor-state"]
    pub(crate) editor_state: Arc<EguiState>,

    /// The loaded sample's file path, if any - not a `FloatParam`, but
    /// persisted the same way `editor_state` is (nih-plug's own state
    /// save/restore, backing both a host's project recall and preset
    /// files - see `FREEZE-PLAN-010`). Without this, reopening a saved
    /// project or preset would restore every param correctly but silently
    /// revert to the built-in placeholder tone. Re-decoded from disk in
    /// `PrismPlugin::initialize()`/`apply_preset()` when present - if the
    /// file has since moved or been deleted, that just falls back to
    /// whatever's already loaded rather than failing outright.
    #[persist = "sample-path"]
    pub(crate) sample_path: Mutex<Option<PathBuf>>,

    /// Name of the last preset loaded or saved, if any - persisted the same
    /// way as `sample_path` above, for the same reason: without this, a
    /// saved host project restores every param correctly (`FloatParam`s
    /// already persist on their own) but the editor's preset browser shows
    /// "(no preset)" after reopening, even though the actual sound came
    /// back fine. Purely a GUI display/navigation aid (drives the combo
    /// box's selection and Prev/Next's starting point) - never read by the
    /// DSP path.
    #[persist = "selected-preset"]
    pub(crate) selected_preset: Mutex<Option<String>>,

    /// Mirrors the editor's typable "Preset name..." field
    /// (`PrismEditorState::preset_name_input`) so its live contents -
    /// whatever's currently typed, whether or not it's been saved yet -
    /// survive a host project reload. `PrismEditorState` itself is
    /// GUI-local and rebuilt fresh every time the editor is (re)opened
    /// (see its own doc comment), so without a persisted copy to
    /// initialize from, that field would always come back blank on
    /// reload even though `selected_preset` above restores the combo
    /// button's own label correctly - they're two different pieces of
    /// state read by two different widgets. Synced from the editor's
    /// field every frame (cheap - one `String` clone into a `Mutex`),
    /// not just on save, so a name that's been typed but not yet saved
    /// also survives a reload.
    #[persist = "preset-name-input"]
    pub(crate) preset_name_input: Mutex<String>,

    /// Sample B's loaded file path, if any - mirrors `sample_path` above for
    /// the same reason (survives a host project save/reload and preset
    /// recall). `None` until the user loads a Sample B; unlike `sample_path`
    /// there's no synthetic placeholder to fall back to.
    #[persist = "sample-path-b"]
    pub(crate) sample_path_b: Mutex<Option<PathBuf>>,

    #[id = "freeze_point"]
    pub(crate) freeze_point: FloatParam,

    /// Post-normalization attenuation for Sample A - files are
    /// peak-normalized on load (`peak_normalize_channels`), so this only
    /// ever turns Sample A down from that normalized level, never up (its
    /// `FloatRange` tops out at 100%). See
    /// `prism_dsp::freeze::analyze_freeze_point`'s `gain_pct` doc comment
    /// for where this is actually applied.
    #[id = "sample_a_volume"]
    pub(crate) sample_a_volume: FloatParam,

    /// Retunes Sample A's input in semitones (decimal for microtonal),
    /// applied before Freezing/Fusion - lets two differently-pitched
    /// samples be lined up to the same tone. See
    /// `prism_dsp::resample::apply_tune`'s doc comment for the vari-speed
    /// technique this uses.
    #[id = "sample_a_tune"]
    pub(crate) sample_a_tune: FloatParam,

    #[id = "formant_shift"]
    pub(crate) formant_shift: FloatParam,

    #[id = "stereo_width"]
    pub(crate) stereo_width: FloatParam,

    /// Which Spectral Fusion algorithm combines Sample A's and Sample B's
    /// frozen spectra - see `FusionMode`/`prism_dsp::fusion`.
    #[id = "fusion_mode"]
    pub(crate) fusion_mode: EnumParam<FusionMode>,

    /// Sample B's own Freeze Point, independent of Sample A's - only
    /// meaningful once `fusion_mode` is anything but `Off`.
    #[id = "freeze_point_b"]
    pub(crate) freeze_point_b: FloatParam,

    /// Sample B's own Volume, mirroring `sample_a_volume` - post-
    /// normalization attenuation only, never a boost.
    #[id = "sample_b_volume"]
    pub(crate) sample_b_volume: FloatParam,

    /// Sample B's own Tune, mirroring `sample_a_tune`.
    #[id = "sample_b_tune"]
    pub(crate) sample_b_tune: FloatParam,

    /// Sample B's own Formant Shift, independent of Sample A's.
    #[id = "formant_shift_b"]
    pub(crate) formant_shift_b: FloatParam,

    /// Mix mode's A/B blend: 0% is plain A, 100% is plain B.
    #[id = "fusion_mix_amount"]
    pub(crate) fusion_mix_amount: FloatParam,

    /// Cross-Synth mode's dry/wet amount: 0% is plain A, 100% is B's
    /// spectral envelope fully imposed.
    #[id = "fusion_cross_synth_amount"]
    pub(crate) fusion_cross_synth_amount: FloatParam,

    /// Convolve mode's dry/wet amount: 0% is plain A, 100% is the full
    /// per-bin complex product of A's and B's spectra.
    #[id = "fusion_convolve_amount"]
    pub(crate) fusion_convolve_amount: FloatParam,

    /// Ring Modulate mode's dry/wet amount: 0% is plain A, 100% is the full
    /// (peak-normalized) sample-by-sample product of A's and B's loops.
    #[id = "fusion_ring_mod_amount"]
    pub(crate) fusion_ring_mod_amount: FloatParam,

    /// Length of the frozen loop buffer itself (not the FreezeFft/playback
    /// duration) - shorter values reduce both the in-memory buffer size and
    /// the exported WAV's file size. Purely a size/perceptual tradeoff, not
    /// a technical one: any hop-aligned length loops with the same
    /// continuity (see `render::render_frozen_loop`'s doc comment).
    #[id = "loop_length"]
    pub(crate) loop_length_seconds: FloatParam,

    #[id = "attack"]
    pub(crate) attack: FloatParam,

    #[id = "decay"]
    pub(crate) decay: FloatParam,

    #[id = "sustain"]
    pub(crate) sustain: FloatParam,

    #[id = "release"]
    pub(crate) release: FloatParam,

    #[id = "velocity_sensitivity"]
    pub(crate) velocity_sensitivity: FloatParam,

    /// Maximum semitones the pitch wheel can bend by (at full deflection
    /// either direction) - the wheel's own raw MIDI position isn't a param
    /// at all (it's a continuous performance control, not a settable
    /// value), only how far it's *allowed* to bend is.
    #[id = "pitch_bend_range"]
    pub(crate) pitch_bend_range_semitones: FloatParam,

    #[id = "pan_center"]
    pub(crate) pan_center_pct: FloatParam,

    #[id = "pan_width"]
    pub(crate) pan_width_pct: FloatParam,
    #[nested(group = "Source")]
    pub(crate) source: SourceParams,
    #[nested(group = "FM")]
    pub(crate) fm: FmParams,
    #[nested(group = "Spectral")]
    pub(crate) spectral: SpectralExtraParams,
    #[nested(array, group = "Modulation")]
    pub(crate) mod_slots: [ModSlotParams; MOD_SLOTS],
    #[nested(group = "Percussion")]
    pub(crate) percussion: PercussionParams,
    #[nested(group = "MicroTextures")]
    pub(crate) micro: MicroTextureParams,
    #[nested(group = "Voice")]
    pub(crate) voice: VoiceParams,
    #[nested(group = "Filter")]
    pub(crate) filter: FilterParams,
    #[nested(group = "Chord")]
    pub(crate) chord: ChordParams,
    #[nested(group = "Chorus")]
    pub(crate) chorus: ChorusParams,
    #[nested(group = "Output")]
    pub(crate) output: OutputParams,
}

impl Default for PrismPluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(
                (BASE_EDITOR_WIDTH as f32 * DEFAULT_SCALE) as u32,
                (BASE_EDITOR_HEIGHT as f32 * DEFAULT_SCALE) as u32,
            ),
            sample_path: Mutex::new(None),
            selected_preset: Mutex::new(None),
            preset_name_input: Mutex::new(String::new()),
            sample_path_b: Mutex::new(None),
            freeze_point: FloatParam::new("Freeze Point", 50.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            // 100% (unity, at the peak-normalized load level) is the
            // correct default - anything lower would silently attenuate a
            // freshly loaded sample for no reason. The range's own 100%
            // ceiling is what enforces "can only reduce, never boost".
            sample_a_volume: FloatParam::new("Volume", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            sample_a_tune: FloatParam::new("Tune", 0.0, FloatRange::Linear { min: -24.0, max: 24.0 })
                .with_unit(" st")
                .with_value_to_string(v2s_1dp()),
            formant_shift: FloatParam::new(
                "Formant Shift",
                0.0,
                FloatRange::Linear { min: -12.0, max: 12.0 },
            )
            .with_unit(" st")
                .with_value_to_string(v2s_1dp()),
            stereo_width: FloatParam::new("Stereo Width", 30.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            fusion_mode: EnumParam::new("Spectral Fusion", FusionMode::Off),
            freeze_point_b: FloatParam::new("Freeze Point B", 50.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            sample_b_volume: FloatParam::new("Volume", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            sample_b_tune: FloatParam::new("Tune", 0.0, FloatRange::Linear { min: -24.0, max: 24.0 })
                .with_unit(" st")
                .with_value_to_string(v2s_1dp()),
            formant_shift_b: FloatParam::new("Formant Shift B", 0.0, FloatRange::Linear { min: -12.0, max: 12.0 })
                .with_unit(" st")
                .with_value_to_string(v2s_1dp()),
            // A blend slider, not an intensity dial - 50/50 is the honest
            // "on" starting point.
            fusion_mix_amount: FloatParam::new("Mix Blend", 50.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            // Intensity dials on an already-deliberately-chosen effect -
            // selecting the mode itself communicates intent, so full-wet is
            // the sensible starting point.
            fusion_cross_synth_amount: FloatParam::new("Cross-Synth Amount", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            fusion_convolve_amount: FloatParam::new("Convolve Amount", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            fusion_ring_mod_amount: FloatParam::new("Ring Mod Amount", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            loop_length_seconds: FloatParam::new(
                "Loop Length",
                prism_dsp::render::DEFAULT_LOOP_SECONDS,
                FloatRange::Linear {
                    min: prism_dsp::render::MIN_LOOP_SECONDS,
                    max: prism_dsp::render::MAX_LOOP_SECONDS,
                },
            )
            .with_unit(" s")
                .with_value_to_string(v2s_1dp()),
            attack: FloatParam::new(
                "Attack",
                prism_dsp::voice::ATTACK_MS,
                FloatRange::Skewed { min: 1.0, max: 2000.0, factor: FloatRange::skew_factor(-2.0) },
            )
            .with_unit(" ms")
                .with_value_to_string(v2s_1dp()),
            decay: FloatParam::new(
                "Decay",
                prism_dsp::voice::DECAY_MS,
                FloatRange::Skewed { min: 1.0, max: 2000.0, factor: FloatRange::skew_factor(-2.0) },
            )
            .with_unit(" ms")
                .with_value_to_string(v2s_1dp()),
            sustain: FloatParam::new(
                "Sustain",
                prism_dsp::voice::SUSTAIN_LEVEL * 100.0,
                FloatRange::Linear { min: 0.0, max: 100.0 },
            )
            .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            release: FloatParam::new(
                "Release",
                prism_dsp::voice::RELEASE_MS,
                FloatRange::Skewed { min: 1.0, max: 5000.0, factor: FloatRange::skew_factor(-2.0) },
            )
            .with_unit(" ms")
                .with_value_to_string(v2s_1dp()),
            velocity_sensitivity: FloatParam::new(
                "Velocity Sensitivity",
                prism_dsp::voice::DEFAULT_VELOCITY_SENSITIVITY * 100.0,
                FloatRange::Linear { min: 0.0, max: 100.0 },
            )
            .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            pitch_bend_range_semitones: FloatParam::new(
                "Pitch Bend Range",
                2.0, // the MIDI/GM standard default of +/-2 semitones
                FloatRange::Linear { min: 0.0, max: 24.0 },
            )
            .with_unit(" st")
                .with_value_to_string(v2s_1dp()),
            pan_center_pct: FloatParam::new("Pan Center", 0.0, FloatRange::Linear { min: -100.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            pan_width_pct: FloatParam::new("Random Pan Width", 0.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %")
                .with_value_to_string(v2s_1dp()),
            source: SourceParams::default(),
            fm: FmParams::default(),
            spectral: SpectralExtraParams::default(),
            mod_slots: Default::default(),
            percussion: PercussionParams::default(),
            micro: MicroTextureParams::default(),
            voice: VoiceParams::default(),
            filter: FilterParams::default(),
            chord: ChordParams::default(),
            chorus: ChorusParams::default(),
            output: OutputParams::default(),
        }
    }
}

/// Every DSP-relevant param bundled into one `RenderRequest`, including the
/// current Spectral Fusion settings - shared by every call site that needs
/// to (re-)request a render (`load_sample_from_path`, `initialize()`,
/// `process()`) so this growing field list only needs to be assembled in
/// one place.
pub(crate) fn current_render_request(params: &PrismPluginParams) -> RenderRequest {
    let sp = &params.spectral;
    let fm = &params.fm;
    let pc = &params.percussion;
    let mt = &params.micro;
    let mut mod_routes = [ModRoute::default(); MOD_SLOTS];
    for (route, slot) in mod_routes.iter_mut().zip(params.mod_slots.iter()) {
        *route = ModRoute {
            target: slot.target.value().dsp_index(),
            shape: match slot.shape.value() {
                ModShapeParam::Lfo => ModShape::Lfo,
                ModShapeParam::Ramp => ModShape::Ramp,
                ModShapeParam::Random => ModShape::Random,
            },
            depth: slot.depth.value(),
            rate_hz: slot.rate.value(),
            phase: slot.phase.value() / 100.0,
            bipolar: slot.bipolar.value(),
        };
    }
    ChainParams {
        trim_start_sec: params.source.trim_start.value(),
        trim_end_sec: params.source.trim_end.value(),

        fm_enabled: fm.enabled.value(),
        fm: FmTransformParams {
            algorithm: match fm.algorithm.value() {
                FmAlgorithmParam::Direct => FmAlgorithm::Direct,
                FmAlgorithmParam::Stack => FmAlgorithm::Stack,
                FmAlgorithmParam::Pair => FmAlgorithm::Pair,
            },
            carrier_ratio: fm.carrier_ratio.value(),
            index: fm.index.value(),
            op2_ratio: fm.op2_ratio.value(),
            op2_index: fm.op2_index.value(),
            op2_feedback: fm.op2_feedback.value(),
            index_sway_hz: fm.sway_rate.value(),
            index_sway_depth: fm.sway_depth.value() / 100.0,
            follow: fm.follow.value() / 100.0,
            stereo_detune_cents: fm.detune.value(),
            mix_pct: fm.mix.value(),
            hold: fm.hold.value().max(1) as u32,
            levels: fm.levels.value().max(0) as u32,
        },
        fm_slice_start_sec: fm.slice_start.value(),
        fm_slice_length_sec: fm.slice_length.value(),

        spectral_enabled: sp.enabled.value(),
        freeze_point_a_pct: params.freeze_point.value(),
        volume_a_pct: params.sample_a_volume.value(),
        tune_a_semitones: params.sample_a_tune.value(),
        formant_shift_a_semitones: params.formant_shift.value(),
        stereo_width_pct: params.stereo_width.value(),
        loop_length_seconds: params.loop_length_seconds.value(),
        fusion: FusionRenderParams {
            mode: params.fusion_mode.value().to_dsp(),
            freeze_point_b_pct: params.freeze_point_b.value(),
            formant_shift_b_semitones: params.formant_shift_b.value(),
            volume_b_pct: params.sample_b_volume.value(),
            tune_b_semitones: params.sample_b_tune.value(),
            mix_amount_pct: params.fusion_mix_amount.value(),
            cross_synth_amount_pct: params.fusion_cross_synth_amount.value(),
            convolve_amount_pct: params.fusion_convolve_amount.value(),
            ring_mod_amount_pct: params.fusion_ring_mod_amount.value(),
            fft_size: sp.fft_size.value().size(),
            shimmer: ShimmerParams {
                enabled: sp.freeze_style.value() == FreezeStyleParam::Shimmer,
                span_pct: sp.shimmer_span.value(),
                low_cut_hz: sp.shimmer_low_cut.value(),
                seed: sp.shimmer_seed.value().max(0) as u32,
            },
        },
        mod_routes,

        percussion_enabled: pc.enabled.value(),
        percussion: DspPercussionParams {
            noise_amount_pct: pc.noise_amount.value(),
            noise_color: match pc.noise_color.value() {
                NoiseColorParam::White => NoiseColor::White,
                NoiseColorParam::Pink => NoiseColor::Pink,
                NoiseColorParam::BandLimited => NoiseColor::BandLimited,
            },
            noise_decay_seconds: pc.noise_decay.value(),
            transient_amount_pct: pc.transient_amount.value(),
            transient_decay_seconds: pc.transient_decay.value(),
            transient_frequency_hz: pc.transient_freq.value(),
            pitch_start_semitones: pc.pitch_start.value(),
            pitch_end_semitones: pc.pitch_end.value(),
            pitch_decay_seconds: pc.pitch_decay.value(),
            amp_decay_seconds: pc.amp_decay.value(),
            body_amount_pct: pc.body_amount.value(),
            partial_count: pc.partials.value().max(1) as usize,
            partial_decay_seconds: pc.partial_decay.value(),
            digital_amount_pct: pc.digital.value(),
            drive_amount_pct: pc.drive.value(),
            compress_amount_pct: pc.compress.value(),
            stereo_width_pct: pc.width.value(),
            length_seconds: pc.length.value(),
        },

        micro_enabled: mt.enabled.value(),
        micro: DspMicroTextureParams {
            grain_seconds: mt.grain.value(),
            density_hz: mt.density.value(),
            jitter: mt.jitter.value() / 100.0,
            reverse_probability: mt.reverse.value() / 100.0,
            pitch_scatter_semitones: mt.pitch_scatter.value(),
            pan_scatter: mt.pan_scatter.value() / 100.0,
            volume_variance: mt.volume_variance.value() / 100.0,
            density_mod_rate_hz: mt.density_mod_rate.value(),
            density_mod_depth: mt.density_mod_depth.value() / 100.0,
            grain_chaos: mt.chaos.value() / 100.0,
            retrigger_hz: mt.retrigger_rate.value(),
            retrigger_amount: mt.retrigger_amount.value() / 100.0,
            bit_depth: mt.bit_depth.value() as f32,
            downsample: mt.downsample.value() as f32,
            formant_shift_semitones: mt.formant_shift.value(),
            formant_resonance: mt.formant_resonance.value() / 100.0,
            formant_mix: mt.formant_mix.value() / 100.0,
        },
    }
}

/// Real-time voice settings. `chain` decides what `Auto` playback means:
/// a one-shot for hits (Percussion, FM without the freeze) and plain
/// samples, a loop for frozen pads and textures.
pub(crate) fn current_voice_settings(params: &PrismPluginParams, chain: &ChainParams) -> VoiceSettings {
    let v = &params.voice;
    VoiceSettings {
        playback: match params.source.playback.value() {
            PlaybackModeParam::Loop => PlaybackMode::Loop,
            PlaybackModeParam::PingPong => PlaybackMode::PingPong,
            PlaybackModeParam::OneShot => PlaybackMode::OneShot,
            PlaybackModeParam::Auto => {
                if chain.suggests_one_shot() || !chain.renders() {
                    PlaybackMode::OneShot
                } else {
                    PlaybackMode::Loop
                }
            }
        },
        transpose_semitones: v.transpose.value(),
        fine_cents: v.fine.value(),
        vibrato_hz: v.vibrato_speed.value(),
        vibrato_semitones: v.vibrato_depth.value(),
        polyphonic: v.polyphonic.value(),
        voice_cap: v.voice_cap.value().max(1) as usize,
        choke: v.choke.value(),
    }
}

pub(crate) fn current_filter_settings(params: &PrismPluginParams) -> FilterSettings {
    let f = &params.filter;
    FilterSettings {
        mode: match f.mode.value() {
            FilterModeParam::Off => FilterMode::Off,
            FilterModeParam::Lowpass => FilterMode::Lowpass,
            FilterModeParam::Highpass => FilterMode::Highpass,
            FilterModeParam::Bandpass => FilterMode::Bandpass,
        },
        // Hz back to the tracker's normalised 20 Hz..20 kHz exponential scale.
        cutoff: ((f.cutoff.value() / 20.0).max(1.0).ln() / 1000f32.ln()).clamp(0.0, 1.0),
        resonance: f.resonance.value() / 100.0,
        env_octaves: f.env_amount.value(),
        attack: f.attack.value(),
        decay: f.decay.value(),
        sustain: f.sustain.value() / 100.0,
        release: f.release.value(),
        key_follow: f.key_follow.value() / 100.0,
    }
}

/// Chord settings. Allocation-free (safe on the audio thread): the custom
/// interval text is parsed straight into the fixed array, and if the GUI
/// happens to hold the lock this block, the previous intervals are kept.
pub(crate) fn current_chord_settings(params: &PrismPluginParams, previous: &ChordSettings) -> ChordSettings {
    let c = &params.chord;
    let mut settings = ChordSettings {
        enabled: c.enabled.value(),
        shape: match c.shape.value() {
            ChordShapeParam::Major => ChordShape::Major,
            ChordShapeParam::Minor => ChordShape::Minor,
            ChordShapeParam::Sus2 => ChordShape::Sus2,
            ChordShapeParam::Sus4 => ChordShape::Sus4,
            ChordShapeParam::Major7 => ChordShape::Major7,
            ChordShapeParam::Minor7 => ChordShape::Minor7,
            ChordShapeParam::Dom7 => ChordShape::Dom7,
            ChordShapeParam::Add9 => ChordShape::Add9,
            ChordShapeParam::Power => ChordShape::Power,
            ChordShapeParam::Open => ChordShape::Open,
            ChordShapeParam::Custom => ChordShape::Custom,
        },
        custom: previous.custom,
        custom_len: previous.custom_len,
        inversion: c.inversion.value(),
        octaves: c.octaves.value(),
        detune_cents: c.detune.value(),
        strum_sec: c.strum.value(),
        pan_spread: c.pan_spread.value() / 100.0,
        voice_cap: c.voices.value().max(1) as usize,
    };
    if let Ok(text) = c.custom_intervals.try_lock() {
        settings.custom_len = parse_intervals_into(&text, &mut settings.custom);
    }
    settings
}

pub(crate) fn current_chorus_settings(params: &PrismPluginParams) -> ChorusSettings {
    let c = &params.chorus;
    ChorusSettings {
        enabled: c.enabled.value(),
        rate_hz: c.rate.value(),
        depth_ms: c.depth.value(),
        delay_ms: c.delay.value(),
        mix: c.mix.value() / 100.0,
        spread: c.spread.value() / 100.0,
    }
}

/// True when mod slot routes something at `target` with non-zero depth -
/// drives the "modulated" dot next to a slider.
pub(crate) fn is_modulated(params: &PrismPluginParams, target: ModTargetParam) -> bool {
    params.mod_slots.iter().any(|s| s.target.value() == target && s.depth.value() != 0.0)
}

/// Hands every real-time setting (envelope, velocity, pan, voice, filter,
/// chord, chorus) to `voices` - shared by `process()` and the WAV export so
/// an exported note is configured exactly like a played one. Returns the
/// chord settings used, to pass back in as `previous` next time.
pub(crate) fn apply_realtime_settings(voices: &mut VoiceManager, params: &PrismPluginParams, chain: &ChainParams, previous_chord: &ChordSettings) -> ChordSettings {
    voices.set_adsr(AdsrSettings {
        attack_ms: params.attack.value(),
        decay_ms: params.decay.value(),
        sustain_level: params.sustain.value() / 100.0,
        release_ms: params.release.value(),
    });
    voices.set_velocity_sensitivity(params.velocity_sensitivity.value() / 100.0);
    voices.set_pan_settings(params.pan_center_pct.value() / 100.0, params.pan_width_pct.value() / 100.0);
    voices.set_voice_settings(current_voice_settings(params, chain));
    voices.set_filter(current_filter_settings(params));
    let chord = current_chord_settings(params, previous_chord);
    voices.set_chord(chord);
    voices.set_chorus(current_chorus_settings(params));
    chord
}
