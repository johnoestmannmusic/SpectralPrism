mod render_worker;

use arc_swap::{ArcSwap, ArcSwapOption};
use prism_dsp::fusion::{render_fused_loop, FusionMode as DspFusionMode, FusionRenderParams};
use prism_dsp::render::{LoopBufferData, DEFAULT_ROOT_NOTE};
use prism_dsp::resample::resample_linear;
use prism_dsp::voice::{AdsrSettings, VoiceManager};
use nih_plug::prelude::*;
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::{create_egui_editor, egui, widgets, EguiState};
use render_worker::{RenderRequest, RenderTrigger, RenderWorker};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

/// Sentinel for "no note has been received yet" in `last_note`, distinct from
/// any real MIDI note number (0-127).
const NO_NOTE: u8 = 255;

/// YYYYMMDD, stamped at compile time by `build.rs` - shown in the editor's
/// heading and bottom-right corner, and saved into every exported preset,
/// so it's possible to tell which plugin version made a given sound.
const BUILD_NUMBER: &str = env!("SPECTRALPRISM_BUILD_NUMBER");

/// The loop buffer is baked from a source sample (the built-in placeholder
/// tone until a real file is loaded via the editor's "Load Sample" button)
/// and played polyphonically through `VoiceManager`, driven by real MIDI
/// note on/off. Freeze Point / Formant Shift / Stereo Width are real
/// automatable params, but re-rendering the frozen loop on every change is
/// too expensive for the audio thread - `process()` only *notices* a param
/// change and hands it to a background `RenderWorker`, which publishes the
/// new loop into `loop_buffer` (an `ArcSwap` so the audio thread can read
/// the latest version lock-free without ever blocking on the render).
pub struct PrismPlugin {
    params: Arc<PrismPluginParams>,
    /// The audio being frozen. An `ArcSwap` (not a plain `Arc`) so the
    /// editor's "Load Sample" button can swap in newly loaded audio without
    /// touching the audio thread - `RenderWorker` always reads whatever is
    /// current at the moment it renders.
    source: Arc<ArcSwap<Vec<Vec<f32>>>>,
    /// The real file's name once the user has loaded one, `None` while
    /// still on the built-in placeholder tone. Kept on the plugin (not in
    /// `PrismEditorState`) specifically so it survives the editor window
    /// being closed and reopened within the same plugin instance -
    /// `PrismEditorState` is recreated fresh every time `editor()` is
    /// called, but `source` itself (and therefore what's actually loaded)
    /// is not. Also doubles as the "[ Load Sample ]" sign's gate in
    /// `draw_freeze_point_waveform`.
    loaded_filename: Arc<ArcSwapOption<String>>,
    /// Sample B, for Spectral Fusion - mirrors `source` exactly, except
    /// there's no synthetic placeholder tone: it starts genuinely empty
    /// until the user loads one (see `RenderWorker::spawn`'s "no Sample B
    /// loaded" guard for what happens if a Fusion mode needing B is
    /// selected before that).
    source_b: Arc<ArcSwap<Vec<Vec<f32>>>>,
    /// Mirrors `loaded_filename` for Sample B.
    loaded_filename_b: Arc<ArcSwapOption<String>>,
    loop_buffer: Arc<ArcSwap<LoopBufferData>>,
    /// Exists from construction, independent of `worker` (which is only
    /// spawned once `initialize()` runs) - see `RenderWorker::spawn`'s doc
    /// comment for why that independence matters. The editor captures a
    /// clone of *this*, not something derived from `worker`, so it can
    /// trigger a render even if it was created before `initialize()` ran.
    trigger: RenderTrigger,
    worker: Option<RenderWorker>,
    voices: VoiceManager,
    sample_rate: f32,
    /// Raw MIDI pitch wheel position, normalized `[0, 1]` with `0.5` = no
    /// bend (nih-plug's own convention for `NoteEvent::MidiPitchBend`, see
    /// `process()`). Audio-thread-only - not a param, since the wheel's
    /// live position isn't a settable value, only how far it's *allowed* to
    /// bend is (`PrismPluginParams::pitch_bend_range_semitones`).
    pitch_bend_normalized: f32,
    /// The params a render has already been requested for (or that were
    /// used for the initial synchronous render). Compared against the
    /// current param values each block to detect a change at all.
    last_requested: RenderRequest,
    /// A detected change waiting for the throttle window to open. Always
    /// overwritten with the latest observed value (never queued), so once
    /// the throttle allows a send, it's always the most current params -
    /// never a stale intermediate value from partway through a drag.
    pending_request: Option<RenderRequest>,
    /// Counts down in samples; a new render is only actually requested once
    /// this reaches zero. See `RENDER_THROTTLE_MS` for why this exists.
    throttle_countdown: i64,
    /// Last MIDI note number received (`NO_NOTE` until the first one
    /// arrives), and the voice count read after this block's
    /// `VoiceManager::process_block` call - so a voice that just finished
    /// this block is already reflected. Shared with the editor via `Arc` so
    /// the GUI can poll them each frame without touching the audio thread,
    /// matching the pattern nih-plug's own `gain_gui_egui` example uses for
    /// its peak meter. Purely a diagnostic display (see FREEZE-BUG-001's
    /// false-alarm follow-up) - not read anywhere in the DSP path itself.
    last_note: Arc<AtomicU8>,
    active_voice_count: Arc<AtomicU8>,
}

/// Minimum spacing between actual render requests, regardless of how often
/// the params themselves change.
///
/// Without this, a host automating (or a human dragging) Freeze Point
/// smoothly triggers a fresh render on nearly every processed block - each
/// landing mid-crossfade from the *previous* one, so the crossfades never
/// get to finish and instead pile up into a continuous warble. This isn't a
/// bug in the crossfade itself: two different freeze-point positions are
/// genuinely different, unrelated spectral snapshots (not points on a smooth
/// continuum), so some character change when scrubbing is inherent to this
/// DSP approach. But letting each committed change's `BUFFER_CROSSFADE_MS`
/// window actually complete before the next one starts turns "constant
/// stutter" into "occasional smooth morph" - throttling to roughly
/// `BUFFER_CROSSFADE_MS` or a bit more keeps them from overlapping.
const RENDER_THROTTLE_MS: f32 = 100.0;

/// The editor's base (1x scale, minimum-drag-size) window size in logical
/// points - the size its two-column layout is designed to exactly fill.
/// Also the reference size `apply_gui_scale` divides the *actual* window
/// size by to compute the current scale factor. The window actually opens
/// at `DEFAULT_SCALE` times this (see `EguiState::from_size` below), not at
/// this size directly - this is just the floor `ResizableWindow`'s
/// `min_size` won't let it shrink past.
/// Widened and shortened from an earlier (620x560) guess, which was too
/// narrow for the two-column layout at scale (elements got cut off
/// horizontally) while leaving too much unused vertical space. Height
/// bumped again (460->600) after Pitch Bend Range/Pan Center/Pan Width
/// (`FREEZE-PLAN-019`/`FREEZE-PLAN-020`) added three more rows to the right
/// column, then once more (600->640) after the Import/Export Preset row
/// (`FREEZE-PLAN-022`) added one more row above the two-column section -
/// each time forcing a scroll to see the new controls at the default size
/// until bumped. The `ScrollArea` safety net (see `editor()`) means nothing
/// is ever actually clipped/lost in the meantime, just not visible without
/// scrolling.
/// Narrowed and shortened after moving every label beside its slider (one
/// row instead of two, via `egui::Grid` - see `param_row`) and moving the
/// MIDI diagnostic line into the bottom status bar - both freed up enough
/// space that the previous three-column-era size (1040x560) left a large
/// empty gap below the content. Height bumped again once Sample B's
/// waveform/Freeze Point/Formant Shift became always-visible (greyed out
/// rather than hidden when Off - see the Sample B block in `editor()`) so
/// the left column's height (and thus the window) no longer depends on
/// which Fusion mode is selected. Each size re-measured directly against a
/// screenshot of the actual rendered content at `DEFAULT_SCALE` rather than
/// guessed.
const BASE_EDITOR_WIDTH: u32 = 950;
const BASE_EDITOR_HEIGHT: u32 = 585;
/// The editor opens at this multiple of the base size by default (matching
/// `apply_gui_scale`'s scale factor, since the two are computed from the
/// same base) - requested directly ("too small to read" at 1x, then a
/// further +25% on top of the first bump). Still freely resizable larger
/// (or back down to 1x) afterward via the corner.
const DEFAULT_SCALE: f32 = 1.875;

/// Shared by `draw_freeze_point_waveform` and `draw_adsr_graph` so the two
/// side-by-side graph boxes always line up at the same height, regardless
/// of scale - requested directly after the ADSR graph's own (taller) 90.0
/// default made the two columns visibly mismatched. Trimmed from 70.0 as
/// part of reclaiming vertical space across the editor - both graphs are
/// informational, not primary controls, so a bit shorter still reads fine.
const GRAPH_HEIGHT: f32 = 55.0;

/// Fixed height (1x-scale points, multiplied by `scale` like everything
/// else) for the Spectral Fusion info box, so its position - right above
/// the Load Sample buttons - and the rest of the layout around it stay put
/// as the selected mode's description changes length. A `ScrollArea` inside
/// it is still the hard backstop against any description ever needing more
/// room than this. 60% of the original 56.0 - every description is capped
/// at two sentences, so the original height was more headroom than any of
/// them actually need.
const INFO_BOX_HEIGHT: f32 = 33.6;

/// Blank space around the outside of the whole editor's content, in 1x-scale
/// points (multiplied by `scale` like everything else) - requested directly
/// in Reaper, where content was otherwise touching the window edges.
/// `16.0 * DEFAULT_SCALE` (1.875) is exactly 30px, matching what was asked
/// for at the default scale.
const EDITOR_MARGIN: f32 = 16.0;

/// Light palette lifted from `src/0006/index.html`'s light-mode `:root`
/// overrides (`--bg`/`--panel`/`--edge`/`--edge-hover`/`--ink`/`--dim`/
/// `--surface-deep`) - this plugin's own visual reference, see
/// `FREEZE-PLAN-011`/`FREEZE-PLAN-013`. `COLOR_BG`/`COLOR_PANEL` are pure
/// white rather than 0006's own slightly-off-white `--bg` (`#fafafa`), per
/// explicit request. The accent is `COLOR_ACCENT`, not 0006's light-mode
/// `--amber` - see that constant's own doc comment.
const COLOR_BG: egui::Color32 = egui::Color32::from_rgb(0xff, 0xff, 0xff);
const COLOR_PANEL: egui::Color32 = egui::Color32::from_rgb(0xff, 0xff, 0xff);
const COLOR_EDGE: egui::Color32 = egui::Color32::from_rgb(0xdc, 0xdd, 0xe0);
const COLOR_EDGE_HOVER: egui::Color32 = egui::Color32::from_rgb(0xb3, 0xb8, 0xbd);
const COLOR_INK: egui::Color32 = egui::Color32::from_rgb(0x2e, 0x34, 0x40);
const COLOR_DIM: egui::Color32 = egui::Color32::from_rgb(0x5e, 0x64, 0x70);
/// Not 0006's own light-theme accent (`#3f9e0e`) - the specific green the
/// user asked for, which 0006 actually uses elsewhere as a hardcoded
/// "on/active" highlight (`.spectral-btn.fusion-on`, the mode-switch toggle).
const COLOR_ACCENT: egui::Color32 = egui::Color32::from_rgb(0x6c, 0xd7, 0x3c);
const COLOR_SURFACE_DEEP: egui::Color32 = egui::Color32::from_rgb(0xee, 0xf0, 0xe9);
const COLOR_ERROR: egui::Color32 = egui::Color32::from_rgb(0xc0, 0x36, 0x2f);

/// Bundled locally (copied from `src/0006/ASSETS/`) rather than referenced
/// across projects by relative path, so this plugin doesn't depend on the
/// rest of the repo layout to build correctly.
const MEDODICA_FONT_BYTES: &[u8] = include_bytes!("../assets/MedodicaRegular.otf");
const MEDODICA_FONT_NAME: &str = "Medodica";

/// Installs the bundled Medodica font (see `MEDODICA_FONT_BYTES`) as the
/// first choice for both the proportional and monospace font families -
/// 0006 uses it as its one font for everything (`--mono: 'Medodica', ...
/// monospace`), so every `egui::TextStyle` (which all default to
/// `FontFamily::Proportional` except `Monospace` itself) should pick it up
/// too. Called once, from the editor's `build` callback (not per-frame -
/// fonts don't need to be reloaded every pass).
fn install_medodica_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        MEDODICA_FONT_NAME.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(MEDODICA_FONT_BYTES)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().insert(0, MEDODICA_FONT_NAME.to_owned());
    }
    ctx.set_fonts(fonts);
}

/// Recolors every default `egui::Visuals` field this editor actually uses
/// to 0006's light palette (see the `COLOR_*` constants above), so buttons,
/// sliders, panels, and text match that project's look instead of egui's
/// stock dark theme. Called once, from the editor's `build` callback - the
/// palette itself never changes at runtime, only `apply_gui_scale` (sizes)
/// needs to run every frame.
fn apply_theme(ctx: &egui::Context) {
    install_medodica_font(ctx);
    ctx.all_styles_mut(|style| {
        let v = &mut style.visuals;
        v.dark_mode = false;
        v.window_fill = COLOR_BG;
        v.panel_fill = COLOR_BG;
        v.faint_bg_color = COLOR_PANEL;
        v.extreme_bg_color = COLOR_SURFACE_DEEP;
        v.code_bg_color = COLOR_SURFACE_DEEP;
        v.error_fg_color = COLOR_ERROR;
        v.warn_fg_color = COLOR_ACCENT;
        v.hyperlink_color = COLOR_ACCENT;
        v.selection.bg_fill = COLOR_ACCENT;
        // Dark ink, not the accent color itself - same reasoning as
        // `widgets.active.fg_stroke` below: the accent fill is bright
        // enough that same-color (or white) text on top of it is
        // unreadable. This is what makes a `selectable_label` (e.g. the
        // Preset Browser's tree) show its text once selected instead of
        // just a solid color swatch.
        v.selection.stroke = egui::Stroke::new(1.0, COLOR_INK);

        v.widgets.noninteractive.bg_fill = COLOR_PANEL;
        v.widgets.noninteractive.weak_bg_fill = COLOR_PANEL;
        v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, COLOR_EDGE);
        v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, COLOR_INK);

        v.widgets.inactive.bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.inactive.weak_bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, COLOR_EDGE);
        v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, COLOR_INK);

        v.widgets.hovered.bg_fill = COLOR_EDGE;
        v.widgets.hovered.weak_bg_fill = COLOR_EDGE;
        v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, COLOR_EDGE_HOVER);
        v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, COLOR_INK);

        v.widgets.active.bg_fill = COLOR_ACCENT;
        v.widgets.active.weak_bg_fill = COLOR_ACCENT;
        v.widgets.active.bg_stroke = egui::Stroke::new(1.0, COLOR_ACCENT);
        // Dark ink on the light-green accent fill reads better than white -
        // the accent is bright enough that white text washes out on it.
        v.widgets.active.fg_stroke = egui::Stroke::new(1.0, COLOR_INK);

        v.widgets.open.bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.open.weak_bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.open.bg_stroke = egui::Stroke::new(1.0, COLOR_ACCENT);
        v.widgets.open.fg_stroke = egui::Stroke::new(1.0, COLOR_INK);
    });
}

/// Which combination of Sample A's and Sample B's frozen spectra becomes
/// the plugin's output - see `prism_dsp::fusion` for what each mode
/// actually does under the hood. This is the automatable, host-visible
/// mirror of `prism_dsp::fusion::FusionMode` (that one has no nih_plug
/// dependency by design - see that crate's module doc comment); `to_dsp()`
/// converts at the render-request boundary (`RenderWorker::spawn`).
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
enum FusionMode {
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
    fn to_dsp(self) -> DspFusionMode {
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
    fn needs_sample_b(self) -> bool {
        self != FusionMode::Off
    }

    /// The stable id this mode is saved/restored as in the plugin's own
    /// preset `.spjson` files - a string, not the enum itself, so a future
    /// variant reordering can't silently reinterpret an old preset (mirrors
    /// nih_plug's own `#[id]`-based automation-safety story for the real
    /// param). An unrecognized id (a preset from a newer plugin version, or
    /// a hand-edited file) falls back to `Off` rather than failing to load.
    fn preset_id(self) -> &'static str {
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

    fn from_preset_id(id: &str) -> Self {
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
    fn info_text(self) -> &'static str {
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
struct PrismPluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

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
    sample_path: Mutex<Option<PathBuf>>,

    /// Name of the last preset loaded or saved, if any - persisted the same
    /// way as `sample_path` above, for the same reason: without this, a
    /// saved host project restores every param correctly (`FloatParam`s
    /// already persist on their own) but the editor's preset browser shows
    /// "(no preset)" after reopening, even though the actual sound came
    /// back fine. Purely a GUI display/navigation aid (drives the combo
    /// box's selection and Prev/Next's starting point) - never read by the
    /// DSP path.
    #[persist = "selected-preset"]
    selected_preset: Mutex<Option<String>>,

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
    preset_name_input: Mutex<String>,

    /// Sample B's loaded file path, if any - mirrors `sample_path` above for
    /// the same reason (survives a host project save/reload and preset
    /// recall). `None` until the user loads a Sample B; unlike `sample_path`
    /// there's no synthetic placeholder to fall back to.
    #[persist = "sample-path-b"]
    sample_path_b: Mutex<Option<PathBuf>>,

    #[id = "freeze_point"]
    pub freeze_point: FloatParam,

    /// Post-normalization attenuation for Sample A - files are
    /// peak-normalized on load (`peak_normalize_channels`), so this only
    /// ever turns Sample A down from that normalized level, never up (its
    /// `FloatRange` tops out at 100%). See
    /// `prism_dsp::freeze::analyze_freeze_point`'s `gain_pct` doc comment
    /// for where this is actually applied.
    #[id = "sample_a_volume"]
    pub sample_a_volume: FloatParam,

    /// Retunes Sample A's input in semitones (decimal for microtonal),
    /// applied before Freezing/Fusion - lets two differently-pitched
    /// samples be lined up to the same tone. See
    /// `prism_dsp::resample::apply_tune`'s doc comment for the vari-speed
    /// technique this uses.
    #[id = "sample_a_tune"]
    pub sample_a_tune: FloatParam,

    #[id = "formant_shift"]
    pub formant_shift: FloatParam,

    #[id = "stereo_width"]
    pub stereo_width: FloatParam,

    /// Which Spectral Fusion algorithm combines Sample A's and Sample B's
    /// frozen spectra - see `FusionMode`/`prism_dsp::fusion`.
    #[id = "fusion_mode"]
    pub fusion_mode: EnumParam<FusionMode>,

    /// Sample B's own Freeze Point, independent of Sample A's - only
    /// meaningful once `fusion_mode` is anything but `Off`.
    #[id = "freeze_point_b"]
    pub freeze_point_b: FloatParam,

    /// Sample B's own Volume, mirroring `sample_a_volume` - post-
    /// normalization attenuation only, never a boost.
    #[id = "sample_b_volume"]
    pub sample_b_volume: FloatParam,

    /// Sample B's own Tune, mirroring `sample_a_tune`.
    #[id = "sample_b_tune"]
    pub sample_b_tune: FloatParam,

    /// Sample B's own Formant Shift, independent of Sample A's.
    #[id = "formant_shift_b"]
    pub formant_shift_b: FloatParam,

    /// Mix mode's A/B blend: 0% is plain A, 100% is plain B.
    #[id = "fusion_mix_amount"]
    pub fusion_mix_amount: FloatParam,

    /// Cross-Synth mode's dry/wet amount: 0% is plain A, 100% is B's
    /// spectral envelope fully imposed.
    #[id = "fusion_cross_synth_amount"]
    pub fusion_cross_synth_amount: FloatParam,

    /// Convolve mode's dry/wet amount: 0% is plain A, 100% is the full
    /// per-bin complex product of A's and B's spectra.
    #[id = "fusion_convolve_amount"]
    pub fusion_convolve_amount: FloatParam,

    /// Ring Modulate mode's dry/wet amount: 0% is plain A, 100% is the full
    /// (peak-normalized) sample-by-sample product of A's and B's loops.
    #[id = "fusion_ring_mod_amount"]
    pub fusion_ring_mod_amount: FloatParam,

    /// Length of the frozen loop buffer itself (not the FreezeFft/playback
    /// duration) - shorter values reduce both the in-memory buffer size and
    /// the exported WAV's file size. Purely a size/perceptual tradeoff, not
    /// a technical one: any hop-aligned length loops with the same
    /// continuity (see `render::render_frozen_loop`'s doc comment).
    #[id = "loop_length"]
    pub loop_length_seconds: FloatParam,

    #[id = "attack"]
    pub attack: FloatParam,

    #[id = "decay"]
    pub decay: FloatParam,

    #[id = "sustain"]
    pub sustain: FloatParam,

    #[id = "release"]
    pub release: FloatParam,

    #[id = "velocity_sensitivity"]
    pub velocity_sensitivity: FloatParam,

    /// Maximum semitones the pitch wheel can bend by (at full deflection
    /// either direction) - the wheel's own raw MIDI position isn't a param
    /// at all (it's a continuous performance control, not a settable
    /// value), only how far it's *allowed* to bend is.
    #[id = "pitch_bend_range"]
    pub pitch_bend_range_semitones: FloatParam,

    #[id = "pan_center"]
    pub pan_center_pct: FloatParam,

    #[id = "pan_width"]
    pub pan_width_pct: FloatParam,
}

fn silent_loop_buffer() -> LoopBufferData {
    LoopBufferData { channels: Vec::new(), sample_rate: 1.0, root_note: DEFAULT_ROOT_NOTE }
}

impl Default for PrismPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(PrismPluginParams::default()),
            source: Arc::new(ArcSwap::new(Arc::new(Vec::new()))),
            loaded_filename: Arc::new(ArcSwapOption::from(None)),
            source_b: Arc::new(ArcSwap::new(Arc::new(Vec::new()))),
            loaded_filename_b: Arc::new(ArcSwapOption::from(None)),
            loop_buffer: Arc::new(ArcSwap::new(Arc::new(silent_loop_buffer()))),
            trigger: RenderTrigger::new(),
            worker: None,
            voices: VoiceManager::new(1.0, DEFAULT_ROOT_NOTE),
            sample_rate: 1.0,
            pitch_bend_normalized: 0.5,
            last_requested: RenderRequest {
                freeze_point_pct: 50.0,
                volume_pct: 100.0,
                tune_semitones: 0.0,
                formant_shift_semitones: 0.0,
                stereo_width_pct: 30.0,
                loop_length_seconds: prism_dsp::render::DEFAULT_LOOP_SECONDS,
                fusion: FusionRenderParams::default(),
            },
            pending_request: None,
            throttle_countdown: 0,
            last_note: Arc::new(AtomicU8::new(NO_NOTE)),
            active_voice_count: Arc::new(AtomicU8::new(0)),
        }
    }
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
                .with_unit(" %"),
            // 100% (unity, at the peak-normalized load level) is the
            // correct default - anything lower would silently attenuate a
            // freshly loaded sample for no reason. The range's own 100%
            // ceiling is what enforces "can only reduce, never boost".
            sample_a_volume: FloatParam::new("Volume", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            sample_a_tune: FloatParam::new("Tune", 0.0, FloatRange::Linear { min: -24.0, max: 24.0 })
                .with_unit(" st"),
            formant_shift: FloatParam::new(
                "Formant Shift",
                0.0,
                FloatRange::Linear { min: -12.0, max: 12.0 },
            )
            .with_unit(" st"),
            stereo_width: FloatParam::new("Stereo Width", 30.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            fusion_mode: EnumParam::new("Spectral Fusion", FusionMode::Off),
            freeze_point_b: FloatParam::new("Freeze Point B", 50.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            sample_b_volume: FloatParam::new("Volume", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            sample_b_tune: FloatParam::new("Tune", 0.0, FloatRange::Linear { min: -24.0, max: 24.0 })
                .with_unit(" st"),
            formant_shift_b: FloatParam::new("Formant Shift B", 0.0, FloatRange::Linear { min: -12.0, max: 12.0 })
                .with_unit(" st"),
            // A blend slider, not an intensity dial - 50/50 is the honest
            // "on" starting point.
            fusion_mix_amount: FloatParam::new("Mix Blend", 50.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            // Intensity dials on an already-deliberately-chosen effect -
            // selecting the mode itself communicates intent, so full-wet is
            // the sensible starting point.
            fusion_cross_synth_amount: FloatParam::new("Cross-Synth Amount", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            fusion_convolve_amount: FloatParam::new("Convolve Amount", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            fusion_ring_mod_amount: FloatParam::new("Ring Mod Amount", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
            loop_length_seconds: FloatParam::new(
                "Loop Length",
                prism_dsp::render::DEFAULT_LOOP_SECONDS,
                FloatRange::Linear {
                    min: prism_dsp::render::MIN_LOOP_SECONDS,
                    max: prism_dsp::render::MAX_LOOP_SECONDS,
                },
            )
            .with_unit(" s"),
            attack: FloatParam::new(
                "Attack",
                prism_dsp::voice::ATTACK_MS,
                FloatRange::Skewed { min: 1.0, max: 2000.0, factor: FloatRange::skew_factor(-2.0) },
            )
            .with_unit(" ms"),
            decay: FloatParam::new(
                "Decay",
                prism_dsp::voice::DECAY_MS,
                FloatRange::Skewed { min: 1.0, max: 2000.0, factor: FloatRange::skew_factor(-2.0) },
            )
            .with_unit(" ms"),
            sustain: FloatParam::new(
                "Sustain",
                prism_dsp::voice::SUSTAIN_LEVEL * 100.0,
                FloatRange::Linear { min: 0.0, max: 100.0 },
            )
            .with_unit(" %"),
            release: FloatParam::new(
                "Release",
                prism_dsp::voice::RELEASE_MS,
                FloatRange::Skewed { min: 1.0, max: 5000.0, factor: FloatRange::skew_factor(-2.0) },
            )
            .with_unit(" ms"),
            velocity_sensitivity: FloatParam::new(
                "Velocity Sensitivity",
                prism_dsp::voice::DEFAULT_VELOCITY_SENSITIVITY * 100.0,
                FloatRange::Linear { min: 0.0, max: 100.0 },
            )
            .with_unit(" %"),
            pitch_bend_range_semitones: FloatParam::new(
                "Pitch Bend Range",
                2.0, // the MIDI/GM standard default of +/-2 semitones
                FloatRange::Linear { min: 0.0, max: 24.0 },
            )
            .with_unit(" st"),
            pan_center_pct: FloatParam::new("Pan Center", 0.0, FloatRange::Linear { min: -100.0, max: 100.0 })
                .with_unit(" %"),
            pan_width_pct: FloatParam::new("Random Pan Width", 0.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit(" %"),
        }
    }
}

/// A short synthetic tone to freeze - the built-in placeholder until a real
/// file is loaded via the editor. Brightness (the balance between the
/// fundamental and its upper harmonics) sweeps over the tone's duration,
/// specifically so that different Freeze Point values capture genuinely
/// different-sounding moments - a *constant* tone would give every Freeze
/// Point nearly identical magnitude content differing only in essentially
/// arbitrary starting phase, which is a poor demonstration of what Freeze
/// Point is for.
fn synthetic_source(sample_rate: f32, seconds: f32) -> Vec<f32> {
    let len = (sample_rate * seconds) as usize;
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let brightness = i as f32 / len as f32; // 0.0 at start, 1.0 at end
            0.5 * (t * 220.0 * std::f32::consts::TAU).sin()
                + (0.05 + 0.5 * brightness) * (t * 440.0 * std::f32::consts::TAU).sin()
                + (0.5 * brightness) * (t * 660.0 * std::f32::consts::TAU).sin()
        })
        .collect()
}

/// Decodes a WAV file into one `Vec<f32>` per channel (interleaved samples
/// deinterleaved), normalizing integer formats to `[-1.0, 1.0]`. Returns the
/// file's own sample rate alongside - the caller is responsible for
/// resampling to the plugin's operating rate if they differ.
fn load_wav_channels(path: &Path) -> Result<(Vec<Vec<f32>>, f32), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("couldn't open WAV: {e}"))?;
    let spec = reader.spec();
    let num_channels = (spec.channels as usize).max(1);
    let sample_rate = spec.sample_rate as f32;

    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); num_channels];
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (i, sample) in reader.samples::<f32>().enumerate() {
                let s = sample.map_err(|e| format!("error reading sample: {e}"))?;
                channels[i % num_channels].push(s);
            }
        }
        hound::SampleFormat::Int => {
            let max_amplitude = (1i64 << (spec.bits_per_sample - 1)) as f32;
            for (i, sample) in reader.samples::<i32>().enumerate() {
                let s = sample.map_err(|e| format!("error reading sample: {e}"))? as f32 / max_amplitude;
                channels[i % num_channels].push(s);
            }
        }
    }

    if channels.iter().all(|c| c.is_empty()) {
        return Err("WAV file contains no audio samples".to_string());
    }

    Ok((channels, sample_rate))
}

/// Brings a loaded file's audio to the plugin's current operating rate -
/// without this, a file whose native rate differs from the host's would
/// play back pitch/speed-shifted, since `VoiceManager` reads the frozen
/// loop assuming it's already at the plugin's operating rate.
fn prepare_source_for_plugin_rate(channels: Vec<Vec<f32>>, file_rate: f32, plugin_rate: f32) -> Vec<Vec<f32>> {
    if (file_rate - plugin_rate).abs() < 0.5 {
        channels
    } else {
        channels.into_iter().map(|c| resample_linear(&c, file_rate, plugin_rate)).collect()
    }
}

/// Scales every channel by the same factor so the loudest sample across all
/// of them hits unity - so a quiet file and a loud file freeze to
/// comparable levels rather than one needing a much higher Freeze Point
/// "loudness" than the other by accident of how it happened to be recorded.
/// A silent file (all-zero, or already effectively silent) is left alone
/// rather than divided by ~0. This runs once at load time; the per-sample
/// Volume slider (`sample_a_volume`/`sample_b_volume`) then only ever
/// *attenuates* from this normalized level, applied at render time (see
/// `prism_dsp::freeze::analyze_freeze_point`'s `gain_pct` parameter) rather
/// than baked into the stored audio, so it can be freely readjusted without
/// re-loading the file.
fn peak_normalize_channels(channels: &mut [Vec<f32>]) {
    let peak = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 1e-6 {
        let scale = 1.0 / peak;
        for channel in channels.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= scale;
            }
        }
    }
}

/// Combines `load_wav_channels`, `prepare_source_for_plugin_rate`, and
/// `peak_normalize_channels` - the full "get a WAV file's audio ready to be
/// frozen at this sample rate" step, shared by the interactive "Load Sample
/// A/B..." flow, project-recall in `initialize()`, and preset recall in
/// `apply_preset()`.
fn load_and_prepare_sample(path: &Path, plugin_rate: f32) -> Result<Vec<Vec<f32>>, String> {
    let (channels, file_rate) = load_wav_channels(path)?;
    let mut prepared = prepare_source_for_plugin_rate(channels, file_rate, plugin_rate);
    peak_normalize_channels(&mut prepared);
    Ok(prepared)
}

/// Every DSP-relevant param bundled into one `RenderRequest`, including the
/// current Spectral Fusion settings - shared by every call site that needs
/// to (re-)request a render (`load_sample_from_path`, `initialize()`,
/// `process()`) so this growing field list only needs to be assembled in
/// one place.
fn current_render_request(params: &PrismPluginParams) -> RenderRequest {
    RenderRequest {
        freeze_point_pct: params.freeze_point.value(),
        volume_pct: params.sample_a_volume.value(),
        tune_semitones: params.sample_a_tune.value(),
        formant_shift_semitones: params.formant_shift.value(),
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
        },
    }
}

/// The full "a real file has just been chosen" flow, shared by the
/// interactive file dialogs (`open_sample_dialog`/`open_sample_dialog_b`)
/// and preset recall (`apply_preset`, when a preset references a sample):
/// decode/resample it, swap it into `source`, persist the path into
/// `sample_path_slot` (`PrismPluginParams::sample_path` for Sample A,
/// `::sample_path_b` for Sample B - see either field's doc comment) so it
/// survives a host project save/reload, trigger a re-render at every
/// current param (including the other sample slot's, untouched), and
/// update `loaded_filename` plus any editor error message. Takes the
/// source/path-slot/filename-handle explicitly (rather than hardcoding
/// Sample A's) so both slots share this one flow instead of duplicating it.
fn load_sample_from_path(
    path: &Path,
    source: &Arc<ArcSwap<Vec<Vec<f32>>>>,
    sample_path_slot: &Mutex<Option<PathBuf>>,
    loop_buffer: &Arc<ArcSwap<LoopBufferData>>,
    trigger: &RenderTrigger,
    params: &PrismPluginParams,
    loaded_filename: &Arc<ArcSwapOption<String>>,
    state: &mut PrismEditorState,
) {
    match load_and_prepare_sample(path, loop_buffer.load().sample_rate) {
        Ok(prepared) => {
            source.store(Arc::new(prepared));
            *sample_path_slot.lock().unwrap() = Some(path.to_path_buf());
            trigger.request_render(current_render_request(params));
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            loaded_filename.store(name.map(Arc::new));
            state.error = None;
        }
        Err(e) => state.error = Some(e),
    }
}

/// Writes the currently-playing frozen loop (`LoopBufferData`, the actual
/// rendered spectral snapshot - not the original source sample) to a WAV
/// file, for a downloader to import into another program (e.g. OpenMPT).
/// 16-bit PCM int, not 32-bit float, to match this project's existing
/// PCM-WAV export convention (0006's own sampler/sample-ZIP exports) and
/// for the widest possible compatibility with trackers/samplers - matching
/// `prism_cli`'s own WAV writer isn't done here since that one deliberately
/// stays float (a fast-iteration DSP test harness, not a distributable
/// export), and the two have no shared dependency to justify merging over.
/// Peak-normalizes the frozen loop before writing it out (same
/// `peak_normalize_channels` used at sample-load time) - the loop's overall
/// level is whatever Volume/Freeze Point/Fusion happened to produce, which
/// is rarely anywhere near 0 dBFS, so an un-normalized export would come out
/// quieter than it needs to for no audible benefit.
fn write_loop_buffer_wav(path: &Path, buffer: &LoopBufferData) -> Result<(), String> {
    if buffer.channels.is_empty() {
        return Err("nothing has been frozen yet - load a sample first".to_string());
    }
    let mut channels = buffer.channels.clone();
    peak_normalize_channels(&mut channels);
    let left = &channels[0];
    let right = channels.get(1).unwrap_or(left);

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: buffer.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| format!("couldn't create WAV file: {e}"))?;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let l_i16 = (l.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        let r_i16 = (r.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(l_i16).map_err(|e| format!("couldn't write sample: {e}"))?;
        writer.write_sample(r_i16).map_err(|e| format!("couldn't write sample: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("couldn't finalize WAV file: {e}"))
}

/// A named snapshot of every DSP-relevant param, plus the sample it was
/// made with (if any) - see `FREEZE-PLAN-010`. Deliberately doesn't include
/// `editor_state` (window geometry has nothing to do with the sound) or any
/// GUI-only state.
#[derive(serde::Serialize, serde::Deserialize)]
struct Preset {
    freeze_point_pct: f32,
    /// `#[serde(default = "default_volume_pct")]` (100%, unity at the
    /// peak-normalized load level) - a preset saved before Volume existed
    /// must still play at its original level, not silently attenuated.
    #[serde(default = "default_volume_pct")]
    volume_a_pct: f32,
    /// `#[serde(default)]` (0.0, i.e. untuned) - presets saved before Tune
    /// existed must still play at their original pitch.
    #[serde(default)]
    tune_a_semitones: f32,
    formant_shift_semitones: f32,
    stereo_width_pct: f32,
    /// `#[serde(default)]` with a custom default fn (not the bare
    /// `0.0` a plain `#[serde(default)]` would give) - unlike
    /// `pan_center_pct`/`pan_width_pct` where 0.0 is the semantically
    /// correct "off" value, 0.0 seconds is a nonsensical loop length. A
    /// preset saved before this param existed comes back at
    /// `DEFAULT_LOOP_SECONDS` instead, matching what that preset's sound
    /// actually used at the time.
    #[serde(default = "default_loop_length_seconds")]
    loop_length_seconds: f32,
    attack_ms: f32,
    decay_ms: f32,
    sustain_pct: f32,
    release_ms: f32,
    velocity_sensitivity_pct: f32,
    /// `#[serde(default)]` on these three: presets saved before they
    /// existed must still load rather than fail outright - they'll just
    /// come back with pitch bend disabled (range 0) and no panning (center/
    /// width 0), the same as a freshly-created instance would have before
    /// anyone touched these controls.
    #[serde(default)]
    pitch_bend_range_semitones: f32,
    #[serde(default)]
    pan_center_pct: f32,
    #[serde(default)]
    pan_width_pct: f32,
    /// `None` if the preset was saved while still on the built-in
    /// placeholder tone - recalling it then leaves whatever sample is
    /// already loaded untouched rather than resetting to the placeholder.
    sample_path: Option<PathBuf>,
    /// Which `BUILD_NUMBER` (YYYYMMDD) made this preset, so it's possible to
    /// tell which plugin version to blame if it doesn't sound right after
    /// importing into a later one. `#[serde(default)]` (an empty string)
    /// for presets saved before this existed - displayed as "unknown"
    /// rather than a blank/confusing value. Purely informational - never
    /// applied to anything, and never blocks an import.
    #[serde(default)]
    build_number: String,
    /// A stable id string (`FusionMode::preset_id`/`from_preset_id`), not
    /// the enum itself - a future variant reordering can't silently
    /// reinterpret an old preset. `#[serde(default)]` (an empty string,
    /// which `from_preset_id` treats the same as `"off"`) for presets saved
    /// before Spectral Fusion existed.
    #[serde(default)]
    fusion_mode: String,
    /// `#[serde(default = "default_freeze_point_b_pct")]`, matching
    /// `freeze_point_b`'s own `FloatParam` default (50.0) - unlike
    /// `formant_shift_b_semitones` below, 0.0 would put Sample B's Freeze
    /// Point at the very start of the sample, which isn't a neutral/no-op
    /// value the way 0.0 is for a shift or a percentage blend.
    #[serde(default = "default_freeze_point_b_pct")]
    freeze_point_b_pct: f32,
    /// Mirrors `volume_a_pct` above for Sample B.
    #[serde(default = "default_volume_pct")]
    volume_b_pct: f32,
    /// Mirrors `tune_a_semitones` for Sample B.
    #[serde(default)]
    tune_b_semitones: f32,
    #[serde(default)]
    formant_shift_b_semitones: f32,
    /// `#[serde(default = "default_fusion_mix_amount_pct")]`: Mix is a
    /// blend slider, not an intensity dial, so its neutral default is 50%
    /// (matching `fusion_mix_amount`'s own `FloatParam` default), not 0.0.
    #[serde(default = "default_fusion_mix_amount_pct")]
    fusion_mix_amount_pct: f32,
    /// `#[serde(default = "default_fusion_full_amount_pct")]` on these
    /// three: intensity dials on an already-deliberately-chosen effect, so
    /// their neutral default is 100% (full amount), matching each control's
    /// own `FloatParam` default - not 0.0, which would silently mean "no
    /// effect" for a preset that predates these controls but still had a
    /// Fusion mode selected (impossible today, but keeps the invariant that
    /// a preset missing a field behaves like a freshly-created instance).
    #[serde(default = "default_fusion_full_amount_pct")]
    fusion_cross_synth_amount_pct: f32,
    #[serde(default = "default_fusion_full_amount_pct")]
    fusion_convolve_amount_pct: f32,
    #[serde(default = "default_fusion_full_amount_pct")]
    fusion_ring_mod_amount_pct: f32,
    /// Mirrors `sample_path` for Sample B - `None` if no Sample B was ever
    /// loaded when this preset was saved.
    #[serde(default)]
    sample_path_b: Option<PathBuf>,
    /// When true, `apply_preset` unloads Sample A and Sample B entirely
    /// (clearing `source`/`source_b`, both `sample_path` params, and both
    /// `loaded_filename`s) instead of its normal behavior of leaving
    /// whatever's currently loaded untouched when `sample_path`/
    /// `sample_path_b` are `None`. Ordinary presets never need this - a
    /// preset without a referenced sample simply doesn't touch the current
    /// one, which is the right default for e.g. a "just the ADSR" preset -
    /// but a true "back to default" Init preset needs an explicit way to
    /// say "no, really, unload everything," which plain `None` can't
    /// express. `#[serde(default)]` (false) for presets saved before this
    /// existed, and left `false` by `Preset::capture` (never set by the
    /// ordinary Save flow) - only meaningful on a hand-authored preset file
    /// like the factory `Init.spjson`.
    #[serde(default)]
    unload_samples: bool,
    /// Browsable/sortable metadata, entered via the Save dialog - see
    /// `PrismEditorState::show_save_dialog`. `#[serde(default)]` (empty
    /// strings) for presets saved before the Preset Browser existed; the
    /// browser and info panel treat an empty `category`/`sub_category` as
    /// "Uncategorized" rather than a blank tree node. `title` defaults to
    /// the on-disk preset name itself when empty, so older presets still
    /// display something meaningful.
    #[serde(default)]
    title: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    sub_category: String,
    /// YYYY-MM-DD, set to `today_utc_date()` every time this preset is
    /// saved via the Save dialog - distinct from `build_number` (which
    /// plugin *version* made it) and computed at runtime, not compile time.
    /// `#[serde(default)]` (empty, displayed as "unknown") for presets
    /// saved before this existed.
    #[serde(default)]
    date_last_updated: String,
}

fn default_loop_length_seconds() -> f32 {
    prism_dsp::render::DEFAULT_LOOP_SECONDS
}

fn default_freeze_point_b_pct() -> f32 {
    50.0
}

fn default_volume_pct() -> f32 {
    100.0
}

fn default_fusion_mix_amount_pct() -> f32 {
    50.0
}

fn default_fusion_full_amount_pct() -> f32 {
    100.0
}

impl Preset {
    fn capture(params: &PrismPluginParams) -> Self {
        Self {
            freeze_point_pct: params.freeze_point.value(),
            volume_a_pct: params.sample_a_volume.value(),
            tune_a_semitones: params.sample_a_tune.value(),
            formant_shift_semitones: params.formant_shift.value(),
            stereo_width_pct: params.stereo_width.value(),
            loop_length_seconds: params.loop_length_seconds.value(),
            attack_ms: params.attack.value(),
            decay_ms: params.decay.value(),
            sustain_pct: params.sustain.value(),
            release_ms: params.release.value(),
            velocity_sensitivity_pct: params.velocity_sensitivity.value(),
            pitch_bend_range_semitones: params.pitch_bend_range_semitones.value(),
            pan_center_pct: params.pan_center_pct.value(),
            pan_width_pct: params.pan_width_pct.value(),
            sample_path: params.sample_path.lock().unwrap().clone(),
            build_number: BUILD_NUMBER.to_string(),
            fusion_mode: params.fusion_mode.value().preset_id().to_string(),
            freeze_point_b_pct: params.freeze_point_b.value(),
            volume_b_pct: params.sample_b_volume.value(),
            tune_b_semitones: params.sample_b_tune.value(),
            formant_shift_b_semitones: params.formant_shift_b.value(),
            fusion_mix_amount_pct: params.fusion_mix_amount.value(),
            fusion_cross_synth_amount_pct: params.fusion_cross_synth_amount.value(),
            fusion_convolve_amount_pct: params.fusion_convolve_amount.value(),
            fusion_ring_mod_amount_pct: params.fusion_ring_mod_amount.value(),
            sample_path_b: params.sample_path_b.lock().unwrap().clone(),
            // Never set by the ordinary Save flow - see the field's own doc
            // comment; only a hand-authored preset file sets this.
            unload_samples: false,
            // Not tied to any param - left blank here; the Save dialog
            // (the only place that actually writes a named library preset
            // to disk) fills these in on the `Preset` this returns before
            // calling `save_preset`.
            title: String::new(),
            author: String::new(),
            category: String::new(),
            sub_category: String::new(),
            date_last_updated: String::new(),
        }
    }
}

/// Today's date (UTC) as YYYY-MM-DD, for `Preset::date_last_updated`.
/// Computed from the system clock in plain Rust rather than shelling out to
/// `date` (unlike `build.rs`, which only ever runs once at compile time on
/// a build machine) - this runs at runtime, potentially every time a preset
/// is saved, from inside a plugin hosted in a DAW, where spawning a
/// subprocess on every save is a needless dependency on an external binary
/// being on `PATH` in whatever environment the host provides.
fn today_utc_date() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (year, month, day) = civil_from_days((secs / 86_400) as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Howard Hinnant's `civil_from_days` algorithm (public domain) - converts
/// a day count since the Unix epoch (1970-01-01) into a proleptic
/// Gregorian (year, month, day). The standard small, dependency-free way to
/// do this conversion without a date/time crate; see
/// <https://howardhinnant.github.io/date_algorithms.html>.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}

/// Presets live in a per-user directory rather than next to the plugin
/// bundle (which may not be writable, depending on install location),
/// created on first use. `None` if `$HOME` isn't set (or the directory
/// can't be created), which just disables preset save/load rather than
/// panicking - there's nowhere sensible to fall back to.
fn presets_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".config").join("SpectralPrism").join("presets");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// `.spjson` (not plain `.json`) - still JSON content, but a distinct
/// extension so SpectralPrism's own presets can be filtered out from a
/// user's other, unrelated `.json` files (in a file picker, a search, a
/// sync folder, etc).
const PRESET_FILE_EXTENSION: &str = "spjson";

fn preset_file_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.{PRESET_FILE_EXTENSION}"))
}

/// Sorted (so Prev/Next and the dropdown have a stable, predictable order)
/// list of preset names, without the `.spjson` extension, found in `dir`. An
/// unreadable directory (shouldn't happen once `presets_dir()` has
/// succeeded once, but e.g. permissions could change) just yields no
/// presets rather than an error - there's no interactive action to blame it
/// on, unlike an explicit save/load.
fn list_presets(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == PRESET_FILE_EXTENSION))
        .filter_map(|entry| entry.path().file_stem().map(|stem| stem.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    names
}

/// Writes a preset to an arbitrary path - shared by the named on-disk
/// library (`save_preset`, always under `presets_dir()`) and the
/// interactive "Export Preset..." file dialog (any path the user picks).
/// JSON, via the same `Preset`/serde type either route captures - see that
/// struct's `#[serde(default)]` fields for how this stays readable by
/// future plugin versions that add more parameters, and by past ones.
fn write_preset_file(path: &Path, preset: &Preset) -> Result<(), String> {
    let json = serde_json::to_string_pretty(preset).map_err(|e| format!("couldn't serialize preset: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("couldn't write preset file: {e}"))
}

/// Reads a preset from an arbitrary path - see `write_preset_file`.
fn read_preset_file(path: &Path) -> Result<Preset, String> {
    let json = std::fs::read_to_string(path).map_err(|e| format!("couldn't read preset file: {e}"))?;
    serde_json::from_str(&json).map_err(|e| format!("couldn't parse preset file: {e}"))
}

fn save_preset(dir: &Path, name: &str, preset: &Preset) -> Result<(), String> {
    write_preset_file(&preset_file_path(dir, name), preset)
}

fn load_preset(dir: &Path, name: &str) -> Result<Preset, String> {
    read_preset_file(&preset_file_path(dir, name))
}

fn delete_preset(dir: &Path, name: &str) -> Result<(), String> {
    std::fs::remove_file(preset_file_path(dir, name)).map_err(|e| format!("couldn't delete preset file: {e}"))
}

/// GUI-thread-only state for the editor - not shared with the audio thread
/// and not persisted. Which file is loaded lives on the plugin itself
/// (`PrismPlugin::loaded_filename`) instead, so it survives the editor
/// window being closed and reopened; the selected-preset name lives on
/// `PrismPluginParams` for the same reason (and so it round-trips through a
/// saved host project - see that field's doc comment). Only the load-error
/// message and the preset-name text field (fine to forget when the editor
/// is reopened - not sound-affecting state) stay here.
#[derive(Default)]
struct PrismEditorState {
    error: Option<String>,
    /// Text field buffer for naming a new preset to save - also doubles as
    /// the Save dialog's prefilled Title.
    preset_name_input: String,
    /// Whether the Save dialog (Title/Author/Category/Sub-category) is
    /// currently open - opened by clicking "Save", closed by its own
    /// Save/Cancel buttons or backdrop click/Escape (`egui::Modal`).
    show_save_dialog: bool,
    save_title_input: String,
    save_author_input: String,
    save_category_input: String,
    save_subcategory_input: String,
    /// Whether the Preset Browser modal is currently open.
    show_preset_browser: bool,
    /// Live search text, filtering the browser's Category/Sub-category/
    /// Preset tree by title/author substring match.
    preset_browser_search: String,
    /// Every named on-disk preset's browsable metadata - loaded once when
    /// the browser is opened (not re-read from disk every frame it's open).
    preset_browser_entries: Vec<PresetBrowserEntry>,
    /// Which preset (by its on-disk name, i.e. `PresetBrowserEntry::name`)
    /// is currently highlighted in the browser's tree, if any - drives the
    /// info panel at the bottom of the modal. Distinct from actually
    /// loading it, which only happens when "Load" is clicked.
    preset_browser_selected: Option<String>,
    /// Set to the on-disk name of a preset the user just clicked "Delete"
    /// on, so the browser can ask "are you sure?" before actually removing
    /// the file - cleared on confirm, cancel, or closing the browser.
    preset_browser_delete_confirm: Option<String>,
    /// Whether the "Export Preset..." strip-sample-paths prompt is
    /// currently open - opened by clicking "Export Preset...", closed by
    /// its own Export/Cancel buttons or backdrop click/Escape.
    show_export_preset_dialog: bool,
    /// Checkbox state in that prompt: whether to reduce
    /// `sample_path`/`sample_path_b` in the exported file down to just the
    /// filename (for sharing a preset publicly, where the recipient's
    /// absolute sample paths won't exist) - the filename itself is kept
    /// (not blanked entirely) so the info panel/relocate dialog can still
    /// show/suggest it; `apply_preset`'s existing missing-sample relocate
    /// flow already handles a path that doesn't resolve gracefully on
    /// re-import, prompting to locate the file rather than failing.
    export_strip_sample_paths: bool,
    /// Title/Author/Category/Sub-category/date-last-updated of whichever
    /// preset is currently active in memory - kept in sync everywhere
    /// `params.selected_preset` itself is (load/import/save), since
    /// `Preset::capture` deliberately leaves these blank (only the Save
    /// dialog fills them in directly, on the `Preset` it's about to write).
    /// "Export Preset..." needs this metadata to carry it into the
    /// exported file instead of exporting it blank.
    current_preset_title: String,
    current_preset_author: String,
    current_preset_category: String,
    current_preset_sub_category: String,
    current_preset_date_last_updated: String,
}

/// One preset's worth of metadata for the Preset Browser's tree/search/info
/// panel, read once per preset file when the browser opens - see
/// `load_preset_browser_entries`. `name` is the on-disk file stem (what
/// `load_preset`/`save_preset` key on); the rest mirror the matching
/// `Preset` fields.
#[derive(Clone)]
struct PresetBrowserEntry {
    name: String,
    title: String,
    author: String,
    category: String,
    sub_category: String,
}

/// Reads every named on-disk preset's metadata from `dir` - used to
/// populate the Preset Browser's tree when it opens. A preset file that
/// fails to parse (corrupted, or some other unrelated `.spjson` file) is
/// silently skipped rather than blocking the whole browser on one bad file.
fn load_preset_browser_entries(dir: &Path) -> Vec<PresetBrowserEntry> {
    list_presets(dir)
        .into_iter()
        .filter_map(|name| {
            let preset = load_preset(dir, &name).ok()?;
            let title = if preset.title.is_empty() { name.clone() } else { preset.title };
            Some(PresetBrowserEntry { name, title, author: preset.author, category: preset.category, sub_category: preset.sub_category })
        })
        .collect()
}

/// `category`/`sub_category` as they should actually be displayed/grouped
/// in the Preset Browser's tree - an empty string becomes "Uncategorized"
/// rather than a blank, easy-to-miss tree node.
fn display_category(category: &str) -> &str {
    if category.is_empty() {
        "Uncategorized"
    } else {
        category
    }
}

/// Every distinct, non-empty value already used for one metadata field
/// across every on-disk preset, sorted - powers the Save dialog's
/// "pick from existing" dropdown for Title/Author/Category/Sub-category,
/// alongside just typing a new value directly into the text field.
fn distinct_preset_values<'a>(entries: &'a [PresetBrowserEntry], field: impl Fn(&'a PresetBrowserEntry) -> &'a str) -> Vec<String> {
    let mut values: Vec<String> = entries.iter().map(field).filter(|s| !s.is_empty()).map(str::to_string).collect();
    values.sort();
    values.dedup();
    values
}

/// Multiplies built-in text sizes and interactive-widget spacing by `scale`,
/// so dragging `ResizableWindow`'s corner (see `editor()` below) genuinely
/// makes buttons/sliders/labels bigger too, not just the custom-drawn
/// waveform/ADSR graphs (which already grow via `ui.available_width()`).
///
/// This is necessary rather than just calling `egui::Context::
/// set_zoom_factor()`: this project's pinned `nih_plug_egui`/`egui_baseview`
/// revision renders using its *own* `pixels_per_point` field, tracked
/// entirely inside `egui_baseview`'s window loop and reset on every host
/// resize event from the fixed `WindowScalePolicy` nih_plug_egui chose at
/// window-creation time (`Some(1.0)` on Linux) - it never reads back
/// whatever `Context::set_zoom_factor()`/`pixels_per_point()` was set to
/// internally, so that call has no visible effect on this platform/version.
/// Scaling the actual "points" sizes (fonts, spacing, and our own custom
/// draw dimensions) is the one lever that reaches every widget regardless.
///
/// Rebuilt from `egui::Style::default()`'s reference values every call
/// (not compounded onto whatever the style already is) so it stays correct
/// as the window is dragged to any size, in either direction, and applied
/// via `all_styles_mut` so it doesn't clobber dark/light `Visuals`.
fn apply_gui_scale(ctx: &egui::Context, scale: f32) {
    let base = egui::Style::default();
    ctx.all_styles_mut(|style| {
        for (text_style, font_id) in style.text_styles.iter_mut() {
            if let Some(base_font) = base.text_styles.get(text_style) {
                font_id.size = base_font.size * scale;
            }
        }
        style.spacing.item_spacing = base.spacing.item_spacing * scale;
        style.spacing.button_padding = base.spacing.button_padding * scale;
        style.spacing.interact_size = base.spacing.interact_size * scale;
        style.spacing.slider_width = base.spacing.slider_width * scale;
    });
}

/// Width (1x-scale points) of `draw_vertical_volume_slider` - drawn inside
/// the same `ui.horizontal` as `draw_freeze_point_waveform`, so the
/// waveform (which sizes itself to whatever `ui.available_width()` is left
/// once this has claimed its own space) automatically ends up narrower by
/// exactly this much, no manual arithmetic needed at the call site.
const VOLUME_SLIDER_WIDTH: f32 = 28.0;

/// One "label | slider" row inside an `egui::Grid` - the label goes in the
/// grid's first column (sized to whichever label in that grid is widest)
/// and the slider in the second (so every slider in the same grid starts
/// at the same x, not just sits below its own label), then advances to the
/// next row. Callers still need their own `egui::Grid::new(...).show(ui,
/// |ui| { ... })` around a run of these - this only draws one row's worth.
fn param_row(ui: &mut egui::Ui, label: &str, param: &FloatParam, setter: &ParamSetter) {
    ui.label(label);
    ui.add(widgets::ParamSlider::for_param(param, setter));
    ui.end_row();
}

/// A vertical fader for a sample's Volume, meant to sit directly to the
/// left of that sample's `draw_freeze_point_waveform` (`nih_plug_egui`'s
/// `ParamSlider` is horizontal-only - see its own doc comment - so this is
/// hand-painted the same way `draw_freeze_point_waveform`/`draw_adsr_graph`
/// already are). A fixed-size percentage readout sits above a click/drag
/// track - dragging (or clicking) anywhere in the track sets the value via
/// the same begin/set/end-normalized pattern every other custom control
/// here uses, with the fill growing up from the bottom (louder = taller, a
/// fader without a legend needed).
fn draw_vertical_volume_slider(ui: &mut egui::Ui, volume: &FloatParam, setter: &ParamSetter, height: f32, scale: f32) {
    let width = VOLUME_SLIDER_WIDTH * scale;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
    let painter = ui.painter();

    let readout_height = 14.0 * scale;
    let track_rect = egui::Rect::from_min_max(egui::pos2(rect.left(), rect.top() + readout_height), rect.max);

    painter.text(
        egui::pos2(rect.center().x, rect.top() + readout_height * 0.5),
        egui::Align2::CENTER_CENTER,
        format!("{:.0}%", volume.value()),
        egui::FontId::proportional(9.0 * scale),
        COLOR_DIM,
    );

    painter.rect_filled(track_rect, 2.0, COLOR_SURFACE_DEEP);
    let value = volume.unmodulated_normalized_value();
    let fill_top = track_rect.bottom() - track_rect.height() * value;
    let fill_rect = egui::Rect::from_min_max(egui::pos2(track_rect.left(), fill_top), track_rect.max);
    painter.rect_filled(fill_rect, 2.0, COLOR_ACCENT);
    painter.rect_stroke(track_rect, 2.0, egui::Stroke::new(1.0, COLOR_EDGE), egui::StrokeKind::Inside);

    if response.drag_started() || response.clicked() {
        setter.begin_set_parameter(volume);
    }
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let normalized = 1.0 - ((pos.y - track_rect.top()) / track_rect.height().max(1.0)).clamp(0.0, 1.0);
            setter.set_parameter_normalized(volume, normalized);
        }
    }
    if response.drag_stopped() || response.clicked() {
        setter.end_set_parameter(volume);
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
}

/// Draws the loaded source's waveform (min/max per pixel column, since the
/// source is almost always much longer than the display is wide) with a
/// vertical marker at Freeze Point's current position, so the user can see
/// *what* they're about to freeze rather than reading a bare percentage.
/// Click/drag directly on it to set Freeze Point, using the same
/// begin/set/end-normalized pattern a built-in `ParamSlider` uses
/// internally, just driven by pixel position instead of a slider track.
///
/// Until the user has explicitly loaded a real file (`has_loaded_sample`)
/// there's nothing meaningful to show or drag - the plugin always has
/// *some* audio in `source` even then (the built-in placeholder tone,
/// loaded at `initialize()` before the editor can even open), so checking
/// whether `source` itself is empty would never actually trigger this.
/// Instead this area becomes a clickable "[ Load Sample ]" sign - returns
/// `true` on the frame it's clicked, so the caller (which owns the actual
/// file dialog / decode logic, shared with the "Load Sample..." button
/// below) can open it in response.
#[allow(clippy::too_many_arguments)]
fn draw_freeze_point_waveform(
    ui: &mut egui::Ui,
    source: &Arc<ArcSwap<Vec<Vec<f32>>>>,
    freeze_point: &FloatParam,
    volume_pct: f32,
    setter: &ParamSetter,
    has_loaded_sample: bool,
    scale: f32,
) -> bool {
    let volume = (volume_pct / 100.0).clamp(0.0, 1.0);
    let desired_size = egui::vec2(ui.available_width(), GRAPH_HEIGHT * scale);
    let painter = ui.painter().clone();

    if !has_loaded_sample {
        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
        let bg = if response.hovered() { COLOR_EDGE } else { COLOR_SURFACE_DEEP };
        painter.rect_filled(rect, 2.0, bg);
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, COLOR_EDGE), egui::StrokeKind::Inside);
        let text_color = if response.hovered() { COLOR_ACCENT } else { COLOR_DIM };
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "[ Load Sample ]",
            egui::FontId::proportional(14.0 * scale),
            text_color,
        );
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        return response.clicked();
    }

    let source_guard = source.load();
    let samples = source_guard.first().filter(|s| !s.is_empty());
    let Some(samples) = samples else {
        // Shouldn't happen while `has_loaded_sample` is true, but avoid a
        // panic on the empty slice below if it somehow does.
        ui.allocate_exact_size(desired_size, egui::Sense::hover());
        return false;
    };

    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click_and_drag());
    painter.rect_filled(rect, 2.0, COLOR_SURFACE_DEEP);
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, COLOR_EDGE), egui::StrokeKind::Inside);

    let width_px = (rect.width().max(1.0) as usize).max(1);
    let mid_y = rect.center().y;
    let half_height = rect.height() * 0.5 * 0.9;
    let samples_per_px = (samples.len() as f32 / width_px as f32).max(1.0);
    let waveform_stroke = egui::Color32::from_rgba_unmultiplied(COLOR_INK.r(), COLOR_INK.g(), COLOR_INK.b(), 160);
    for px in 0..width_px {
        let start = ((px as f32) * samples_per_px) as usize;
        if start >= samples.len() {
            break;
        }
        let end = (((px + 1) as f32) * samples_per_px).ceil() as usize;
        let end = end.clamp(start + 1, samples.len());
        let slice = &samples[start..end];
        let (min_v, max_v) =
            slice.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &s| (mn.min(s), mx.max(s)));
        // Scaled by the sample's own Volume slider so the drawn waveform's
        // height reflects what will actually be heard, not the underlying
        // peak-normalized audio's own (always-100%) amplitude.
        let (min_v, max_v) = (min_v * volume, max_v * volume);
        let x = rect.left() + px as f32;
        let y_top = mid_y - max_v.clamp(-1.0, 1.0) * half_height;
        let y_bottom = (mid_y - min_v.clamp(-1.0, 1.0) * half_height).max(y_top + 1.0);
        painter.line_segment([egui::pos2(x, y_top), egui::pos2(x, y_bottom)], egui::Stroke::new(1.0, waveform_stroke));
    }

    let freeze_normalized = freeze_point.unmodulated_normalized_value();
    let marker_x = rect.left() + freeze_normalized * rect.width();
    painter.line_segment(
        [egui::pos2(marker_x, rect.top()), egui::pos2(marker_x, rect.bottom())],
        egui::Stroke::new(2.0, COLOR_ACCENT),
    );

    if response.drag_started() || response.clicked() {
        setter.begin_set_parameter(freeze_point);
    }
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let normalized = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
            setter.set_parameter_normalized(freeze_point, normalized);
        }
    }
    if response.drag_stopped() || response.clicked() {
        setter.end_set_parameter(freeze_point);
    }

    false
}

/// Draws the ADSR envelope shape with three grabbable handles instead of
/// four plain sliders: the attack-end point (drags horizontally only - it
/// always tops out at level 1.0), the decay-end/sustain-level point (drags
/// both axes), and the release-end point (drags horizontally only - it
/// always returns to level 0.0). Attack/Decay/Release each get an equal
/// fixed-width "slot" for layout (not sized by their own current value, so
/// the graph doesn't rescale itself while you're dragging it) with a fixed-
/// width flat plateau for Sustain, which has no time/duration of its own.
fn draw_adsr_graph(
    ui: &mut egui::Ui,
    attack: &FloatParam,
    decay: &FloatParam,
    sustain: &FloatParam,
    release: &FloatParam,
    setter: &ParamSetter,
    scale: f32,
) {
    let desired_size = egui::vec2(ui.available_width(), GRAPH_HEIGHT * scale);
    let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, COLOR_SURFACE_DEEP);
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, COLOR_EDGE), egui::StrokeKind::Inside);

    let sustain_plateau_width: f32 = 30.0 * scale;
    let segment_width = ((rect.width() - sustain_plateau_width) / 3.0).max(1.0);
    let margin = 10.0 * scale;
    let plot_top = rect.top() + margin;
    let plot_bottom = rect.bottom() - margin;
    let plot_height = (plot_bottom - plot_top).max(1.0);
    let y_for_level = |level: f32| plot_bottom - level.clamp(0.0, 1.0) * plot_height;
    let level_for_y = |y: f32| ((plot_bottom - y) / plot_height).clamp(0.0, 1.0);

    let attack_start_x = rect.left();
    let decay_start_x = attack_start_x + segment_width;
    let sustain_start_x = decay_start_x + segment_width;
    let release_start_x = sustain_start_x + sustain_plateau_width;

    let attack_norm = attack.unmodulated_normalized_value();
    let decay_norm = decay.unmodulated_normalized_value();
    let sustain_norm = sustain.unmodulated_normalized_value();
    let release_norm = release.unmodulated_normalized_value();

    let p0 = egui::pos2(attack_start_x, y_for_level(0.0));
    let p1 = egui::pos2(attack_start_x + segment_width * attack_norm, y_for_level(1.0));
    let p2 = egui::pos2(decay_start_x + segment_width * decay_norm, y_for_level(sustain_norm));
    let p3 = egui::pos2(sustain_start_x + sustain_plateau_width, y_for_level(sustain_norm));
    let p4 = egui::pos2(release_start_x + segment_width * release_norm, y_for_level(0.0));

    let curve_stroke = egui::Stroke::new(2.0, COLOR_ACCENT);
    painter.line_segment([p0, p1], curve_stroke);
    painter.line_segment([p1, p2], curve_stroke);
    painter.line_segment([p2, p3], curve_stroke);
    painter.line_segment([p3, p4], curve_stroke);

    let handle_radius = 5.0 * scale;
    let handle_color = COLOR_ACCENT;
    painter.circle_filled(p1, handle_radius, handle_color);
    painter.circle_filled(p2, handle_radius, handle_color);
    painter.circle_filled(p4, handle_radius, handle_color);

    let handle_rect = |center: egui::Pos2| egui::Rect::from_center_size(center, egui::vec2(handle_radius * 3.0, handle_radius * 3.0));

    // Attack handle: horizontal-only drag within its slot.
    let attack_id = ui.make_persistent_id("adsr_attack_handle");
    let attack_response = ui.interact(handle_rect(p1), attack_id, egui::Sense::click_and_drag());
    if attack_response.drag_started() || attack_response.clicked() {
        setter.begin_set_parameter(attack);
    }
    if attack_response.dragged() || attack_response.clicked() {
        if let Some(pos) = attack_response.interact_pointer_pos() {
            setter.set_parameter_normalized(attack, ((pos.x - attack_start_x) / segment_width).clamp(0.0, 1.0));
        }
    }
    if attack_response.drag_stopped() || attack_response.clicked() {
        setter.end_set_parameter(attack);
    }

    // Decay/Sustain handle: both axes at once (X -> decay time, Y -> sustain level).
    let decay_sustain_id = ui.make_persistent_id("adsr_decay_sustain_handle");
    let ds_response = ui.interact(handle_rect(p2), decay_sustain_id, egui::Sense::click_and_drag());
    if ds_response.drag_started() || ds_response.clicked() {
        setter.begin_set_parameter(decay);
        setter.begin_set_parameter(sustain);
    }
    if ds_response.dragged() || ds_response.clicked() {
        if let Some(pos) = ds_response.interact_pointer_pos() {
            setter.set_parameter_normalized(decay, ((pos.x - decay_start_x) / segment_width).clamp(0.0, 1.0));
            setter.set_parameter_normalized(sustain, level_for_y(pos.y));
        }
    }
    if ds_response.drag_stopped() || ds_response.clicked() {
        setter.end_set_parameter(decay);
        setter.end_set_parameter(sustain);
    }

    // Release handle: horizontal-only drag within its slot.
    let release_id = ui.make_persistent_id("adsr_release_handle");
    let release_response = ui.interact(handle_rect(p4), release_id, egui::Sense::click_and_drag());
    if release_response.drag_started() || release_response.clicked() {
        setter.begin_set_parameter(release);
    }
    if release_response.dragged() || release_response.clicked() {
        if let Some(pos) = release_response.interact_pointer_pos() {
            setter.set_parameter_normalized(release, ((pos.x - release_start_x) / segment_width).clamp(0.0, 1.0));
        }
    }
    if release_response.drag_stopped() || release_response.clicked() {
        setter.end_set_parameter(release);
    }
}

impl Plugin for PrismPlugin {
    const NAME: &'static str = "SpectralPrism";
    const VENDOR: &'static str = "John Oestmann";
    const URL: &'static str = "https://johnoestmannmusic.com";
    const EMAIL: &'static str = "contact@johnoestmannmusic.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    // MidiCCs (not just Basic) is required to actually receive
    // NoteEvent::MidiPitchBend - Basic alone silently never delivers it.
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let source = self.source.clone();
        let source_b = self.source_b.clone();
        let loop_buffer = self.loop_buffer.clone();
        let trigger = self.trigger.clone();
        let last_note = self.last_note.clone();
        let active_voice_count = self.active_voice_count.clone();
        let loaded_filename = self.loaded_filename.clone();
        let loaded_filename_b = self.loaded_filename_b.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            PrismEditorState {
                // `PrismEditorState` itself is purely GUI-local and
                // rebuilt fresh every time the editor is (re)opened - see
                // `PrismPluginParams::preset_name_input`'s doc comment for
                // why this field specifically needs to be seeded from
                // that persisted mirror here, rather than just starting
                // blank via `Default::default()`.
                preset_name_input: self.params.preset_name_input.lock().unwrap().clone(),
                ..Default::default()
            },
            |ctx, _state| apply_theme(ctx),
            move |egui_ctx, setter, state| {
                // GUI scaling: the corner of `ResizableWindow` below lets the
                // user drag the window to any size (through nih-plug's real
                // host-negotiated resize, not just adding blank space); here
                // we read back the *actual* current size and derive how much
                // bigger than the base design size it now is, then scale
                // every widget's actual "points" dimensions by that (see
                // `apply_gui_scale` for why - not `set_zoom_factor`, which
                // this nih_plug_egui/egui_baseview revision doesn't apply at
                // render time). Never scales below 1x (`min_size` below also
                // stops the window from being dragged smaller than the base
                // size in the first place).
                let (current_width, current_height) = params.editor_state.size();
                let scale = (((current_width as f32 / BASE_EDITOR_WIDTH as f32)
                    + (current_height as f32 / BASE_EDITOR_HEIGHT as f32))
                    / 2.0)
                    .max(1.0);
                apply_gui_scale(egui_ctx, scale);

                // Shared by the waveform-area "[ Load Sample ]" sign (shown
                // before anything is loaded) and the "Load Sample..." button
                // below it (always available) - both just need to trigger the
                // same file dialog / decode / re-render flow.
                let open_sample_dialog = |state: &mut PrismEditorState| {
                    if let Some(path) = rfd::FileDialog::new().add_filter("WAV", &["wav", "WAV"]).pick_file() {
                        load_sample_from_path(&path, &source, &params.sample_path, &loop_buffer, &trigger, &params, &loaded_filename, state);
                    }
                };
                // Sample B's counterpart, for Spectral Fusion - same flow,
                // its own source/path-slot/filename handle.
                let open_sample_dialog_b = |state: &mut PrismEditorState| {
                    if let Some(path) = rfd::FileDialog::new().add_filter("WAV", &["wav", "WAV"]).pick_file() {
                        load_sample_from_path(&path, &source_b, &params.sample_path_b, &loop_buffer, &trigger, &params, &loaded_filename_b, state);
                    }
                };

                // Shared by `apply_preset` for both Sample A and Sample B
                // when `preset.unload_samples` is set (see that field's doc
                // comment) - clears the persisted sample path and displayed
                // filename, stores `replacement` as the new source audio,
                // then re-renders so the change actually takes effect
                // immediately. `replacement` is a parameter (not always
                // empty) because Sample A can never safely go fully
                // empty - `render_fused_loop`/`render_frozen_loop` assume
                // at least one channel and panic on `Vec::new()` (which
                // silently kills the background render worker thread,
                // leaving `loop_buffer` stuck on whatever was last
                // rendered - exactly `initialize()`'s own reasoning for
                // always falling back to `synthetic_source` for Sample A
                // specifically, never for Sample B).
                let unload_sample = |source: &Arc<ArcSwap<Vec<Vec<f32>>>>,
                                      sample_path_slot: &Mutex<Option<PathBuf>>,
                                      loaded_filename: &Arc<ArcSwapOption<String>>,
                                      replacement: Vec<Vec<f32>>| {
                    source.store(Arc::new(replacement));
                    *sample_path_slot.lock().unwrap() = None;
                    loaded_filename.store(None);
                    trigger.request_render(current_render_request(&params));
                };

                // Shared by `apply_preset` for both Sample A and Sample B:
                // loads `path_opt` if it's `Some`, and if that fails,
                // prompts the user to locate the moved/missing file - see
                // `apply_preset`'s own doc comment for why a preset stores
                // (and may need to re-resolve) an absolute sample path.
                let load_or_relocate = |path_opt: &Option<PathBuf>,
                                        source: &Arc<ArcSwap<Vec<Vec<f32>>>>,
                                        sample_path_slot: &Mutex<Option<PathBuf>>,
                                        loaded_filename: &Arc<ArcSwapOption<String>>,
                                        label: &str,
                                        state: &mut PrismEditorState|
                 -> Option<PathBuf> {
                    let path = path_opt.as_ref()?;
                    load_sample_from_path(path, source, sample_path_slot, &loop_buffer, &trigger, &params, loaded_filename, state);
                    if state.error.is_none() {
                        return None;
                    }

                    rfd::MessageDialog::new()
                        .set_title("Sample not found")
                        .set_description(format!(
                            "{label}'s sample couldn't be found:\n{}\n\nLocate it to continue.",
                            path.display()
                        ))
                        .show();
                    let relocated = rfd::FileDialog::new().add_filter("WAV", &["wav", "WAV"]).pick_file()?;
                    load_sample_from_path(&relocated, source, sample_path_slot, &loop_buffer, &trigger, &params, loaded_filename, state);
                    state.error.is_none().then_some(relocated)
                };

                // Recalls a saved snapshot: every DSP param via the same
                // begin/set/end pattern the custom graph widgets use (so
                // host automation recording sees a proper set, not a silent
                // jump), plus the referenced sample if the preset has one -
                // sharing `load_sample_from_path` with the interactive
                // dialog so both paths persist `sample_path` and re-render
                // identically. A preset saved without a sample (`None`)
                // leaves whatever's currently loaded untouched.
                //
                // Presets deliberately still store the sample's full
                // absolute path (privacy tradeoff accepted, since stripping
                // it would break the common case of reloading your *own*
                // presets) - but a shared preset's path naturally won't
                // resolve on someone else's machine. Rather than just
                // failing, a missing sample prompts the user to locate it
                // via a file dialog; if they do, that corrected path is
                // returned here so the caller can write it back into
                // whichever preset file this came from, so it doesn't have
                // to be re-located every time on this machine specifically.
                let apply_preset = |preset: &Preset, state: &mut PrismEditorState| -> (Option<PathBuf>, Option<PathBuf>) {
                    let set = |param: &FloatParam, value: f32| {
                        setter.begin_set_parameter(param);
                        setter.set_parameter(param, value);
                        setter.end_set_parameter(param);
                    };
                    set(&params.freeze_point, preset.freeze_point_pct);
                    set(&params.sample_a_volume, preset.volume_a_pct);
                    set(&params.sample_a_tune, preset.tune_a_semitones);
                    set(&params.formant_shift, preset.formant_shift_semitones);
                    set(&params.stereo_width, preset.stereo_width_pct);
                    set(&params.loop_length_seconds, preset.loop_length_seconds);
                    set(&params.attack, preset.attack_ms);
                    set(&params.decay, preset.decay_ms);
                    set(&params.sustain, preset.sustain_pct);
                    set(&params.release, preset.release_ms);
                    set(&params.velocity_sensitivity, preset.velocity_sensitivity_pct);
                    set(&params.pitch_bend_range_semitones, preset.pitch_bend_range_semitones);
                    set(&params.pan_center_pct, preset.pan_center_pct);
                    set(&params.pan_width_pct, preset.pan_width_pct);
                    set(&params.freeze_point_b, preset.freeze_point_b_pct);
                    set(&params.sample_b_volume, preset.volume_b_pct);
                    set(&params.sample_b_tune, preset.tune_b_semitones);
                    set(&params.formant_shift_b, preset.formant_shift_b_semitones);
                    set(&params.fusion_mix_amount, preset.fusion_mix_amount_pct);
                    set(&params.fusion_cross_synth_amount, preset.fusion_cross_synth_amount_pct);
                    set(&params.fusion_convolve_amount, preset.fusion_convolve_amount_pct);
                    set(&params.fusion_ring_mod_amount, preset.fusion_ring_mod_amount_pct);
                    setter.begin_set_parameter(&params.fusion_mode);
                    setter.set_parameter(&params.fusion_mode, FusionMode::from_preset_id(&preset.fusion_mode));
                    setter.end_set_parameter(&params.fusion_mode);

                    if preset.unload_samples {
                        let sample_rate = loop_buffer.load().sample_rate;
                        unload_sample(&source, &params.sample_path, &loaded_filename, vec![synthetic_source(sample_rate, 1.0)]);
                        unload_sample(&source_b, &params.sample_path_b, &loaded_filename_b, Vec::new());
                        state.error = None;
                        return (None, None);
                    }

                    // Sample A's own relocate error takes priority over
                    // Sample B's (or its absence) if both need attention -
                    // whichever ran last would otherwise silently overwrite
                    // `state.error`, see `load_or_relocate`.
                    let relocated_a = load_or_relocate(&preset.sample_path, &source, &params.sample_path, &loaded_filename, "Sample A", state);
                    let error_after_a = state.error.take();
                    let relocated_b =
                        load_or_relocate(&preset.sample_path_b, &source_b, &params.sample_path_b, &loaded_filename_b, "Sample B", state);
                    state.error = error_after_a.or(state.error.take());
                    (relocated_a, relocated_b)
                };

                // A persistent bottom strip, added *before* the central
                // content below (egui panels must be added before
                // `CentralPanel` - which `ResizableWindow` uses internally -
                // so it can shrink to leave room) so the build number stays
                // fixed in the corner regardless of the content's own
                // scroll position, rather than just being the last thing in
                // the scrollable area.
                // The build number now only lives in the heading
                // ("SpectralPrism | v{BUILD_NUMBER}") - showing it here too
                // was redundant. The freed-up right side of this strip is
                // reserved for something else later.
                egui::TopBottomPanel::bottom("spectral_prism_status_bar")
                    .frame(
                        egui::Frame::default()
                            // `Frame::default()` has no fill of its own, so
                            // without this it falls through to the raw
                            // (black) clear color behind the whole window
                            // instead of the theme's background - this is
                            // what made the strip unreadable (light-grey
                            // text on black instead of on the theme's own
                            // surface color).
                            .fill(COLOR_SURFACE_DEEP)
                            .inner_margin(egui::Margin::symmetric((6.0 * scale) as i8, (3.0 * scale) as i8)),
                    )
                    .show_separator_line(false)
                    .show(egui_ctx, |ui| {
                        // Diagnostic, not something being adjusted - moved
                        // here (out of the scrollable content area) so it
                        // doesn't cost a full-width row up top and stays
                        // visible regardless of scroll position.
                        let note = last_note.load(Ordering::Relaxed);
                        let voice_count = active_voice_count.load(Ordering::Relaxed);
                        let note_label = if note == NO_NOTE { "--".to_string() } else { note.to_string() };
                        // COLOR_INK (dark charcoal), not COLOR_DIM - this
                        // strip has its own light background rather than
                        // sharing the main content's white, so it needs the
                        // stronger of the two text colors to stay legible.
                        ui.label(egui::RichText::new(format!("MIDI: note {note_label} | {voice_count} voice(s) active")).small().color(COLOR_INK));
                    });

                ResizableWindow::new("spectral_prism_window")
                    .min_size(egui::vec2(BASE_EDITOR_WIDTH as f32, BASE_EDITOR_HEIGHT as f32))
                    .show(egui_ctx, &params.editor_state, |ui| {
                        egui::Frame::default().inner_margin(egui::Margin::same((EDITOR_MARGIN * scale) as i8)).show(ui, |ui| {
                        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        ui.heading(format!("SpectralPrism | v{BUILD_NUMBER}"));

                        let presets_dir = presets_dir();
                        let preset_names = presets_dir.as_deref().map(list_presets).unwrap_or_default();
                        // Cloned out of the lock immediately - held only
                        // long enough to read, not across the rest of the
                        // frame (see `PrismPluginParams::selected_preset`'s
                        // doc comment for why this lives there, not in
                        // `PrismEditorState`).
                        let selected_preset_name = params.selected_preset.lock().unwrap().clone();
                        let current_preset_idx =
                            selected_preset_name.as_ref().and_then(|name| preset_names.iter().position(|n| n == name));

                        let load_preset_at = |idx: usize, state: &mut PrismEditorState| {
                            let (Some(name), Some(dir)) = (preset_names.get(idx), &presets_dir) else { return };
                            match load_preset(dir, name) {
                                Ok(mut preset) => {
                                    // `apply_preset` already leaves `state.error`
                                    // in the right final state (set if the
                                    // sample couldn't be found/relocated,
                                    // cleared otherwise) - not overwritten here.
                                    let (relocated_a, relocated_b) = apply_preset(&preset, state);
                                    if relocated_a.is_some() || relocated_b.is_some() {
                                        if let Some(relocated) = relocated_a {
                                            preset.sample_path = Some(relocated);
                                        }
                                        if let Some(relocated) = relocated_b {
                                            preset.sample_path_b = Some(relocated);
                                        }
                                        // Best-effort: a failed resave isn't
                                        // worth surfacing as an error - the
                                        // preset still applied correctly this
                                        // time, it just won't remember the fix
                                        // for next time.
                                        let _ = save_preset(dir, name, &preset);
                                    }
                                    *params.selected_preset.lock().unwrap() = Some(name.clone());
                                    // Prefills the Save field with the just-
                                    // loaded preset's own name, so clicking
                                    // Save immediately overwrites it instead
                                    // of requiring the name to be retyped.
                                    state.preset_name_input = name.clone();
                                    state.current_preset_title = preset.title.clone();
                                    state.current_preset_author = preset.author.clone();
                                    state.current_preset_category = preset.category.clone();
                                    state.current_preset_sub_category = preset.sub_category.clone();
                                    state.current_preset_date_last_updated = preset.date_last_updated.clone();
                                }
                                Err(e) => state.error = Some(e),
                            }
                        };

                        // One row for the whole preset toolbar - Prev/combo/
                        // Next, the Save-as-name field, and Import/Export -
                        // there was plenty of horizontal room to avoid
                        // stacking these three groups as separate rows.
                        // Thin vertical separators mark the group
                        // boundaries.
                        ui.horizontal(|ui| {
                            if ui.add_enabled(!preset_names.is_empty(), egui::Button::new("◀")).clicked() {
                                let idx = current_preset_idx.map(|i| i.saturating_sub(1)).unwrap_or(0);
                                load_preset_at(idx, state);
                            }
                            // Opens the Preset Browser modal instead of a
                            // plain dropdown - still shows the current
                            // preset's name, just as a button.
                            let combo_label = selected_preset_name.as_deref().unwrap_or("(no preset)");
                            if ui.button(combo_label).clicked() {
                                state.preset_browser_entries = presets_dir.as_deref().map(load_preset_browser_entries).unwrap_or_default();
                                state.preset_browser_search.clear();
                                state.preset_browser_selected = selected_preset_name.clone();
                                state.show_preset_browser = true;
                            }
                            if ui.add_enabled(!preset_names.is_empty(), egui::Button::new("▶")).clicked() {
                                let idx = current_preset_idx
                                    .map(|i| (i + 1).min(preset_names.len() - 1))
                                    .unwrap_or(0);
                                load_preset_at(idx, state);
                            }

                            ui.separator();

                            ui.add(
                                egui::TextEdit::singleline(&mut state.preset_name_input)
                                    .hint_text("Preset name...")
                                    .desired_width(140.0 * scale),
                            );
                            // Mirrors the field's live contents into
                            // persisted state every frame (not just on
                            // save) - see
                            // `PrismPluginParams::preset_name_input`'s doc
                            // comment. Covers every way this field can
                            // change (typing here, or Prev/Next/Import/
                            // Save assigning it above/below earlier in
                            // this same frame) with one sync point.
                            *params.preset_name_input.lock().unwrap() = state.preset_name_input.clone();
                            let name = state.preset_name_input.trim().to_string();
                            // Opens the Save dialog (Title/Author/Category/
                            // Sub-category) instead of saving immediately -
                            // prefilled from the currently-loaded preset's
                            // own metadata when re-saving over it, or blank
                            // for a brand new one.
                            if ui.add_enabled(!name.is_empty(), egui::Button::new("Save")).clicked() {
                                let existing = selected_preset_name
                                    .as_ref()
                                    .filter(|selected| selected.as_str() == name)
                                    .and_then(|selected| presets_dir.as_deref().and_then(|dir| load_preset(dir, selected).ok()));
                                state.save_title_input = name.clone();
                                state.save_author_input = existing.as_ref().map(|p| p.author.clone()).unwrap_or_default();
                                state.save_category_input = existing.as_ref().map(|p| p.category.clone()).unwrap_or_default();
                                state.save_subcategory_input = existing.as_ref().map(|p| p.sub_category.clone()).unwrap_or_default();
                                state.show_save_dialog = true;
                            }

                            ui.separator();

                            // Separate from the named on-disk library above
                            // (Save/◀/▶/combo, all under `presets_dir()`) -
                            // these go to/from any file the user picks, for
                            // sharing a preset outside this machine (e.g.
                            // alongside a sample-pack WAV export, see
                            // "Export WAV Sample..." below).
                            if ui.button("Import Preset...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("SpectralPrism Preset", &[PRESET_FILE_EXTENSION])
                                    .pick_file()
                                {
                                    match read_preset_file(&path) {
                                        Ok(mut preset) => {
                                            // See `load_preset_at` for why
                                            // `state.error` isn't touched here.
                                            let (relocated_a, relocated_b) = apply_preset(&preset, state);
                                            if relocated_a.is_some() || relocated_b.is_some() {
                                                if let Some(relocated) = relocated_a {
                                                    preset.sample_path = Some(relocated);
                                                }
                                                if let Some(relocated) = relocated_b {
                                                    preset.sample_path_b = Some(relocated);
                                                }
                                                let _ = write_preset_file(&path, &preset);
                                            }
                                            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
                                            // Prefills the Save field the same
                                            // way `load_preset_at` does, so
                                            // Save immediately overwrites a
                                            // re-imported preset without
                                            // retyping its name - only
                                            // meaningful for a preset saved
                                            // back into the named on-disk
                                            // library, not this arbitrary file
                                            // path, but that's exactly what
                                            // typing a name into that field
                                            // and clicking Save already does.
                                            if let Some(stem) = &stem {
                                                state.preset_name_input = stem.clone();
                                            }
                                            *params.selected_preset.lock().unwrap() = stem;
                                            state.current_preset_title = preset.title.clone();
                                            state.current_preset_author = preset.author.clone();
                                            state.current_preset_category = preset.category.clone();
                                            state.current_preset_sub_category = preset.sub_category.clone();
                                            state.current_preset_date_last_updated = preset.date_last_updated.clone();
                                        }
                                        Err(e) => state.error = Some(e),
                                    }
                                }
                            }
                            if ui.button("Export Preset...").clicked() {
                                state.export_strip_sample_paths = false;
                                state.show_export_preset_dialog = true;
                            }
                        });

                        if state.show_save_dialog {
                            // Scanned fresh every frame the dialog is open
                            // (cheap - a handful of small on-disk files) so
                            // the dropdowns below always reflect the current
                            // library, including a preset saved moments ago
                            // in this same session.
                            let save_dialog_entries = presets_dir.as_deref().map(load_preset_browser_entries).unwrap_or_default();
                            let existing_titles = distinct_preset_values(&save_dialog_entries, |e| &e.title);
                            let existing_authors = distinct_preset_values(&save_dialog_entries, |e| &e.author);
                            let existing_categories = distinct_preset_values(&save_dialog_entries, |e| &e.category);
                            let existing_subcategories = distinct_preset_values(&save_dialog_entries, |e| &e.sub_category);
                            // A plain text field (typing = "add a new
                            // value") plus a small dropdown button listing
                            // every distinct value already used for this
                            // field across the on-disk library (already
                            // alphabetical - `distinct_preset_values`
                            // sorts) - clicking one overwrites the text
                            // field with it. Deliberately *not*
                            // `egui::ComboBox`: its popup always wraps
                            // content in a fixed-height `ScrollArea`
                            // (`combo_box.rs`), which for a library with
                            // more than a handful of entries meant most of
                            // the list was hidden behind an easy-to-miss
                            // scrollbar. Built directly on the lower-level
                            // `egui::popup` primitives `ComboBox` itself
                            // uses internally instead, so the whole list is
                            // always visible at once - wrapping into a new
                            // column every 15 entries rather than scrolling.
                            const DROPDOWN_COLUMN_HEIGHT: usize = 15;
                            let field_with_dropdown = |ui: &mut egui::Ui, id_salt: &str, value: &mut String, options: &[String]| {
                                ui.horizontal(|ui| {
                                    ui.text_edit_singleline(value);
                                    let popup_id = ui.make_persistent_id(id_salt);
                                    // A hand-painted triangle (mirroring
                                    // `egui::ComboBox`'s own
                                    // `paint_default_icon`, which does the
                                    // same thing internally) rather than a
                                    // Unicode arrow character - this app's
                                    // bundled Medodica font doesn't cover
                                    // the geometric-shapes block at all, and
                                    // egui's own fallback font's coverage of
                                    // it turned out to be inconsistent (▼
                                    // rendered as a missing-glyph tofu box
                                    // despite ◀/▶ elsewhere in this same
                                    // toolbar working fine), so a painted
                                    // shape is the only way to guarantee
                                    // this actually renders.
                                    // Explicit size (matching the width the
                                    // original `egui::ComboBox` version of
                                    // this button used) rather than
                                    // whatever an empty-label `ui.button`
                                    // happens to size itself to - and the
                                    // same 0.7/0.45 width/height ratio
                                    // `paint_default_icon` uses, not an
                                    // arbitrary smaller one, which is what
                                    // made the first version of this look
                                    // squished/undersized.
                                    let toggle = ui.add_sized(egui::vec2(18.0 * scale, ui.spacing().interact_size.y), egui::Button::new(""));
                                    if ui.is_rect_visible(toggle.rect) {
                                        let tri = egui::Rect::from_center_size(
                                            toggle.rect.center(),
                                            egui::vec2(toggle.rect.width() * 0.7, toggle.rect.height() * 0.45),
                                        );
                                        ui.painter().add(egui::Shape::convex_polygon(
                                            vec![tri.left_top(), tri.right_top(), tri.center_bottom()],
                                            COLOR_INK,
                                            egui::Stroke::NONE,
                                        ));
                                    }
                                    if toggle.clicked() {
                                        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                                    }
                                    egui::popup::popup_below_widget(
                                        ui,
                                        popup_id,
                                        &toggle,
                                        egui::popup::PopupCloseBehavior::CloseOnClick,
                                        |ui| {
                                            // The toggle button itself is
                                            // tiny (just the arrow glyph),
                                            // and `popup_below_widget` sizes
                                            // the popup to match its
                                            // triggering widget - without
                                            // this, wrapping stays on by
                                            // default and every option's
                                            // text wraps almost immediately
                                            // inside that narrow starting
                                            // width, one or two characters
                                            // per line (reads as "vertical"
                                            // text). Same fix `egui::ComboBox`
                                            // itself applies internally to
                                            // its own popup, for the same
                                            // reason (see `combo_box.rs`).
                                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                                            if options.is_empty() {
                                                ui.label("No existing values yet");
                                                return;
                                            }
                                            let num_columns = options.len().div_ceil(DROPDOWN_COLUMN_HEIGHT).max(1);
                                            ui.columns(num_columns, |columns| {
                                                for (col, chunk) in options.chunks(DROPDOWN_COLUMN_HEIGHT).enumerate() {
                                                    for option in chunk {
                                                        if columns[col].selectable_label(value == option, option).clicked() {
                                                            *value = option.clone();
                                                        }
                                                    }
                                                }
                                            });
                                        },
                                    );
                                });
                            };
                            let modal = egui::Modal::new(egui::Id::new("spectral_prism_save_modal")).show(egui_ctx, |ui| {
                                ui.set_min_width(320.0 * scale);
                                ui.heading("Save Preset");
                                ui.add_space(4.0);
                                egui::Grid::new("spectral_prism_save_modal_grid").num_columns(2).spacing([8.0 * scale, 6.0 * scale]).show(
                                    ui,
                                    |ui| {
                                        ui.label("Title");
                                        field_with_dropdown(ui, "spectral_prism_save_title_dropdown", &mut state.save_title_input, &existing_titles);
                                        ui.end_row();
                                        ui.label("Author");
                                        field_with_dropdown(
                                            ui,
                                            "spectral_prism_save_author_dropdown",
                                            &mut state.save_author_input,
                                            &existing_authors,
                                        );
                                        ui.end_row();
                                        ui.label("Category");
                                        field_with_dropdown(
                                            ui,
                                            "spectral_prism_save_category_dropdown",
                                            &mut state.save_category_input,
                                            &existing_categories,
                                        );
                                        ui.end_row();
                                        ui.label("Sub-category");
                                        field_with_dropdown(
                                            ui,
                                            "spectral_prism_save_subcategory_dropdown",
                                            &mut state.save_subcategory_input,
                                            &existing_subcategories,
                                        );
                                        ui.end_row();
                                    },
                                );
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    let title = state.save_title_input.trim().to_string();
                                    if ui.add_enabled(!title.is_empty(), egui::Button::new("Save")).clicked() {
                                        match &presets_dir {
                                            Some(dir) => {
                                                let mut preset = Preset::capture(&params);
                                                preset.title = title.clone();
                                                preset.author = state.save_author_input.trim().to_string();
                                                preset.category = state.save_category_input.trim().to_string();
                                                preset.sub_category = state.save_subcategory_input.trim().to_string();
                                                preset.date_last_updated = today_utc_date();
                                                match save_preset(dir, &title, &preset) {
                                                    Ok(()) => {
                                                        *params.selected_preset.lock().unwrap() = Some(title.clone());
                                                        state.preset_name_input = title;
                                                        state.current_preset_title = preset.title.clone();
                                                        state.current_preset_author = preset.author.clone();
                                                        state.current_preset_category = preset.category.clone();
                                                        state.current_preset_sub_category = preset.sub_category.clone();
                                                        state.current_preset_date_last_updated = preset.date_last_updated.clone();
                                                        state.error = None;
                                                        state.show_save_dialog = false;
                                                    }
                                                    Err(e) => state.error = Some(e),
                                                }
                                            }
                                            None => state.error = Some("couldn't find a presets directory ($HOME not set?)".to_string()),
                                        }
                                    }
                                    if ui.button("Cancel").clicked() {
                                        state.show_save_dialog = false;
                                    }
                                });
                            });
                            if modal.should_close() {
                                state.show_save_dialog = false;
                            }
                        }

                        if state.show_export_preset_dialog {
                            let modal = egui::Modal::new(egui::Id::new("spectral_prism_export_preset_modal")).show(egui_ctx, |ui| {
                                ui.set_min_width(340.0 * scale);
                                ui.heading("Export Preset");
                                ui.add_space(4.0);
                                ui.checkbox(&mut state.export_strip_sample_paths, "Strip sample file path(s)");
                                ui.add_space(4.0);
                                ui.colored_label(
                                    COLOR_DIM,
                                    "For sharing this preset publicly, since your absolute sample path(s) won't exist on \
                                     another machine (the sample's filename is kept either way). Leave unchecked to keep \
                                     the full path for your own use. Either way, re-importing a preset whose sample \
                                     can't be found will prompt to locate it.",
                                );
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    if ui.button("Export...").clicked() {
                                        state.show_export_preset_dialog = false;
                                        let default_name = if state.preset_name_input.trim().is_empty() {
                                            "preset".to_string()
                                        } else {
                                            state.preset_name_input.trim().to_string()
                                        };
                                        if let Some(path) = rfd::FileDialog::new()
                                            .add_filter("SpectralPrism Preset", &[PRESET_FILE_EXTENSION])
                                            .set_file_name(&format!("{default_name}.{PRESET_FILE_EXTENSION}"))
                                            .save_file()
                                        {
                                            let mut preset = Preset::capture(&params);
                                            // `Preset::capture` deliberately
                                            // leaves these blank (only the
                                            // Save dialog fills them in
                                            // directly) - carry forward
                                            // whichever preset is currently
                                            // active in memory instead of
                                            // exporting empty metadata.
                                            preset.title = state.current_preset_title.clone();
                                            preset.author = state.current_preset_author.clone();
                                            preset.category = state.current_preset_category.clone();
                                            preset.sub_category = state.current_preset_sub_category.clone();
                                            preset.date_last_updated = state.current_preset_date_last_updated.clone();
                                            if state.export_strip_sample_paths {
                                                // Keeps just the filename (not blanked entirely) so the
                                                // recipient - and this preset's own info panel/relocate
                                                // dialog - still knows what to look for.
                                                preset.sample_path = preset.sample_path.as_ref().and_then(|p| p.file_name()).map(PathBuf::from);
                                                preset.sample_path_b =
                                                    preset.sample_path_b.as_ref().and_then(|p| p.file_name()).map(PathBuf::from);
                                            }
                                            match write_preset_file(&path, &preset) {
                                                Ok(()) => state.error = None,
                                                Err(e) => state.error = Some(e),
                                            }
                                        }
                                    }
                                    if ui.button("Cancel").clicked() {
                                        state.show_export_preset_dialog = false;
                                    }
                                });
                            });
                            if modal.should_close() {
                                state.show_export_preset_dialog = false;
                            }
                        }

                        if state.show_preset_browser {
                            let modal = egui::Modal::new(egui::Id::new("spectral_prism_preset_browser")).show(egui_ctx, |ui| {
                                ui.set_min_width(520.0 * scale);
                                ui.heading("Preset Browser");
                                ui.add_space(4.0);
                                ui.add(egui::TextEdit::singleline(&mut state.preset_browser_search).hint_text("Search by title or author..."));
                                ui.add_space(4.0);

                                let search = state.preset_browser_search.trim().to_lowercase();
                                let mut by_category: std::collections::BTreeMap<String, std::collections::BTreeMap<String, Vec<&PresetBrowserEntry>>> =
                                    Default::default();
                                for entry in &state.preset_browser_entries {
                                    if !search.is_empty()
                                        && !entry.title.to_lowercase().contains(&search)
                                        && !entry.author.to_lowercase().contains(&search)
                                    {
                                        continue;
                                    }
                                    by_category
                                        .entry(display_category(&entry.category).to_string())
                                        .or_default()
                                        .entry(display_category(&entry.sub_category).to_string())
                                        .or_default()
                                        .push(entry);
                                }

                                egui::ScrollArea::vertical().max_height(240.0 * scale).show(ui, |ui| {
                                    if by_category.is_empty() {
                                        ui.colored_label(COLOR_DIM, "No presets found.");
                                    }
                                    for (category, sub_categories) in &by_category {
                                        egui::CollapsingHeader::new(category).default_open(false).show(ui, |ui| {
                                            for (sub_category, entries) in sub_categories {
                                                egui::CollapsingHeader::new(sub_category).default_open(false).show(ui, |ui| {
                                                    for entry in entries {
                                                        let is_selected = state.preset_browser_selected.as_deref() == Some(entry.name.as_str());
                                                        if ui.selectable_label(is_selected, &entry.title).clicked() {
                                                            state.preset_browser_selected = Some(entry.name.clone());
                                                            state.preset_browser_delete_confirm = None;
                                                        }
                                                    }
                                                });
                                            }
                                        });
                                    }
                                });

                                ui.separator();
                                let selected_entry = state
                                    .preset_browser_selected
                                    .as_deref()
                                    .and_then(|name| presets_dir.as_deref().and_then(|dir| load_preset(dir, name).ok()));
                                // Always exactly 3 rows, whether or not
                                // anything is selected ("-" placeholders
                                // otherwise), so the modal's overall height -
                                // and the position of the Load/Delete/Close
                                // row below - never jumps around as
                                // different presets (with different amounts
                                // of metadata) are highlighted. Each row uses
                                // a truncating label so an unusually long
                                // value can't wrap and grow the row either.
                                let row = |ui: &mut egui::Ui, text: String| {
                                    ui.add(egui::Label::new(egui::RichText::new(text).color(COLOR_DIM)).truncate());
                                };
                                match &selected_entry {
                                    Some(preset) => {
                                        let sample_a_name = preset
                                            .sample_path
                                            .as_ref()
                                            .and_then(|p| p.file_name())
                                            .map(|n| n.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| "(none)".to_string());
                                        let sample_b_name = preset
                                            .sample_path_b
                                            .as_ref()
                                            .and_then(|p| p.file_name())
                                            .map(|n| n.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| "(none)".to_string());
                                        let algorithm = FusionMode::from_preset_id(&preset.fusion_mode);
                                        let title = if preset.title.is_empty() { "(untitled)" } else { &preset.title };
                                        let author = if preset.author.is_empty() { "-" } else { &preset.author };
                                        let build = if preset.build_number.is_empty() { "unknown" } else { &preset.build_number };
                                        let updated = if preset.date_last_updated.is_empty() { "unknown" } else { &preset.date_last_updated };
                                        row(
                                            ui,
                                            format!(
                                                "Title: {title}    Author: {author}    Category: {} / {}",
                                                display_category(&preset.category),
                                                display_category(&preset.sub_category)
                                            ),
                                        );
                                        row(
                                            ui,
                                            format!(
                                                "Sample A: {sample_a_name}    Sample B: {sample_b_name}    Algorithm: {}",
                                                FusionMode::variants()[algorithm.to_index()]
                                            ),
                                        );
                                        row(ui, format!("Build: {build}    Last updated: {updated}"));
                                    }
                                    None => {
                                        row(ui, "Title: -    Author: -    Category: -".to_string());
                                        row(ui, "Sample A: -    Sample B: -    Algorithm: -".to_string());
                                        row(ui, "Build: -    Last updated: -".to_string());
                                    }
                                }

                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    let can_load = state.preset_browser_selected.is_some();
                                    if ui.add_enabled(can_load, egui::Button::new("Load")).clicked() {
                                        if let Some(name) = state.preset_browser_selected.clone() {
                                            if let Some(idx) = preset_names.iter().position(|n| n == &name) {
                                                load_preset_at(idx, state);
                                            }
                                        }
                                        state.show_preset_browser = false;
                                        state.preset_browser_delete_confirm = None;
                                    }

                                    let pending_delete = state.preset_browser_delete_confirm.clone().filter(|pending| {
                                        state.preset_browser_selected.as_deref() == Some(pending.as_str())
                                    });
                                    if let Some(name) = pending_delete {
                                        ui.colored_label(COLOR_ERROR, "Delete this preset?");
                                        if ui.button("Confirm Delete").clicked() {
                                            if let Some(dir) = &presets_dir {
                                                match delete_preset(dir, &name) {
                                                    Ok(()) => {
                                                        state.preset_browser_entries = load_preset_browser_entries(dir);
                                                        state.preset_browser_selected = None;
                                                    }
                                                    Err(e) => state.error = Some(e),
                                                }
                                            }
                                            state.preset_browser_delete_confirm = None;
                                        }
                                        if ui.button("Cancel").clicked() {
                                            state.preset_browser_delete_confirm = None;
                                        }
                                    } else if ui.add_enabled(can_load, egui::Button::new("Delete")).clicked() {
                                        state.preset_browser_delete_confirm = state.preset_browser_selected.clone();
                                    }

                                    if ui.button("Close").clicked() {
                                        state.show_preset_browser = false;
                                        state.preset_browser_delete_confirm = None;
                                    }
                                });
                            });
                            if modal.should_close() {
                                state.preset_browser_delete_confirm = None;
                                state.show_preset_browser = false;
                            }
                        }

                        ui.add_space(8.0);

                        // Three sections side by side rather than two long
                        // stacked columns - requested directly, so the
                        // window grows wider instead of needing to scroll
                        // to see the bottom row of controls.
                        ui.columns(3, |columns| {
                            let left = &mut columns[0];
                            left.label("Sample A");
                            let has_loaded_sample = loaded_filename.load().is_some();
                            let clicked_load_a = left
                                .horizontal(|ui| {
                                    draw_vertical_volume_slider(ui, &params.sample_a_volume, setter, GRAPH_HEIGHT * scale, scale);
                                    draw_freeze_point_waveform(ui, &source, &params.freeze_point, params.sample_a_volume.value(), setter, has_loaded_sample, scale)
                                })
                                .inner;
                            if clicked_load_a {
                                open_sample_dialog(state);
                            }

                            let current_fusion = params.fusion_mode.value();
                            egui::Grid::new("spectral_prism_sample_a_grid").num_columns(2).spacing([8.0 * scale, 6.0 * scale]).show(left, |ui| {
                                param_row(ui, "Freeze Point", &params.freeze_point, setter);
                                param_row(ui, "Tune", &params.sample_a_tune, setter);
                                param_row(ui, "Formant Shift", &params.formant_shift, setter);

                                ui.label("Spectral Fusion");
                                egui::ComboBox::from_id_salt("spectral_prism_fusion_combo")
                                    .selected_text(FusionMode::variants()[current_fusion.to_index()])
                                    .show_ui(ui, |ui| {
                                        for idx in 0..FusionMode::variants().len() {
                                            let variant = FusionMode::from_index(idx);
                                            if ui.selectable_label(current_fusion == variant, FusionMode::variants()[idx]).clicked() {
                                                setter.begin_set_parameter(&params.fusion_mode);
                                                setter.set_parameter(&params.fusion_mode, variant);
                                                setter.end_set_parameter(&params.fusion_mode);
                                            }
                                        }
                                    });
                                ui.end_row();
                            });

                            // Always rendered (not just while a Fusion mode
                            // needing Sample B is selected) and only greyed
                            // out/non-interactive when Off, via
                            // `add_enabled_ui` - so the left column's total
                            // height stays the same across every mode,
                            // which is what lets the default window size
                            // (below) actually fit every state without
                            // needing to grow for Fusion controls. The
                            // Amount row and the "needs Sample B" line are
                            // both still reserved (as a blank row/label)
                            // even when not applicable, for the same reason.
                            left.add_space(8.0);
                            let has_loaded_sample_b = loaded_filename_b.load().is_some();
                            let clicked_load_b = left
                                .add_enabled_ui(current_fusion != FusionMode::Off, |ui| {
                                    ui.label("Sample B");
                                    let clicked = ui
                                        .horizontal(|ui| {
                                            draw_vertical_volume_slider(ui, &params.sample_b_volume, setter, GRAPH_HEIGHT * scale, scale);
                                            draw_freeze_point_waveform(
                                                ui,
                                                &source_b,
                                                &params.freeze_point_b,
                                                params.sample_b_volume.value(),
                                                setter,
                                                has_loaded_sample_b,
                                                scale,
                                            )
                                        })
                                        .inner;

                                    egui::Grid::new("spectral_prism_sample_b_grid").num_columns(2).spacing([8.0 * scale, 6.0 * scale]).show(
                                        ui,
                                        |ui| {
                                            param_row(ui, "Freeze Point", &params.freeze_point_b, setter);
                                            param_row(ui, "Tune", &params.sample_b_tune, setter);
                                            param_row(ui, "Formant Shift", &params.formant_shift_b, setter);
                                            match current_fusion {
                                                FusionMode::Mix => param_row(ui, "Mix Blend (A \u{2194} B)", &params.fusion_mix_amount, setter),
                                                FusionMode::CrossSynth => {
                                                    param_row(ui, "Cross-Synth Amount", &params.fusion_cross_synth_amount, setter)
                                                }
                                                FusionMode::Convolve => param_row(ui, "Convolve Amount", &params.fusion_convolve_amount, setter),
                                                FusionMode::RingModulate => {
                                                    param_row(ui, "Ring Mod Amount", &params.fusion_ring_mod_amount, setter)
                                                }
                                                // No Amount control for this mode - an empty row reserves
                                                // the same height anyway, so the grid (and everything
                                                // below it) doesn't shift between modes.
                                                _ => {
                                                    ui.label("");
                                                    ui.label("");
                                                    ui.end_row();
                                                }
                                            }
                                        },
                                    );

                                    clicked
                                })
                                .inner;
                            if clicked_load_b {
                                open_sample_dialog_b(state);
                            }

                            left.add_space(4.0);
                            if !has_loaded_sample_b && current_fusion.needs_sample_b() {
                                left.colored_label(COLOR_ERROR, "This mode needs Sample B - load one above to hear it.");
                            } else {
                                left.label("");
                            }

                            // Envelope + its 5 sliders in the middle column,
                            // the remaining 5 (Stereo Width through Random
                            // Pan Width) in the right column - splitting
                            // what used to be one 10-slider column in half
                            // so the window grows wider instead of taller.
                            // Each column's rows share one `egui::Grid` so
                            // every label lines up to the same width and
                            // every slider starts at the same x.
                            let middle = &mut columns[1];
                            middle.label("Envelope (Attack / Decay / Sustain / Release)");
                            draw_adsr_graph(middle, &params.attack, &params.decay, &params.sustain, &params.release, setter, scale);
                            egui::Grid::new("spectral_prism_envelope_grid").num_columns(2).spacing([8.0 * scale, 6.0 * scale]).show(middle, |ui| {
                                param_row(ui, "Attack", &params.attack, setter);
                                param_row(ui, "Decay", &params.decay, setter);
                                param_row(ui, "Sustain", &params.sustain, setter);
                                param_row(ui, "Release", &params.release, setter);
                                param_row(ui, "Velocity Sensitivity", &params.velocity_sensitivity, setter);
                            });

                            let right = &mut columns[2];
                            egui::Grid::new("spectral_prism_right_grid").num_columns(2).spacing([8.0 * scale, 6.0 * scale]).show(right, |ui| {
                                param_row(ui, "Stereo Width", &params.stereo_width, setter);
                                param_row(ui, "Loop Length", &params.loop_length_seconds, setter);
                                param_row(ui, "Pitch Bend Range", &params.pitch_bend_range_semitones, setter);
                                param_row(ui, "Pan Center", &params.pan_center_pct, setter);
                                param_row(ui, "Random Pan Width", &params.pan_width_pct, setter);
                            });
                        });

                        // Always visible (even in Off mode, describing plain
                        // Spectral Freeze) rather than only while a Fusion
                        // mode is selected, and pinned to a fixed height/
                        // position (right above the Load Sample buttons)
                        // rather than living inside the conditional Sample B
                        // block, so the rest of the layout doesn't shift
                        // around as the mode (and its description length)
                        // changes.
                        ui.add_space(8.0);
                        let current_fusion = params.fusion_mode.value();
                        egui::Frame::default()
                            .fill(COLOR_SURFACE_DEEP)
                            .stroke(egui::Stroke::new(1.0, COLOR_EDGE))
                            .inner_margin(egui::Margin::same((6.0 * scale) as i8))
                            .show(ui, |ui| {
                                ui.set_height(INFO_BOX_HEIGHT * scale);
                                egui::ScrollArea::vertical().max_height(INFO_BOX_HEIGHT * scale).show(ui, |ui| {
                                    ui.colored_label(COLOR_DIM, current_fusion.info_text());
                                });
                            });

                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(8.0);

                        ui.horizontal(|ui| {
                            if ui.button("Load Sample A...").clicked() {
                                open_sample_dialog(state);
                            }
                            if ui.button("Load Sample B...").clicked() {
                                open_sample_dialog_b(state);
                            }
                            if ui.button("Export WAV Sample...").clicked() {
                                let default_name = if state.preset_name_input.trim().is_empty() {
                                    "SpectralPrism-export".to_string()
                                } else {
                                    state.preset_name_input.trim().to_string()
                                };
                                if let Some(path) =
                                    rfd::FileDialog::new().add_filter("WAV", &["wav"]).set_file_name(&format!("{default_name}.wav")).save_file()
                                {
                                    match write_loop_buffer_wav(&path, &loop_buffer.load()) {
                                        Ok(()) => state.error = None,
                                        Err(e) => state.error = Some(e),
                                    }
                                }
                            }
                        });

                        ui.add_space(4.0);
                        match loaded_filename.load().as_deref() {
                            Some(name) => {
                                ui.colored_label(COLOR_DIM, format!("Loaded: {name}"));
                            }
                            None => {
                                ui.colored_label(COLOR_DIM, "Using built-in placeholder tone");
                            }
                        }
                        if let Some(error) = &state.error {
                            ui.colored_label(COLOR_ERROR, error);
                        }
                        });
                        });
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;

        // `sample_path` (if any) is restored from persisted state before
        // `initialize()` runs, same as every `FloatParam` - so a saved host
        // project or a recalled preset (see `FREEZE-PLAN-010`) gets its
        // actual sample back too, not just the built-in placeholder tone.
        // A missing/moved file falls back to the placeholder rather than
        // failing outright; there's no editor open yet to show an error to,
        // so this is logged instead.
        let restored_path = self.params.sample_path.lock().unwrap().clone();
        match restored_path.as_deref().map(|path| (path, load_and_prepare_sample(path, sample_rate))) {
            Some((path, Ok(prepared))) => {
                self.source.store(Arc::new(prepared));
                self.loaded_filename.store(path.file_name().map(|n| Arc::new(n.to_string_lossy().into_owned())));
            }
            Some((path, Err(e))) => {
                nih_log!("SpectralPrism: couldn't restore sample from {}: {e}", path.display());
                self.source.store(Arc::new(vec![synthetic_source(sample_rate, 1.0)]));
                self.loaded_filename.store(None);
            }
            None => {
                self.source.store(Arc::new(vec![synthetic_source(sample_rate, 1.0)]));
                self.loaded_filename.store(None);
            }
        }

        // Sample B mirrors Sample A's restore above, except there's no
        // synthetic placeholder to fall back to - a missing/moved file (or
        // simply never having loaded one) just leaves `source_b` empty,
        // which `prism_dsp::fusion::effective_mode` (used below and by
        // `RenderWorker`) already treats as "Fusion mode needs B but none is
        // loaded" and degrades gracefully rather than failing.
        let restored_path_b = self.params.sample_path_b.lock().unwrap().clone();
        match restored_path_b.as_deref().map(|path| (path, load_and_prepare_sample(path, sample_rate))) {
            Some((path, Ok(prepared))) => {
                self.source_b.store(Arc::new(prepared));
                self.loaded_filename_b.store(path.file_name().map(|n| Arc::new(n.to_string_lossy().into_owned())));
            }
            Some((path, Err(e))) => {
                nih_log!("SpectralPrism: couldn't restore Sample B from {}: {e}", path.display());
                self.loaded_filename_b.store(None);
            }
            None => {
                self.loaded_filename_b.store(None);
            }
        }

        let request = current_render_request(&self.params);
        // First render happens synchronously here (initialize() runs before
        // playback starts, so blocking is fine) so process() never sees the
        // placeholder silent buffer once the host actually starts playing.
        let effective_fusion =
            FusionRenderParams { mode: prism_dsp::fusion::effective_mode(request.fusion.mode, &self.source_b.load()), ..request.fusion };
        self.loop_buffer.store(Arc::new(render_fused_loop(
            &self.source.load(),
            &self.source_b.load(),
            sample_rate,
            request.freeze_point_pct,
            request.volume_pct,
            request.tune_semitones,
            request.formant_shift_semitones,
            &effective_fusion,
            request.stereo_width_pct,
            request.loop_length_seconds,
            DEFAULT_ROOT_NOTE,
        )));
        self.last_requested = request;

        self.worker = Some(RenderWorker::spawn(
            self.trigger.clone(),
            self.source.clone(),
            self.source_b.clone(),
            sample_rate,
            DEFAULT_ROOT_NOTE,
            self.loop_buffer.clone(),
        ));
        self.voices = VoiceManager::new(sample_rate, DEFAULT_ROOT_NOTE);
        self.sample_rate = sample_rate;
        self.pending_request = None;
        self.throttle_countdown = 0;

        true
    }

    fn reset(&mut self) {
        self.voices.choke_all();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Freeze Point / Formant Shift / Stereo Width are checked once per
        // block (not per-sample - they're not audio-rate, and re-rendering
        // the frozen loop is too expensive to consider on every sample
        // anyway). A detected change is only actually sent to the worker
        // once every RENDER_THROTTLE_MS, so a smooth drag/automation sweep
        // can't land a new buffer swap before the previous one's crossfade
        // has finished (see RENDER_THROTTLE_MS for why that matters).
        let current_request = current_render_request(&self.params);
        if current_request != self.last_requested {
            self.pending_request = Some(current_request);
        }

        self.throttle_countdown -= buffer.samples() as i64;
        if self.throttle_countdown <= 0 {
            if let Some(request) = self.pending_request.take() {
                self.trigger.request_render(request);
                self.last_requested = request;
            }
            self.throttle_countdown = ((RENDER_THROTTLE_MS / 1000.0) * self.sample_rate) as i64;
        }

        // Attack/Decay/Sustain/Release are cheap (per-sample, not
        // render-time like the three above) - no worker/crossfade/throttle
        // needed, just apply the current values to any voices triggered
        // from this point on. Like most synths, a change here doesn't
        // reshape a note already mid-envelope - only affects new note-ons.
        self.voices.set_adsr(AdsrSettings {
            attack_ms: self.params.attack.value(),
            decay_ms: self.params.decay.value(),
            sustain_level: self.params.sustain.value() / 100.0,
            release_ms: self.params.release.value(),
        });
        self.voices.set_velocity_sensitivity(self.params.velocity_sensitivity.value() / 100.0);
        self.voices.set_pan_settings(self.params.pan_center_pct.value() / 100.0, self.params.pan_width_pct.value() / 100.0);

        // Block-level MIDI handling: every event pending for this buffer is
        // applied before rendering, rather than split at the exact sample it
        // arrived on. Good enough for proving polyphony/pitch/voice-stealing
        // here; sample-accurate note timing can be revisited later if a
        // fast arpeggio/chord attack audibly needs it.
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::NoteOn { note, channel, velocity, voice_id, .. } => {
                    self.voices.note_on(note, channel, velocity, voice_id.unwrap_or(note as i32));
                    self.last_note.store(note, Ordering::Relaxed);
                }
                NoteEvent::NoteOff { note, channel, .. } => {
                    self.voices.note_off(note, channel);
                }
                NoteEvent::Choke { .. } => {
                    self.voices.choke_all();
                }
                NoteEvent::MidiPitchBend { value, .. } => {
                    // `value` is normalized [0, 1] with 0.5 = no bend (see
                    // nih-plug's own doc comment on the variant) - stored
                    // raw and converted to semitones below, after the loop,
                    // so a same-block wheel movement still takes effect for
                    // this block rather than being delayed to the next one.
                    self.pitch_bend_normalized = value;
                }
                _ => (),
            }
        }
        self.voices.set_pitch_bend_semitones(
            (self.pitch_bend_normalized - 0.5) * 2.0 * self.params.pitch_bend_range_semitones.value(),
        );

        let loop_buffer = self.loop_buffer.load();
        let channels = buffer.as_slice();
        let (left, right) = channels.split_at_mut(1);
        self.voices.process_block(&loop_buffer, left[0], right[0]);
        // Read *after* process_block so a voice that finished this block is
        // already reflected, not the stale pre-block count.
        self.active_voice_count.store(self.voices.active_voice_count() as u8, Ordering::Relaxed);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for PrismPlugin {
    const CLAP_ID: &'static str = "com.johnoestmannmusic.spectral-prism";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Freezes a spectral snapshot of a sample into a sustained pad/drone");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for PrismPlugin {
    // Deliberately different from the old "SpectralFreeze01" - this is a
    // rename (SpectralPrism), not just a relabeling, and a new class ID is
    // the correct way to signal that to a host (a host distinguishes VST3
    // plugins by this ID, not by name). Any already-saved project
    // referencing the old ID will need the renamed plugin re-added.
    const VST3_CLASS_ID: [u8; 16] = *b"SpectralPrism001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth];
}

nih_export_clap!(PrismPlugin);
nih_export_vst3!(PrismPlugin);

#[cfg(test)]
mod date_tests {
    use super::*;

    #[test]
    fn civil_from_days_matches_known_reference_dates() {
        // Unix epoch itself.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // A pre-epoch date (negative day count) - exercises the `era`
        // branch's negative-side rounding.
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        // A leap-year February 29th.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        // The date this feature was actually built on, cross-checked
        // against the real calendar by hand.
        assert_eq!(civil_from_days(20_707), (2026, 9, 11));
    }

    #[test]
    fn today_utc_date_matches_yyyy_mm_dd_shape() {
        let date = today_utc_date();
        assert_eq!(date.len(), 10, "expected YYYY-MM-DD, got {date:?}");
        assert_eq!(date.as_bytes()[4], b'-');
        assert_eq!(date.as_bytes()[7], b'-');
        assert!(date.chars().enumerate().all(|(i, c)| (i == 4 || i == 7) || c.is_ascii_digit()));
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    /// A fresh, empty directory under the OS temp dir, unique per test run
    /// (no `tempfile` dependency needed for this small a use). Not cleaned
    /// up automatically - left for the OS/user to reclaim, same as any
    /// other stray `/tmp` file, since these tests don't run often enough
    /// for that to matter.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "spectralprism_test_{name}_{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create temp test dir");
        dir
    }

    fn sample_preset(sample_path: Option<PathBuf>) -> Preset {
        Preset {
            freeze_point_pct: 42.0,
            volume_a_pct: 85.0,
            tune_a_semitones: 2.0,
            formant_shift_semitones: -3.5,
            stereo_width_pct: 60.0,
            loop_length_seconds: 2.5,
            attack_ms: 12.0,
            decay_ms: 250.0,
            sustain_pct: 70.0,
            release_ms: 500.0,
            velocity_sensitivity_pct: 80.0,
            pitch_bend_range_semitones: 4.0,
            pan_center_pct: -10.0,
            pan_width_pct: 35.0,
            sample_path,
            build_number: "20260101".to_string(),
            fusion_mode: "cross-synth".to_string(),
            freeze_point_b_pct: 65.0,
            volume_b_pct: 90.0,
            tune_b_semitones: -1.5,
            formant_shift_b_semitones: 1.5,
            fusion_mix_amount_pct: 30.0,
            fusion_cross_synth_amount_pct: 80.0,
            fusion_convolve_amount_pct: 90.0,
            fusion_ring_mod_amount_pct: 70.0,
            sample_path_b: Some(PathBuf::from("/some/sample-b.wav")),
            unload_samples: false,
            title: "My Test Preset".to_string(),
            author: "Test Author".to_string(),
            category: "Pads".to_string(),
            sub_category: "Evolving".to_string(),
            date_last_updated: "2026-09-11".to_string(),
        }
    }

    #[test]
    fn preset_round_trips_through_json() {
        let original = sample_preset(Some(PathBuf::from("/some/sample.wav")));
        let json = serde_json::to_string(&original).unwrap();
        let restored: Preset = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.freeze_point_pct, original.freeze_point_pct);
        assert_eq!(restored.volume_a_pct, original.volume_a_pct);
        assert_eq!(restored.tune_a_semitones, original.tune_a_semitones);
        assert_eq!(restored.formant_shift_semitones, original.formant_shift_semitones);
        assert_eq!(restored.stereo_width_pct, original.stereo_width_pct);
        assert_eq!(restored.loop_length_seconds, original.loop_length_seconds);
        assert_eq!(restored.attack_ms, original.attack_ms);
        assert_eq!(restored.decay_ms, original.decay_ms);
        assert_eq!(restored.sustain_pct, original.sustain_pct);
        assert_eq!(restored.release_ms, original.release_ms);
        assert_eq!(restored.velocity_sensitivity_pct, original.velocity_sensitivity_pct);
        assert_eq!(restored.pitch_bend_range_semitones, original.pitch_bend_range_semitones);
        assert_eq!(restored.pan_center_pct, original.pan_center_pct);
        assert_eq!(restored.pan_width_pct, original.pan_width_pct);
        assert_eq!(restored.sample_path, original.sample_path);
        assert_eq!(restored.build_number, original.build_number);
        assert_eq!(restored.fusion_mode, original.fusion_mode);
        assert_eq!(restored.freeze_point_b_pct, original.freeze_point_b_pct);
        assert_eq!(restored.volume_b_pct, original.volume_b_pct);
        assert_eq!(restored.tune_b_semitones, original.tune_b_semitones);
        assert_eq!(restored.formant_shift_b_semitones, original.formant_shift_b_semitones);
        assert_eq!(restored.fusion_mix_amount_pct, original.fusion_mix_amount_pct);
        assert_eq!(restored.fusion_cross_synth_amount_pct, original.fusion_cross_synth_amount_pct);
        assert_eq!(restored.fusion_convolve_amount_pct, original.fusion_convolve_amount_pct);
        assert_eq!(restored.fusion_ring_mod_amount_pct, original.fusion_ring_mod_amount_pct);
        assert_eq!(restored.sample_path_b, original.sample_path_b);
        assert_eq!(restored.unload_samples, original.unload_samples);
        assert_eq!(restored.title, original.title);
        assert_eq!(restored.author, original.author);
        assert_eq!(restored.category, original.category);
        assert_eq!(restored.sub_category, original.sub_category);
        assert_eq!(restored.date_last_updated, original.date_last_updated);
    }

    #[test]
    fn preset_without_fusion_fields_still_deserializes() {
        // Regression test for backward compatibility: a preset saved before
        // Spectral Fusion existed must still load, defaulting to Fusion off
        // and every new control at the same default its `FloatParam`/
        // `EnumParam` counterpart has for a freshly-created instance (see
        // each field's own `#[serde(default = "...")]` doc comment).
        let old_json = r#"{
            "freeze_point_pct": 42.0,
            "formant_shift_semitones": -3.5,
            "stereo_width_pct": 60.0,
            "attack_ms": 12.0,
            "decay_ms": 250.0,
            "sustain_pct": 70.0,
            "release_ms": 500.0,
            "velocity_sensitivity_pct": 80.0,
            "sample_path": null
        }"#;
        let restored: Preset = serde_json::from_str(old_json).expect("old-format preset should still deserialize");
        assert_eq!(restored.volume_a_pct, 100.0);
        assert_eq!(restored.tune_a_semitones, 0.0);
        assert_eq!(restored.fusion_mode, "");
        assert_eq!(FusionMode::from_preset_id(&restored.fusion_mode), FusionMode::Off);
        assert_eq!(restored.freeze_point_b_pct, 50.0);
        assert_eq!(restored.volume_b_pct, 100.0);
        assert_eq!(restored.tune_b_semitones, 0.0);
        assert_eq!(restored.formant_shift_b_semitones, 0.0);
        assert_eq!(restored.fusion_mix_amount_pct, 50.0);
        assert_eq!(restored.fusion_cross_synth_amount_pct, 100.0);
        assert_eq!(restored.fusion_convolve_amount_pct, 100.0);
        assert_eq!(restored.fusion_ring_mod_amount_pct, 100.0);
        assert_eq!(restored.sample_path_b, None);
        assert!(!restored.unload_samples);
        assert_eq!(restored.title, "");
        assert_eq!(restored.author, "");
        assert_eq!(restored.category, "");
        assert_eq!(restored.sub_category, "");
        assert_eq!(restored.date_last_updated, "");
    }

    #[test]
    fn preset_without_pitch_bend_or_pan_fields_still_deserializes() {
        // Regression test for backward compatibility: a preset saved before
        // FREEZE-PLAN-018 added these fields must still load, defaulting to
        // pitch bend disabled and no panning (see the `Preset` struct's doc
        // comment on why 0.0 is an acceptable fallback for both); before
        // FREEZE-PLAN-023 added build_number, defaulting to an empty string
        // (displayed as "unknown", never applied to anything); and before
        // FREEZE-PLAN-024 added loop_length_seconds, defaulting to
        // DEFAULT_LOOP_SECONDS rather than 0.0 (see that field's doc
        // comment on why 0.0 would be nonsensical here).
        let old_json = r#"{
            "freeze_point_pct": 42.0,
            "formant_shift_semitones": -3.5,
            "stereo_width_pct": 60.0,
            "attack_ms": 12.0,
            "decay_ms": 250.0,
            "sustain_pct": 70.0,
            "release_ms": 500.0,
            "velocity_sensitivity_pct": 80.0,
            "sample_path": null
        }"#;
        let restored: Preset = serde_json::from_str(old_json).expect("old-format preset should still deserialize");
        assert_eq!(restored.pitch_bend_range_semitones, 0.0);
        assert_eq!(restored.pan_center_pct, 0.0);
        assert_eq!(restored.pan_width_pct, 0.0);
        assert_eq!(restored.build_number, "");
        assert_eq!(restored.loop_length_seconds, prism_dsp::render::DEFAULT_LOOP_SECONDS);
    }

    #[test]
    fn save_then_load_preset_round_trips_on_disk() {
        let dir = temp_dir("save_load");
        let original = sample_preset(None);

        save_preset(&dir, "My Test Preset", &original).expect("save should succeed");
        let restored = load_preset(&dir, "My Test Preset").expect("load should succeed");

        assert_eq!(restored.freeze_point_pct, original.freeze_point_pct);
        assert_eq!(restored.sample_path, None);
    }

    #[test]
    fn delete_preset_removes_it_from_disk() {
        let dir = temp_dir("delete_preset");
        let original = sample_preset(None);
        save_preset(&dir, "My Test Preset", &original).expect("save should succeed");
        assert!(list_presets(&dir).contains(&"My Test Preset".to_string()));

        delete_preset(&dir, "My Test Preset").expect("delete should succeed");

        assert!(!list_presets(&dir).contains(&"My Test Preset".to_string()));
        assert!(load_preset(&dir, "My Test Preset").is_err());
    }

    #[test]
    fn list_presets_returns_sorted_names_without_extension() {
        let dir = temp_dir("list");
        save_preset(&dir, "Zebra", &sample_preset(None)).unwrap();
        save_preset(&dir, "Alpha", &sample_preset(None)).unwrap();
        save_preset(&dir, "Mango", &sample_preset(None)).unwrap();
        // A non-JSON file in the same directory should be ignored.
        std::fs::write(dir.join("notes.txt"), "not a preset").unwrap();

        let names = list_presets(&dir);

        assert_eq!(names, vec!["Alpha".to_string(), "Mango".to_string(), "Zebra".to_string()]);
    }

    #[test]
    fn list_presets_on_missing_directory_returns_empty() {
        let dir = std::env::temp_dir().join("spectralprism_test_definitely_does_not_exist");
        assert!(list_presets(&dir).is_empty());
    }

    #[test]
    fn load_preset_with_unknown_name_fails_with_message() {
        let dir = temp_dir("missing_preset");
        let result = load_preset(&dir, "does_not_exist");
        assert!(result.is_err());
    }

    #[test]
    fn write_then_read_preset_file_round_trips_at_an_arbitrary_path() {
        // The file-dialog Import/Export flow, as opposed to save_preset/
        // load_preset's fixed presets_dir()-relative naming.
        let dir = temp_dir("preset_file");
        let path = dir.join("my-shared-preset.spjson");
        let original = sample_preset(Some(PathBuf::from("/some/sample.wav")));

        write_preset_file(&path, &original).expect("write should succeed");
        let restored = read_preset_file(&path).expect("read should succeed");

        assert_eq!(restored.freeze_point_pct, original.freeze_point_pct);
        assert_eq!(restored.pan_width_pct, original.pan_width_pct);
        assert_eq!(restored.sample_path, original.sample_path);
    }

    #[test]
    fn write_loop_buffer_wav_produces_a_readable_stereo_wav() {
        let dir = temp_dir("wav_export");
        let path = dir.join("export.wav");
        let buffer = LoopBufferData {
            channels: vec![vec![0.5f32; 100], vec![-0.25f32; 100]],
            sample_rate: 48000.0,
            root_note: DEFAULT_ROOT_NOTE,
        };

        write_loop_buffer_wav(&path, &buffer).expect("export should succeed");

        let mut reader = hound::WavReader::open(&path).expect("exported file should be a valid WAV");
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, 48000);
        let samples: Vec<i32> = reader.samples::<i32>().map(|s| s.unwrap()).collect();
        // Interleaved L/R - spot-check the first frame rather than every
        // sample, converting back from i16 to approximately the original
        // f32 range to allow for quantization rounding. The export is
        // peak-normalized, so the 0.5/-0.25 source (peak 0.5) comes out
        // scaled 2x to 1.0/-0.5.
        assert!((samples[0] as f32 / i16::MAX as f32 - 1.0).abs() < 1e-3);
        assert!((samples[1] as f32 / i16::MAX as f32 - -0.5).abs() < 1e-3);
    }

    #[test]
    fn write_loop_buffer_wav_peak_normalizes_a_quiet_buffer() {
        let dir = temp_dir("wav_export_normalize");
        let path = dir.join("export.wav");
        let buffer = LoopBufferData { channels: vec![vec![0.1f32; 100], vec![-0.05f32; 100]], sample_rate: 48000.0, root_note: DEFAULT_ROOT_NOTE };

        write_loop_buffer_wav(&path, &buffer).expect("export should succeed");

        let mut reader = hound::WavReader::open(&path).expect("exported file should be a valid WAV");
        let samples: Vec<i32> = reader.samples::<i32>().map(|s| s.unwrap()).collect();
        // Peak was 0.1, so the export should be scaled 10x to hit unity.
        assert!((samples[0] as f32 / i16::MAX as f32 - 1.0).abs() < 1e-3);
        assert!((samples[1] as f32 / i16::MAX as f32 - -0.5).abs() < 1e-3);
    }

    #[test]
    fn write_loop_buffer_wav_fails_with_message_when_nothing_frozen() {
        let dir = temp_dir("wav_export_empty");
        let path = dir.join("export.wav");
        let buffer = LoopBufferData { channels: Vec::new(), sample_rate: 48000.0, root_note: DEFAULT_ROOT_NOTE };
        assert!(write_loop_buffer_wav(&path, &buffer).is_err());
    }
}
