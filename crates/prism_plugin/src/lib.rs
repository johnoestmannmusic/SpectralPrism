mod editor;
mod params;
mod preset;
mod render_worker;
mod samples;
mod stage_params;
mod stages;
mod theme;
mod widgets;

use arc_swap::{ArcSwap, ArcSwapOption};
use params::*;
use prism_dsp::chain::render_chain;
use prism_dsp::chord::ChordSettings;
use prism_dsp::fusion::FusionRenderParams;
use prism_dsp::render::{LoopBufferData, DEFAULT_ROOT_NOTE};
use prism_dsp::voice::VoiceManager;
use nih_plug::prelude::*;
use render_worker::{RenderRequest, RenderTrigger, RenderWorker};
use samples::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

/// Sentinel for "no note has been received yet" in `last_note`, distinct from
/// any real MIDI note number (0-127).
pub(crate) const NO_NOTE: u8 = 255;

/// YYYYMMDD, stamped at compile time by `build.rs` - shown in the editor's
/// heading and bottom-right corner, and saved into every exported preset,
/// so it's possible to tell which plugin version made a given sound.
pub(crate) const BUILD_NUMBER: &str = env!("SPECTRALPRISM_BUILD_NUMBER");

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
    /// Last chord settings handed to the voices - keeps the custom
    /// intervals if the GUI holds their lock during a block.
    chord: ChordSettings,
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
    /// Output peak level (linear, `f32` bits) for the editor's meter, with
    /// a falling decay, and a flag set whenever the output reaches 0 dBFS
    /// (the editor clears it once it has shown it).
    output_peak: Arc<AtomicU32>,
    output_clipped: Arc<AtomicBool>,
    /// Per-sample multiplier for the meter's fall-off.
    meter_decay_per_sample: f32,
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
            last_requested: RenderRequest::default(),
            chord: ChordSettings::default(),
            pending_request: None,
            throttle_countdown: 0,
            last_note: Arc::new(AtomicU8::new(NO_NOTE)),
            active_voice_count: Arc::new(AtomicU8::new(0)),
            output_peak: Arc::new(AtomicU32::new(0)),
            output_clipped: Arc::new(AtomicBool::new(false)),
            meter_decay_per_sample: 0.0,
        }
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
        editor::create(editor::EditorData {
            params: self.params.clone(),
            source: self.source.clone(),
            source_b: self.source_b.clone(),
            loop_buffer: self.loop_buffer.clone(),
            trigger: self.trigger.clone(),
            last_note: self.last_note.clone(),
            active_voice_count: self.active_voice_count.clone(),
            loaded_filename: self.loaded_filename.clone(),
            loaded_filename_b: self.loaded_filename_b.clone(),
            output_peak: self.output_peak.clone(),
            output_clipped: self.output_clipped.clone(),
        })
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
        self.loop_buffer.store(Arc::new(render_chain(
            &RenderRequest { fusion: effective_fusion, ..request },
            &self.source.load(),
            &self.source_b.load(),
            sample_rate,
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
        // The meter falls by 12 dB (x0.25) in ~150 ms, as in nih-plug's own examples.
        self.meter_decay_per_sample = 0.25f32.powf(1.0 / (sample_rate * 0.15));
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

        // Real-time settings (envelope, velocity, pan, Voice / Filter /
        // Chord / Chorus) are cheap per-block reads - no render needed.
        // `current_request` decides what Auto playback means (one-shot hits
        // vs looping pads). The same helper configures the WAV export.
        self.chord = apply_realtime_settings(&mut self.voices, &self.params, &current_request, &self.chord);

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

        // Output Volume: the very end of the chain, after the soft limiter -
        // so boosting past 0 dB really can clip, which the meter shows.
        let mut peak = 0.0f32;
        for (l, r) in left[0].iter_mut().zip(right[0].iter_mut()) {
            let gain = output_gain(self.params.output.volume.smoothed.next());
            *l *= gain;
            *r *= gain;
            peak = peak.max(l.abs()).max(r.abs());
        }
        let shown = f32::from_bits(self.output_peak.load(Ordering::Relaxed));
        let decayed = shown * self.meter_decay_per_sample.powi(buffer.samples() as i32);
        self.output_peak.store(peak.max(decayed).to_bits(), Ordering::Relaxed);
        if peak >= 1.0 {
            self.output_clipped.store(true, Ordering::Relaxed);
        }
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
