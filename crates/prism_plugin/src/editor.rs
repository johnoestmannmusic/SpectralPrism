//! The plugin editor.

use crate::params::*;
use crate::preset::*;
use crate::render_worker::RenderTrigger;
use crate::samples::*;
use crate::stages::*;
use crate::theme::*;
use crate::widgets::*;
use crate::NO_NOTE;
use crate::BUILD_NUMBER;
use arc_swap::{ArcSwap, ArcSwapOption};
use nih_plug::prelude::*;
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::{create_egui_editor, egui};
use prism_dsp::render::LoopBufferData;
use prism_dsp::voice::PlaybackMode;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

/// GUI-thread-only state for the editor - not shared with the audio thread
/// and not persisted. Which file is loaded lives on the plugin itself
/// (`PrismPlugin::loaded_filename`) instead, so it survives the editor
/// window being closed and reopened; the selected-preset name lives on
/// `PrismPluginParams` for the same reason (and so it round-trips through a
/// saved host project - see that field's doc comment). Only the load-error
/// message and the preset-name text field (fine to forget when the editor
/// is reopened - not sound-affecting state) stay here.
#[derive(Default)]
pub(crate) struct PrismEditorState {
    pub(crate) error: Option<String>,
    /// Text field buffer for naming a new preset to save - also doubles as
    /// the Save dialog's prefilled Title.
    pub(crate) preset_name_input: String,
    /// Whether the Save dialog (Title/Author/Category/Sub-category) is
    /// currently open - opened by clicking "Save", closed by its own
    /// Save/Cancel buttons or backdrop click/Escape (`egui::Modal`).
    pub(crate) show_save_dialog: bool,
    pub(crate) save_title_input: String,
    pub(crate) save_author_input: String,
    pub(crate) save_category_input: String,
    pub(crate) save_subcategory_input: String,
    /// Whether the Preset Browser modal is currently open.
    pub(crate) show_preset_browser: bool,
    /// Live search text, filtering the browser's Category/Sub-category/
    /// Preset tree by title/author substring match.
    pub(crate) preset_browser_search: String,
    /// Every named on-disk preset's browsable metadata - loaded once when
    /// the browser is opened (not re-read from disk every frame it's open).
    pub(crate) preset_browser_entries: Vec<PresetBrowserEntry>,
    /// Which preset (by its on-disk name, i.e. `PresetBrowserEntry::name`)
    /// is currently highlighted in the browser's tree, if any - drives the
    /// info panel at the bottom of the modal. Distinct from actually
    /// loading it, which only happens when "Load" is clicked.
    pub(crate) preset_browser_selected: Option<String>,
    /// Set to the on-disk name of a preset the user just clicked "Delete"
    /// on, so the browser can ask "are you sure?" before actually removing
    /// the file - cleared on confirm, cancel, or closing the browser.
    pub(crate) preset_browser_delete_confirm: Option<String>,
    /// Whether the "Export Preset..." strip-sample-paths prompt is
    /// currently open - opened by clicking "Export Preset...", closed by
    /// its own Export/Cancel buttons or backdrop click/Escape.
    pub(crate) show_export_preset_dialog: bool,
    /// Which chain stage's panel is open.
    pub(crate) selected_stage: Stage,
    /// Cached peaks of the output display.
    pub(crate) peak_cache: PeakCache,
    /// Chord mode's Custom intervals text field buffer.
    pub(crate) chord_text: String,
    /// When the output last hit 0 dBFS - keeps the meter red for a moment.
    pub(crate) last_clip: Option<std::time::Instant>,
    /// Checkbox state in that prompt: whether to reduce
    /// `sample_path`/`sample_path_b` in the exported file down to just the
    /// filename (for sharing a preset publicly, where the recipient's
    /// absolute sample paths won't exist) - the filename itself is kept
    /// (not blanked entirely) so the info panel/relocate dialog can still
    /// show/suggest it; `apply_preset`'s existing missing-sample relocate
    /// flow already handles a path that doesn't resolve gracefully on
    /// re-import, prompting to locate the file rather than failing.
    pub(crate) export_strip_sample_paths: bool,
    /// Title/Author/Category/Sub-category/date-last-updated of whichever
    /// preset is currently active in memory - kept in sync everywhere
    /// `params.selected_preset` itself is (load/import/save), since
    /// `Preset::capture` deliberately leaves these blank (only the Save
    /// dialog fills them in directly, on the `Preset` it's about to write).
    /// "Export Preset..." needs this metadata to carry it into the
    /// exported file instead of exporting it blank.
    pub(crate) current_preset_title: String,
    pub(crate) current_preset_author: String,
    pub(crate) current_preset_category: String,
    pub(crate) current_preset_sub_category: String,
    pub(crate) current_preset_date_last_updated: String,
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
pub(crate) fn load_sample_from_path(
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

/// Everything the editor needs from the plugin, cloned out of it.
pub(crate) struct EditorData {
    pub(crate) params: Arc<PrismPluginParams>,
    pub(crate) source: Arc<ArcSwap<Vec<Vec<f32>>>>,
    pub(crate) source_b: Arc<ArcSwap<Vec<Vec<f32>>>>,
    pub(crate) loop_buffer: Arc<ArcSwap<LoopBufferData>>,
    pub(crate) trigger: RenderTrigger,
    pub(crate) last_note: Arc<AtomicU8>,
    pub(crate) active_voice_count: Arc<AtomicU8>,
    pub(crate) loaded_filename: Arc<ArcSwapOption<String>>,
    pub(crate) loaded_filename_b: Arc<ArcSwapOption<String>>,
    pub(crate) output_peak: Arc<AtomicU32>,
    pub(crate) output_clipped: Arc<AtomicBool>,
}

pub(crate) fn create(data: EditorData) -> Option<Box<dyn Editor>> {
    let EditorData { params, source, source_b, loop_buffer, trigger, last_note, active_voice_count, loaded_filename, loaded_filename_b, output_peak, output_clipped } = data;

    let initial_preset_name = params.preset_name_input.lock().unwrap().clone();
    create_egui_editor(
        params.editor_state.clone(),
        PrismEditorState {
            // `PrismEditorState` itself is purely GUI-local and
            // rebuilt fresh every time the editor is (re)opened - see
            // `PrismPluginParams::preset_name_input`'s doc comment for
            // why this field specifically needs to be seeded from
            // that persisted mirror here, rather than just starting
            // blank via `Default::default()`.
            preset_name_input: initial_preset_name,
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
                apply_stage_params(preset, &params, setter);

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
                    ui.horizontal(|ui| {
                    // Output Volume + meter: the end of the chain, always
                    // visible here (outside the scrolling content).
                    if output_clipped.swap(false, Ordering::Relaxed) {
                        state.last_clip = Some(std::time::Instant::now());
                    }
                    let clipping = state.last_clip.is_some_and(|t| t.elapsed().as_secs_f32() < 1.5);
                    ui.label(egui::RichText::new("OUTPUT").small().color(COLOR_INK));
                    draw_output_meter(
                        ui,
                        &params.output.volume,
                        setter,
                        f32::from_bits(output_peak.load(Ordering::Relaxed)),
                        clipping,
                        OUTPUT_MIN_DB,
                        OUTPUT_MAX_DB,
                        220.0 * scale,
                        scale,
                    );
                    ui.label(egui::RichText::new(params.output.volume.to_string()).small().color(if clipping { COLOR_ERROR } else { COLOR_INK }));
                    if clipping {
                        ui.label(egui::RichText::new("CLIP").small().strong().color(COLOR_ERROR));
                    }
                    ui.separator();
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
                });
            // The meter moves with the audio, not with input events.
            egui_ctx.request_repaint();

            ResizableWindow::new("spectral_prism_window")
                .min_size(egui::vec2(BASE_EDITOR_WIDTH as f32, BASE_EDITOR_HEIGHT as f32))
                .show(egui_ctx, &params.editor_state, |ui| {
                    egui::Frame::default().inner_margin(egui::Margin::same((EDITOR_MARGIN * scale) as i8)).show(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {

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
                        ui.heading("SpectralPrism");
                        ui.colored_label(COLOR_DIM, egui::RichText::new(format!("v{BUILD_NUMBER}")).small());
                        ui.separator();
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

                    ui.add_space(6.0 * scale);

                    // --- Output: the loop/hit MIDI notes play right now. ---
                    let current = loop_buffer.load_full();
                    let frames = current.channels.first().map_or(0, Vec::len);
                    let secs = frames as f32 / current.sample_rate.max(1.0);
                    let chain = current_render_request(&params);
                    let mode = match current_voice_settings(&params, &chain).playback {
                        PlaybackMode::Loop => "loop",
                        PlaybackMode::PingPong => "ping-pong loop",
                        PlaybackMode::OneShot => "one-shot",
                    };
                    let caption = format!("OUTPUT  ·  {secs:.2} s {mode}");
                    draw_output_display(ui, &current, &mut state.peak_cache, trigger.is_busy(), &caption, GRAPH_HEIGHT * 1.3 * scale, scale);
                    ui.add_space(8.0 * scale);

                    // --- Signal chain: pick a stage, toggle stages on/off. ---
                    let mut help: Option<&'static str> = None;
                    let name_a = loaded_filename.load().as_deref().map(|s| s.to_string());
                    let name_b = loaded_filename_b.load().as_deref().map(|s| s.to_string());
                    if let Some(stage) = draw_chain_strip(ui, &params, setter, state.selected_stage, name_a.as_deref(), scale, &mut help) {
                        state.selected_stage = stage;
                    }
                    ui.add_space(4.0 * scale);

                    // --- The selected stage's controls. ---
                    // Re-read every frame so a preset load shows up; edits in
                    // the text field write straight back (see `chord_panel`).
                    state.chord_text = params.chord.custom_intervals.lock().unwrap().clone();
                    let stage = state.selected_stage;
                    let cx = PanelCtx {
                        params: &params,
                        setter,
                        scale,
                        source: &source,
                        source_b: &source_b,
                        sample_rate: current.sample_rate,
                        name_a,
                        name_b,
                    };
                    let actions = ui.scope(|ui| draw_stage_panel(ui, stage, &cx, &mut help, &mut state.chord_text)).inner;
                    if actions.load_a {
                        open_sample_dialog(state);
                    }
                    if actions.load_b {
                        open_sample_dialog_b(state);
                    }

                    // --- Info line: hovered control's help, else the stage's. ---
                    ui.add_space(6.0 * scale);
                    let info = help.unwrap_or(stage.description());
                    egui::Frame::default()
                        .fill(COLOR_SURFACE_DEEP)
                        .stroke(egui::Stroke::new(1.0_f32, COLOR_EDGE))
                        .inner_margin(egui::Margin::same((6.0 * scale) as i8))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.set_height(INFO_BOX_HEIGHT * scale);
                            ui.add(egui::Label::new(egui::RichText::new(info).color(COLOR_DIM)).wrap());
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
                        let export = ui.button("Export WAV...").on_hover_text(
                            "Exports one note at C4 through the whole chain - envelope, chord and strum, filter, chorus \
                             and Output Volume - until its release has died away, at the level you hear.",
                        );
                        if export.clicked() {
                            let default_name = if state.preset_name_input.trim().is_empty() {
                                "SpectralPrism-export".to_string()
                            } else {
                                state.preset_name_input.trim().to_string()
                            };
                            if let Some(path) =
                                rfd::FileDialog::new().add_filter("WAV", &["wav"]).set_file_name(&format!("{default_name}.wav")).save_file()
                            {
                                let note = render_played_note(&params, &loop_buffer.load_full());
                                match write_loop_buffer_wav(&path, &note, false) {
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
