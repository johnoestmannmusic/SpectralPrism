//! Hand-painted controls shared by the stage panels.

use crate::theme::*;
use arc_swap::ArcSwap;
use nih_plug::prelude::*;
use nih_plug_egui::{egui, widgets};
use std::sync::Arc;

/// Width (1x-scale points) of `draw_vertical_volume_slider` - drawn inside
/// the same `ui.horizontal` as `draw_freeze_point_waveform`, so the
/// waveform (which sizes itself to whatever `ui.available_width()` is left
/// once this has claimed its own space) automatically ends up narrower by
/// exactly this much, no manual arithmetic needed at the call site.
pub(crate) const VOLUME_SLIDER_WIDTH: f32 = 28.0;


/// A vertical fader for a sample's Volume, meant to sit directly to the
/// left of that sample's `draw_freeze_point_waveform` (`nih_plug_egui`'s
/// `ParamSlider` is horizontal-only - see its own doc comment - so this is
/// hand-painted the same way `draw_freeze_point_waveform`/`draw_adsr_graph`
/// already are). A fixed-size percentage readout sits above a click/drag
/// track - dragging (or clicking) anywhere in the track sets the value via
/// the same begin/set/end-normalized pattern every other custom control
/// here uses, with the fill growing up from the bottom (louder = taller, a
/// fader without a legend needed).
pub(crate) fn draw_vertical_volume_slider(ui: &mut egui::Ui, volume: &FloatParam, setter: &ParamSetter, height: f32, scale: f32) {
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
    painter.rect_stroke(track_rect, 2.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);

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
pub(crate) fn draw_freeze_point_waveform(
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
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);
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
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);

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
        painter.line_segment([egui::pos2(x, y_top), egui::pos2(x, y_bottom)], egui::Stroke::new(1.0_f32, waveform_stroke));
    }

    let freeze_normalized = freeze_point.unmodulated_normalized_value();
    let marker_x = rect.left() + freeze_normalized * rect.width();
    painter.line_segment(
        [egui::pos2(marker_x, rect.top()), egui::pos2(marker_x, rect.bottom())],
        egui::Stroke::new(2.0_f32, COLOR_ACCENT),
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
pub(crate) fn draw_adsr_graph(
    ui: &mut egui::Ui,
    id_salt: &str,
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
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);

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

    let curve_stroke = egui::Stroke::new(2.0_f32, COLOR_ACCENT);
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
    let attack_id = ui.make_persistent_id(("adsr_attack_handle", id_salt));
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
    let decay_sustain_id = ui.make_persistent_id(("adsr_decay_sustain_handle", id_salt));
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
    let release_id = ui.make_persistent_id(("adsr_release_handle", id_salt));
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

// ------------------------------------------------------------------------
// Rows with hover help. Every control writes its one-line explanation into
// `help` while hovered; the editor shows it in the info line under the
// stage panel (falling back to the stage's own description).
// ------------------------------------------------------------------------

/// Small accent dot drawn after a label whose param a mod route targets.
fn mod_dot(ui: &mut egui::Ui) {
    let size = ui.text_style_height(&egui::TextStyle::Body) * 0.45;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size * 1.6, size), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), size * 0.5, COLOR_ACCENT);
    response.on_hover_text("Modulated by a route in the Spectral stage's Modulation section");
}

fn label_cell(ui: &mut egui::Ui, label: &str, modulated: bool) -> egui::Response {
    ui.horizontal(|ui| {
        let r = ui.label(label);
        if modulated {
            mod_dot(ui);
        }
        r
    })
    .inner
}

/// "label | slider" grid row (any param type) with hover help.
pub(crate) fn row<P: Param>(ui: &mut egui::Ui, label: &str, param: &P, setter: &ParamSetter, tip: &'static str, help: &mut Option<&'static str>) {
    row_mod(ui, label, param, setter, false, tip, help);
}

/// `row` plus a "modulated" dot after the label.
pub(crate) fn row_mod<P: Param>(
    ui: &mut egui::Ui,
    label: &str,
    param: &P,
    setter: &ParamSetter,
    modulated: bool,
    tip: &'static str,
    help: &mut Option<&'static str>,
) {
    let l = label_cell(ui, label, modulated);
    let s = ui.add(widgets::ParamSlider::for_param(param, setter));
    if l.hovered() || s.hovered() {
        *help = Some(tip);
    }
    ui.end_row();
}

/// "label | checkbox" grid row for a `BoolParam`.
pub(crate) fn toggle_row(ui: &mut egui::Ui, label: &str, param: &BoolParam, setter: &ParamSetter, tip: &'static str, help: &mut Option<&'static str>) {
    let l = ui.label(label);
    let mut value = param.value();
    let text = if value { "On" } else { "Off" };
    let c = ui.checkbox(&mut value, text);
    if c.changed() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, value);
        setter.end_set_parameter(param);
    }
    if l.hovered() || c.hovered() {
        *help = Some(tip);
    }
    ui.end_row();
}

/// Drop-down for any `EnumParam`, outside a grid.
pub(crate) fn enum_combo<E: Enum + PartialEq + 'static>(ui: &mut egui::Ui, id_salt: impl std::hash::Hash, param: &EnumParam<E>, setter: &ParamSetter, width: f32) -> egui::Response {
    let current_index = param.value().to_index();
    let names = E::variants();
    egui::ComboBox::from_id_salt(id_salt)
        .selected_text(names[current_index])
        .width(width)
        .show_ui(ui, |ui| {
            for (idx, name) in names.iter().enumerate() {
                if ui.selectable_label(current_index == idx, *name).clicked() {
                    setter.begin_set_parameter(param);
                    setter.set_parameter(param, E::from_index(idx));
                    setter.end_set_parameter(param);
                }
            }
        })
        .response
}

/// "label | drop-down" grid row for an `EnumParam`.
pub(crate) fn enum_row<E: Enum + PartialEq + 'static>(
    ui: &mut egui::Ui,
    label: &str,
    param: &EnumParam<E>,
    setter: &ParamSetter,
    tip: &'static str,
    help: &mut Option<&'static str>,
) {
    let l = ui.label(label);
    let width = ui.spacing().slider_width + ui.spacing().interact_size.x;
    let c = enum_combo(ui, (label, param.name()), param, setter, width);
    if l.hovered() || c.hovered() {
        *help = Some(tip);
    }
    ui.end_row();
}

/// A titled, outlined group of controls.
pub(crate) fn section<R>(ui: &mut egui::Ui, title: &str, scale: f32, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::default()
        .stroke(egui::Stroke::new(1.0_f32, COLOR_EDGE))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::same((6.0 * scale) as i8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new(title).color(COLOR_DIM).small());
            ui.add_space(2.0 * scale);
            add_contents(ui)
        })
        .inner
}

/// Starts the two-column label/control grid every section uses.
pub(crate) fn grid(ui: &mut egui::Ui, id: impl std::hash::Hash, scale: f32, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id).num_columns(2).spacing([8.0 * scale, 5.0 * scale]).show(ui, add_contents);
}

// ------------------------------------------------------------------------
// Waveforms
// ------------------------------------------------------------------------

/// Min/max peaks of `samples` across `width_px` columns.
pub(crate) fn compute_peaks(samples: &[f32], width_px: usize) -> Vec<(f32, f32)> {
    let width_px = width_px.max(1);
    if samples.is_empty() {
        return Vec::new();
    }
    let per_px = (samples.len() as f32 / width_px as f32).max(1.0);
    (0..width_px)
        .map_while(|px| {
            let start = (px as f32 * per_px) as usize;
            if start >= samples.len() {
                return None;
            }
            let end = (((px + 1) as f32 * per_px).ceil() as usize).clamp(start + 1, samples.len());
            Some(samples[start..end].iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &s| (mn.min(s), mx.max(s))))
        })
        .collect()
}

/// Paints pre-computed peaks as vertical bars into `rect`.
pub(crate) fn paint_peaks(painter: &egui::Painter, rect: egui::Rect, peaks: &[(f32, f32)], gain: f32, color: egui::Color32) {
    let mid_y = rect.center().y;
    let half = rect.height() * 0.5 * 0.9;
    for (px, &(mn, mx)) in peaks.iter().enumerate() {
        let x = rect.left() + px as f32 + 0.5;
        let y_top = mid_y - (mx * gain).clamp(-1.0, 1.0) * half;
        let y_bottom = (mid_y - (mn * gain).clamp(-1.0, 1.0) * half).max(y_top + 1.0);
        painter.line_segment([egui::pos2(x, y_top), egui::pos2(x, y_bottom)], egui::Stroke::new(1.0_f32, color));
    }
}

fn wave_color() -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(COLOR_INK.r(), COLOR_INK.g(), COLOR_INK.b(), 160)
}

/// Peaks of the rendered output, cached by buffer identity and width so a
/// long loop isn't re-scanned every frame.
#[derive(Default)]
pub(crate) struct PeakCache {
    key: (usize, usize),
    peaks: Vec<(f32, f32)>,
}

/// The rendered instrument output (what MIDI notes actually play), with a
/// caption on the left and a "Rendering..." chip while the worker is busy.
pub(crate) fn draw_output_display(
    ui: &mut egui::Ui,
    buffer: &Arc<prism_dsp::render::LoopBufferData>,
    cache: &mut PeakCache,
    busy: bool,
    caption: &str,
    height: f32,
    scale: f32,
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, COLOR_SURFACE_DEEP);
    painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);

    let width_px = rect.width().max(1.0) as usize;
    let key = (Arc::as_ptr(buffer) as usize, width_px);
    if cache.key != key {
        // Mid (L+R)/2 so a wide stereo render still reads as one shape.
        let mid: Vec<f32> = match (buffer.channels.first(), buffer.channels.get(1)) {
            (Some(l), Some(r)) => l.iter().zip(r).map(|(a, b)| 0.5 * (a + b)).collect(),
            (Some(l), None) => l.clone(),
            _ => Vec::new(),
        };
        let peaks = compute_peaks(&mid, width_px);
        let max = peaks.iter().fold(0.0f32, |m, &(a, b)| m.max(a.abs()).max(b.abs()));
        // Normalised for display: the shape matters here, not the level.
        let norm = if max > 1e-6 { 1.0 / max } else { 1.0 };
        cache.peaks = peaks.into_iter().map(|(a, b)| (a * norm, b * norm)).collect();
        cache.key = key;
    }
    let inner = rect.shrink2(egui::vec2(0.0, 14.0 * scale));
    paint_peaks(painter, inner, &cache.peaks, 1.0, wave_color());
    if cache.peaks.is_empty() {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "(silent)", egui::FontId::proportional(12.0 * scale), COLOR_DIM);
    }

    let font = egui::FontId::proportional(10.0 * scale);
    painter.text(rect.left_top() + egui::vec2(6.0, 3.0) * scale, egui::Align2::LEFT_TOP, caption, font.clone(), COLOR_DIM);
    if busy {
        let galley = painter.layout_no_wrap("Rendering...".to_string(), font, COLOR_INK);
        let chip = egui::Rect::from_min_size(
            egui::pos2(rect.right() - galley.size().x - 14.0 * scale, rect.top() + 3.0 * scale),
            galley.size() + egui::vec2(8.0, 2.0) * scale,
        );
        painter.rect_filled(chip, 3.0, COLOR_ACCENT);
        painter.galley(chip.min + egui::vec2(4.0, 1.0) * scale, galley, COLOR_INK);
    }
}

/// Sample A's waveform with everything outside the trim range shaded.
/// Returns `true` when the "[ Load Sample ]" placeholder is clicked.
pub(crate) fn draw_trim_waveform(
    ui: &mut egui::Ui,
    source: &Arc<ArcSwap<Vec<Vec<f32>>>>,
    sample_rate: f32,
    trim_start_sec: f32,
    trim_end_sec: f32,
    has_loaded_sample: bool,
    scale: f32,
) -> bool {
    let size = egui::vec2(ui.available_width(), GRAPH_HEIGHT * scale);
    if !has_loaded_sample {
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
        let painter = ui.painter();
        painter.rect_filled(rect, 2.0, if response.hovered() { COLOR_EDGE } else { COLOR_SURFACE_DEEP });
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "[ Load Sample ]",
            egui::FontId::proportional(14.0 * scale),
            if response.hovered() { COLOR_ACCENT } else { COLOR_DIM },
        );
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        return response.clicked();
    }
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, COLOR_SURFACE_DEEP);
    let guard = source.load();
    let samples = guard.first().map(Vec::as_slice).unwrap_or(&[]);
    paint_peaks(painter, rect, &compute_peaks(samples, rect.width() as usize), 1.0, wave_color());
    if !samples.is_empty() && sample_rate > 0.0 {
        let total = samples.len() as f32 / sample_rate;
        let x_for = |sec: f32| rect.left() + (sec / total).clamp(0.0, 1.0) * rect.width();
        let start_x = x_for(trim_start_sec);
        let end_x = if trim_end_sec > trim_start_sec { x_for(trim_end_sec) } else { rect.right() };
        let shade = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 170);
        painter.rect_filled(egui::Rect::from_min_max(rect.min, egui::pos2(start_x, rect.bottom())), 0.0, shade);
        painter.rect_filled(egui::Rect::from_min_max(egui::pos2(end_x, rect.top()), rect.max), 0.0, shade);
        for x in [start_x, end_x] {
            painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], egui::Stroke::new(2.0_f32, COLOR_ACCENT));
        }
    }
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);
    false
}

/// A three-octave keyboard strip with the chord's tones (relative to a C
/// root) highlighted - a picture of what one played note turns into.
pub(crate) fn draw_chord_keys(ui: &mut egui::Ui, intervals: &[i32], scale: f32) {
    const WHITE: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
    let lowest = intervals.iter().copied().min().unwrap_or(0).min(0);
    let base = 12 * lowest.div_euclid(12);
    let octaves = ((intervals.iter().copied().max().unwrap_or(0) - base) / 12 + 1).clamp(2, 4);
    let white_count = 7 * octaves;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 46.0 * scale), egui::Sense::hover());
    let painter = ui.painter();
    let key_w = rect.width() / white_count as f32;
    let on = |semi: i32| intervals.contains(&semi);
    for o in 0..octaves {
        for (i, &w) in WHITE.iter().enumerate() {
            let semi = base + 12 * o + w;
            let x = rect.left() + (o * 7 + i as i32) as f32 * key_w;
            let r = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(key_w, rect.height()));
            painter.rect_filled(r.shrink(0.5), 1.0, if on(semi) { COLOR_ACCENT } else { COLOR_PANEL });
            painter.rect_stroke(r.shrink(0.5), 1.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);
            if semi == 0 {
                painter.text(r.center_bottom() - egui::vec2(0.0, 3.0 * scale), egui::Align2::CENTER_BOTTOM, "root", egui::FontId::proportional(8.0 * scale), COLOR_DIM);
            }
        }
        for (i, &w) in WHITE.iter().enumerate() {
            if w == 4 || w == 11 {
                continue;
            }
            let semi = base + 12 * o + w + 1;
            let x = rect.left() + (o * 7 + i as i32 + 1) as f32 * key_w - key_w * 0.3;
            let r = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(key_w * 0.6, rect.height() * 0.6));
            painter.rect_filled(r, 1.0, if on(semi) { COLOR_ACCENT } else { COLOR_INK });
        }
    }
}

/// The Output Volume control: one horizontal bar that is both the meter
/// (the fill is the current output peak, on the same dB scale as the
/// control) and the slider (the dark marker is the volume setting - click
/// or drag anywhere to move it, double-click to reset to 0 dB). The fill
/// turns red while `clipping` (the level going into the safety limiter
/// went over 0 dBFS recently - the limiter was squashing it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_output_meter(
    ui: &mut egui::Ui,
    volume: &FloatParam,
    setter: &ParamSetter,
    peak_linear: f32,
    clipping: bool,
    min_db: f32,
    max_db: f32,
    width: f32,
    scale: f32,
) -> egui::Response {
    let height = 16.0 * scale;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
    let painter = ui.painter();
    let x_for_db = |db: f32| rect.left() + ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0) * rect.width();

    painter.rect_filled(rect, 2.0, COLOR_SURFACE_DEEP);
    let peak_db = if peak_linear > 1e-6 { 20.0 * peak_linear.log10() } else { min_db };
    if peak_db > min_db {
        let fill = egui::Rect::from_min_max(rect.min, egui::pos2(x_for_db(peak_db), rect.bottom()));
        painter.rect_filled(fill, 2.0, if clipping { COLOR_ERROR } else { COLOR_ACCENT });
    }
    // 0 dBFS tick.
    let zero_x = x_for_db(0.0);
    painter.line_segment([egui::pos2(zero_x, rect.top()), egui::pos2(zero_x, rect.bottom())], egui::Stroke::new(1.0_f32, COLOR_EDGE_HOVER));
    // Volume setting marker.
    let marker_x = x_for_db(volume.unmodulated_plain_value());
    let marker = egui::Rect::from_center_size(egui::pos2(marker_x, rect.center().y), egui::vec2(4.0 * scale, rect.height() + 4.0 * scale));
    painter.rect_filled(marker, 1.0, COLOR_INK);
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, if clipping { COLOR_ERROR } else { COLOR_EDGE }), egui::StrokeKind::Inside);

    if response.double_clicked() {
        setter.begin_set_parameter(volume);
        setter.set_parameter(volume, 0.0);
        setter.end_set_parameter(volume);
    } else {
        if response.drag_started() || response.clicked() {
            setter.begin_set_parameter(volume);
        }
        if response.dragged() || response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                setter.set_parameter_normalized(volume, ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0));
            }
        }
        if response.drag_stopped() || response.clicked() {
            setter.end_set_parameter(volume);
        }
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    response.on_hover_text("Output Volume - drag to set, double-click for 0 dB. The bar shows the output level; red means the signal went over 0 dB and the safety limiter is squashing it (audible distortion) - turn the volume down.")
}

// ------------------------------------------------------------------------
// FM Synth source
// ------------------------------------------------------------------------

/// One routing-matrix cell: a small box whose fill shows the amount
/// (0-100 %). Drag up/down to set, double-click for 0. `feedback` marks the
/// diagonal (an operator modulating itself).
pub(crate) fn matrix_cell(ui: &mut egui::Ui, param: &FloatParam, setter: &ParamSetter, feedback: bool, scale: f32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(38.0, 22.0) * scale, egui::Sense::click_and_drag());
    let painter = ui.painter();
    let value = param.unmodulated_normalized_value();
    painter.rect_filled(rect, 2.0, if feedback { COLOR_PANEL } else { COLOR_SURFACE_DEEP });
    if value > 0.0 {
        let top = rect.bottom() - rect.height() * value;
        painter.rect_filled(egui::Rect::from_min_max(egui::pos2(rect.left(), top), rect.max), 2.0, COLOR_ACCENT);
    }
    let stroke = if response.hovered() || response.dragged() { COLOR_INK } else { COLOR_EDGE };
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0_f32, stroke), egui::StrokeKind::Inside);
    let text = if value > 0.0 { format!("{:.0}", param.unmodulated_plain_value()) } else if feedback { "fb".into() } else { "·".into() };
    painter.text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(10.0 * scale), if value > 0.0 { COLOR_INK } else { COLOR_DIM });

    if response.double_clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, 0.0);
        setter.end_set_parameter(param);
    } else {
        if response.drag_started() {
            setter.begin_set_parameter(param);
        }
        if response.dragged() {
            let delta = -response.drag_delta().y / (120.0 * scale);
            setter.set_parameter_normalized(param, (param.unmodulated_normalized_value() + delta).clamp(0.0, 1.0));
        }
        if response.drag_stopped() {
            setter.end_set_parameter(param);
        }
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    response
}

fn arrow_head(painter: &egui::Painter, tip: egui::Pos2, dir: egui::Vec2, size: f32, color: egui::Color32) {
    let d = dir.normalized();
    let n = egui::vec2(-d.y, d.x);
    painter.add(egui::Shape::convex_polygon(vec![tip, tip - d * size + n * size * 0.55, tip - d * size - n * size * 0.55], color, egui::Stroke::NONE));
}

/// A live picture of the FM Synth routing: four operator nodes in a row
/// (filled = heard), an arc for every modulation (thicker = stronger;
/// higher-numbered sources arc over the top, lower-numbered underneath), a
/// loop above an operator with feedback, and drops into the OUT bar for
/// every operator with output.
pub(crate) fn draw_fm_routing(ui: &mut egui::Ui, fm: &prism_dsp::fm_synth::FmSynthParams, scale: f32) {
    use prism_dsp::fm_synth::OPS;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 120.0 * scale), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, COLOR_SURFACE_DEEP);
    let r = 11.0 * scale;
    let node_y = rect.top() + rect.height() * 0.42;
    let bar_y = rect.bottom() - 12.0 * scale;
    let node = |i: usize| egui::pos2(rect.left() + rect.width() * (i as f32 * 2.0 + 1.0) / (OPS as f32 * 2.0), node_y);
    let width = |pct: f32| (pct / 100.0) * 3.0 * scale + 0.75;

    // OUT bar and drops.
    painter.line_segment([egui::pos2(rect.left() + 8.0 * scale, bar_y), egui::pos2(rect.right() - 8.0 * scale, bar_y)], egui::Stroke::new(2.0_f32, COLOR_EDGE_HOVER));
    painter.text(egui::pos2(rect.right() - 8.0 * scale, bar_y - 3.0 * scale), egui::Align2::RIGHT_BOTTOM, "OUT", egui::FontId::proportional(8.5 * scale), COLOR_DIM);
    for (i, op) in fm.ops.iter().enumerate() {
        if op.output_pct > 0.0 {
            let p = node(i);
            painter.line_segment([egui::pos2(p.x, p.y + r), egui::pos2(p.x, bar_y)], egui::Stroke::new(width(op.output_pct), COLOR_INK));
        }
    }
    // Modulation arcs.
    for (dst, op) in fm.ops.iter().enumerate() {
        for (src, &amt) in op.mod_from.iter().enumerate() {
            if amt <= 0.0 {
                continue;
            }
            let stroke = egui::Stroke::new(width(amt), COLOR_ACCENT);
            if src == dst {
                let c = node(dst) - egui::vec2(0.0, r * 1.7);
                painter.circle_stroke(c, r * 0.7, stroke);
                continue;
            }
            let (a, b) = (node(src), node(dst));
            let up = if src > dst { -1.0 } else { 1.0 };
            let span = (b.x - a.x).abs();
            let lift = up * (r * 1.2 + span * 0.18);
            let start = a + egui::vec2(0.0, up * r);
            let end = b + egui::vec2(0.0, up * r);
            let ctrl = egui::pos2((a.x + b.x) * 0.5, a.y + lift * 1.6);
            painter.add(egui::epaint::QuadraticBezierShape::from_points_stroke([start, ctrl, end], false, egui::Color32::TRANSPARENT, stroke));
            arrow_head(painter, end, end - ctrl, 6.0 * scale, COLOR_ACCENT);
        }
    }
    // Nodes on top.
    for (i, op) in fm.ops.iter().enumerate() {
        let p = node(i);
        let used = op.output_pct > 0.0 || fm.ops.iter().enumerate().any(|(d, o)| d != i && o.mod_from[i] > 0.0);
        let fill = if op.output_pct > 0.0 { COLOR_ACCENT } else { COLOR_PANEL };
        painter.circle(p, r, fill, egui::Stroke::new(1.0_f32, if used { COLOR_INK } else { COLOR_EDGE }));
        painter.text(p, egui::Align2::CENTER_CENTER, (i + 1).to_string(), egui::FontId::proportional(11.0 * scale), if used { COLOR_INK } else { COLOR_DIM });
    }
}

/// A few cycles of a waveform (normalised samples).
pub(crate) fn draw_cycle_preview(ui: &mut egui::Ui, samples: &[f32], scale: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 52.0 * scale), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, COLOR_SURFACE_DEEP);
    painter.line_segment([egui::pos2(rect.left(), rect.center().y), egui::pos2(rect.right(), rect.center().y)], egui::Stroke::new(1.0_f32, COLOR_EDGE));
    if samples.len() > 1 {
        let half = rect.height() * 0.42;
        let points: Vec<egui::Pos2> = (0..rect.width().max(2.0) as usize)
            .map(|px| {
                let i = (px as f32 / rect.width() * (samples.len() - 1) as f32) as usize;
                egui::pos2(rect.left() + px as f32, rect.center().y - samples[i].clamp(-1.0, 1.0) * half)
            })
            .collect();
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5_f32, COLOR_INK)));
    } else {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "(silent - give an operator some Output)", egui::FontId::proportional(10.0 * scale), COLOR_DIM);
    }
    painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0_f32, COLOR_EDGE), egui::StrokeKind::Inside);
}
