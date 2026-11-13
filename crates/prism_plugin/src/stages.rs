//! The signal-chain GUI: a strip of stage cards in processing order, and
//! one panel per stage.
//!
//! ```text
//! SOURCE ─▶ FM ─▶ SPECTRAL ─▶ PERCUSSION ─▶ TEXTURE ║ VOICE ─▶ CHORD ─▶ CHORUS
//! └──────────── render-time (baked into the loop) ─┘ └──── real-time (per note) ┘
//! ```
//!
//! Each card shows whether its stage is on (a clickable LED) and a one-line
//! summary; clicking a card opens its panel. Controls that don't apply to
//! the current settings are hidden (Source B outside fusion modes, Op2
//! controls for the Direct algorithm, ...), and a switched-off stage's panel
//! is dimmed but stays editable so it can be set up before switching it on.

use crate::params::*;
use crate::theme::*;
use crate::widgets::*;
use arc_swap::ArcSwap;
use nih_plug::prelude::*;
use nih_plug_egui::egui;
use prism_dsp::chord::{chord_intervals, ChordSettings};
use prism_dsp::percussion::{hat_preset, kick_preset, metal_preset, snare_preset, PercussionParams as DspPerc};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum Stage {
    Source,
    Fm,
    #[default]
    Spectral,
    Percussion,
    Texture,
    Voice,
    Chord,
    Chorus,
}

impl Stage {
    pub(crate) const ALL: [Stage; 8] = [Stage::Source, Stage::Fm, Stage::Spectral, Stage::Percussion, Stage::Texture, Stage::Voice, Stage::Chord, Stage::Chorus];

    pub(crate) fn title(self) -> &'static str {
        match self {
            Stage::Source => "SOURCE",
            Stage::Fm => "FM",
            Stage::Spectral => "SPECTRAL",
            Stage::Percussion => "PERCUSSION",
            Stage::Texture => "TEXTURE",
            Stage::Voice => "VOICE",
            Stage::Chord => "CHORD",
            Stage::Chorus => "CHORUS",
        }
    }

    /// Shown in the info line when no control is hovered.
    pub(crate) fn description(self) -> &'static str {
        match self {
            Stage::Source => "Source: the sample(s) everything starts from. With every render stage off, the plugin is a plain sampler - trim and playback mode apply here.",
            Stage::Fm => "FM: the sample phase-modulates a sine carrier at the played pitch before anything else, turning any sound into a metallic, bell-like or growling tone. Alone it transforms a hit; with Spectral on it feeds the freeze.",
            Stage::Spectral => "Spectral: freezes one instant of the sample into an endless, evolving loop - optionally fused with Sample B and swept by modulation routes. The heart of SpectralPrism.",
            Stage::Percussion => "Percussion: re-synthesises whatever the earlier stages produced as a short drum hit - modal body, noise, transient and a pitch sweep. Pick a preset as a starting point.",
            Stage::Texture => "Texture (MicroTextures): granular post-processing - scatters short grains of the result in time, pitch and space, with retrigger stutters, a formant filter bank and lo-fi crush.",
            Stage::Voice => "Voice: how each note plays the rendered loop - amp envelope, tuning, vibrato, polyphony and a resonant filter with its own envelope. Changes here are instant (no re-render).",
            Stage::Chord => "Chord: every note you play becomes a whole chord - choose a shape, voice it with inversions and octaves, and strum or detune it.",
            Stage::Chorus => "Chorus: a stereo ensemble effect on the whole instrument - slowly modulated delays thicken and widen the sound.",
        }
    }

    /// The stage's on/off param, if it has one.
    pub(crate) fn enable_param(self, p: &PrismPluginParams) -> Option<&BoolParam> {
        match self {
            Stage::Fm => Some(&p.fm.enabled),
            Stage::Spectral => Some(&p.spectral.enabled),
            Stage::Percussion => Some(&p.percussion.enabled),
            Stage::Texture => Some(&p.micro.enabled),
            Stage::Chord => Some(&p.chord.enabled),
            Stage::Chorus => Some(&p.chorus.enabled),
            Stage::Source | Stage::Voice => None,
        }
    }

    pub(crate) fn is_on(self, p: &PrismPluginParams) -> bool {
        self.enable_param(p).is_none_or(BoolParam::value)
    }
}

fn variant_name<E: Enum>(e: E) -> &'static str {
    E::variants()[e.to_index()]
}

/// One-line card summary.
pub(crate) fn summary(stage: Stage, p: &PrismPluginParams, sample_a: Option<&str>) -> String {
    if !stage.is_on(p) {
        return "off".to_string();
    }
    match stage {
        Stage::Source => {
            let name = sample_a.unwrap_or("placeholder tone");
            let renders = p.spectral.enabled.value() || p.fm.enabled.value() || p.percussion.enabled.value() || p.micro.enabled.value();
            if renders {
                name.to_string()
            } else {
                format!("{name} · sampler")
            }
        }
        Stage::Fm => match p.fm.algorithm.value() {
            FmAlgorithmParam::Direct => "direct".into(),
            FmAlgorithmParam::Stack => "stack".into(),
            FmAlgorithmParam::Pair => "pair".into(),
        },
        Stage::Spectral => {
            let mut s = variant_name(p.fusion_mode.value()).to_string();
            if s == "Off" {
                s = "Freeze".into();
            }
            if p.spectral.freeze_style.value() == FreezeStyleParam::Shimmer {
                s.push_str(" · shimmer");
            }
            let mods = p.mod_slots.iter().filter(|m| m.target.value() != ModTargetParam::None && m.depth.value() != 0.0).count();
            if mods > 0 {
                s.push_str(&format!(" · {mods} mod"));
            }
            s
        }
        Stage::Percussion => format!("hit {:.2} s", p.percussion.length.value()),
        Stage::Texture => format!("grains {:.0} ms", p.micro.grain.value() * 1000.0),
        Stage::Voice => {
            let poly = if p.voice.polyphonic.value() { "poly" } else { "mono" };
            match p.filter.mode.value() {
                FilterModeParam::Off => poly.to_string(),
                FilterModeParam::Lowpass => format!("{poly} · LP"),
                FilterModeParam::Highpass => format!("{poly} · HP"),
                FilterModeParam::Bandpass => format!("{poly} · BP"),
            }
        }
        Stage::Chord => variant_name(p.chord.shape.value()).to_lowercase(),
        Stage::Chorus => format!("mix {:.0}%", p.chorus.mix.value()),
    }
}

/// The chain strip. Returns the newly selected stage, if a card was clicked.
pub(crate) fn draw_chain_strip(
    ui: &mut egui::Ui,
    p: &PrismPluginParams,
    setter: &ParamSetter,
    selected: Stage,
    sample_a: Option<&str>,
    scale: f32,
    help: &mut Option<&'static str>,
) -> Option<Stage> {
    let mut clicked = None;
    let gap = 14.0 * scale;
    let divider = 22.0 * scale;
    let card_h = 44.0 * scale;
    let n = Stage::ALL.len() as f32;
    let total_w = ui.available_width();
    let card_w = ((total_w - gap * (n - 2.0) - divider) / n).max(40.0);
    let (strip, _) = ui.allocate_exact_size(egui::vec2(total_w, card_h + 12.0 * scale), egui::Sense::hover());
    let painter = ui.painter().clone();
    let mut x = strip.left();
    let mut render_span = (strip.left(), strip.left());
    for (i, stage) in Stage::ALL.iter().copied().enumerate() {
        let rect = egui::Rect::from_min_size(egui::pos2(x, strip.top()), egui::vec2(card_w, card_h));
        let on = stage.is_on(p);
        let is_sel = stage == selected;
        let card = ui.interact(rect, ui.id().with(("chain_card", i)), egui::Sense::click());
        let fill = if is_sel { COLOR_SURFACE_DEEP } else if card.hovered() { COLOR_PANEL } else { COLOR_BG };
        painter.rect_filled(rect, 4.0, fill);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(if is_sel { 2.0_f32 } else { 1.0_f32 }, if is_sel { COLOR_ACCENT } else { COLOR_EDGE }),
            egui::StrokeKind::Inside,
        );
        let ink = if on { COLOR_INK } else { COLOR_DIM };
        painter.text(rect.left_top() + egui::vec2(8.0, 6.0) * scale, egui::Align2::LEFT_TOP, stage.title(), egui::FontId::proportional(11.0 * scale), ink);
        let text = summary(stage, p, sample_a);
        let galley = painter.layout(text, egui::FontId::proportional(9.0 * scale), COLOR_DIM, rect.width() - 12.0 * scale);
        painter.galley(rect.left_bottom() + egui::vec2(8.0 * scale, -6.0 * scale - galley.size().y.min(11.0 * scale)), galley, COLOR_DIM);

        // Power LED: a click toggles the stage without changing the selection.
        let led_c = rect.right_top() + egui::vec2(-10.0, 10.0) * scale;
        let led_r = 4.5 * scale;
        if let Some(param) = stage.enable_param(p) {
            let led_rect = egui::Rect::from_center_size(led_c, egui::vec2(led_r * 4.0, led_r * 4.0));
            let led = ui.interact(led_rect, ui.id().with(("chain_led", i)), egui::Sense::click());
            painter.circle(led_c, led_r, if on { COLOR_ACCENT } else { COLOR_SURFACE_DEEP }, egui::Stroke::new(1.0_f32, if led.hovered() { COLOR_INK } else { COLOR_EDGE_HOVER }));
            if led.clicked() {
                setter.begin_set_parameter(param);
                setter.set_parameter(param, !on);
                setter.end_set_parameter(param);
            }
            if led.hovered() {
                *help = Some("Switch this stage on or off.");
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
        }
        if card.clicked() {
            clicked = Some(stage);
        }
        if card.hovered() && help.is_none() {
            *help = Some(stage.description());
        }

        if stage == Stage::Texture {
            render_span.1 = rect.right();
            // Double bar: render-time stages | real-time stages.
            let mx = rect.right() + divider * 0.5;
            for dx in [-2.0, 2.0] {
                painter.line_segment([egui::pos2(mx + dx * scale, rect.top()), egui::pos2(mx + dx * scale, rect.bottom())], egui::Stroke::new(1.0_f32, COLOR_EDGE_HOVER));
            }
            x = rect.right() + divider;
        } else {
            if i + 1 < Stage::ALL.len() {
                // Arrow to the next stage.
                let y = rect.center().y;
                let (a, b) = (rect.right() + 3.0 * scale, rect.right() + gap - 3.0 * scale);
                painter.line_segment([egui::pos2(a, y), egui::pos2(b, y)], egui::Stroke::new(1.0_f32, COLOR_EDGE_HOVER));
                painter.add(egui::Shape::convex_polygon(
                    vec![egui::pos2(b, y), egui::pos2(b - 4.0 * scale, y - 3.0 * scale), egui::pos2(b - 4.0 * scale, y + 3.0 * scale)],
                    COLOR_EDGE_HOVER,
                    egui::Stroke::NONE,
                ));
            }
            x = rect.right() + gap;
        }
    }
    let label_y = strip.top() + card_h + 2.0 * scale;
    let small = egui::FontId::proportional(8.5 * scale);
    painter.text(egui::pos2((render_span.0 + render_span.1) * 0.5, label_y), egui::Align2::CENTER_TOP, "rendered into the loop  ·  a moment to update", small.clone(), COLOR_DIM);
    painter.text(egui::pos2((render_span.1 + divider + strip.right()) * 0.5, label_y), egui::Align2::CENTER_TOP, "per note  ·  instant", small, COLOR_DIM);
    clicked
}

/// Everything a stage panel needs.
pub(crate) struct PanelCtx<'a> {
    pub(crate) params: &'a PrismPluginParams,
    pub(crate) setter: &'a ParamSetter<'a>,
    pub(crate) scale: f32,
    pub(crate) source: &'a Arc<ArcSwap<Vec<Vec<f32>>>>,
    pub(crate) source_b: &'a Arc<ArcSwap<Vec<Vec<f32>>>>,
    pub(crate) sample_rate: f32,
    pub(crate) name_a: Option<String>,
    pub(crate) name_b: Option<String>,
}

/// What a panel asks the editor to do (file dialogs live in the editor).
#[derive(Default)]
pub(crate) struct PanelActions {
    pub(crate) load_a: bool,
    pub(crate) load_b: bool,
}

/// Draws the selected stage's panel.
pub(crate) fn draw_stage_panel(ui: &mut egui::Ui, stage: Stage, cx: &PanelCtx, help: &mut Option<&'static str>, chord_text: &mut String) -> PanelActions {
    let mut actions = PanelActions::default();
    let p = cx.params;
    if let Some(param) = stage.enable_param(p) {
        if !param.value() {
            ui.horizontal(|ui| {
                ui.colored_label(COLOR_DIM, format!("{} is off - its settings below are kept but not heard.", stage.title()));
                if ui.button("Turn on").clicked() {
                    cx.setter.begin_set_parameter(param);
                    cx.setter.set_parameter(param, true);
                    cx.setter.end_set_parameter(param);
                }
            });
            ui.add_space(4.0 * cx.scale);
            ui.multiply_opacity(0.55);
        }
    }
    match stage {
        Stage::Source => source_panel(ui, cx, help, &mut actions),
        Stage::Fm => fm_panel(ui, cx, help),
        Stage::Spectral => spectral_panel(ui, cx, help, &mut actions),
        Stage::Percussion => percussion_panel(ui, cx, help),
        Stage::Texture => texture_panel(ui, cx, help),
        Stage::Voice => voice_panel(ui, cx, help),
        Stage::Chord => chord_panel(ui, cx, help, chord_text),
        Stage::Chorus => chorus_panel(ui, cx, help),
    }
    actions
}

fn source_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>, actions: &mut PanelActions) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let renders = p.spectral.enabled.value() || p.fm.enabled.value() || p.percussion.enabled.value() || p.micro.enabled.value();
    ui.columns(3, |cols| {
        section(&mut cols[0], "SAMPLE A", sc, |ui| {
            if draw_trim_waveform(ui, cx.source, cx.sample_rate, p.source.trim_start.value(), p.source.trim_end.value(), cx.name_a.is_some(), sc) {
                actions.load_a = true;
            }
            ui.horizontal(|ui| {
                if ui.button("Load Sample A...").clicked() {
                    actions.load_a = true;
                }
                ui.colored_label(COLOR_DIM, cx.name_a.as_deref().unwrap_or("built-in placeholder tone"));
            });
        });
        section(&mut cols[1], "PLAYBACK", sc, |ui| {
            grid(ui, "src_grid", sc, |ui| {
                enum_row(ui, "Playback", &p.source.playback, s, "Auto plays hits (Percussion, FM alone, plain samples) once and loops frozen pads. Or force Loop, Ping-pong or One-shot.", help);
                row(ui, "Trim Start", &p.source.trim_start, s, "Where plain-sampler playback starts in the sample.", help);
                row(ui, "Trim End", &p.source.trim_end, s, "Where plain-sampler playback stops ('end' = the end of the sample).", help);
            });
            ui.add_space(4.0 * sc);
            let note = if renders {
                "Trim applies to the plain sampler. With a render stage on, the stages read the whole sample (Spectral picks its moment with Freeze Point)."
            } else {
                "Every render stage is off: notes play the trimmed sample directly, pitched by key."
            };
            ui.add(egui::Label::new(egui::RichText::new(note).color(COLOR_DIM).small()).wrap());
        });
        section(&mut cols[2], "SAMPLE B (for Spectral Fusion)", sc, |ui| {
            if draw_trim_waveform(ui, cx.source_b, cx.sample_rate, 0.0, 0.0, cx.name_b.is_some(), sc) {
                actions.load_b = true;
            }
            ui.horizontal(|ui| {
                if ui.button("Load Sample B...").clicked() {
                    actions.load_b = true;
                }
                ui.colored_label(COLOR_DIM, cx.name_b.as_deref().unwrap_or("none loaded"));
            });
        });
    });
}

fn fm_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let fm = &p.fm;
    let direct = fm.algorithm.value() == FmAlgorithmParam::Direct;
    ui.columns(3, |cols| {
        section(&mut cols[0], "ROUTING", sc, |ui| {
            grid(ui, "fm_routing", sc, |ui| {
                enum_row(ui, "Algorithm", &fm.algorithm, s, "Direct: the sample modulates the carrier. Stack: the sample modulates a second operator that modulates the carrier. Pair: both modulate the carrier side by side.", help);
                row(ui, "Slice Start", &fm.slice_start, s, "Where in Sample A the FM transform starts reading.", help);
                row(ui, "Slice Length", &fm.slice_length, s, "How much of Sample A to transform ('to end' = the rest of it).", help);
            });
        });
        section(&mut cols[1], "OPERATORS", sc, |ui| {
            grid(ui, "fm_ops", sc, |ui| {
                row(ui, "Carrier Ratio", &fm.carrier_ratio, s, "Carrier pitch as a multiple of the played note.", help);
                let label = if fm.algorithm.value() == FmAlgorithmParam::Stack { "Sample > Op2" } else { "Sample Index" };
                row(ui, label, &fm.index, s, "How hard the sample modulates: low = subtle colour, high = dense, noisy sidebands.", help);
                if !direct {
                    row(ui, "Op2 Ratio", &fm.op2_ratio, s, "Second operator's pitch as a multiple of the played note.", help);
                    row(ui, "Op2 Index", &fm.op2_index, s, "How hard the second operator modulates the carrier.", help);
                    row(ui, "Op2 Feedback", &fm.op2_feedback, s, "The second operator modulating itself - adds grit and noise.", help);
                }
            });
        });
        section(&mut cols[2], "MOTION & LO-FI", sc, |ui| {
            grid(ui, "fm_motion", sc, |ui| {
                row(ui, "Follow Envelope", &fm.follow, s, "0% = a steady carrier; 100% = the carrier's level follows the sample's own loudness.", help);
                row(ui, "Index Sway Rate", &fm.sway_rate, s, "A slow LFO on the modulation index, for movement ('off' = none).", help);
                row(ui, "Index Sway Depth", &fm.sway_depth, s, "How far the sway LFO moves the index.", help);
                row(ui, "Stereo Detune", &fm.detune, s, "Detunes the left and right carriers apart for width.", help);
                row(ui, "Mix (dry/FM)", &fm.mix, s, "Blend of the untouched sample and the FM result.", help);
                row(ui, "Sample Hold", &fm.hold, s, "Sample-and-hold length: holds each value for N samples, for aliased crunch.", help);
                row(ui, "Quantise Levels", &fm.levels, s, "Rounds the output to this many levels, for bit-crushed grit.", help);
            });
        });
    });
}

fn spectral_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>, actions: &mut PanelActions) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let fusion = p.fusion_mode.value();
    let needs_b = fusion.needs_sample_b();
    let m = |t: ModTargetParam| is_modulated(p, t);
    ui.columns(3, |cols| {
        section(&mut cols[0], "SAMPLE A", sc, |ui| {
            let clicked = ui
                .horizontal(|ui| {
                    draw_vertical_volume_slider(ui, &p.sample_a_volume, s, GRAPH_HEIGHT * sc, sc);
                    draw_freeze_point_waveform(ui, cx.source, &p.freeze_point, p.sample_a_volume.value(), s, cx.name_a.is_some(), sc)
                })
                .inner;
            actions.load_a |= clicked;
            grid(ui, "spec_a", sc, |ui| {
                row_mod(ui, "Freeze Point", &p.freeze_point, s, m(ModTargetParam::FreezePointA), "The instant of Sample A that gets frozen - drag on the waveform too.", help);
                row_mod(ui, "Tune", &p.sample_a_tune, s, m(ModTargetParam::TuneA), "Retunes Sample A before freezing, e.g. to line it up with Sample B.", help);
                row_mod(ui, "Formant Shift", &p.formant_shift, s, m(ModTargetParam::FormantA), "Moves the spectral envelope (vowel/body character) without changing pitch.", help);
            });
        });
        section(&mut cols[1], "FUSION", sc, |ui| {
            grid(ui, "spec_fusion", sc, |ui| {
                enum_row(ui, "Mode", &p.fusion_mode, s, fusion.info_text(), help);
            });
            if needs_b {
                ui.add_space(2.0 * sc);
                let clicked = ui
                    .horizontal(|ui| {
                        draw_vertical_volume_slider(ui, &p.sample_b_volume, s, GRAPH_HEIGHT * sc, sc);
                        draw_freeze_point_waveform(ui, cx.source_b, &p.freeze_point_b, p.sample_b_volume.value(), s, cx.name_b.is_some(), sc)
                    })
                    .inner;
                actions.load_b |= clicked;
                grid(ui, "spec_b", sc, |ui| {
                    row_mod(ui, "Freeze Point B", &p.freeze_point_b, s, m(ModTargetParam::FreezePointB), "The instant of Sample B that gets frozen.", help);
                    row_mod(ui, "Tune B", &p.sample_b_tune, s, m(ModTargetParam::TuneB), "Retunes Sample B before freezing.", help);
                    row_mod(ui, "Formant B", &p.formant_shift_b, s, m(ModTargetParam::FormantB), "Moves Sample B's spectral envelope.", help);
                    match fusion {
                        FusionMode::Mix => row_mod(ui, "Mix (A/B)", &p.fusion_mix_amount, s, m(ModTargetParam::MixAmount), "0% = only A, 100% = only B.", help),
                        FusionMode::CrossSynth => row_mod(ui, "Amount", &p.fusion_cross_synth_amount, s, m(ModTargetParam::CrossSynthAmount), "How much of B's spectral shape is imposed on A.", help),
                        FusionMode::Convolve => row_mod(ui, "Amount", &p.fusion_convolve_amount, s, m(ModTargetParam::ConvolveAmount), "Dry A vs the full A×B convolution.", help),
                        FusionMode::RingModulate => row_mod(ui, "Amount", &p.fusion_ring_mod_amount, s, m(ModTargetParam::RingModAmount), "Dry A vs the full ring modulation.", help),
                        _ => {}
                    }
                });
                if cx.name_b.is_none() {
                    ui.colored_label(COLOR_ERROR, "This mode needs Sample B - click the box above to load one.");
                }
            } else {
                ui.add(egui::Label::new(egui::RichText::new(fusion.info_text()).color(COLOR_DIM).small()).wrap());
            }
        });
        section(&mut cols[2], "FREEZE", sc, |ui| {
            grid(ui, "spec_freeze", sc, |ui| {
                row_mod(ui, "Loop Length", &p.loop_length_seconds, s, m(ModTargetParam::LoopLength), "Length of the rendered loop - also the period every modulation route sweeps over.", help);
                row_mod(ui, "Stereo Width", &p.stereo_width, s, m(ModTargetParam::StereoWidth), "Spreads the frozen spectrum across left and right.", help);
                enum_row(ui, "Smear (FFT)", &p.spectral.fft_size, s, "Analysis size: Tight keeps detail and transients, Smear blurs them into a smoother wash.", help);
                enum_row(ui, "Freeze Style", &p.spectral.freeze_style, s, "Phase-lock: a stable, exact freeze. Shimmer: an averaged spectrum with random phases - airy and glittering.", help);
                if p.spectral.freeze_style.value() == FreezeStyleParam::Shimmer {
                    row(ui, "Shimmer Span", &p.spectral.shimmer_span, s, "How much of the sample around the freeze point is averaged (100% = the whole sample).", help);
                    row(ui, "Shimmer Low Cut", &p.spectral.shimmer_low_cut, s, "Removes rumble below this frequency from the shimmer.", help);
                    row(ui, "Shimmer Seed", &p.spectral.shimmer_seed, s, "A different random-phase pattern for each seed.", help);
                }
            });
        });
    });
    ui.add_space(6.0 * sc);
    mod_matrix(ui, cx, help);
}

fn mod_matrix(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    section(ui, "MODULATION - routes sweep a parameter over each loop (depth is in the target's units: st / % / s)", sc, |ui| {
        egui::Grid::new("mod_matrix").num_columns(7).spacing([10.0 * sc, 4.0 * sc]).show(ui, |ui| {
            for h in ["", "Target", "Shape", "Depth", "Rate", "Phase", "Bipolar"] {
                ui.colored_label(COLOR_DIM, egui::RichText::new(h).small());
            }
            ui.end_row();
            for (i, slot) in p.mod_slots.iter().enumerate() {
                ui.label(format!("{}", i + 1));
                let used = slot.target.value() != ModTargetParam::None;
                let t = enum_combo(ui, ("mod_target", i), &slot.target, s, 150.0 * sc);
                if t.hovered() {
                    *help = Some("The parameter this route moves. Freeze Point / Tune targets re-analyse the sample as they move; Loop Length tiles loops of different lengths.");
                }
                ui.add_enabled_ui(used, |ui| {
                    let r = enum_combo(ui, ("mod_shape", i), &slot.shape, s, 80.0 * sc);
                    if r.hovered() {
                        *help = Some("LFO: a sine at Rate. Ramp: one sweep per loop. Random: a new value at every control point (Phase seeds it).");
                    }
                });
                for (param, tip) in [
                    (&slot.depth, "How far the route moves its target, in the target's own units."),
                    (&slot.rate, "LFO cycles per second (snapped so the loop still joins cleanly)."),
                    (&slot.phase, "Where in its cycle the LFO starts; also the Random shape's seed."),
                ] {
                    ui.add_enabled_ui(used, |ui| {
                        if ui.add(nih_plug_egui::widgets::ParamSlider::for_param(param, s).with_width(110.0 * sc)).hovered() {
                            *help = Some(tip);
                        }
                    });
                }
                ui.add_enabled_ui(used, |ui| {
                    let mut b = slot.bipolar.value();
                    let r = ui.checkbox(&mut b, "");
                    if r.changed() {
                        s.begin_set_parameter(&slot.bipolar);
                        s.set_parameter(&slot.bipolar, b);
                        s.end_set_parameter(&slot.bipolar);
                    }
                    if r.hovered() {
                        *help = Some("On: sweeps +/- depth around the set value. Off: sweeps from the set value up to + depth.");
                    }
                });
                ui.end_row();
            }
        });
    });
}

fn apply_percussion_preset(cx: &PanelCtx, preset: DspPerc) {
    let (pc, s) = (&cx.params.percussion, cx.setter);
    let set = |param: &FloatParam, v: f32| {
        s.begin_set_parameter(param);
        s.set_parameter(param, v);
        s.end_set_parameter(param);
    };
    set(&pc.noise_amount, preset.noise_amount_pct);
    set(&pc.noise_decay, preset.noise_decay_seconds);
    set(&pc.transient_amount, preset.transient_amount_pct);
    set(&pc.transient_decay, preset.transient_decay_seconds);
    set(&pc.transient_freq, preset.transient_frequency_hz);
    set(&pc.pitch_start, preset.pitch_start_semitones);
    set(&pc.pitch_end, preset.pitch_end_semitones);
    set(&pc.pitch_decay, preset.pitch_decay_seconds);
    set(&pc.amp_decay, preset.amp_decay_seconds);
    set(&pc.body_amount, preset.body_amount_pct);
    set(&pc.partial_decay, preset.partial_decay_seconds);
    set(&pc.digital, preset.digital_amount_pct);
    set(&pc.drive, preset.drive_amount_pct);
    set(&pc.compress, preset.compress_amount_pct);
    set(&pc.width, preset.stereo_width_pct);
    set(&pc.length, preset.length_seconds);
    s.begin_set_parameter(&pc.partials);
    s.set_parameter(&pc.partials, preset.partial_count as i32);
    s.end_set_parameter(&pc.partials);
    let color = match preset.noise_color {
        prism_dsp::percussion::NoiseColor::White => NoiseColorParam::White,
        prism_dsp::percussion::NoiseColor::Pink => NoiseColorParam::Pink,
        prism_dsp::percussion::NoiseColor::BandLimited => NoiseColorParam::BandLimited,
    };
    s.begin_set_parameter(&pc.noise_color);
    s.set_parameter(&pc.noise_color, color);
    s.end_set_parameter(&pc.noise_color);
    if !pc.enabled.value() {
        s.begin_set_parameter(&pc.enabled);
        s.set_parameter(&pc.enabled, true);
        s.end_set_parameter(&pc.enabled);
    }
}

fn percussion_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let pc = &p.percussion;
    ui.horizontal(|ui| {
        ui.colored_label(COLOR_DIM, "Start from:");
        for (name, preset) in [("Kick", kick_preset()), ("Snare", snare_preset()), ("Metal", metal_preset()), ("Hat", hat_preset())] {
            let r = ui.button(name);
            if r.clicked() {
                apply_percussion_preset(cx, preset);
            }
            if r.hovered() {
                *help = Some("Loads this drum's settings into every Percussion control (and switches the stage on).");
            }
        }
    });
    ui.add_space(4.0 * sc);
    ui.columns(3, |cols| {
        section(&mut cols[0], "NOISE & TRANSIENT", sc, |ui| {
            grid(ui, "perc_noise", sc, |ui| {
                row(ui, "Noise Amount", &pc.noise_amount, s, "Level of the noise layer (snares, hats).", help);
                enum_row(ui, "Noise Colour", &pc.noise_color, s, "White: bright hiss. Pink: darker. Band-limited: a focused high band for hats and metal.", help);
                row(ui, "Noise Decay", &pc.noise_decay, s, "How long the noise rings.", help);
                row(ui, "Transient Amount", &pc.transient_amount, s, "Level of the click at the start of the hit.", help);
                row(ui, "Transient Decay", &pc.transient_decay, s, "Length of the click.", help);
                row(ui, "Transient Freq", &pc.transient_freq, s, "Pitch of the click - higher is snappier.", help);
            });
        });
        section(&mut cols[1], "PITCH & AMP", sc, |ui| {
            grid(ui, "perc_pitch", sc, |ui| {
                row(ui, "Pitch Start", &pc.pitch_start, s, "Pitch offset at the start of the sweep (a kick starts high).", help);
                row(ui, "Pitch End", &pc.pitch_end, s, "Pitch offset the sweep settles at.", help);
                row(ui, "Pitch Decay", &pc.pitch_decay, s, "How fast the pitch sweeps from start to end.", help);
                row(ui, "Amp Decay", &pc.amp_decay, s, "How long the whole hit takes to fade.", help);
                row(ui, "Hit Length", &pc.length, s, "Length of the rendered hit.", help);
            });
        });
        section(&mut cols[2], "BODY & CHARACTER", sc, |ui| {
            grid(ui, "perc_body", sc, |ui| {
                row(ui, "Body Amount", &pc.body_amount, s, "Level of the tuned body, built from the earlier stages' strongest partials.", help);
                row(ui, "Partials", &pc.partials, s, "How many partials make up the body - more is richer and more metallic.", help);
                row(ui, "Partial Decay", &pc.partial_decay, s, "How long the strongest partial rings (higher ones fade faster).", help);
                row(ui, "Digital", &pc.digital, s, "Ring-modulates the body with an inharmonic ratio, for metallic or digital tones.", help);
                row(ui, "Drive", &pc.drive, s, "Saturation for weight and grit.", help);
                row(ui, "Compression", &pc.compress, s, "Squashes the peak so the body comes forward.", help);
                row(ui, "Hit Width", &pc.width, s, "Stereo width of the hit.", help);
            });
        });
    });
}

fn texture_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let mt = &p.micro;
    ui.columns(3, |cols| {
        section(&mut cols[0], "GRAINS", sc, |ui| {
            grid(ui, "mt_grains", sc, |ui| {
                row(ui, "Grain Size", &mt.grain, s, "Length of each grain.", help);
                row(ui, "Density", &mt.density, s, "Grains per second.", help);
                row(ui, "Jitter", &mt.jitter, s, "Randomises when grains start.", help);
                row(ui, "Reverse", &mt.reverse, s, "Chance of a grain playing backwards.", help);
                row(ui, "Pitch Scatter", &mt.pitch_scatter, s, "Random pitch offset per grain.", help);
                row(ui, "Pan Scatter", &mt.pan_scatter, s, "Random stereo position per grain.", help);
                row(ui, "Volume Variance", &mt.volume_variance, s, "Random level per grain.", help);
            });
        });
        section(&mut cols[1], "MOTION", sc, |ui| {
            grid(ui, "mt_motion", sc, |ui| {
                row(ui, "Density Mod Rate", &mt.density_mod_rate, s, "An LFO speeding up and slowing down the grain rate.", help);
                row(ui, "Density Mod Depth", &mt.density_mod_depth, s, "How far that LFO moves the grain rate.", help);
                row(ui, "Grain Chaos", &mt.chaos, s, "Per-grain digital variation: quantised pitch jumps, hard pans, gain swings, bit reduction.", help);
                row(ui, "Retrigger Rate", &mt.retrigger_rate, s, "Rate of stutter-style retriggers.", help);
                row(ui, "Retrigger Amount", &mt.retrigger_amount, s, "How strongly the retrigger stutter is applied.", help);
            });
        });
        section(&mut cols[2], "FORMANT & LO-FI", sc, |ui| {
            grid(ui, "mt_formant", sc, |ui| {
                row(ui, "Formant Shift", &mt.formant_shift, s, "Moves a vowel-like resonant filter bank up or down.", help);
                row(ui, "Resonance", &mt.formant_resonance, s, "How peaky the formant filters are.", help);
                row(ui, "Formant Mix", &mt.formant_mix, s, "Dry grains vs formant-filtered grains.", help);
                row(ui, "Bit Depth", &mt.bit_depth, s, "Bit-crush (16 = clean).", help);
                row(ui, "Downsample", &mt.downsample, s, "Sample-rate reduction (1x = clean).", help);
            });
        });
    });
}

fn voice_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let (v, f) = (&p.voice, &p.filter);
    ui.columns(3, |cols| {
        section(&mut cols[0], "AMP ENVELOPE", sc, |ui| {
            draw_adsr_graph(ui, "amp", &p.attack, &p.decay, &p.sustain, &p.release, s, sc);
            grid(ui, "voice_amp", sc, |ui| {
                row(ui, "Attack", &p.attack, s, "Fade-in time of each note.", help);
                row(ui, "Decay", &p.decay, s, "Time to fall to the sustain level.", help);
                row(ui, "Sustain", &p.sustain, s, "Level held while the key is down.", help);
                row(ui, "Release", &p.release, s, "Fade-out after the key is released.", help);
                row(ui, "Velocity Sens.", &p.velocity_sensitivity, s, "How much key velocity affects loudness (0% = always full).", help);
            });
        });
        section(&mut cols[1], "PITCH & PLAY", sc, |ui| {
            grid(ui, "voice_pitch", sc, |ui| {
                row(ui, "Transpose", &v.transpose, s, "Shifts every note by semitones.", help);
                row(ui, "Fine Tune", &v.fine, s, "Fine pitch offset in cents.", help);
                row(ui, "Pitch Bend Range", &p.pitch_bend_range_semitones, s, "How far the pitch wheel bends, in semitones.", help);
                row(ui, "Vibrato Speed", &v.vibrato_speed, s, "Vibrato LFO rate.", help);
                row(ui, "Vibrato Depth", &v.vibrato_depth, s, "Vibrato depth in semitones (0 = off).", help);
                toggle_row(ui, "Polyphonic", &v.polyphonic, s, "Off = one note at a time: a new note releases the previous one.", help);
                row(ui, "Voice Cap", &v.voice_cap, s, "Most voices that can sound at once (chord voices count individually).", help);
                toggle_row(ui, "Choke", &v.choke, s, "A new note quickly fades out the previous one (mono) or a retriggered note (poly) instead of letting it ring.", help);
                row(ui, "Pan Center", &p.pan_center_pct, s, "Where notes sit in the stereo field.", help);
                row(ui, "Random Pan Width", &p.pan_width_pct, s, "Each note lands at a random position this far around the centre.", help);
            });
        });
        section(&mut cols[2], "FILTER", sc, |ui| {
            grid(ui, "voice_filter_mode", sc, |ui| {
                enum_row(ui, "Mode", &f.mode, s, "A resonant filter on every note, with its own envelope.", help);
            });
            if f.mode.value() != FilterModeParam::Off {
                draw_adsr_graph(ui, "filter", &f.attack, &f.decay, &f.sustain, &f.release, s, sc);
                grid(ui, "voice_filter", sc, |ui| {
                    row(ui, "Cutoff", &f.cutoff, s, "Filter frequency before the envelope.", help);
                    row(ui, "Resonance", &f.resonance, s, "Emphasis at the cutoff.", help);
                    row(ui, "Env Amount", &f.env_amount, s, "How far the envelope sweeps the cutoff, in octaves (negative closes it).", help);
                    row(ui, "Env Attack", &f.attack, s, "Filter envelope attack.", help);
                    row(ui, "Env Decay", &f.decay, s, "Filter envelope decay.", help);
                    row(ui, "Env Sustain", &f.sustain, s, "Filter envelope sustain level.", help);
                    row(ui, "Env Release", &f.release, s, "Filter envelope release.", help);
                    row(ui, "Key Follow", &f.key_follow, s, "How much the cutoff follows the note's pitch.", help);
                });
            }
        });
    });
}

fn chord_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>, chord_text: &mut String) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let c = &p.chord;
    let custom = c.shape.value() == ChordShapeParam::Custom;
    let settings = current_chord_settings(p, &ChordSettings::default());
    let intervals = chord_intervals(&settings);
    ui.columns(3, |cols| {
        section(&mut cols[0], "SHAPE", sc, |ui| {
            grid(ui, "chord_shape", sc, |ui| {
                enum_row(ui, "Shape", &c.shape, s, "The chord each played note becomes.", help);
                if custom {
                    let l = ui.label("Intervals");
                    let r = ui.add(egui::TextEdit::singleline(chord_text).hint_text("0, 4, 7, 11"));
                    if r.changed() {
                        *c.custom_intervals.lock().unwrap() = chord_text.clone();
                    }
                    if l.hovered() || r.hovered() {
                        *help = Some("Semitones above the played note, comma-separated (e.g. 0, 3, 7, 10).");
                    }
                    ui.end_row();
                }
            });
        });
        section(&mut cols[1], "VOICING", sc, |ui| {
            grid(ui, "chord_voicing", sc, |ui| {
                row(ui, "Inversion", &c.inversion, s, "Moves the lowest tones up (or highest down) an octave.", help);
                row(ui, "Octaves", &c.octaves, s, "Stacks copies of the chord in higher octaves.", help);
                row(ui, "Voices", &c.voices, s, "Most tones one chord can use.", help);
                row(ui, "Detune", &c.detune, s, "Alternately detunes the tones up and down, for a lush ensemble.", help);
                row(ui, "Strum", &c.strum, s, "Delay between successive tones - a guitar-like roll.", help);
                row(ui, "Pan Spread", &c.pan_spread, s, "Spreads the tones across the stereo field, low to high.", help);
            });
        });
        section(&mut cols[2], "PLAYS (relative to the played note)", sc, |ui| {
            draw_chord_keys(ui, &intervals, sc);
            let list: Vec<String> = intervals.iter().map(|i| format!("{i:+}")).collect();
            ui.colored_label(COLOR_DIM, egui::RichText::new(format!("{} tones: {}", intervals.len(), list.join(" "))).small());
        });
    });
}

fn chorus_panel(ui: &mut egui::Ui, cx: &PanelCtx, help: &mut Option<&'static str>) {
    let (p, s, sc) = (cx.params, cx.setter, cx.scale);
    let c = &p.chorus;
    ui.columns(3, |cols| {
        section(&mut cols[0], "MOVEMENT", sc, |ui| {
            grid(ui, "cho_a", sc, |ui| {
                row(ui, "Rate", &c.rate, s, "Speed of the delay modulation.", help);
                row(ui, "Depth", &c.depth, s, "How far the delay time swings.", help);
                row(ui, "Delay", &c.delay, s, "Base delay time.", help);
            });
        });
        section(&mut cols[1], "BLEND", sc, |ui| {
            grid(ui, "cho_b", sc, |ui| {
                row(ui, "Mix", &c.mix, s, "Dry vs chorused signal.", help);
                row(ui, "Spread", &c.spread, s, "Phase offset between left and right - 100% is the widest.", help);
            });
        });
        section(&mut cols[2], "ABOUT", sc, |ui| {
            ui.add(
                egui::Label::new(egui::RichText::new("Runs on the whole instrument after the voices are mixed, so it thickens chords and pads without a re-render.").color(COLOR_DIM).small())
                    .wrap(),
            );
        });
    });
}
