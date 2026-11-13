//! Look and feel: palette, the bundled Medodica font, theme and GUI scaling.

use nih_plug_egui::egui;

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
pub(crate) const BASE_EDITOR_WIDTH: u32 = 1040;
pub(crate) const BASE_EDITOR_HEIGHT: u32 = 660;
/// The editor opens at this multiple of the base size by default (matching
/// `apply_gui_scale`'s scale factor, since the two are computed from the
/// same base) - requested directly ("too small to read" at 1x, then a
/// further +25% on top of the first bump). Still freely resizable larger
/// (or back down to 1x) afterward via the corner.
pub(crate) const DEFAULT_SCALE: f32 = 1.7;

/// Shared by `draw_freeze_point_waveform` and `draw_adsr_graph` so the two
/// side-by-side graph boxes always line up at the same height, regardless
/// of scale - requested directly after the ADSR graph's own (taller) 90.0
/// default made the two columns visibly mismatched. Trimmed from 70.0 as
/// part of reclaiming vertical space across the editor - both graphs are
/// informational, not primary controls, so a bit shorter still reads fine.
pub(crate) const GRAPH_HEIGHT: f32 = 55.0;

/// Fixed height (1x-scale points, multiplied by `scale` like everything
/// else) for the Spectral Fusion info box, so its position - right above
/// the Load Sample buttons - and the rest of the layout around it stay put
/// as the selected mode's description changes length. A `ScrollArea` inside
/// it is still the hard backstop against any description ever needing more
/// room than this. 60% of the original 56.0 - every description is capped
/// at two sentences, so the original height was more headroom than any of
/// them actually need.
pub(crate) const INFO_BOX_HEIGHT: f32 = 33.6;

/// Blank space around the outside of the whole editor's content, in 1x-scale
/// points (multiplied by `scale` like everything else) - requested directly
/// in Reaper, where content was otherwise touching the window edges.
/// `16.0 * DEFAULT_SCALE` (1.875) is exactly 30px, matching what was asked
/// for at the default scale.
pub(crate) const EDITOR_MARGIN: f32 = 16.0;

/// Light palette lifted from `src/0006/index.html`'s light-mode `:root`
/// overrides (`--bg`/`--panel`/`--edge`/`--edge-hover`/`--ink`/`--dim`/
/// `--surface-deep`) - this plugin's own visual reference, see
/// `FREEZE-PLAN-011`/`FREEZE-PLAN-013`. `COLOR_BG`/`COLOR_PANEL` are pure
/// white rather than 0006's own slightly-off-white `--bg` (`#fafafa`), per
/// explicit request. The accent is `COLOR_ACCENT`, not 0006's light-mode
/// `--amber` - see that constant's own doc comment.
pub(crate) const COLOR_BG: egui::Color32 = egui::Color32::from_rgb(0xff, 0xff, 0xff);
pub(crate) const COLOR_PANEL: egui::Color32 = egui::Color32::from_rgb(0xff, 0xff, 0xff);
pub(crate) const COLOR_EDGE: egui::Color32 = egui::Color32::from_rgb(0xdc, 0xdd, 0xe0);
pub(crate) const COLOR_EDGE_HOVER: egui::Color32 = egui::Color32::from_rgb(0xb3, 0xb8, 0xbd);
pub(crate) const COLOR_INK: egui::Color32 = egui::Color32::from_rgb(0x2e, 0x34, 0x40);
pub(crate) const COLOR_DIM: egui::Color32 = egui::Color32::from_rgb(0x5e, 0x64, 0x70);
/// Not 0006's own light-theme accent (`#3f9e0e`) - the specific green the
/// user asked for, which 0006 actually uses elsewhere as a hardcoded
/// "on/active" highlight (`.spectral-btn.fusion-on`, the mode-switch toggle).
pub(crate) const COLOR_ACCENT: egui::Color32 = egui::Color32::from_rgb(0x6c, 0xd7, 0x3c);
pub(crate) const COLOR_SURFACE_DEEP: egui::Color32 = egui::Color32::from_rgb(0xee, 0xf0, 0xe9);
pub(crate) const COLOR_ERROR: egui::Color32 = egui::Color32::from_rgb(0xc0, 0x36, 0x2f);

/// Bundled locally (copied from `src/0006/ASSETS/`) rather than referenced
/// across projects by relative path, so this plugin doesn't depend on the
/// rest of the repo layout to build correctly.
pub(crate) const MEDODICA_FONT_BYTES: &[u8] = include_bytes!("../assets/MedodicaRegular.otf");
pub(crate) const MEDODICA_FONT_NAME: &str = "Medodica";

/// Installs the bundled Medodica font (see `MEDODICA_FONT_BYTES`) as the
/// first choice for both the proportional and monospace font families -
/// 0006 uses it as its one font for everything (`--mono: 'Medodica', ...
/// monospace`), so every `egui::TextStyle` (which all default to
/// `FontFamily::Proportional` except `Monospace` itself) should pick it up
/// too. Called once, from the editor's `build` callback (not per-frame -
/// fonts don't need to be reloaded every pass).
pub(crate) fn install_medodica_font(ctx: &egui::Context) {
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
pub(crate) fn apply_theme(ctx: &egui::Context) {
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
        v.selection.stroke = egui::Stroke::new(1.0_f32, COLOR_INK);

        v.widgets.noninteractive.bg_fill = COLOR_PANEL;
        v.widgets.noninteractive.weak_bg_fill = COLOR_PANEL;
        v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, COLOR_EDGE);
        v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32, COLOR_INK);

        v.widgets.inactive.bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.inactive.weak_bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0_f32, COLOR_EDGE);
        v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0_f32, COLOR_INK);

        v.widgets.hovered.bg_fill = COLOR_EDGE;
        v.widgets.hovered.weak_bg_fill = COLOR_EDGE;
        v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, COLOR_EDGE_HOVER);
        v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0_f32, COLOR_INK);

        v.widgets.active.bg_fill = COLOR_ACCENT;
        v.widgets.active.weak_bg_fill = COLOR_ACCENT;
        v.widgets.active.bg_stroke = egui::Stroke::new(1.0_f32, COLOR_ACCENT);
        // Dark ink on the light-green accent fill reads better than white -
        // the accent is bright enough that white text washes out on it.
        v.widgets.active.fg_stroke = egui::Stroke::new(1.0_f32, COLOR_INK);

        v.widgets.open.bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.open.weak_bg_fill = COLOR_SURFACE_DEEP;
        v.widgets.open.bg_stroke = egui::Stroke::new(1.0_f32, COLOR_ACCENT);
        v.widgets.open.fg_stroke = egui::Stroke::new(1.0_f32, COLOR_INK);
    });
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
pub(crate) fn apply_gui_scale(ctx: &egui::Context, scale: f32) {
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
