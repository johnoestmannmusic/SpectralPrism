//! Preset (`.spjson`) format, the on-disk preset library, and browser metadata.

use crate::params::PrismPluginParams;
use crate::BUILD_NUMBER;
use nih_plug::prelude::{ParamSetter, Params};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A named snapshot of every DSP-relevant param, plus the sample it was
/// made with (if any) - see `FREEZE-PLAN-010`. Deliberately doesn't include
/// `editor_state` (window geometry has nothing to do with the sound) or any
/// GUI-only state.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Preset {
    pub(crate) freeze_point_pct: f32,
    /// `#[serde(default = "default_volume_pct")]` (100%, unity at the
    /// peak-normalized load level) - a preset saved before Volume existed
    /// must still play at its original level, not silently attenuated.
    #[serde(default = "default_volume_pct")]
    pub(crate) volume_a_pct: f32,
    /// `#[serde(default)]` (0.0, i.e. untuned) - presets saved before Tune
    /// existed must still play at their original pitch.
    #[serde(default)]
    pub(crate) tune_a_semitones: f32,
    pub(crate) formant_shift_semitones: f32,
    pub(crate) stereo_width_pct: f32,
    /// `#[serde(default)]` with a custom default fn (not the bare
    /// `0.0` a plain `#[serde(default)]` would give) - unlike
    /// `pan_center_pct`/`pan_width_pct` where 0.0 is the semantically
    /// correct "off" value, 0.0 seconds is a nonsensical loop length. A
    /// preset saved before this param existed comes back at
    /// `DEFAULT_LOOP_SECONDS` instead, matching what that preset's sound
    /// actually used at the time.
    #[serde(default = "default_loop_length_seconds")]
    pub(crate) loop_length_seconds: f32,
    pub(crate) attack_ms: f32,
    pub(crate) decay_ms: f32,
    pub(crate) sustain_pct: f32,
    pub(crate) release_ms: f32,
    pub(crate) velocity_sensitivity_pct: f32,
    /// `#[serde(default)]` on these three: presets saved before they
    /// existed must still load rather than fail outright - they'll just
    /// come back with pitch bend disabled (range 0) and no panning (center/
    /// width 0), the same as a freshly-created instance would have before
    /// anyone touched these controls.
    #[serde(default)]
    pub(crate) pitch_bend_range_semitones: f32,
    #[serde(default)]
    pub(crate) pan_center_pct: f32,
    #[serde(default)]
    pub(crate) pan_width_pct: f32,
    /// `None` if the preset was saved while still on the built-in
    /// placeholder tone - recalling it then leaves whatever sample is
    /// already loaded untouched rather than resetting to the placeholder.
    pub(crate) sample_path: Option<PathBuf>,
    /// Which `BUILD_NUMBER` (YYYYMMDD) made this preset, so it's possible to
    /// tell which plugin version to blame if it doesn't sound right after
    /// importing into a later one. `#[serde(default)]` (an empty string)
    /// for presets saved before this existed - displayed as "unknown"
    /// rather than a blank/confusing value. Purely informational - never
    /// applied to anything, and never blocks an import.
    #[serde(default)]
    pub(crate) build_number: String,
    /// A stable id string (`FusionMode::preset_id`/`from_preset_id`), not
    /// the enum itself - a future variant reordering can't silently
    /// reinterpret an old preset. `#[serde(default)]` (an empty string,
    /// which `from_preset_id` treats the same as `"off"`) for presets saved
    /// before Spectral Fusion existed.
    #[serde(default)]
    pub(crate) fusion_mode: String,
    /// `#[serde(default = "default_freeze_point_b_pct")]`, matching
    /// `freeze_point_b`'s own `FloatParam` default (50.0) - unlike
    /// `formant_shift_b_semitones` below, 0.0 would put Sample B's Freeze
    /// Point at the very start of the sample, which isn't a neutral/no-op
    /// value the way 0.0 is for a shift or a percentage blend.
    #[serde(default = "default_freeze_point_b_pct")]
    pub(crate) freeze_point_b_pct: f32,
    /// Mirrors `volume_a_pct` above for Sample B.
    #[serde(default = "default_volume_pct")]
    pub(crate) volume_b_pct: f32,
    /// Mirrors `tune_a_semitones` for Sample B.
    #[serde(default)]
    pub(crate) tune_b_semitones: f32,
    #[serde(default)]
    pub(crate) formant_shift_b_semitones: f32,
    /// `#[serde(default = "default_fusion_mix_amount_pct")]`: Mix is a
    /// blend slider, not an intensity dial, so its neutral default is 50%
    /// (matching `fusion_mix_amount`'s own `FloatParam` default), not 0.0.
    #[serde(default = "default_fusion_mix_amount_pct")]
    pub(crate) fusion_mix_amount_pct: f32,
    /// `#[serde(default = "default_fusion_full_amount_pct")]` on these
    /// three: intensity dials on an already-deliberately-chosen effect, so
    /// their neutral default is 100% (full amount), matching each control's
    /// own `FloatParam` default - not 0.0, which would silently mean "no
    /// effect" for a preset that predates these controls but still had a
    /// Fusion mode selected (impossible today, but keeps the invariant that
    /// a preset missing a field behaves like a freshly-created instance).
    #[serde(default = "default_fusion_full_amount_pct")]
    pub(crate) fusion_cross_synth_amount_pct: f32,
    #[serde(default = "default_fusion_full_amount_pct")]
    pub(crate) fusion_convolve_amount_pct: f32,
    #[serde(default = "default_fusion_full_amount_pct")]
    pub(crate) fusion_ring_mod_amount_pct: f32,
    /// Mirrors `sample_path` for Sample B - `None` if no Sample B was ever
    /// loaded when this preset was saved.
    #[serde(default)]
    pub(crate) sample_path_b: Option<PathBuf>,
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
    pub(crate) unload_samples: bool,
    /// Browsable/sortable metadata, entered via the Save dialog - see
    /// `PrismEditorState::show_save_dialog`. `#[serde(default)]` (empty
    /// strings) for presets saved before the Preset Browser existed; the
    /// browser and info panel treat an empty `category`/`sub_category` as
    /// "Uncategorized" rather than a blank tree node. `title` defaults to
    /// the on-disk preset name itself when empty, so older presets still
    /// display something meaningful.
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) author: String,
    #[serde(default)]
    pub(crate) category: String,
    #[serde(default)]
    pub(crate) sub_category: String,
    /// YYYY-MM-DD, set to `today_utc_date()` every time this preset is
    /// saved via the Save dialog - distinct from `build_number` (which
    /// plugin *version* made it) and computed at runtime, not compile time.
    /// `#[serde(default)]` (empty, displayed as "unknown") for presets
    /// saved before this existed.
    #[serde(default)]
    pub(crate) date_last_updated: String,
    /// Every parameter added with the tracker's instrument stages (FM,
    /// Spectral extras, Modulation, Percussion, MicroTextures, Voice,
    /// Filter, Chord, Chorus, Source), keyed by its stable param id with its
    /// plain value (an enum's value is its variant index). A generic map
    /// rather than one field per param so new params never need preset
    /// plumbing. Missing ids - e.g. every one of them in a preset saved
    /// before these stages existed - recall at their defaults, so an old
    /// preset doesn't inherit whatever FM/Chord/... the previous patch had.
    #[serde(default)]
    pub(crate) stage_params: BTreeMap<String, f32>,
    /// Chord mode's Custom intervals text ("0, 4, 7, 11"); `None` = default.
    #[serde(default)]
    pub(crate) chord_custom_intervals: Option<String>,
}

pub(crate) fn default_loop_length_seconds() -> f32 {
    prism_dsp::render::DEFAULT_LOOP_SECONDS
}

pub(crate) fn default_freeze_point_b_pct() -> f32 {
    50.0
}

pub(crate) fn default_volume_pct() -> f32 {
    100.0
}

pub(crate) fn default_fusion_mix_amount_pct() -> f32 {
    50.0
}

pub(crate) fn default_fusion_full_amount_pct() -> f32 {
    100.0
}

impl Preset {
    pub(crate) fn capture(params: &PrismPluginParams) -> Self {
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
            stage_params: capture_stage_params(params),
            chord_custom_intervals: Some(params.chord.custom_intervals.lock().unwrap().clone()),
        }
    }
}

/// Ids of the original params, which keep their own named `Preset` fields
/// (so old preset files stay readable) - everything else goes through
/// `Preset::stage_params`.
pub(crate) const LEGACY_PARAM_IDS: &[&str] = &[
    "freeze_point",
    "sample_a_volume",
    "sample_a_tune",
    "formant_shift",
    "stereo_width",
    "fusion_mode",
    "freeze_point_b",
    "sample_b_volume",
    "sample_b_tune",
    "formant_shift_b",
    "fusion_mix_amount",
    "fusion_cross_synth_amount",
    "fusion_convolve_amount",
    "fusion_ring_mod_amount",
    "loop_length",
    "attack",
    "decay",
    "sustain",
    "release",
    "velocity_sensitivity",
    "pitch_bend_range",
    "pan_center",
    "pan_width",
];

/// Plain values of every non-legacy param, keyed by id.
pub(crate) fn capture_stage_params(params: &PrismPluginParams) -> BTreeMap<String, f32> {
    params
        .param_map()
        .into_iter()
        .filter(|(id, _, _)| !LEGACY_PARAM_IDS.contains(&id.as_str()))
        // SAFETY: the pointers come from `params`, which outlives this call.
        .map(|(id, ptr, _)| (id, unsafe { ptr.unmodulated_plain_value() }))
        .collect()
}

/// Recalls `Preset::stage_params` through the host-visible begin/set/end
/// path (so automation recording sees a proper change); ids missing from
/// the map go back to their defaults.
pub(crate) fn apply_stage_params(preset: &Preset, params: &PrismPluginParams, setter: &ParamSetter) {
    for (id, ptr, _) in params.param_map() {
        if LEGACY_PARAM_IDS.contains(&id.as_str()) {
            continue;
        }
        // SAFETY: the pointers come from `params`, which outlives this call.
        unsafe {
            let normalized = match preset.stage_params.get(&id) {
                Some(&plain) => ptr.preview_normalized(plain),
                None => ptr.default_normalized_value(),
            };
            setter.raw_context.raw_begin_set_parameter(ptr);
            setter.raw_context.raw_set_parameter_normalized(ptr, normalized);
            setter.raw_context.raw_end_set_parameter(ptr);
        }
    }
    *params.chord.custom_intervals.lock().unwrap() = preset.chord_custom_intervals.clone().unwrap_or_else(|| "0, 4, 7".to_string());
}

/// Today's date (UTC) as YYYY-MM-DD, for `Preset::date_last_updated`.
/// Computed from the system clock in plain Rust rather than shelling out to
/// `date` (unlike `build.rs`, which only ever runs once at compile time on
/// a build machine) - this runs at runtime, potentially every time a preset
/// is saved, from inside a plugin hosted in a DAW, where spawning a
/// subprocess on every save is a needless dependency on an external binary
/// being on `PATH` in whatever environment the host provides.
pub(crate) fn today_utc_date() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (year, month, day) = civil_from_days((secs / 86_400) as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Howard Hinnant's `civil_from_days` algorithm (public domain) - converts
/// a day count since the Unix epoch (1970-01-01) into a proleptic
/// Gregorian (year, month, day). The standard small, dependency-free way to
/// do this conversion without a date/time crate; see
/// <https://howardhinnant.github.io/date_algorithms.html>.
pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
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
pub(crate) fn presets_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".config").join("SpectralPrism").join("presets");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// `.spjson` (not plain `.json`) - still JSON content, but a distinct
/// extension so SpectralPrism's own presets can be filtered out from a
/// user's other, unrelated `.json` files (in a file picker, a search, a
/// sync folder, etc).
pub(crate) const PRESET_FILE_EXTENSION: &str = "spjson";

pub(crate) fn preset_file_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.{PRESET_FILE_EXTENSION}"))
}

/// Sorted (so Prev/Next and the dropdown have a stable, predictable order)
/// list of preset names, without the `.spjson` extension, found in `dir`. An
/// unreadable directory (shouldn't happen once `presets_dir()` has
/// succeeded once, but e.g. permissions could change) just yields no
/// presets rather than an error - there's no interactive action to blame it
/// on, unlike an explicit save/load.
pub(crate) fn list_presets(dir: &Path) -> Vec<String> {
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
pub(crate) fn write_preset_file(path: &Path, preset: &Preset) -> Result<(), String> {
    let json = serde_json::to_string_pretty(preset).map_err(|e| format!("couldn't serialize preset: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("couldn't write preset file: {e}"))
}

/// Reads a preset from an arbitrary path - see `write_preset_file`.
pub(crate) fn read_preset_file(path: &Path) -> Result<Preset, String> {
    let json = std::fs::read_to_string(path).map_err(|e| format!("couldn't read preset file: {e}"))?;
    serde_json::from_str(&json).map_err(|e| format!("couldn't parse preset file: {e}"))
}

pub(crate) fn save_preset(dir: &Path, name: &str, preset: &Preset) -> Result<(), String> {
    write_preset_file(&preset_file_path(dir, name), preset)
}

pub(crate) fn load_preset(dir: &Path, name: &str) -> Result<Preset, String> {
    read_preset_file(&preset_file_path(dir, name))
}

pub(crate) fn delete_preset(dir: &Path, name: &str) -> Result<(), String> {
    std::fs::remove_file(preset_file_path(dir, name)).map_err(|e| format!("couldn't delete preset file: {e}"))
}

/// One preset's worth of metadata for the Preset Browser's tree/search/info
/// panel, read once per preset file when the browser opens - see
/// `load_preset_browser_entries`. `name` is the on-disk file stem (what
/// `load_preset`/`save_preset` key on); the rest mirror the matching
/// `Preset` fields.
#[derive(Clone)]
pub(crate) struct PresetBrowserEntry {
    pub(crate) name: String,
    pub(crate) title: String,
    pub(crate) author: String,
    pub(crate) category: String,
    pub(crate) sub_category: String,
}

/// Reads every named on-disk preset's metadata from `dir` - used to
/// populate the Preset Browser's tree when it opens. A preset file that
/// fails to parse (corrupted, or some other unrelated `.spjson` file) is
/// silently skipped rather than blocking the whole browser on one bad file.
pub(crate) fn load_preset_browser_entries(dir: &Path) -> Vec<PresetBrowserEntry> {
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
pub(crate) fn display_category(category: &str) -> &str {
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
pub(crate) fn distinct_preset_values<'a>(entries: &'a [PresetBrowserEntry], field: impl Fn(&'a PresetBrowserEntry) -> &'a str) -> Vec<String> {
    let mut values: Vec<String> = entries.iter().map(field).filter(|s| !s.is_empty()).map(str::to_string).collect();
    values.sort();
    values.dedup();
    values
}

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
    use crate::params::FusionMode;
    use crate::samples::write_loop_buffer_wav;
    use prism_dsp::render::{LoopBufferData, DEFAULT_ROOT_NOTE};

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
            stage_params: BTreeMap::new(),
            chord_custom_intervals: None,
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

        write_loop_buffer_wav(&path, &buffer, true).expect("export should succeed");

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

        write_loop_buffer_wav(&path, &buffer, true).expect("export should succeed");

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
        assert!(write_loop_buffer_wav(&path, &buffer, true).is_err());
    }

    #[test]
    fn capture_stores_every_new_param_and_no_legacy_ones() {
        let params = PrismPluginParams::default();
        let captured = capture_stage_params(&params);
        for id in ["fm_enabled", "spectral_enabled", "perc_noise_amount", "mt_grain", "voice_cap", "flt_cutoff", "chord_shape", "cho_mix", "src_playback", "mod_target_1", "mod_depth_4"] {
            assert!(captured.contains_key(id), "missing {id}: {:?}", captured.keys().collect::<Vec<_>>());
        }
        for id in LEGACY_PARAM_IDS {
            assert!(!captured.contains_key(*id), "legacy id {id} should keep its own field");
        }
        assert_eq!(captured["spectral_enabled"], 1.0, "Spectral defaults on so old sessions keep freezing");
    }

    #[test]
    fn legacy_ids_list_matches_every_param_not_in_a_stage_group() {
        // Every param on the root struct (no group) must be a legacy id, or
        // its value would be saved twice / not at all.
        let params = PrismPluginParams::default();
        for (id, _, group) in params.param_map() {
            if group.is_empty() {
                assert!(LEGACY_PARAM_IDS.contains(&id.as_str()), "{id} has no group and isn't in LEGACY_PARAM_IDS");
            }
        }
    }

    #[test]
    fn stage_params_round_trip_and_old_presets_get_an_empty_map() {
        let mut preset = sample_preset(None);
        preset.stage_params.insert("fm_enabled".into(), 1.0);
        preset.chord_custom_intervals = Some("0, 3, 7".into());
        let restored: Preset = serde_json::from_str(&serde_json::to_string(&preset).unwrap()).unwrap();
        assert_eq!(restored.stage_params.get("fm_enabled"), Some(&1.0));
        assert_eq!(restored.chord_custom_intervals.as_deref(), Some("0, 3, 7"));

        let mut json: serde_json::Value = serde_json::to_value(&preset).unwrap();
        json.as_object_mut().unwrap().remove("stage_params");
        json.as_object_mut().unwrap().remove("chord_custom_intervals");
        let old: Preset = serde_json::from_value(json).unwrap();
        assert!(old.stage_params.is_empty());
        assert!(old.chord_custom_intervals.is_none());
    }
}
