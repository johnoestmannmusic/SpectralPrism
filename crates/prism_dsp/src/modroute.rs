//! Modulation routes for the Spectral render: a small, fixed set of
//! LFO / ramp / random routes, each driving one `modulate::MOD_*` target over
//! the length of the rendered loop.
//!
//! Ports `sampleSpectralModulation` / `modShapeValue` / `SPECTRAL_PARAMS`
//! from SpectralPrism Tracker's `src/core/spectral.ts` 1:1, so a patch
//! modulates the same way in the plugin as in the tracker. The output is the
//! flat `MOD_TARGET_COUNT * num_points` control-point array plus a bitmask of
//! targets that actually vary - exactly what
//! `modulate::render_fused_modulated_loop` and
//! `modulate::render_fused_loop_length_modulated_loop` consume.

use crate::modulate::MOD_TARGET_COUNT;
use crate::render::{MAX_LOOP_SECONDS, MIN_LOOP_SECONDS};

/// Number of route slots the plugin exposes.
pub const MOD_SLOTS: usize = 4;

/// Control points supplied per target (the minimum; long loops get more).
pub const MIN_CONTROL_POINTS: usize = 128;
/// Upper bound on control points per target, however long the loop.
pub const MAX_CONTROL_POINTS: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModShape {
    Lfo,
    Ramp,
    Random,
}

/// One modulation route. `target == None` (or `depth == 0`) is an empty slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModRoute {
    /// A `modulate::MOD_*` index, or `None` for an unused slot.
    pub target: Option<usize>,
    pub shape: ModShape,
    /// Modulation amount, in the target's own units (semitones / % / s).
    pub depth: f32,
    /// Cycles per second (LFO shape).
    pub rate_hz: f32,
    /// LFO phase offset (0..1); also seeds the Random shape.
    pub phase: f32,
    /// true = sweep +/-depth; false = sweep 0..depth.
    pub bipolar: bool,
}

impl Default for ModRoute {
    fn default() -> Self {
        Self { target: None, shape: ModShape::Lfo, depth: 0.0, rate_hz: 0.25, phase: 0.0, bipolar: true }
    }
}

impl ModRoute {
    pub fn is_active(&self) -> bool {
        self.target.is_some_and(|t| t < MOD_TARGET_COUNT) && self.depth != 0.0
    }
}

/// Display/clamp metadata for one modulation target, in `MOD_*` order.
#[derive(Clone, Copy, Debug)]
pub struct ModTargetMeta {
    pub label: &'static str,
    pub unit: &'static str,
    pub min: f32,
    pub max: f32,
}

/// Mirrors the tracker's `SPECTRAL_PARAMS` (index == position).
pub const MOD_TARGETS: [ModTargetMeta; MOD_TARGET_COUNT] = [
    ModTargetMeta { label: "Freeze Point A", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Freeze Point B", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Tune A", unit: "st", min: -24.0, max: 24.0 },
    ModTargetMeta { label: "Tune B", unit: "st", min: -24.0, max: 24.0 },
    ModTargetMeta { label: "Volume A", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Volume B", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Formant A", unit: "st", min: -12.0, max: 12.0 },
    ModTargetMeta { label: "Formant B", unit: "st", min: -12.0, max: 12.0 },
    ModTargetMeta { label: "Stereo Width", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Mix Amount", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Cross-Synth Amount", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Convolve Amount", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Ring Mod Amount", unit: "%", min: 0.0, max: 100.0 },
    ModTargetMeta { label: "Loop Length", unit: "s", min: MIN_LOOP_SECONDS, max: MAX_LOOP_SECONDS },
];

/// Control points per target for a loop of `duration_seconds`: 128 up to
/// 8 s, then 16 per second so a fast LFO on a long loop still resolves.
pub fn mod_control_points(duration_seconds: f32) -> usize {
    ((duration_seconds as f64 * 16.0).ceil() as usize).clamp(MIN_CONTROL_POINTS, MAX_CONTROL_POINTS)
}

/// Same hash as the tracker's `pseudoRandom` (f64 so the values match).
fn pseudo_random(n: f64) -> f64 {
    let x = (n * 12.9898 + 78.233).sin() * 43758.5453;
    (x - x.floor()) * 2.0 - 1.0
}

/// Raw shape value in -1..1 at control point `i` of `num_points`.
pub fn mod_shape_value(shape: ModShape, rate_hz: f32, phase: f32, i: usize, num_points: usize, duration_seconds: f32) -> f32 {
    let pos = i as f64 / num_points.max(1) as f64;
    match shape {
        ModShape::Lfo => {
            let t = pos * duration_seconds as f64;
            (std::f64::consts::TAU * (rate_hz as f64 * t + phase as f64)).sin() as f32
        }
        ModShape::Ramp => (pos * 2.0 - 1.0) as f32,
        ModShape::Random => pseudo_random(phase as f64 * 1000.0 + i as f64) as f32,
    }
}

/// Sampled modulation tracks, ready for the `modulate` renderers.
#[derive(Clone, Debug, PartialEq)]
pub struct ModTracks {
    /// `MOD_TARGET_COUNT * num_points` absolute values in `MOD_*` order.
    pub points: Vec<f32>,
    /// Bit `i` set = target `i` varies.
    pub mask: u32,
    pub num_points: usize,
}

/// Samples every route into absolute per-control-point values around
/// `base` (the unmodulated value of each target, in `MOD_*` order), clamped
/// to each target's range. Targets with no active route are filled with
/// their base value and left out of the mask.
pub fn sample_mod_tracks(base: &[f32; MOD_TARGET_COUNT], routes: &[ModRoute], loop_seconds: f32) -> ModTracks {
    let duration = loop_seconds.max(0.001);
    let num_points = mod_control_points(duration);
    let mut points = vec![0.0f32; MOD_TARGET_COUNT * num_points];
    let mut mask = 0u32;

    for (target, meta) in MOD_TARGETS.iter().enumerate() {
        let row = &mut points[target * num_points..(target + 1) * num_points];
        let active: Vec<&ModRoute> = routes.iter().filter(|r| r.target == Some(target) && r.depth != 0.0).collect();
        if active.is_empty() {
            row.fill(base[target]);
            continue;
        }
        for (i, slot) in row.iter_mut().enumerate() {
            let mut value = base[target];
            for route in &active {
                let raw = mod_shape_value(route.shape, route.rate_hz, route.phase, i, num_points, duration);
                let shaped = if route.bipolar { raw } else { (raw + 1.0) / 2.0 };
                value += route.depth * shaped;
            }
            *slot = value.clamp(meta.min, meta.max);
        }
        mask |= 1 << target;
    }

    ModTracks { points, mask, num_points }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulate::{MOD_LOOP_LENGTH, MOD_TUNE_A};

    fn base() -> [f32; MOD_TARGET_COUNT] {
        let mut b = [0.0; MOD_TARGET_COUNT];
        b[MOD_LOOP_LENGTH] = 4.0;
        b
    }

    #[test]
    fn no_routes_means_flat_tracks_and_empty_mask() {
        let tracks = sample_mod_tracks(&base(), &[ModRoute::default(); MOD_SLOTS], 4.0);
        assert_eq!(tracks.mask, 0);
        assert_eq!(tracks.num_points, MIN_CONTROL_POINTS);
        let ll = &tracks.points[MOD_LOOP_LENGTH * tracks.num_points..(MOD_LOOP_LENGTH + 1) * tracks.num_points];
        assert!(ll.iter().all(|&v| v == 4.0));
    }

    #[test]
    fn zero_depth_route_is_ignored() {
        let route = ModRoute { target: Some(MOD_TUNE_A), depth: 0.0, ..Default::default() };
        assert_eq!(sample_mod_tracks(&base(), &[route], 4.0).mask, 0);
    }

    #[test]
    fn lfo_route_sets_mask_bit_and_clamps_to_target_range() {
        let route = ModRoute { target: Some(MOD_TUNE_A), depth: 100.0, rate_hz: 1.0, ..Default::default() };
        let tracks = sample_mod_tracks(&base(), &[route], 4.0);
        assert_eq!(tracks.mask, 1 << MOD_TUNE_A);
        let row = &tracks.points[MOD_TUNE_A * tracks.num_points..(MOD_TUNE_A + 1) * tracks.num_points];
        assert!(row.iter().all(|&v| (-24.0..=24.0).contains(&v)));
        assert!(row.iter().any(|&v| v == 24.0) && row.iter().any(|&v| v == -24.0));
    }

    #[test]
    fn unipolar_ramp_sweeps_zero_to_depth() {
        let route = ModRoute { target: Some(MOD_TUNE_A), shape: ModShape::Ramp, depth: 12.0, bipolar: false, ..Default::default() };
        let tracks = sample_mod_tracks(&base(), &[route], 4.0);
        let row = &tracks.points[MOD_TUNE_A * tracks.num_points..(MOD_TUNE_A + 1) * tracks.num_points];
        assert_eq!(row[0], 0.0);
        assert!(row[row.len() - 1] > 11.5 && row[row.len() - 1] <= 12.0);
    }

    #[test]
    fn long_loops_get_more_control_points() {
        assert_eq!(mod_control_points(4.0), 128);
        assert_eq!(mod_control_points(20.0), 320);
        assert_eq!(mod_control_points(1000.0), MAX_CONTROL_POINTS);
    }

    #[test]
    fn random_shape_matches_tracker_hash() {
        // pseudoRandom(0) in the tracker: fract(sin(78.233) * 43758.5453) * 2 - 1
        let expected = {
            let x = (78.233f64).sin() * 43758.5453;
            ((x - x.floor()) * 2.0 - 1.0) as f32
        };
        assert_eq!(mod_shape_value(ModShape::Random, 0.0, 0.0, 0, 128, 4.0), expected);
    }
}
