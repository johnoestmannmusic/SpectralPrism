use crate::analysis::analyze_frame;
use crate::fft::{FreezeFft, HOP_SIZE};
use crate::phase_advance::compute_advance;
use crate::window::sine_window;

pub struct FrozenSpectrum {
    pub mag: Vec<f32>,
    pub phase0: Vec<f32>,
    pub advance: Vec<f32>,
}

/// Analyzes the seed frame at `freeze_point_pct` (0-100) of `signal`'s
/// usable length and derives a frozen magnitude + per-bin phase-advance
/// description, matching src/0006/index.html renderFreeze (lines 3378-3394):
/// magnitude is frozen outright (the seed frame's `mag`), phase keeps
/// advancing per-bin at the measured true rate between the seed frame and
/// the next hop's frame.
///
/// `gain_pct` (0-100, clamped) is the per-sample Volume control - a pure
/// attenuation applied directly to the frozen magnitude here rather than to
/// `signal` itself. This is deliberately equivalent to scaling the raw
/// audio (frequency-domain magnitude is exactly linear in input amplitude),
/// but far cheaper (`NUM_BINS` multiplies instead of the whole signal), and
/// it's the one place shared by every downstream consumer of a
/// `FrozenSpectrum` - `render::render_frozen_loop`'s single-source path and
/// `fusion::render_pre_resynth_fusion`'s two-source combine path both get
/// correct, consistent gain handling for free, including through formant
/// shift (cepstral envelope/fine-structure split, both scale linearly with
/// gain) and Fusion combines that multiply magnitudes together (Convolve),
/// where scaling here rather than after the fact makes each source's own
/// Volume contribute its own share of the product, not just an overall
/// output trim.
pub fn analyze_freeze_point(signal: &[f32], freeze_point_pct: f32, gain_pct: f32, fft: &FreezeFft) -> FrozenSpectrum {
    let n = fft.size();
    let window = sine_window(n);
    let len = signal.len();
    let max_pos = len.saturating_sub(n);
    let pct = freeze_point_pct.clamp(0.0, 100.0) / 100.0;
    let pos0 = ((pct * max_pos as f32).round() as usize).min(max_pos);
    let pos1 = (pos0 + HOP_SIZE).min(max_pos);

    let f0 = analyze_frame(signal, pos0, fft, &window);
    let f1 = analyze_frame(signal, pos1, fft, &window);

    let advance = compute_advance(&f0.phase, &f1.phase, HOP_SIZE, n);

    // Clamped to [0, 1] (never a boost) - the Volume control's own
    // `FloatParam` range already enforces 0-100%, this is just defense in
    // depth against an out-of-range value reaching here some other way.
    let gain = (gain_pct / 100.0).clamp(0.0, 1.0);
    let mag = if gain == 1.0 { f0.mag } else { f0.mag.into_iter().map(|m| m * gain).collect() };

    FrozenSpectrum {
        mag,
        phase0: f0.phase,
        advance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32;
                0.5 * (t * 0.02).sin() + 0.3 * (t * 0.007).sin() + 0.05 * (t * 1.3).sin()
            })
            .collect()
    }

    #[test]
    fn magnitude_frozen_across_frames() {
        // The FrozenSpectrum's `mag` IS the magnitude every resynthesized
        // frame will reuse forever - this test locks down that it is
        // derived from exactly the seed frame (pos0), not some blend with
        // pos1 or a running average.
        let fft = FreezeFft::new();
        let signal = make_test_signal(fft.size() * 8);

        let frozen = analyze_freeze_point(&signal, 25.0, 100.0, &fft);
        let window = sine_window(fft.size());
        let expected_f0 = analyze_frame(&signal, {
            let max_pos = signal.len() - fft.size();
            ((0.25 * max_pos as f32).round() as usize).min(max_pos)
        }, &fft, &window);

        for (a, b) in frozen.mag.iter().zip(expected_f0.mag.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn freeze_point_clamped_to_valid_range() {
        let fft = FreezeFft::new();
        let signal = make_test_signal(fft.size() * 4);

        // Should not panic and should behave like 0/100 respectively.
        let low = analyze_freeze_point(&signal, -50.0, 100.0, &fft);
        let clamped_low = analyze_freeze_point(&signal, 0.0, 100.0, &fft);
        for (a, b) in low.mag.iter().zip(clamped_low.mag.iter()) {
            assert!((a - b).abs() < 1e-4);
        }

        let high = analyze_freeze_point(&signal, 500.0, 100.0, &fft);
        let clamped_high = analyze_freeze_point(&signal, 100.0, 100.0, &fft);
        for (a, b) in high.mag.iter().zip(clamped_high.mag.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn gain_pct_scales_magnitude_linearly_and_never_boosts() {
        let fft = FreezeFft::new();
        let signal = make_test_signal(fft.size() * 4);

        let full = analyze_freeze_point(&signal, 30.0, 100.0, &fft);
        let half = analyze_freeze_point(&signal, 30.0, 50.0, &fft);
        let silent = analyze_freeze_point(&signal, 30.0, 0.0, &fft);
        let over = analyze_freeze_point(&signal, 30.0, 150.0, &fft);

        for ((f, h), s) in full.mag.iter().zip(half.mag.iter()).zip(silent.mag.iter()) {
            assert!((h - f * 0.5).abs() < 1e-4, "50% gain should exactly halve magnitude");
            assert!(*s < 1e-6, "0% gain should produce ~silent magnitude, got {}", s);
        }
        for (f, o) in full.mag.iter().zip(over.mag.iter()) {
            assert_eq!(*f, *o, "gain_pct above 100 must clamp to 100 (never boost)");
        }
    }
}
