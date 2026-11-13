//! Per-voice resonant filter with its own envelope and key follow.
//!
//! Ports the tracker's layer filter (`src/core/layers.ts`: `filterEnvAt`,
//! `layerFilterBaseHz`, `layerFilterEnvCents`, `bandpassQ`, `layerBiquad`,
//! and `channelFilter.ts`: `cutoffToHz`, `resonanceToQDb`). The coefficient
//! formulas match the Web Audio `BiquadFilterNode` the tracker plays through,
//! so a patch sounds the same in both. Coefficients are recomputed every
//! `CONTROL_FRAMES` samples while the envelope moves.

/// Coefficient update interval in samples (the tracker's `DRIFT_CONTROL_FRAMES`).
pub const CONTROL_FRAMES: usize = 32;
const FILTER_BASE_HZ: f32 = 20.0;
const FILTER_SPAN: f32 = 1000.0;
const Q_MIN_DB: f32 = -3.0103;
const Q_MAX_DB: f32 = 24.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    Off,
    Lowpass,
    Highpass,
    Bandpass,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterSettings {
    pub mode: FilterMode,
    /// Normalised 0..1 (20 Hz..20 kHz, exponential).
    pub cutoff: f32,
    /// Normalised 0..1.
    pub resonance: f32,
    /// Envelope depth in octaves (-6..6).
    pub env_octaves: f32,
    /// Envelope times in seconds, sustain 0..1.
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
    /// 0 = fixed cutoff, 1 = cutoff follows the note's pitch fully.
    pub key_follow: f32,
}

impl Default for FilterSettings {
    /// The tracker's `defaultLayerFilter()`.
    fn default() -> Self {
        Self { mode: FilterMode::Off, cutoff: 0.75, resonance: 0.2, env_octaves: 0.0, attack: 0.005, decay: 0.3, sustain: 0.0, release: 0.3, key_follow: 0.0 }
    }
}

/// Normalised cutoff -> Hz.
pub fn cutoff_to_hz(cutoff: f32) -> f32 {
    FILTER_BASE_HZ * FILTER_SPAN.powf(cutoff.clamp(0.0, 1.0))
}

/// Normalised resonance -> Web Audio LP/HP Q in dB.
pub fn resonance_to_q_db(resonance: f32) -> f32 {
    Q_MIN_DB + resonance.clamp(0.0, 1.0) * (Q_MAX_DB - Q_MIN_DB)
}

/// Normalised resonance -> linear band-pass Q.
pub fn bandpass_q(resonance: f32) -> f32 {
    0.5 * 40f32.powf(resonance.clamp(0.0, 1.0))
}

/// Envelope value (0..1) `t` seconds after the note starts; after a release
/// at `release_at` it falls linearly from where it was to 0.
pub fn filter_env_at(s: &FilterSettings, t: f32, release_at: Option<f32>) -> f32 {
    let held = |time: f32| -> f32 {
        let a = s.attack.max(0.001);
        let d = s.decay.max(0.001);
        let sus = s.sustain.clamp(0.0, 1.0);
        if time <= 0.0 {
            0.0
        } else if time < a {
            time / a
        } else if time < a + d {
            1.0 + (sus - 1.0) * ((time - a) / d)
        } else {
            sus
        }
    };
    match release_at {
        Some(r_at) if t >= r_at => held(r_at) * (1.0 - (t - r_at) / s.release.max(0.001)).max(0.0),
        _ => held(t),
    }
}

/// Normalised biquad coefficients (a0 == 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Biquad {
    const PASS: Biquad = Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
    const SILENT: Biquad = Biquad { b0: 0.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
}

/// Web Audio-compatible coefficients for `mode` at `hz` (Q in dB for LP/HP,
/// linear for BP), as in the tracker's `layerBiquad`.
pub fn biquad(mode: FilterMode, hz: f32, q: f32, sample_rate: f32) -> Biquad {
    let nyquist = sample_rate / 2.0;
    let f0 = hz.clamp(0.0, nyquist) / nyquist;
    match mode {
        FilterMode::Off => Biquad::PASS,
        FilterMode::Lowpass | FilterMode::Highpass => {
            let lowpass = mode == FilterMode::Lowpass;
            if f0 >= 1.0 {
                return if lowpass { Biquad::PASS } else { Biquad::SILENT };
            }
            if f0 <= 0.0 {
                return if lowpass { Biquad::SILENT } else { Biquad::PASS };
            }
            let w0 = std::f32::consts::PI * f0;
            let cos = w0.cos();
            let alpha = w0.sin() / (2.0 * 10f32.powf(q / 20.0));
            let a0 = 1.0 + alpha;
            let b = (if lowpass { (1.0 - cos) / 2.0 } else { (1.0 + cos) / 2.0 }) / a0;
            Biquad { b0: b, b1: if lowpass { 2.0 * b } else { -2.0 * b }, b2: b, a1: (-2.0 * cos) / a0, a2: (1.0 - alpha) / a0 }
        }
        FilterMode::Bandpass => {
            if f0 <= 0.0 || f0 >= 1.0 {
                return Biquad::SILENT;
            }
            let w0 = std::f32::consts::PI * f0;
            let alpha = w0.sin() / (2.0 * q.max(1e-4));
            let a0 = 1.0 + alpha;
            Biquad { b0: alpha / a0, b1: 0.0, b2: -alpha / a0, a1: (-2.0 * w0.cos()) / a0, a2: (1.0 - alpha) / a0 }
        }
    }
}

/// Coefficients for `settings` at envelope value `env` for a note playing
/// at `rate` (1.0 = the root note).
pub fn coefficients_for(settings: &FilterSettings, env: f32, rate: f32, sample_rate: f32) -> Biquad {
    let follow = settings.key_follow.clamp(0.0, 1.0);
    let base_hz = cutoff_to_hz(settings.cutoff) * rate.max(1e-6).powf(follow);
    let env_cents = 1200.0 * settings.env_octaves.clamp(-6.0, 6.0) * env;
    let hz = base_hz * 2f32.powf(env_cents / 1200.0);
    let q = if settings.mode == FilterMode::Bandpass { bandpass_q(settings.resonance) } else { resonance_to_q_db(settings.resonance) };
    biquad(settings.mode, hz, q, sample_rate)
}

/// One voice's stereo filter + envelope state.
#[derive(Clone, Debug)]
pub struct VoiceFilter {
    coeffs: Biquad,
    // Direct form I history per channel: x1, x2, y1, y2.
    state: [[f32; 4]; 2],
    /// Samples since the note started.
    elapsed: u64,
    /// Samples at which note-off arrived.
    released_at: Option<u64>,
    countdown: usize,
}

impl Default for VoiceFilter {
    fn default() -> Self {
        Self { coeffs: Biquad::PASS, state: [[0.0; 4]; 2], elapsed: 0, released_at: None, countdown: 0 }
    }
}

impl VoiceFilter {
    pub fn note_off(&mut self) {
        if self.released_at.is_none() {
            self.released_at = Some(self.elapsed);
        }
    }

    /// Filters one stereo frame. `rate` is the voice's playback rate
    /// relative to the root note (for key follow).
    #[inline]
    pub fn process(&mut self, settings: &FilterSettings, rate: f32, sample_rate: f32, l: f32, r: f32) -> (f32, f32) {
        if settings.mode == FilterMode::Off {
            self.elapsed += 1;
            return (l, r);
        }
        if self.countdown == 0 {
            let t = self.elapsed as f32 / sample_rate;
            let release_at = self.released_at.map(|s| s as f32 / sample_rate);
            let env = filter_env_at(settings, t, release_at);
            self.coeffs = coefficients_for(settings, env, rate, sample_rate);
            self.countdown = CONTROL_FRAMES;
        }
        self.countdown -= 1;
        self.elapsed += 1;
        let c = self.coeffs;
        let mut run = |ch: usize, x: f32| -> f32 {
            let s = &mut self.state[ch];
            let y = c.b0 * x + c.b1 * s[0] + c.b2 * s[1] - c.a1 * s[2] - c.a2 * s[3];
            // Flush denormals / recover from any blow-up.
            let y = if y.is_finite() && y.abs() > 1e-20 { y } else { 0.0 };
            s[1] = s[0];
            s[0] = x;
            s[3] = s[2];
            s[2] = y;
            y
        };
        (run(0, l), run(1, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn sine(hz: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * hz * std::f32::consts::TAU / SR).sin()).collect()
    }

    fn rms_through(settings: &FilterSettings, input: &[f32]) -> f32 {
        let mut f = VoiceFilter::default();
        let out: Vec<f32> = input.iter().map(|&x| f.process(settings, 1.0, SR, x, x).0).collect();
        let tail = &out[out.len() / 2..];
        (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt()
    }

    #[test]
    fn cutoff_mapping_spans_20hz_to_20khz() {
        assert!((cutoff_to_hz(0.0) - 20.0).abs() < 1e-3);
        assert!((cutoff_to_hz(1.0) - 20_000.0).abs() < 1.0);
    }

    #[test]
    fn lowpass_attenuates_highs_and_passes_lows() {
        let s = FilterSettings { mode: FilterMode::Lowpass, cutoff: 0.4, ..FilterSettings::default() }; // ~315 Hz
        assert!(rms_through(&s, &sine(5000.0, 9600)) < 0.05);
        assert!(rms_through(&s, &sine(60.0, 9600)) > 0.6);
    }

    #[test]
    fn highpass_attenuates_lows() {
        let s = FilterSettings { mode: FilterMode::Highpass, cutoff: 0.7, ..FilterSettings::default() };
        assert!(rms_through(&s, &sine(60.0, 9600)) < 0.05);
    }

    #[test]
    fn stays_stable_at_full_resonance_with_a_sweeping_envelope() {
        let s = FilterSettings { mode: FilterMode::Lowpass, cutoff: 0.2, resonance: 1.0, env_octaves: 6.0, attack: 0.01, decay: 0.05, ..FilterSettings::default() };
        let mut f = VoiceFilter::default();
        let noise: Vec<f32> = (0..48_000u32).map(|i| ((i.wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0).collect();
        for (i, &x) in noise.iter().enumerate() {
            if i == 24_000 {
                f.note_off();
            }
            let (l, r) = f.process(&s, 1.0, SR, x, x);
            assert!(l.is_finite() && r.is_finite() && l.abs() < 100.0);
        }
    }

    #[test]
    fn envelope_shape_and_release() {
        let s = FilterSettings { attack: 0.1, decay: 0.1, sustain: 0.5, release: 0.2, ..FilterSettings::default() };
        assert_eq!(filter_env_at(&s, 0.0, None), 0.0);
        assert!((filter_env_at(&s, 0.05, None) - 0.5).abs() < 1e-5);
        assert!((filter_env_at(&s, 0.1, None) - 1.0).abs() < 1e-5);
        assert!((filter_env_at(&s, 1.0, None) - 0.5).abs() < 1e-5);
        assert!((filter_env_at(&s, 1.1, Some(1.0)) - 0.25).abs() < 1e-5);
        assert_eq!(filter_env_at(&s, 2.0, Some(1.0)), 0.0);
    }

    #[test]
    fn off_mode_is_identity() {
        let mut f = VoiceFilter::default();
        assert_eq!(f.process(&FilterSettings::default(), 1.0, SR, 0.3, -0.2), (0.3, -0.2));
    }
}
