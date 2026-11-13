//! Stereo ensemble chorus: each channel is delayed by a slowly modulated
//! amount, with the right channel's LFO offset by up to a quarter cycle.
//!
//! Ports the delay curve from the tracker's `chorusDelayAt` /
//! `chorusMaxDelay` (`src/core/layers.ts`) and the wet/dry law from its
//! offline renderer (`(1 - mix) * dry + mix * delayed`). One difference: the
//! tracker runs a chorus per note, while here it runs once on the summed
//! instrument output (a free-running LFO). That sounds very close for an
//! ensemble effect and costs one delay line instead of one per voice.

/// Delay line length in seconds (covers the 40 ms + 10 ms maximum with margin).
const MAX_DELAY_SECONDS: f32 = 0.06;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChorusSettings {
    pub enabled: bool,
    pub rate_hz: f32,
    pub depth_ms: f32,
    pub delay_ms: f32,
    /// Wet amount 0..1.
    pub mix: f32,
    /// L/R LFO phase spread 0..1 (1 = quadrature).
    pub spread: f32,
}

impl Default for ChorusSettings {
    /// The tracker's `defaultChorus()`.
    fn default() -> Self {
        Self { enabled: false, rate_hz: 0.6, depth_ms: 2.5, delay_ms: 12.0, mix: 0.5, spread: 1.0 }
    }
}

/// Delay in seconds on channel `ch` (0 = L, 1 = R) at LFO phase `phase` (radians).
pub fn chorus_delay(settings: &ChorusSettings, ch: usize, phase: f32) -> f32 {
    let offset = if ch == 0 { 0.0 } else { settings.spread.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2 };
    let depth = settings.depth_ms.max(0.0) / 1000.0;
    let base = (settings.delay_ms / 1000.0).max(depth + 0.001);
    (base + depth * (phase + offset).sin()).min(MAX_DELAY_SECONDS - 0.001)
}

pub struct Chorus {
    buf: [Vec<f32>; 2],
    write: usize,
    phase: f32,
    sample_rate: f32,
}

impl Chorus {
    /// Allocates the delay line - call off the audio thread (e.g. in `initialize`).
    pub fn new(sample_rate: f32) -> Self {
        let len = ((MAX_DELAY_SECONDS * sample_rate) as usize).max(4) + 2;
        Self { buf: [vec![0.0; len], vec![0.0; len]], write: 0, phase: 0.0, sample_rate }
    }

    pub fn reset(&mut self) {
        for b in &mut self.buf {
            b.fill(0.0);
        }
        self.write = 0;
        self.phase = 0.0;
    }

    /// Processes a block in place. A disabled chorus still writes its delay
    /// line (cheap) so turning it on doesn't replay stale audio.
    pub fn process(&mut self, settings: &ChorusSettings, left: &mut [f32], right: &mut [f32]) {
        let len = self.buf[0].len();
        let wet = settings.mix.clamp(0.0, 1.0);
        let inc = std::f32::consts::TAU * settings.rate_hz.max(0.0) / self.sample_rate;
        for i in 0..left.len() {
            let dry = [left[i], right[i]];
            self.buf[0][self.write] = dry[0];
            self.buf[1][self.write] = dry[1];
            if settings.enabled {
                for (ch, d) in dry.iter().enumerate() {
                    let delay = chorus_delay(settings, ch, self.phase) * self.sample_rate;
                    let pos = self.write as f32 - delay;
                    let pos = if pos < 0.0 { pos + len as f32 } else { pos };
                    let i0 = pos.floor() as usize % len;
                    let i1 = (i0 + 1) % len;
                    let frac = pos - pos.floor();
                    let delayed = self.buf[ch][i0] + (self.buf[ch][i1] - self.buf[ch][i0]) * frac;
                    let out = (1.0 - wet) * d + wet * delayed;
                    if ch == 0 {
                        left[i] = out;
                    } else {
                        right[i] = out;
                    }
                }
            }
            self.write = (self.write + 1) % len;
            self.phase = (self.phase + inc) % std::f32::consts::TAU;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_chorus_is_identity() {
        let mut c = Chorus::new(48_000.0);
        let mut l = vec![0.5; 64];
        let mut r = vec![-0.5; 64];
        c.process(&ChorusSettings::default(), &mut l, &mut r);
        assert!(l.iter().all(|&v| v == 0.5) && r.iter().all(|&v| v == -0.5));
    }

    #[test]
    fn enabled_chorus_delays_an_impulse_into_the_wet_path() {
        let sr = 48_000.0;
        let mut c = Chorus::new(sr);
        let s = ChorusSettings { enabled: true, mix: 1.0, depth_ms: 0.0, delay_ms: 10.0, ..ChorusSettings::default() };
        let mut l = vec![0.0; 1024];
        let mut r = vec![0.0; 1024];
        l[0] = 1.0;
        r[0] = 1.0;
        c.process(&s, &mut l, &mut r);
        let peak = l.iter().enumerate().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).unwrap().0;
        assert_eq!(peak, 480);
        assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn delay_curve_respects_spread_and_bounds() {
        let s = ChorusSettings::default();
        assert!((chorus_delay(&s, 0, 0.0) - 0.012).abs() < 1e-6);
        assert!((chorus_delay(&s, 1, 0.0) - 0.0145).abs() < 1e-6);
        let extreme = ChorusSettings { delay_ms: 40.0, depth_ms: 10.0, ..s };
        assert!(chorus_delay(&extreme, 0, std::f32::consts::FRAC_PI_2) < MAX_DELAY_SECONDS);
    }
}
