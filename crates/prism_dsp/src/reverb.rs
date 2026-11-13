//! A large stereo reverb for ambient tails: an 8-line feedback delay network
//! (FDN) with a Hadamard feedback matrix.
//!
//! ```text
//! (L+R)/2 ─▶ pre-delay ─▶ 4 diffusion allpasses ─▶ ┌──────── 8 delay lines ◀─┐
//!                                                  │  (slowly modulated)     │
//!                                                  └─▶ damping ─▶ Hadamard ──┘
//!                                    even lines ─▶ L, odd lines ─▶ R ─▶ width ─▶ wet/dry
//! ```
//!
//! - Decay is an RT60 time: each line's feedback gain is set so the tail
//!   falls 60 dB in `decay_seconds`, whatever the line length.
//! - Damping is a one-pole low-pass in every feedback path, so highs die
//!   away faster than lows, as in a real space.
//! - Each line's read point drifts slowly (a fraction of a millisecond, at
//!   a different rate per line), which smears the comb-like ringing long
//!   FDN tails otherwise have.
//! - All memory is allocated in `new` (sized for the largest Size and
//!   Pre-delay), so `process` never allocates.

const LINES: usize = 8;
/// Base delay lengths in ms (mutually prime-ish so the echoes don't line up).
const BASE_MS: [f32; LINES] = [37.1, 41.9, 47.3, 53.9, 61.7, 71.3, 79.9, 89.3];
const DIFFUSER_MS: [f32; 4] = [4.7, 3.6, 12.7, 9.3];
const DIFFUSER_GAIN: f32 = 0.7;
/// Size 0..100 % scales the lines by this range.
const SIZE_MIN: f32 = 0.35;
const SIZE_MAX: f32 = 2.2;
const MOD_DEPTH_MS: f32 = 0.35;
const MAX_PREDELAY_SECONDS: f32 = 0.5;
/// Brings 100 % wet to about 0.8x the dry signal's loudness (broadband).
const WET_GAIN: f32 = 1.4;
/// A longer decay builds up more energy in the tank (measured with noise:
/// level ~ (1 - g^2)^-0.265 for line feedback gain g). Multiplying by
/// (1 - g^2)^this keeps the wet level steady as Decay is swept.
const BUILDUP_COMPENSATION: f32 = 0.265;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverbSettings {
    pub enabled: bool,
    /// Room size 0..100 %.
    pub size_pct: f32,
    /// RT60 decay time in seconds.
    pub decay_seconds: f32,
    /// Damping: the feedback paths' low-pass cutoff in Hz (lower = darker).
    pub damping_hz: f32,
    pub predelay_seconds: f32,
    /// Wet/dry: 0 = dry only, 100 = wet only.
    pub mix_pct: f32,
    /// Stereo width of the tail, 0 (mono) .. 100 %.
    pub width_pct: f32,
}

impl Default for ReverbSettings {
    fn default() -> Self {
        Self { enabled: false, size_pct: 70.0, decay_seconds: 8.0, damping_hz: 6000.0, predelay_seconds: 0.03, mix_pct: 35.0, width_pct: 100.0 }
    }
}

/// A fixed-length circular delay line.
struct Delay {
    buf: Vec<f32>,
    write: usize,
}

impl Delay {
    fn new(len: usize) -> Self {
        Self { buf: vec![0.0; len.max(4)], write: 0 }
    }

    fn push(&mut self, x: f32) {
        self.buf[self.write] = x;
        self.write = (self.write + 1) % self.buf.len();
    }

    /// Reads `delay` samples back (fractional, linear interpolation).
    fn read(&self, delay: f32) -> f32 {
        let len = self.buf.len();
        let d = delay.clamp(1.0, (len - 2) as f32);
        let pos = self.write as f32 - d;
        let pos = if pos < 0.0 { pos + len as f32 } else { pos };
        let i0 = pos as usize % len;
        let i1 = (i0 + 1) % len;
        let frac = pos - pos.floor();
        self.buf[i0] + (self.buf[i1] - self.buf[i0]) * frac
    }

    fn clear(&mut self) {
        self.buf.fill(0.0);
        self.write = 0;
    }
}

/// Schroeder allpass diffuser.
struct Allpass {
    line: Delay,
    delay: f32,
}

impl Allpass {
    fn process(&mut self, x: f32) -> f32 {
        let delayed = self.line.read(self.delay);
        let v = x + DIFFUSER_GAIN * delayed;
        self.line.push(v);
        delayed - DIFFUSER_GAIN * v
    }
}

/// In-place fast Walsh-Hadamard transform, normalised (orthogonal).
fn hadamard(x: &mut [f32; LINES]) {
    let mut h = 1;
    while h < LINES {
        for i in (0..LINES).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let norm = 1.0 / (LINES as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= norm);
}

pub struct Reverb {
    sample_rate: f32,
    predelay: Delay,
    diffusers: Vec<Allpass>,
    lines: Vec<Delay>,
    damp_state: [f32; LINES],
    lfo_phase: [f32; LINES],
    was_enabled: bool,
}

impl Reverb {
    /// Allocates every buffer - call off the audio thread.
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate.max(1.0);
        let ms = |m: f32| (m / 1000.0 * sr) as usize;
        Self {
            sample_rate: sr,
            predelay: Delay::new((MAX_PREDELAY_SECONDS * sr) as usize + 4),
            diffusers: DIFFUSER_MS.iter().map(|&m| Allpass { line: Delay::new(ms(m) + 4), delay: m / 1000.0 * sr }).collect(),
            lines: BASE_MS.iter().map(|&m| Delay::new(ms(m * SIZE_MAX + MOD_DEPTH_MS * 2.0) + 8)).collect(),
            damp_state: [0.0; LINES],
            // Spread the modulation phases so the lines don't move together.
            lfo_phase: std::array::from_fn(|i| i as f32 / LINES as f32),
            was_enabled: false,
        }
    }

    pub fn reset(&mut self) {
        self.predelay.clear();
        self.diffusers.iter_mut().for_each(|a| a.line.clear());
        self.lines.iter_mut().for_each(Delay::clear);
        self.damp_state = [0.0; LINES];
    }

    /// Processes a block in place. Disabled = untouched (and the tail is
    /// cleared, so switching it back on doesn't replay a stale one).
    pub fn process(&mut self, s: &ReverbSettings, left: &mut [f32], right: &mut [f32]) {
        if !s.enabled {
            if self.was_enabled {
                self.reset();
                self.was_enabled = false;
            }
            return;
        }
        self.was_enabled = true;
        let sr = self.sample_rate;
        let size = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * (s.size_pct.clamp(0.0, 100.0) / 100.0);
        let decay = s.decay_seconds.clamp(0.1, 120.0);
        let lengths: [f32; LINES] = std::array::from_fn(|i| BASE_MS[i] * size / 1000.0 * sr);
        // RT60: -60 dB over `decay` seconds, per pass through each line.
        let gains: [f32; LINES] = std::array::from_fn(|i| 10f32.powf(-3.0 * lengths[i] / (decay * sr)));
        let damp = (-std::f32::consts::TAU * s.damping_hz.clamp(200.0, 20_000.0) / sr).exp();
        let predelay = (s.predelay_seconds.clamp(0.0, MAX_PREDELAY_SECONDS) * sr).max(1.0);
        let mod_depth = MOD_DEPTH_MS / 1000.0 * sr;
        let lfo_inc: [f32; LINES] = std::array::from_fn(|i| (0.07 + 0.031 * i as f32) / sr);
        let mean_len = lengths.iter().sum::<f32>() / LINES as f32;
        let g_mean = 10f32.powf(-3.0 * mean_len / (decay * sr));
        let buildup = (1.0 - g_mean * g_mean).max(1e-6).powf(BUILDUP_COMPENSATION);
        let wet = (s.mix_pct.clamp(0.0, 100.0) / 100.0) * WET_GAIN * buildup;
        let dry = 1.0 - s.mix_pct.clamp(0.0, 100.0) / 100.0;
        let width = s.width_pct.clamp(0.0, 100.0) / 100.0;

        for n in 0..left.len() {
            let input = 0.5 * (left[n] + right[n]);
            self.predelay.push(input);
            let mut x = self.predelay.read(predelay);
            for ap in &mut self.diffusers {
                x = ap.process(x);
            }

            let mut taps = [0.0f32; LINES];
            for i in 0..LINES {
                let wobble = (std::f32::consts::TAU * self.lfo_phase[i]).sin() * mod_depth;
                self.lfo_phase[i] = (self.lfo_phase[i] + lfo_inc[i]).fract();
                taps[i] = self.lines[i].read(lengths[i] + mod_depth + wobble);
            }
            let (mut wl, mut wr) = (0.0, 0.0);
            for (i, &t) in taps.iter().enumerate() {
                if i % 2 == 0 {
                    wl += t;
                } else {
                    wr += t;
                }
            }

            let mut fb = taps;
            for i in 0..LINES {
                // One-pole low-pass (damping), then the RT60 gain.
                self.damp_state[i] = fb[i] * (1.0 - damp) + self.damp_state[i] * damp;
                fb[i] = self.damp_state[i] * gains[i];
            }
            hadamard(&mut fb);
            for (i, (line, &f)) in self.lines.iter_mut().zip(&fb).enumerate() {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let v = f + sign * x;
                // Flush denormals in long, quiet tails.
                line.push(if v.abs() > 1e-20 { v } else { 0.0 });
            }

            let scale = 1.0 / (LINES as f32 / 2.0).sqrt();
            let (wl, wr) = (wl * scale, wr * scale);
            let mid = 0.5 * (wl + wr);
            let side = 0.5 * (wl - wr) * width;
            left[n] = left[n] * dry + (mid + side) * wet;
            right[n] = right[n] * dry + (mid - side) * wet;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn impulse_response(settings: &ReverbSettings, seconds: f32) -> (Vec<f32>, Vec<f32>) {
        let mut r = Reverb::new(SR);
        let n = (seconds * SR) as usize;
        let mut l = vec![0.0; n];
        let mut rr = vec![0.0; n];
        l[0] = 1.0;
        rr[0] = 1.0;
        for start in (0..n).step_by(512) {
            let end = (start + 512).min(n);
            r.process(settings, &mut l[start..end], &mut rr[start..end]);
        }
        (l, rr)
    }

    fn energy(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32
    }

    #[test]
    fn disabled_is_identity() {
        let mut r = Reverb::new(SR);
        let mut l = vec![0.3; 256];
        let mut rr = vec![-0.2; 256];
        r.process(&ReverbSettings::default(), &mut l, &mut rr);
        assert!(l.iter().all(|&v| v == 0.3) && rr.iter().all(|&v| v == -0.2));
    }

    #[test]
    fn tail_decays_by_about_60_db_over_the_decay_time() {
        let s = ReverbSettings { enabled: true, mix_pct: 100.0, decay_seconds: 2.0, damping_hz: 20_000.0, ..ReverbSettings::default() };
        let (l, _) = impulse_response(&s, 3.0);
        let early = energy(&l[(0.2 * SR) as usize..(0.4 * SR) as usize]);
        let late = energy(&l[(2.2 * SR) as usize..(2.4 * SR) as usize]);
        let drop_db = 10.0 * (early / late).log10();
        assert!((45.0..75.0).contains(&drop_db), "dropped {drop_db} dB over 2 s");
    }

    #[test]
    fn huge_decay_rings_for_a_long_time_and_stays_finite() {
        let s = ReverbSettings { enabled: true, mix_pct: 100.0, decay_seconds: 60.0, size_pct: 100.0, ..ReverbSettings::default() };
        let (l, r) = impulse_response(&s, 10.0);
        assert!(l.iter().chain(&r).all(|v| v.is_finite() && v.abs() < 2.0));
        assert!(energy(&l[(9.0 * SR) as usize..]) > 1e-8, "a 60 s reverb should still ring after 9 s");
    }

    #[test]
    fn predelay_delays_the_tail() {
        let s = ReverbSettings { enabled: true, mix_pct: 100.0, predelay_seconds: 0.2, ..ReverbSettings::default() };
        let (l, _) = impulse_response(&s, 0.5);
        assert!(l[..(0.19 * SR) as usize].iter().all(|v| v.abs() < 1e-6));
        assert!(l[(0.2 * SR) as usize..].iter().any(|v| v.abs() > 1e-4));
    }

    #[test]
    fn stereo_tail_decorrelates_and_width_zero_is_mono() {
        let s = ReverbSettings { enabled: true, mix_pct: 100.0, ..ReverbSettings::default() };
        let (l, r) = impulse_response(&s, 1.0);
        assert!(l.iter().zip(&r).any(|(a, b)| (a - b).abs() > 1e-3));
        let mono = ReverbSettings { width_pct: 0.0, ..s };
        let (l, r) = impulse_response(&mono, 1.0);
        assert!(l.iter().zip(&r).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn wet_level_stays_near_the_dry_level_across_decay_and_size() {
        for decay in [0.5f32, 2.0, 8.0, 20.0, 60.0] {
            for size in [0.0f32, 70.0, 100.0] {
                let s = ReverbSettings { enabled: true, mix_pct: 100.0, decay_seconds: decay, size_pct: size, ..ReverbSettings::default() };
                let mut r = Reverb::new(SR);
                let n = (6.0 * SR) as usize;
                let mut seed = 12345u32;
                let noise: Vec<f32> = (0..n)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        (seed as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.6
                    })
                    .collect();
                let (mut l, mut rr) = (noise.clone(), noise.clone());
                for start in (0..n).step_by(512) {
                    let end = (start + 512).min(n);
                    r.process(&s, &mut l[start..end], &mut rr[start..end]);
                }
                let ratio = energy(&l[(4.0 * SR) as usize..]).sqrt() / energy(&noise).sqrt();
                assert!((0.5..1.3).contains(&ratio), "decay {decay} size {size}: wet/dry {ratio}");
            }
        }
    }

    #[test]
    fn switching_off_and_on_does_not_replay_a_stale_tail() {
        let on = ReverbSettings { enabled: true, mix_pct: 100.0, ..ReverbSettings::default() };
        let mut r = Reverb::new(SR);
        let (mut l, mut rr) = (vec![1.0; 512], vec![1.0; 512]);
        r.process(&on, &mut l, &mut rr);
        let off = ReverbSettings { enabled: false, ..on };
        let (mut l, mut rr) = (vec![0.0; 512], vec![0.0; 512]);
        r.process(&off, &mut l, &mut rr);
        let (mut l, mut rr) = (vec![0.0; 4096], vec![0.0; 4096]);
        r.process(&on, &mut l, &mut rr);
        assert!(l.iter().all(|v| v.abs() < 1e-9));
    }
}
