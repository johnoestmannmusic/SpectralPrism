//! MicroTextures: the granular/glitch instrument mode for Cycles.
//!
//! Runs as a **post-fusion stage** (like Percussion): it takes the stereo loop
//! `render::LoopBufferData` produced by the spectral stage and re-textures it
//! with:
//!
//! * a granular engine (grain size, density, jitter, reverse, pitch/pan
//!   scatter, per-grain volume variance),
//! * a retrigger/stutter gate,
//! * a resonant formant filter bank (F1/F2 with resonance and mix) as the
//!   tone-shaper instead of a plain low-pass, and
//! * bit-crush / downsample artefacts.
//!
//! The output is a loopable buffer, so the normal sampler can play it like any
//! other instrument source. All randomness uses a fixed seed so realtime
//! playback and offline WAV export produce the same result.

use crate::render::LoopBufferData;
use std::f32::consts::TAU;

/// Parameters for the MicroTextures stage (all plain, no serde here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicroTextureParams {
    pub grain_seconds: f32,
    pub density_hz: f32,
    pub jitter: f32,
    pub reverse_probability: f32,
    pub pitch_scatter_semitones: f32,
    pub pan_scatter: f32,
    pub volume_variance: f32,
    /// LFO that modulates grain density (glitch speed-ups/slow-downs).
    pub density_mod_rate_hz: f32,
    pub density_mod_depth: f32,
    /// Per-grain digital variation: quantised pitch jumps, random reverse,
    /// hard pan, gain swings and bit reduction.
    pub grain_chaos: f32,
    pub retrigger_hz: f32,
    pub retrigger_amount: f32,
    pub bit_depth: f32,
    pub downsample: f32,
    /// Formant envelope shift in semitones (moves the filter bank).
    pub formant_shift_semitones: f32,
    pub formant_resonance: f32,
    pub formant_mix: f32,
}

impl Default for MicroTextureParams {
    fn default() -> Self {
        Self {
            grain_seconds: 0.08,
            density_hz: 12.0,
            jitter: 0.25,
            reverse_probability: 0.2,
            pitch_scatter_semitones: 0.0,
            pan_scatter: 0.4,
            volume_variance: 0.3,
            density_mod_rate_hz: 0.5,
            density_mod_depth: 0.0,
            grain_chaos: 0.0,
            retrigger_hz: 0.0,
            retrigger_amount: 0.0,
            bit_depth: 16.0,
            downsample: 1.0,
            formant_shift_semitones: 0.0,
            formant_resonance: 0.4,
            formant_mix: 0.0,
        }
    }
}

/// Deterministic xorshift PRNG (same seed means realtime == export).
struct Prng(u32);

impl Prng {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

/// One active grain.
struct Grain {
    pos: f32,
    rate: f32,
    remaining: usize,
    length: usize,
    env_pos: usize,
    gain: f32,
    pan: f32,
    /// Per-grain bit reduction (16 = none) for digital glitch timbre.
    bits: f32,
    /// Per-grain one-pole low-pass (1.0 = bypass) for timbre variation.
    lp: f32,
    lp_coeff: f32,
    /// Per-grain sample-and-hold downsampling (1 = none).
    crush: usize,
    crush_count: usize,
    crush_value: f32,
}

/// Direct form-I biquad. Used for the two formant band-passes.
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn bandpass(f0: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = TAU * (f0.clamp(20.0, sample_rate * 0.45) / sample_rate);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q.max(0.1));
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn peak(channels: &[Vec<f32>]) -> f32 {
    let mut max = 0.0f32;
    for channel in channels {
        for &sample in channel {
            let abs = sample.abs();
            if abs > max {
                max = abs;
            }
        }
    }
    max
}

/// Applies the MicroTextures stage to a fused loop.
pub fn render_microtextures(
    input: &LoopBufferData,
    params: &MicroTextureParams,
) -> LoopBufferData {
    let sample_rate = input.sample_rate.max(1.0);
    let frames = input
        .channels
        .first()
        .map(|channel| channel.len())
        .unwrap_or(0)
        .max(1);
    let channel_count = input.channels.len().max(1);

    // Mono source for the grain reader (the output is re-panned below).
    let mut mono = vec![0.0f32; frames];
    for channel in &input.channels {
        for (i, &sample) in channel.iter().enumerate().take(frames) {
            mono[i] += sample;
        }
    }
    let inv_channels = 1.0 / channel_count as f32;
    for sample in &mut mono {
        *sample *= inv_channels;
    }

    let grain_length = ((params.grain_seconds.clamp(0.005, 0.5) * sample_rate) as usize).max(2);
    let density = params.density_hz.clamp(0.5, 80.0);
    let density_mod_rate = params.density_mod_rate_hz.clamp(0.01, 20.0);
    let density_mod_depth = params.density_mod_depth.clamp(0.0, 1.0);
    let grain_chaos = params.grain_chaos.clamp(0.0, 1.0);
    let jitter = params.jitter.clamp(0.0, 1.0);
    let reverse_probability = params.reverse_probability.clamp(0.0, 1.0);
    let pitch_scatter = params.pitch_scatter_semitones.clamp(0.0, 48.0);
    let pan_scatter = params.pan_scatter.clamp(0.0, 1.0);
    let volume_variance = params.volume_variance.clamp(0.0, 1.0);

    let mut grains: Vec<Grain> = Vec::new();
    let mut prng = Prng::new(0x4C41_4E54);
    let mut left = vec![0.0f32; frames];
    let mut right = vec![0.0f32; frames];
    let mut next_grain = 0usize;

    for i in 0..frames {
        if i >= next_grain {
            // Density LFO: speed the granular stream up/down over time.
            let lfo = (TAU * density_mod_rate * i as f32 / sample_rate).sin();
            let density_now = (density * (1.0 + density_mod_depth * lfo)).clamp(0.2, 120.0);
            next_grain = i + ((sample_rate / density_now) as usize).max(1);
            let reverse = prng.next_f32() < reverse_probability;
            let jitter_offset =
                (prng.next_f32() * 2.0 - 1.0) * jitter * grain_length as f32;
            let start =
                (prng.next_f32() * frames as f32 + jitter_offset).rem_euclid(frames as f32);
            // Digital glitch chaos changes each grain's *timbre* (filter,
            // bit depth, sample-rate) rather than its tuning. Pitch only moves
            // with `pitch_scatter`.
            let chaos = grain_chaos;
            let semitones = (prng.next_f32() * 2.0 - 1.0) * pitch_scatter;
            let base_rate = (semitones / 12.0).exp2();
            let rate = if reverse || (chaos > 0.0 && prng.next_f32() < chaos * 0.5) {
                -base_rate
            } else {
                base_rate
            };
            let gain = 1.0 - volume_variance * prng.next_f32();
            let pan = ((prng.next_f32() * 2.0 - 1.0) * pan_scatter
                + (prng.next_f32() * 2.0 - 1.0) * chaos)
                .clamp(-1.0, 1.0);
            let length = if chaos > 0.0 {
                ((grain_length as f32
                    * (0.4 + prng.next_f32() * 1.2 * chaos))
                    .round() as usize)
                    .max(2)
            } else {
                grain_length
            };
            let bits = if chaos > 0.0 {
                (16.0 - chaos * (4.0 + prng.next_f32() * 8.0)).clamp(2.0, 16.0)
            } else {
                16.0
            };
            let lp_coeff = if chaos > 0.0 {
                let cutoff =
                    300.0 + (1.0 - prng.next_f32()) * 11000.0 * (0.2 + 0.8 * chaos);
                (1.0 - (-(TAU * cutoff / sample_rate)).exp()).clamp(0.01, 1.0)
            } else {
                1.0
            };
            let crush = if chaos > 0.0 {
                // Scatter the per-grain downsample amount around the Lo-fi
                // Downsample setting, so chaos changes the sample-rate too.
                let base = params.downsample.round().max(1.0);
                (base * (1.0 + prng.next_f32() * chaos * 4.0))
                    .round()
                    .max(1.0) as usize
            } else {
                1
            };
            grains.push(Grain {
                pos: start,
                rate,
                remaining: length,
                length,
                env_pos: 0,
                gain,
                pan,
                bits,
                lp: 0.0,
                lp_coeff,
                crush,
                crush_count: 0,
                crush_value: 0.0,
            });
        }

        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for grain in grains.iter_mut() {
            if grain.remaining == 0 {
                continue;
            }
            let read = grain.pos.rem_euclid(frames as f32);
            let i0 = read.floor() as usize % frames;
            let i1 = (i0 + 1) % frames;
            let frac = read - read.floor();
            let sample = mono[i0] * (1.0 - frac) + mono[i1] * frac;
            let t = if grain.length > 1 {
                grain.env_pos as f32 / (grain.length - 1) as f32
            } else {
                0.0
            };
            // Hann envelope across the grain.
            let env = 0.5 - 0.5 * (TAU * t).cos();
            let mut sample = sample;
            // Per-grain timbre: one-pole low-pass, bit reduction, then
            // sample-and-hold downsampling.
            if grain.lp_coeff < 1.0 {
                grain.lp += grain.lp_coeff * (sample - grain.lp);
                sample = grain.lp;
            }
            if grain.bits < 16.0 {
                let levels = 2.0f32.powf(grain.bits);
                sample = (sample * levels).round() / levels;
            }
            if grain.crush > 1 {
                if grain.crush_count == 0 {
                    grain.crush_value = sample;
                }
                grain.crush_count = (grain.crush_count + 1) % grain.crush;
                sample = grain.crush_value;
            }
            let amp = sample * env * grain.gain;
            let pan = grain.pan.clamp(-1.0, 1.0);
            l += amp * (1.0 - pan.max(0.0));
            r += amp * (1.0 + pan.min(0.0));
            grain.pos += grain.rate;
            grain.remaining -= 1;
            grain.env_pos += 1;
        }
        grains.retain(|grain| grain.remaining > 0);

        // Retrigger/stutter gate.
        let retrigger_amount = params.retrigger_amount.clamp(0.0, 1.0);
        if retrigger_amount > 0.0 {
            let phase = (i as f32 / sample_rate * params.retrigger_hz.max(0.001)).fract();
            let decay = (-phase * 6.0).exp();
            let gate = 1.0 - retrigger_amount + retrigger_amount * decay;
            l *= gate;
            r *= gate;
        }

        left[i] = l;
        right[i] = r;
    }

    // Formant filter bank: two resonant band-passes with their centre
    // frequencies shifted by `formant_shift_semitones`. This replaces a plain
    // low-pass so the texture keeps a vowel-like, shiftable body.
    let formant_mix = params.formant_mix.clamp(0.0, 1.0);
    if formant_mix > 0.0 {
        let q = 1.0 + params.formant_resonance.clamp(0.0, 1.0) * 14.0;
        let ratio = (params.formant_shift_semitones.clamp(-48.0, 48.0) / 12.0).exp2();
        let f1 = 500.0 * ratio;
        let f2 = 1500.0 * ratio;
        let mut f1_l = Biquad::bandpass(f1, q, sample_rate);
        let mut f2_l = Biquad::bandpass(f2, q, sample_rate);
        let mut f1_r = Biquad::bandpass(f1, q, sample_rate);
        let mut f2_r = Biquad::bandpass(f2, q, sample_rate);
        for i in 0..frames {
            let wet_l = (f1_l.process(left[i]) + f2_l.process(left[i])) * 0.7;
            let wet_r = (f1_r.process(right[i]) + f2_r.process(right[i])) * 0.7;
            left[i] = left[i] * (1.0 - formant_mix) + wet_l * formant_mix;
            right[i] = right[i] * (1.0 - formant_mix) + wet_r * formant_mix;
        }
    }

    // Bit-crush + downsample artefacts.
    let bits = params.bit_depth.clamp(2.0, 16.0);
    let levels = 2.0f32.powf(bits);
    let downsample = (params.downsample.round() as usize).max(1);
    let mut hold_l = 0.0f32;
    let mut hold_r = 0.0f32;
    for i in 0..frames {
        if i % downsample == 0 {
            hold_l = (left[i] * levels).round() / levels;
            hold_r = (right[i] * levels).round() / levels;
        }
        left[i] = hold_l;
        right[i] = hold_r;
    }

    // Peak-normalise so the texture sits at the same level as other renders.
    let peak_value = peak(&[left.clone(), right.clone()]);
    if peak_value > 1e-6 {
        let gain = 0.99 / peak_value;
        for sample in &mut left {
            *sample *= gain;
        }
        for sample in &mut right {
            *sample *= gain;
        }
    }

    LoopBufferData {
        channels: vec![left, right],
        sample_rate,
        root_note: input.root_note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::LoopBufferData;

    fn input(frames: usize) -> LoopBufferData {
        let mut left = vec![0.0f32; frames];
        let mut right = vec![0.0f32; frames];
        for i in 0..frames {
            let s = (TAU * 440.0 * i as f32 / 44100.0).sin();
            left[i] = s;
            right[i] = s * 0.8;
        }
        LoopBufferData {
            channels: vec![left, right],
            sample_rate: 44100.0,
            root_note: 60,
        }
    }

    #[test]
    fn renders_same_length_and_not_silent() {
        let output = render_microtextures(&input(4410), &MicroTextureParams::default());
        assert_eq!(output.channels.len(), 2);
        assert_eq!(output.channels[0].len(), 4410);
        assert!(output.channels[0].iter().any(|sample| sample.abs() > 1e-4));
    }

    #[test]
    fn is_deterministic_and_stutter_changes_output() {
        let source = input(4410);
        let a = render_microtextures(&source, &MicroTextureParams::default());
        let b = render_microtextures(&source, &MicroTextureParams::default());
        assert_eq!(a.channels[0], b.channels[0]);
        let params = MicroTextureParams {
            retrigger_hz: 12.0,
            retrigger_amount: 1.0,
            ..MicroTextureParams::default()
        };
        let c = render_microtextures(&source, &params);
        assert_ne!(a.channels[0], c.channels[0]);
    }

    #[test]
    fn grain_chaos_changes_the_texture() {
        let source = input(4410);
        let plain = render_microtextures(&source, &MicroTextureParams::default());
        let chaotic = render_microtextures(
            &source,
            &MicroTextureParams {
                grain_chaos: 0.8,
                downsample: 4.0,
                ..MicroTextureParams::default()
            },
        );
        assert!(chaotic.channels[0].iter().any(|s| s.is_finite()));
        assert_ne!(plain.channels[0], chaotic.channels[0]);
    }
}
