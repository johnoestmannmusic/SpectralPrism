//! Chord mode: one played note becomes a whole chord of voices.
//!
//! Ports `CHORD_PRESET_INTERVALS` / `chordIntervals` / `chordVoices` from
//! SpectralPrism Tracker's `src/core/sampler.ts` 1:1.

/// Upper bound on intervals in a custom chord / voices in one chord.
pub const MAX_CHORD_VOICES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChordShape {
    Major,
    Minor,
    Sus2,
    Sus4,
    Major7,
    Minor7,
    Dom7,
    Add9,
    Power,
    Open,
    Custom,
}

impl ChordShape {
    /// Root-relative semitone intervals of a built-in shape (`Custom`
    /// falls back to a major triad; its real intervals live in `ChordSettings`).
    pub fn intervals(self) -> &'static [i32] {
        match self {
            ChordShape::Major | ChordShape::Custom => &[0, 4, 7],
            ChordShape::Minor => &[0, 3, 7],
            ChordShape::Sus2 => &[0, 2, 7],
            ChordShape::Sus4 => &[0, 5, 7],
            ChordShape::Major7 => &[0, 4, 7, 11],
            ChordShape::Minor7 => &[0, 3, 7, 10],
            ChordShape::Dom7 => &[0, 4, 7, 10],
            ChordShape::Add9 => &[0, 4, 7, 14],
            ChordShape::Power => &[0, 7],
            ChordShape::Open => &[0, 7, 12, 19],
        }
    }
}

/// Fixed-capacity (`Copy`, allocation-free) chord settings, so they can be
/// handed to the audio thread every block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChordSettings {
    pub enabled: bool,
    pub shape: ChordShape,
    /// Custom intervals (used when `shape == Custom`); first `custom_len` are valid.
    pub custom: [i32; MAX_CHORD_VOICES],
    pub custom_len: usize,
    /// Inversions: move the lowest/highest tone by octaves (-2..2).
    pub inversion: i32,
    /// Octave copies of the whole chord (1..3).
    pub octaves: i32,
    /// Alternating per-voice detune in cents (0..50).
    pub detune_cents: f32,
    /// Seconds between successive voices (0..0.2).
    pub strum_sec: f32,
    /// Voice stereo spread around the note's pan (0..1).
    pub pan_spread: f32,
    /// Max voices in one chord (1..16).
    pub voice_cap: usize,
}

impl Default for ChordSettings {
    fn default() -> Self {
        let mut custom = [0; MAX_CHORD_VOICES];
        custom[..3].copy_from_slice(&[0, 4, 7]);
        Self {
            enabled: false,
            shape: ChordShape::Major,
            custom,
            custom_len: 3,
            inversion: 0,
            octaves: 1,
            detune_cents: 0.0,
            strum_sec: 0.0,
            pan_spread: 0.3,
            voice_cap: 12,
        }
    }
}

impl ChordSettings {
    /// Sets the custom intervals from a slice (truncated to `MAX_CHORD_VOICES`).
    pub fn set_custom(&mut self, intervals: &[i32]) {
        let n = intervals.len().min(MAX_CHORD_VOICES);
        self.custom = [0; MAX_CHORD_VOICES];
        self.custom[..n].copy_from_slice(&intervals[..n]);
        self.custom_len = n;
    }
}

/// Parses "0, 4, 7, 11" style text into intervals (invalid tokens skipped).
pub fn parse_intervals(text: &str) -> Vec<i32> {
    text.split(',').filter_map(|t| t.trim().parse::<f32>().ok()).filter(|v| v.is_finite()).map(|v| v.round() as i32).take(MAX_CHORD_VOICES).collect()
}

/// Allocation-free variant of `parse_intervals` (safe on an audio thread):
/// writes into `out` and returns how many intervals were parsed.
pub fn parse_intervals_into(text: &str, out: &mut [i32; MAX_CHORD_VOICES]) -> usize {
    let mut n = 0;
    for token in text.split(',') {
        if n == MAX_CHORD_VOICES {
            break;
        }
        if let Ok(v) = token.trim().parse::<f32>() {
            if v.is_finite() {
                out[n] = v.round() as i32;
                n += 1;
            }
        }
    }
    n
}

/// The root-relative interval list a chord plays, after inversion and
/// octave copies, sorted ascending and capped at `voice_cap`.
pub fn chord_intervals(settings: &ChordSettings) -> Vec<i32> {
    let mut set: Vec<i32> = if settings.shape == ChordShape::Custom { settings.custom[..settings.custom_len.min(MAX_CHORD_VOICES)].to_vec() } else { settings.shape.intervals().to_vec() };
    if set.is_empty() {
        set.push(0);
    }
    let inversion = settings.inversion.clamp(-2, 2);
    for _ in 0..inversion.max(0) {
        set.sort_unstable();
        set[0] += 12;
    }
    for _ in 0..(-inversion).max(0) {
        set.sort_unstable();
        let last = set.len() - 1;
        set[last] -= 12;
    }
    let octaves = settings.octaves.clamp(1, 3);
    let mut out = Vec::with_capacity(set.len() * octaves as usize);
    for octave in 0..octaves {
        out.extend(set.iter().map(|i| i + 12 * octave));
    }
    out.sort_unstable();
    out.truncate(settings.voice_cap.clamp(1, MAX_CHORD_VOICES));
    out
}

/// One voice of a chord.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChordVoice {
    /// Pitch offset from the played note, in semitones (detune included).
    pub semitones: f32,
    /// Gain multiplier (1/count, so dense chords can't clip at the onset).
    pub gain: f32,
    /// Pan offset in -1..1.
    pub pan: f32,
    /// Strum delay in seconds.
    pub delay_sec: f32,
}

/// Expands one note into its chord voices.
pub fn chord_voices(settings: &ChordSettings) -> Vec<ChordVoice> {
    let intervals = chord_intervals(settings);
    let count = intervals.len().max(1);
    let pan_spread = settings.pan_spread.clamp(0.0, 1.0);
    let detune = settings.detune_cents.clamp(0.0, 50.0);
    let strum = settings.strum_sec.clamp(0.0, 0.2);
    intervals
        .iter()
        .enumerate()
        .map(|(index, &interval)| {
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            ChordVoice {
                semitones: interval as f32 + sign * detune / 100.0,
                gain: 1.0 / count as f32,
                pan: if count <= 1 { 0.0 } else { pan_spread * ((index as f32 / (count - 1) as f32) * 2.0 - 1.0) },
                delay_sec: strum * index as f32,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(shape: ChordShape) -> ChordSettings {
        ChordSettings { enabled: true, shape, ..ChordSettings::default() }
    }

    #[test]
    fn major_triad() {
        assert_eq!(chord_intervals(&with(ChordShape::Major)), vec![0, 4, 7]);
    }

    #[test]
    fn first_inversion_moves_the_root_up_an_octave() {
        let s = ChordSettings { inversion: 1, ..with(ChordShape::Major) };
        assert_eq!(chord_intervals(&s), vec![4, 7, 12]);
    }

    #[test]
    fn negative_inversion_drops_the_top_tone() {
        let s = ChordSettings { inversion: -1, ..with(ChordShape::Major) };
        assert_eq!(chord_intervals(&s), vec![-5, 0, 4]);
    }

    #[test]
    fn octaves_and_voice_cap() {
        let s = ChordSettings { octaves: 2, voice_cap: 5, ..with(ChordShape::Major) };
        assert_eq!(chord_intervals(&s), vec![0, 4, 7, 12, 16]);
    }

    #[test]
    fn custom_intervals_and_parse() {
        let mut s = with(ChordShape::Custom);
        s.set_custom(&parse_intervals("0, 3, x, 10"));
        assert_eq!(chord_intervals(&s), vec![0, 3, 10]);
    }

    #[test]
    fn empty_custom_falls_back_to_root() {
        let mut s = with(ChordShape::Custom);
        s.set_custom(&[]);
        assert_eq!(chord_intervals(&s), vec![0]);
    }

    #[test]
    fn voices_are_gain_normalised_spread_strummed_and_detuned() {
        let s = ChordSettings { detune_cents: 10.0, strum_sec: 0.05, pan_spread: 1.0, ..with(ChordShape::Major) };
        let v = chord_voices(&s);
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|x| (x.gain - 1.0 / 3.0).abs() < 1e-6));
        assert_eq!((v[0].pan, v[1].pan, v[2].pan), (-1.0, 0.0, 1.0));
        assert!((v[2].delay_sec - 0.1).abs() < 1e-6);
        assert!((v[0].semitones - 0.1).abs() < 1e-6 && (v[1].semitones - 3.9).abs() < 1e-6);
    }
}
