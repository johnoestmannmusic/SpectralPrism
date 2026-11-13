//! Sample file I/O: WAV decode/resample/normalise on load, WAV export.

use crate::params::*;
use prism_dsp::chord::{chord_voices, ChordSettings};
use prism_dsp::render::LoopBufferData;
use prism_dsp::voice::{render_note_offline, PlaybackMode, VoiceManager};
use std::sync::Arc;
use prism_dsp::resample::resample_linear;
use std::path::Path;

/// A short synthetic tone to freeze - the built-in placeholder until a real
/// file is loaded via the editor. Brightness (the balance between the
/// fundamental and its upper harmonics) sweeps over the tone's duration,
/// specifically so that different Freeze Point values capture genuinely
/// different-sounding moments - a *constant* tone would give every Freeze
/// Point nearly identical magnitude content differing only in essentially
/// arbitrary starting phase, which is a poor demonstration of what Freeze
/// Point is for.
pub(crate) fn synthetic_source(sample_rate: f32, seconds: f32) -> Vec<f32> {
    let len = (sample_rate * seconds) as usize;
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let brightness = i as f32 / len as f32; // 0.0 at start, 1.0 at end
            0.5 * (t * 220.0 * std::f32::consts::TAU).sin()
                + (0.05 + 0.5 * brightness) * (t * 440.0 * std::f32::consts::TAU).sin()
                + (0.5 * brightness) * (t * 660.0 * std::f32::consts::TAU).sin()
        })
        .collect()
}

/// Decodes a WAV file into one `Vec<f32>` per channel (interleaved samples
/// deinterleaved), normalizing integer formats to `[-1.0, 1.0]`. Returns the
/// file's own sample rate alongside - the caller is responsible for
/// resampling to the plugin's operating rate if they differ.
pub(crate) fn load_wav_channels(path: &Path) -> Result<(Vec<Vec<f32>>, f32), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("couldn't open WAV: {e}"))?;
    let spec = reader.spec();
    let num_channels = (spec.channels as usize).max(1);
    let sample_rate = spec.sample_rate as f32;

    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); num_channels];
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (i, sample) in reader.samples::<f32>().enumerate() {
                let s = sample.map_err(|e| format!("error reading sample: {e}"))?;
                channels[i % num_channels].push(s);
            }
        }
        hound::SampleFormat::Int => {
            let max_amplitude = (1i64 << (spec.bits_per_sample - 1)) as f32;
            for (i, sample) in reader.samples::<i32>().enumerate() {
                let s = sample.map_err(|e| format!("error reading sample: {e}"))? as f32 / max_amplitude;
                channels[i % num_channels].push(s);
            }
        }
    }

    if channels.iter().all(|c| c.is_empty()) {
        return Err("WAV file contains no audio samples".to_string());
    }

    Ok((channels, sample_rate))
}

/// Brings a loaded file's audio to the plugin's current operating rate -
/// without this, a file whose native rate differs from the host's would
/// play back pitch/speed-shifted, since `VoiceManager` reads the frozen
/// loop assuming it's already at the plugin's operating rate.
pub(crate) fn prepare_source_for_plugin_rate(channels: Vec<Vec<f32>>, file_rate: f32, plugin_rate: f32) -> Vec<Vec<f32>> {
    if (file_rate - plugin_rate).abs() < 0.5 {
        channels
    } else {
        channels.into_iter().map(|c| resample_linear(&c, file_rate, plugin_rate)).collect()
    }
}

/// Scales every channel by the same factor so the loudest sample across all
/// of them hits unity - so a quiet file and a loud file freeze to
/// comparable levels rather than one needing a much higher Freeze Point
/// "loudness" than the other by accident of how it happened to be recorded.
/// A silent file (all-zero, or already effectively silent) is left alone
/// rather than divided by ~0. This runs once at load time; the per-sample
/// Volume slider (`sample_a_volume`/`sample_b_volume`) then only ever
/// *attenuates* from this normalized level, applied at render time (see
/// `prism_dsp::freeze::analyze_freeze_point`'s `gain_pct` parameter) rather
/// than baked into the stored audio, so it can be freely readjusted without
/// re-loading the file.
pub(crate) fn peak_normalize_channels(channels: &mut [Vec<f32>]) {
    let peak = channels.iter().flatten().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 1e-6 {
        let scale = 1.0 / peak;
        for channel in channels.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= scale;
            }
        }
    }
}

/// Combines `load_wav_channels`, `prepare_source_for_plugin_rate`, and
/// `peak_normalize_channels` - the full "get a WAV file's audio ready to be
/// frozen at this sample rate" step, shared by the interactive "Load Sample
/// A/B..." flow, project-recall in `initialize()`, and preset recall in
/// `apply_preset()`.
pub(crate) fn load_and_prepare_sample(path: &Path, plugin_rate: f32) -> Result<Vec<Vec<f32>>, String> {
    let (channels, file_rate) = load_wav_channels(path)?;
    let mut prepared = prepare_source_for_plugin_rate(channels, file_rate, plugin_rate);
    peak_normalize_channels(&mut prepared);
    Ok(prepared)
}

/// Writes the currently-playing frozen loop (`LoopBufferData`, the actual
/// rendered spectral snapshot - not the original source sample) to a WAV
/// file, for a downloader to import into another program (e.g. OpenMPT).
/// 16-bit PCM int, not 32-bit float, to match this project's existing
/// PCM-WAV export convention (0006's own sampler/sample-ZIP exports) and
/// for the widest possible compatibility with trackers/samplers - matching
/// `prism_cli`'s own WAV writer isn't done here since that one deliberately
/// stays float (a fast-iteration DSP test harness, not a distributable
/// export), and the two have no shared dependency to justify merging over.
/// Peak-normalizes the frozen loop before writing it out (same
/// `peak_normalize_channels` used at sample-load time) - the loop's overall
/// level is whatever Volume/Freeze Point/Fusion happened to produce, which
/// is rarely anywhere near 0 dBFS, so an un-normalized export would come out
/// quieter than it needs to for no audible benefit.
///
/// `normalize: false` writes the audio at its own level instead (samples
/// beyond full scale are clipped by the 16-bit conversion).
pub(crate) fn write_loop_buffer_wav(path: &Path, buffer: &LoopBufferData, normalize: bool) -> Result<(), String> {
    if buffer.channels.first().is_none_or(Vec::is_empty) {
        return Err("nothing to export - the instrument rendered silence".to_string());
    }
    let mut channels = buffer.channels.clone();
    if normalize {
        peak_normalize_channels(&mut channels);
    }
    let left = &channels[0];
    let right = channels.get(1).unwrap_or(left);

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: buffer.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| format!("couldn't create WAV file: {e}"))?;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let l_i16 = (l.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        let r_i16 = (r.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(l_i16).map_err(|e| format!("couldn't write sample: {e}"))?;
        writer.write_sample(r_i16).map_err(|e| format!("couldn't write sample: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("couldn't finalize WAV file: {e}"))
}

/// The WAV export: one note at the root pitch (C4, where the rendered loop
/// plays back unshifted) through the *whole* chain - the rendered loop
/// played by a fresh voice engine configured exactly like live playback
/// (envelope, chord voices and strum, vibrato, filter, chorus, reverb,
/// Output Volume, soft limiter), rendered until the reverb tail has died away. A loop is held for one pass of the
/// loop (plus the strum), a one-shot until it ends; either is then
/// released and rendered until its tail has died away.
pub(crate) fn render_played_note(params: &PrismPluginParams, buffer: &Arc<LoopBufferData>) -> LoopBufferData {
    let sample_rate = buffer.sample_rate;
    let mut voices = VoiceManager::new(sample_rate, buffer.root_note);
    let chain = current_render_request(params);
    let chord = apply_realtime_settings(&mut voices, params, &chain, &ChordSettings::default());
    let strum_samples = if chord.enabled { chord_voices(&chord).last().map_or(0.0, |v| v.delay_sec) * sample_rate } else { 0.0 } as usize;
    let frames = buffer.channels.first().map_or(0, Vec::len);
    let hold = match current_voice_settings(params, &chain).playback {
        // The voices end themselves; this is only a safety cap.
        PlaybackMode::OneShot => (120.0 * sample_rate) as usize,
        PlaybackMode::Loop | PlaybackMode::PingPong => frames + strum_samples,
    };
    // Output Volume is applied by the voice engine (set above), before its limiter.
    // Room for the release plus the reverb tail to fall to -80 dB (~1.4x its
    // RT60), capped at two minutes.
    let reverb = current_reverb_settings(params);
    let tail_seconds = 30f32.max(if reverb.enabled { reverb.decay_seconds * 1.4 + reverb.predelay_seconds + 10.0 } else { 0.0 }).min(120.0);
    let [left, right] = render_note_offline(&mut voices, buffer, buffer.root_note, 1.0, hold, (tail_seconds * sample_rate) as usize);
    LoopBufferData { channels: vec![left, right], sample_rate, root_note: buffer.root_note }
}

#[cfg(test)]
mod export_tests {
    use super::*;
    use nih_plug::prelude::*;
    use prism_dsp::render::DEFAULT_ROOT_NOTE;

    fn loop_buffer() -> Arc<LoopBufferData> {
        let sr = 48_000.0;
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / sr).sin() * 0.5).collect();
        Arc::new(LoopBufferData { channels: vec![tone.clone(), tone], sample_rate: sr, root_note: DEFAULT_ROOT_NOTE })
    }

    #[test]
    fn default_patch_exports_one_loop_pass_plus_release() {
        let params = PrismPluginParams::default();
        let out = render_played_note(&params, &loop_buffer());
        let len = out.channels[0].len();
        // 1 s loop held, then the 150 ms default release.
        assert!(len > 48_000 && len < 48_000 + 24_000, "len {len}");
        assert!(out.channels[0].iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn strummed_chord_lengthens_the_export_and_volume_is_applied() {
        let chord_params = || ChordParams {
            enabled: BoolParam::new("Chord", true),
            strum: FloatParam::new("Strum", 0.2, FloatRange::Linear { min: 0.0, max: 0.2 }),
            ..ChordParams::default()
        };
        let params = PrismPluginParams { chord: chord_params(), ..PrismPluginParams::default() };
        let chord = render_played_note(&params, &loop_buffer());
        // Major triad strummed 0.2 s apart: the last tone starts 0.4 s late.
        assert!(chord.channels[0].len() > 48_000 + (0.39 * 48_000.0) as usize, "len {}", chord.channels[0].len());

        let muted = PrismPluginParams {
            chord: chord_params(),
            output: OutputParams { volume: FloatParam::new("Output Volume", OUTPUT_MIN_DB, FloatRange::Linear { min: OUTPUT_MIN_DB, max: OUTPUT_MAX_DB }) },
            ..PrismPluginParams::default()
        };
        let silent = render_played_note(&muted, &loop_buffer());
        assert!(silent.channels[0].iter().all(|&v| v == 0.0));
    }
}
