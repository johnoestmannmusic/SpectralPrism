# SpectralPrism

<img width="1778" height="1094" alt="image" src="https://github.com/user-attachments/assets/aac628a5-045b-4f66-9d94-3a714c7d473f" />


A phase-vocoder "freeze" instrument: load a sample, freeze a spectral
snapshot of it into a sustained, evolving pad/drone, and play it back
polyphonically via MIDI. Available as a VST3 and CLAP plugin, and as a
standalone application.

Built in Rust on top of [nih-plug](https://github.com/robbert-vdh/nih-plug).

## Key Features

- **Freeze Point** - where in the source sample the spectral snapshot is taken.
- **Formant Shift** - reshapes the frozen spectrum's envelope independently of pitch.
- **Spectral Fusion** - load a second sample ("Sample B", with its own Freeze
  Point/Volume/Tune/Formant Shift) and combine it with Sample A's frozen spectrum
  via seven algorithms - Mix, Cross-Synth, Convolve, Ring Modulate, Spectral Max,
  Spectral Min, or Cycle - or leave it Off for plain single-sample Freeze.
- **Export WAV** - export the current frozen loop as a peak-normalized WAV file,
  e.g. for use in a tracker or sampler.

### Instrument stages (from SpectralPrism Tracker)

The instrument is a signal chain, shown in the editor as a strip of stage
cards - click a card to edit it, click its LED to switch it on or off:

```
SOURCE ─▶ FM ─▶ SPECTRAL ─▶ PERCUSSION ─▶ TEXTURE ║ VOICE ─▶ CHORD ─▶ POST-FX
└──── rendered into the loop (a moment to update) ┘ └─ per note (instant) ─┘
```

- **Source** - Sample A/B, plus trim and playback mode (Auto / Loop /
  Ping-pong / One-shot). With every render stage off the plugin is a plain
  sampler. Instead of Sample A you can switch to an **FM Synth** source: four
  operators (Sine / Square / Triangle / Saw / Noise, ratio, fine, detune,
  output) with a free routing matrix - any operator can modulate any other,
  including itself (feedback). Its tone is rendered into a buffer that stands
  in for the sample, so it runs through the rest of the chain the same way;
  the loaded sample is kept for when you switch back.
- **FM** - the sample phase-modulates a sine carrier (Direct / Stack / Pair
  algorithms, op2 feedback, index sway, stereo detune, sample-hold and
  quantise lo-fi).
- **Spectral** - the freeze and Spectral Fusion above, plus FFT "smear" size,
  a **Shimmer** freeze style, and four **modulation routes** (LFO / ramp /
  random) that sweep any spectral parameter - including Loop Length - over
  the loop.
- **Percussion** - re-synthesises the result as a drum hit (modal body,
  noise, transient, pitch sweep, drive, compression), with Kick / Snare /
  Metal / Hat starting points.
- **Texture (MicroTextures)** - granular post-processing: grain scatter,
  density modulation, chaos, retrigger stutter, formant filter bank, bit-crush.
- **Voice** - amp envelope, transpose / fine tune, vibrato, mono/poly with
  voice cap and choke, pan spread, and a resonant **filter** (LP/HP/BP) with
  its own envelope and key follow.
- **Chord** - each note plays a chord (11 shapes or custom intervals) with
  inversions, octaves, detune, strum and pan spread.
- **Post-FX** - on the whole instrument: a stereo ensemble **Chorus**, then a
  large **Reverb** (8-line feedback delay network; size, decay up to 60 s,
  damping, pre-delay, width, wet/dry mix) for big ambient spaces.
- **Output Volume** (bottom-left, always visible) - the master level, applied
  before the safety limiter; its meter turns red when the limiter is
  squashing the signal (turn it down).

Every control explains itself in the info line when hovered. For layered
sounds, run several instances of the plugin.

## Building from source

Requires a recent stable Rust toolchain ([rustup.rs](https://rustup.rs)).

```sh
cargo xtask bundle prism_plugin --release
```

This produces `SpectralPrism.vst3` and `SpectralPrism.clap` under
`target/bundled/`. Build natively on each target OS you want a bundle for -
this is a GUI plugin with real platform-specific windowing code, so
cross-compiling from a different OS is not supported.

A standalone (non-plugin) build is also available for quick testing outside a DAW:

```sh
cargo run --release --bin prism_plugin_standalone
```

## Project structure

- `crates/prism_dsp` - the DSP core (spectral freeze, fusion, shimmer,
  modulation, FM transform, percussion, microtextures, the full render
  `chain`, plus the real-time voice engine: playback modes, chord, filter,
  chorus). Pure Rust, no dependency on nih-plug or any plugin format -
  unit-tested independently of the plugin wrapper. SpectralPrism Tracker
  vendors this crate (`scripts/vendor-prism-dsp.mjs`); this repo is its
  source of truth.
- `crates/prism_plugin` - the nih-plug/egui wrapper: parameters
  (`params.rs`, `stage_params.rs`), presets (`preset.rs`), the signal-chain
  editor (`editor.rs`, `stages.rs`, `widgets.rs`), the background render
  worker, MIDI/voice handling, VST3/CLAP export.
- `crates/prism_cli` - a WAV-in/WAV-out command-line harness for fast by-ear DSP
  iteration without needing a plugin host.
- `xtask` - the bundling tool (`cargo xtask bundle ...`), provided by nih-plug.

## License

This project's own source code (`crates/prism_dsp`, `crates/prism_plugin`,
`crates/prism_cli`, `xtask`) is licensed under the [MIT License](LICENSE) -
take and extend any of it however you'd like.

One nuance worth knowing: producing a **VST3** build links against nih-plug's
VST3 bindings, which are an external dependency licensed under
**GPL-3.0-or-later** by their own author (not by this project). As a result, a
*compiled* VST3 binary built from this project is a combined work subject to
GPL-3.0's terms, independent of this project's own MIT license on its source.
This doesn't affect the DSP core (`prism_dsp`) or the CLI harness
(`prism_cli`), neither of which touch nih-plug's VST3 bindings, and it
doesn't restrict what you can do with this project's own source code. It's
specifically about the license terms that apply to a compiled, distributed
VST3 binary.

_Note that as of 2025/10/20, Steinberg relicensed VST3 under MIT (see https://steinbergmedia.github.io/vst3_dev_portal/pages/Versions/Version+3.8.0.html). It appears that nih-plug has not been relicensed at this stage._


## AI Disclosure
GenAI was used to assist the programming of this software. To learn more about my current thoughts around this, please read this post: https://johnoestmannmusic.com/ai-building-exoskeletons
