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

- `crates/prism_dsp` - the DSP core (spectral freeze, formant shift, stereo
  width, envelopes, voice management, resampling). Pure Rust, no dependency on
  nih-plug or any plugin format - unit-tested independently of the plugin wrapper.
- `crates/prism_plugin` - the nih-plug/egui wrapper: parameters, GUI, presets,
  MIDI/voice handling, VST3/CLAP export.
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
