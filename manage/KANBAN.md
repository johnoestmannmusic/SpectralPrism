# SpectralPrism — Project Kanban

How to use:
This file is the shared Kanban Markdown coordination board for this project.
When starting work, agents should familiarize themselves with the state of the project as described below.
Agents should only assign themselves to cards that are not already assigned to another agent. They should move cards between buckets instead of duplicating them, preserve
card IDs, and append dated process comments after each completed feature.
Buckets: Ideas, Bugs, Planned Features, Assigned, Completed. A found bug gets a card in Bugs (symptom, root cause once known, fix approach); once verified fixed it moves to Completed like any other card, keeping its ID.
Dates throughout this file include a time (HH:MM), not just a date, since multiple agents may work on the same project on the same day.

---

## Project: SpectralPrism

**Project Title:** SpectralPrism — Spectral Freeze Instrument Plugin
**Project Description:** A Rust nih-plug/egui VST3/CLAP instrument that freezes a spectral snapshot of a loaded sample at a chosen "Freeze Point" and loops it indefinitely as a sustained pad/drone, shaped by Formant Shift, Stereo Width, Loop Length, and a full ADSR/velocity/pitch-bend/pan voice architecture.
**Implementation Repository:** this repository (`crates/prism_dsp`, `crates/prism_plugin`, `crates/prism_cli`, `xtask`).
**Board Last Updated:** 2026-09-11 08:08 by Claude Sonnet 5

### Ideas

#### SP-IDEA-001 — FM fusion mode ("waveforms act as operators")

- **Card Title:** FM fusion mode
- **Description:** A 10th Spectral Fusion algorithm where Sample A's and Sample B's frozen spectra modulate each other continuously (FM synthesis-style, "operators") rather than combining once. Architecturally bigger than every other Fusion mode shipped in SP-PLAN-001: those are all one-shot combines (either two `FrozenSpectrum`s combined once before resynthesis, or two fully-resynthesized audio buffers combined once after). FM instead needs one signal to continuously modulate the other's instantaneous phase/frequency *during* the per-hop resynthesis loop (`prism_dsp::resynth::FreezeResynth::next_frame`), which means changing the resynthesis engine itself (e.g. a second phase accumulator feeding into the first's phase advance every hop), not adding a sibling combine function next to `prism_dsp::fusion`'s existing ones. Deserves its own dedicated design pass rather than being bolted on.
- **Assigned Agent:** Unassigned
- **Card Creation Date:** 2026-09-11 08:08
- **Card Completion Note:** Pending; idea has not been scheduled.
- **Process Comments:** 2026-09-11 08:08 — Recorded during Spectral Fusion's design pass (SP-PLAN-001) after the user proposed it as a further idea; deliberately excluded from that implementation for the architectural reasons above.

#### SP-IDEA-002 — Cycle-mode alternation beyond A/B

- **Card Title:** N-way Cycle alternation
- **Description:** The shipped Cycle Fusion mode alternates exactly one loop of A then one loop of B (concatenation of two `render_frozen_loop` outputs, see `prism_dsp::fusion::cycle_buffers`). A future extension could support more than two sources, or a configurable per-source repeat count, if there's ever a reason to support more than one secondary sample slot.
- **Assigned Agent:** Unassigned
- **Card Creation Date:** 2026-09-11 08:08
- **Card Completion Note:** Pending; idea has not been scheduled. Blocked on there being any reason to support more than a single Sample B slot at all.
- **Process Comments:** 2026-09-11 08:08 — Recorded as a possible extension while implementing SP-PLAN-001's two-sample Cycle mode.

### Bugs

_None currently open — see Completed for resolved bugs._

### Planned Features

_None currently planned — see Completed and Ideas above._

### Assigned

_None currently assigned._

### Completed

#### SP-PLAN-001 — Spectral Fusion

- **Card Title:** Spectral Fusion
- **Description:** Adds a second sample slot ("Sample B") with its own Freeze Point/Formant Shift, and a "Spectral Fusion" dropdown that combines Sample A's and Sample B's independently frozen spectra nine ways: Off (A alone), Audition (B alone), Mix (A/B crossfade with its own blend slider), Cross-Synth (B's spectral envelope imposed on A's fine structure/phase, reusing `formant::compute_spectral_envelope`'s existing cepstral-liftering machinery), Convolve (per-bin complex multiply of the two frozen spectra, peak-normalized against A), Ring Modulate (time-domain sample multiply of the two independently resynthesized loops, peak-normalized against A), Spectral Max/Min (per-bin louder/quieter-wins selection), and Cycle (one full loop of A then one of B, then repeat — a plain buffer concatenation, verified compatible with `resample::PlaybackReader`'s length-agnostic wraparound). Cross-Synth/Convolve/Ring Modulate each got their own dry/wet Amount slider; Max/Min did not (per-bin comparisons, no natural "amount" to dial). Moved Stereo Width/Loop Length from the left column to the right column (under Velocity Sensitivity) to make room for Sample B's waveform/controls. Added a new "info box" UI pattern (none existed before) giving a 2-sentence explanation of whichever mode is selected. FM ("waveforms act as operators") was explicitly scoped out — see SP-IDEA-001.
- **Assigned Agent:** Claude Sonnet 5
- **Card Creation Date:** 2026-09-11 07:14
- **Card Completion Note:** Fully implemented and verified. DSP: `crates/prism_dsp/src/fusion.rs` (new module, 14 unit tests) plus a behavior-preserving refactor of `render.rs` (`apply_formant_shift`/`render_channel`/`quantize_advance_for_loop` extracted to `pub(crate)` so the fusion path reuses the exact same resynthesis/OLA machinery — regression-checked against all 61 pre-existing `prism_dsp` tests, which passed unmodified). Plugin: new `FusionMode` `EnumParam` (this codebase's first use of that nih_plug pattern) plus 6 new `FloatParam`s, `source_b`/`loaded_filename_b` fields mirroring Sample A's, `render_worker.rs` extended with a graceful-degrade guard (`prism_dsp::fusion::effective_mode`) so selecting a Sample-B-requiring mode with no Sample B loaded falls back to Off for that render rather than panicking (shown to the user via an in-editor warning label, not silently). Both persistence surfaces updated: nih_plug's automatic DAW-project state (new `#[id]`/`#[persist]` fields) and the plugin's own preset `.json` format (`Preset` struct +8 fields, all with backward-compatible `#[serde(default = "...")]` values matching each control's own default, verified by a new `preset_without_fusion_fields_still_deserializes` test against the pre-existing old-format JSON literal). Full workspace test suite: 91/91 passing (`cargo test --workspace`). `cargo xtask bundle prism_plugin --release` produces both CLAP and VST3 bundles cleanly. Manually smoke-tested via the standalone binary (screenshots): verified the column move, the Fusion dropdown (all 9 modes listed and selectable), the conditional Sample B section appearing/disappearing correctly, each mode's own Amount/Blend slider swapping in correctly (checked Mix and Cross-Synth), the "needs Sample B" warning label, and the info box text updating per mode.
- **Process Comments:** 2026-09-11 07:14 — Three parallel Explore passes (UI editor layout, DSP freeze/render pipeline, sibling "Lantern" project's Cross-Synth/Convolve/Ring Modulate implementations) plus one Plan pass produced the implementation plan; user confirmed Amount sliders on Cross-Synth/Convolve/Ring Modulate and added Spectral Max+Min as extra modes during review. 2026-09-11 08:08 — Implemented sequentially per the plan (DSP+tests → params → render worker → plugin wiring → presets → UI), verifying tests green at each step; manual GUI smoke test performed via screenshots (X11 `import`/synthetic click events, no project-specific run skill existed yet for this native baseview/egui app).

#### SP-BUG-001 — Linux CI build failing on `wayland-client`/`x11-xcb` pkg-config errors

- **Card Title:** Linux CI build failing on missing native dependencies
- **Description:** The GitHub Actions Linux build job failed twice in a row: first with `wayland-sys`'s build script unable to find `wayland-client.pc` via pkg-config, then (after fixing that) with `x11`'s build script unable to find `x11-xcb.pc`. Root cause: the workflow's `apt-get install` list for Linux was missing `libwayland-dev` (provides `wayland-client.pc`) and `libx11-xcb-dev`/`libxcb1-dev` (the latter two confirmed against `baseview`'s own upstream CI config, which installs exactly `libx11-dev libxcb1-dev libx11-xcb-dev libgl1-mesa-dev` — this project already had the first and last of those four).
- **Assigned Agent:** Claude Sonnet 5
- **Card Completion Note:** Fixed by adding all three missing packages to `.github/workflows/build.yml`'s Linux dependency install step. Linux build confirmed green after the second fix.
- **Card Creation Date:** 2026-09-07 (approximate, prior session)
- **Process Comments:** 2026-09-07 — Diagnosed and fixed across two round-trips as each missing package surfaced in turn; user confirmed the Linux build succeeded after both fixes.

#### SP-PLAN-000 — GitHub Releases automation

- **Card Title:** Publish all-OS Releases on version tags
- **Description:** Added a `release` job to `.github/workflows/build.yml` that runs after the three-OS `build` matrix completes, gated on `refs/tags/v*` pushes: downloads all three platform artifacts, zips each, and publishes a GitHub Release via `softprops/action-gh-release` with auto-generated release notes.
- **Assigned Agent:** Claude Sonnet 5
- **Card Completion Note:** Implemented; not yet exercised by an actual tag push at time of writing.
- **Card Creation Date:** 2026-09-07 (approximate, prior session)
- **Process Comments:** 2026-09-07 — Added alongside the CI bug fixes above, once the Linux build was confirmed passing.
