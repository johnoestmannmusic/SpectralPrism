# baseview (SpectralPrism-vendored fork)

**This is a locally vendored, patched copy of [RustAudio/baseview](https://github.com/RustAudio/baseview)**,
checked out from commit `9a0b42c09d712777b2edb4c5e0cb6baf21e988f0` (the revision pinned by
`egui-baseview`, which is what `nih_plug_egui` - and therefore this plugin's actual editor -
uses), overriding *every* `baseview` dependency in this workspace via `Cargo.toml`'s
`[patch."https://github.com/RustAudio/baseview.git"]` regardless of which git revision each
dependent originally pinned (Cargo's `[patch]` mechanism keys on source URL, not revision).

Patched here to fix three real, upstream X11-backend bugs (see `manage/KANBAN.md`'s
`SP-BUG-002` for the full user-facing symptoms and root-cause writeup):

1. **Keyboard focus was never requested for the window.** `Window::focus()`/`has_focus()` were
   both `unimplemented!()` stubs, and nothing in the window-creation path called
   `XSetInputFocus` - so the editor window only ever received keyboard focus as an indirect
   side effect of some *other* native dialog closing. Fixed in `src/x11/window.rs`: an
   `XSetInputFocus` call right after the window is created/mapped, plus real (non-panicking)
   implementations of `focus()`/`has_focus()`.
2. **X11 classic auto-repeat wasn't made "detectable".** Without
   `XkbSetDetectableAutoRepeat`, X11's core protocol can deliver a key release+press pair
   during ordinary auto-repeat that's indistinguishable from a genuine second keypress -
   `baseview` then hardcoded every event's `repeat` field to `false` regardless, so `egui`
   had no way to tell a real keystroke from a duplicate. Fixed in `src/x11/xcb_connection.rs`:
   calls `XkbSetDetectableAutoRepeat` right after opening the X display connection.
3. **Alt/NumLock/Super were read from the wrong half of the X11 modifier bitmask.**
   `src/x11/keyboard.rs`'s `key_mods` used `KeyButMask::BUTTON1`/`BUTTON2`/`BUTTON4` (bits
   8/9/11 - the *current mouse button* state) instead of `MOD1`/`MOD2`/`MOD4` (bits 3/4/6 -
   the actual Alt/NumLock/Super *modifier key* state); `KeyButMask` packs both into one
   bitmask, and this picked the wrong bits. In practice, Alt got spuriously reported as held
   any time the left mouse button happened to be down at the same moment as a key event - as
   it very often still is right after clicking into a text field to focus it and then
   immediately typing - which made `egui`'s `TextEdit` treat a plain Backspace as "delete
   previous word" (`Key::Backspace` + `modifiers.alt` → `delete_previous_word`, see
   `egui`'s own `text_edit/builder.rs`) instead of "delete previous character".

Everything else in this tree is unmodified upstream `baseview` source (dual MIT/Apache-2.0,
see `LICENSE-MIT`/`LICENSE-APACHE`) - the `examples/` directory and its own nested
`[workspace]`/`[dev-dependencies]` were stripped from `Cargo.toml` since this vendored copy
only needs to build as a library dependency, not stand alone.

To refresh this vendored copy against a newer upstream `baseview` (e.g. once these fixes land
upstream and this patch is no longer needed): delete this directory, re-copy from a fresh
checkout, and either drop the `[patch]` section in the root `Cargo.toml` entirely (if fixed
upstream) or re-apply the two patches above.

---

A low-level windowing system geared towards making audio plugin UIs.

`baseview` abstracts the platform-specific windowing APIs (winapi, cocoa, xcb) into a platform-independent API, but otherwise gets out of your way so you can write plugin UIs.

Interested in learning more about the project? Join us on [discord](https://discord.gg/b3hjnGw), channel `#plugin-gui`.

## Roadmap

Below is a proposed list of milestones (roughly in-order) and their status. Subject to change at any time.

| Feature                                               | Windows            | Mac OS             | Linux              |
| ----------------------------------------------------- | ------------------ | ------------------ | ------------------ |
| Spawns a window, no parent                            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Cross-platform API for window spawning                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Can find DPI scale factor                             |                    | :heavy_check_mark: | :heavy_check_mark: |
| Basic event handling (mouse, keyboard)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Parent window support                                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| OpenGL context creation (behind the `opengl` feature) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

## Prerequisites

### Linux

Install dependencies, e.g.:

```sh
sudo apt-get install libx11-dev libxcb1-dev libx11-xcb-dev libgl1-mesa-dev
```

## License

Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Baseview by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
