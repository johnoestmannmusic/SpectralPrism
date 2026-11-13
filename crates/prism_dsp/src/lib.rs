//! Pure-Rust DSP core for the SpectralPrism plugin. No dependency on audio
//! hardware, a DAW host, or nih-plug - every algorithm here is a plain
//! function/struct over `Vec<f32>`/`&[f32]` that `cargo test` can verify
//! numerically. Ports the phase-vocoder freeze + cepstral formant-shift
//! algorithm from src/0006/index.html's Spectral Fusion "Freeze" feature.

pub mod analysis;
pub mod chain;
pub mod chord;
pub mod chorus;
pub mod envelope;
pub mod fft;
pub mod filter;
pub mod fm_synth;
pub mod fm_transform;
pub mod formant;
pub mod freeze;
pub mod fusion;
pub mod microtextures;
pub mod modroute;
pub mod modulate;
pub mod percussion;
pub mod phase_advance;
pub mod render;
pub mod resample;
pub mod resynth;
pub mod reverb;
pub mod shimmer;
pub mod stereo;
pub mod voice;
pub mod window;
