//! Stamps the build date (YYYYMMDD) into the binary at compile time, via
//! `env!("SPECTRALPRISM_BUILD_NUMBER")` in `src/lib.rs` - shown in the
//! editor's heading and bottom-right corner, and saved into every exported
//! preset, so it's possible to tell which version of the plugin made a
//! given sound.
//!
//! Shells out to the `date` command rather than pulling in a date/time
//! crate - `date +%Y%m%d` is available and has been exercised on all three
//! of this project's CI targets (Linux, Windows, macOS), so a dependency
//! for this single call is unwarranted.
//!
//! No `cargo:rerun-if-changed` directives are emitted deliberately: cargo's
//! default (rerun whenever any file in the package changes) is exactly
//! what's wanted here - a fresh build number on every real rebuild, not a
//! stale one cached from whenever this file was last touched.

fn main() {
    let output = std::process::Command::new("date")
        .arg("+%Y%m%d")
        .output()
        .expect("failed to run `date` to compute the build number");
    let build_number = String::from_utf8(output.stdout).expect("`date` output was not valid UTF-8").trim().to_string();
    println!("cargo:rustc-env=SPECTRALPRISM_BUILD_NUMBER={build_number}");
}
