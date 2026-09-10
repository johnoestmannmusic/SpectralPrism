//! Stamps the build date (YYYYMMDD) into the binary at compile time, via
//! `env!("SPECTRALPRISM_BUILD_NUMBER")` in `src/lib.rs` - shown in the
//! editor's heading and bottom-right corner, and saved into every exported
//! preset, so it's possible to tell which version of the plugin made a
//! given sound. `SPECTRALPRISM_BUILD_DATE_HUMAN` is the same date spelled
//! out ("11 September 2026") for the heading, computed here in plain Rust
//! (not via a second `date` invocation with a different format string) so
//! it can't drift from `SPECTRALPRISM_BUILD_NUMBER` and doesn't depend on
//! which `date` implementation's format-string extensions happen to be
//! available on a given CI runner.
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

const MONTH_NAMES: [&str; 12] =
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

fn main() {
    let output = std::process::Command::new("date")
        .arg("+%Y%m%d")
        .output()
        .expect("failed to run `date` to compute the build number");
    let build_number = String::from_utf8(output.stdout).expect("`date` output was not valid UTF-8").trim().to_string();

    println!("cargo:rustc-env=SPECTRALPRISM_BUILD_NUMBER={build_number}");
    println!("cargo:rustc-env=SPECTRALPRISM_BUILD_DATE_HUMAN={}", human_readable_date(&build_number));
}

/// Turns "20260911" into "11 September 2026" - plain string slicing/parsing
/// rather than a `date` format string, so this can't disagree with
/// `build_number` and doesn't rely on any particular `date` binary's
/// locale/extension support.
fn human_readable_date(build_number: &str) -> String {
    assert_eq!(build_number.len(), 8, "expected an 8-digit YYYYMMDD build number, got {build_number:?}");
    let year = &build_number[0..4];
    let month: usize = build_number[4..6].parse().expect("build number's month digits were not numeric");
    let day: u32 = build_number[6..8].parse().expect("build number's day digits were not numeric");
    let month_name = MONTH_NAMES.get(month.wrapping_sub(1)).copied().unwrap_or("Unknown");
    format!("{day} {month_name} {year}")
}
