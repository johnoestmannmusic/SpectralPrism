use arc_swap::ArcSwap;
use prism_dsp::fusion::{effective_mode, render_fused_loop, FusionMode, FusionRenderParams};
use prism_dsp::render::LoopBufferData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Freeze Point / Formant Shift / Stereo Width / Spectral Fusion are all
/// render-time operations on the source sample(s) (see
/// `prism_dsp::fusion::render_fused_loop`) - none of them are cheap
/// per-sample transforms, so they can't just be read on the audio thread. A
/// single background thread renders on demand and publishes the result into
/// a shared `ArcSwap` the audio thread reads lock-free.
#[derive(Clone, Copy, PartialEq)]
pub struct RenderRequest {
    pub freeze_point_pct: f32,
    pub volume_pct: f32,
    pub formant_shift_semitones: f32,
    pub stereo_width_pct: f32,
    pub loop_length_seconds: f32,
    pub fusion: FusionRenderParams,
}

/// A cheap-to-clone handle for asking the render worker to render again -
/// separated from `RenderWorker` itself (which owns the thread and stops it
/// on drop) so it can be handed to the GUI thread (e.g. so a "Load Sample"
/// button can trigger a re-render after swapping in new source audio)
/// without giving the GUI any control over the worker's lifetime.
#[derive(Clone)]
pub struct RenderTrigger {
    pending: Arc<(Mutex<Option<RenderRequest>>, Condvar)>,
}

impl RenderTrigger {
    /// Creates a standalone mailbox, independent of any `RenderWorker` -
    /// see `RenderWorker::spawn`'s doc comment for why this independence
    /// matters (a request sent before the worker thread even exists yet is
    /// simply the first thing it processes once it does).
    pub fn new() -> Self {
        Self { pending: Arc::new((Mutex::new(None), Condvar::new())) }
    }

    /// Overwrites any not-yet-started pending request with this one.
    pub fn request_render(&self, request: RenderRequest) {
        let (lock, cvar) = &*self.pending;
        *lock.lock().unwrap() = Some(request);
        cvar.notify_one();
    }
}

impl Default for RenderTrigger {
    fn default() -> Self {
        Self::new()
    }
}

/// Owns the background render thread. Only ever holds the *latest*
/// requested params (`request_render` overwrites any not-yet-started
/// request) so a fast automation sweep can't back the worker up with a
/// queue of stale renders to work through.
///
/// `source`/`source_b` are `ArcSwap`s (not a fixed `Arc` captured at spawn
/// time) so loading a new sample (Phase E; Sample B for Spectral Fusion)
/// can swap it out - the worker always reads whatever the *current* source
/// is at the moment a request comes in, not whatever it was when the thread
/// started.
pub struct RenderWorker {
    trigger: RenderTrigger,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RenderWorker {
    /// Spawns the background thread reusing `trigger`'s own mailbox rather
    /// than creating a fresh one - so a `RenderTrigger` created up front
    /// (e.g. `PrismPlugin::default()`, before `initialize()` has spawned
    /// this worker at all) and handed to the editor keeps working
    /// regardless of whether the editor or `initialize()` runs first. A
    /// request sent before this thread exists just sits in the mailbox
    /// (`pending` starts `Some` in that case, not `None`) and is the first
    /// thing the loop below picks up once it does start. This is exactly
    /// what closed the real bug this fixes: nih-plug hosts aren't
    /// guaranteed to call `initialize()` before creating the editor, so a
    /// `RenderTrigger` captured only from `self.worker` (`None` until
    /// `initialize()` runs) could end up permanently unusable for a given
    /// editor session - loading a sample would swap the source in but never
    /// actually trigger a re-render, silently continuing to play whatever
    /// was frozen before.
    pub fn spawn(
        trigger: RenderTrigger,
        source: Arc<ArcSwap<Vec<Vec<f32>>>>,
        source_b: Arc<ArcSwap<Vec<Vec<f32>>>>,
        sample_rate: f32,
        root_note: u8,
        output: Arc<ArcSwap<LoopBufferData>>,
    ) -> Self {
        let pending = trigger.pending.clone();
        let stop = Arc::new(AtomicBool::new(false));

        let pending_thread = pending.clone();
        let stop_thread = stop.clone();
        let handle = thread::spawn(move || {
            let (lock, cvar) = &*pending_thread;
            loop {
                let request = {
                    let mut guard = lock.lock().unwrap();
                    while guard.is_none() && !stop_thread.load(Ordering::Acquire) {
                        guard = cvar.wait(guard).unwrap();
                    }
                    if stop_thread.load(Ordering::Acquire) {
                        return;
                    }
                    guard.take().expect("woke with no request and no stop signal")
                };

                let current_source = source.load();
                let current_source_b = source_b.load();
                // See `prism_dsp::fusion::effective_mode`'s doc comment for
                // why this degrade happens here (not to the param itself).
                let effective_fusion = FusionRenderParams { mode: effective_mode(request.fusion.mode, &current_source_b), ..request.fusion };
                let rendered = render_fused_loop(
                    &current_source,
                    &current_source_b,
                    sample_rate,
                    request.freeze_point_pct,
                    request.volume_pct,
                    request.formant_shift_semitones,
                    &effective_fusion,
                    request.stereo_width_pct,
                    request.loop_length_seconds,
                    root_note,
                );
                output.store(Arc::new(rendered));
            }
        });

        Self { trigger, stop, handle: Some(handle) }
    }
}

impl Drop for RenderWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let (lock, cvar) = &*self.trigger.pending;
        drop(lock.lock().unwrap());
        cvar.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_dsp::render::{DEFAULT_LOOP_SECONDS, DEFAULT_ROOT_NOTE};
    use std::time::{Duration, Instant};

    fn make_source(sample_rate: f32, seconds: f32) -> Arc<ArcSwap<Vec<Vec<f32>>>> {
        let len = (sample_rate * seconds) as usize;
        let tone: Vec<f32> = (0..len)
            .map(|i| (i as f32 / sample_rate * 220.0 * std::f32::consts::TAU).sin())
            .collect();
        Arc::new(ArcSwap::new(Arc::new(vec![tone])))
    }

    /// Sample B starts genuinely empty until the user loads one - see
    /// `worker_falls_back_to_off_when_b_requiring_mode_has_no_sample_b_loaded`.
    fn make_empty_source() -> Arc<ArcSwap<Vec<Vec<f32>>>> {
        Arc::new(ArcSwap::new(Arc::new(Vec::new())))
    }

    fn wait_for_render(output: &ArcSwap<LoopBufferData>, timeout: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if !output.load().channels.is_empty() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        false
    }

    #[test]
    fn published_buffer_starts_empty_and_updates_after_request() {
        let sample_rate = 48000.0;
        let output = Arc::new(ArcSwap::new(Arc::new(LoopBufferData {
            channels: Vec::new(),
            sample_rate,
            root_note: DEFAULT_ROOT_NOTE,
        })));
        assert!(output.load().channels.is_empty());

        let trigger = RenderTrigger::new();
        let _worker =
            RenderWorker::spawn(trigger.clone(), make_source(sample_rate, 1.0), make_empty_source(), sample_rate, DEFAULT_ROOT_NOTE, output.clone());
        trigger.request_render(RenderRequest {
            freeze_point_pct: 50.0,
            volume_pct: 100.0,
            formant_shift_semitones: 0.0,
            stereo_width_pct: 30.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams::default(),
        });

        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish a render in time");
        assert_eq!(output.load().channels.len(), 2, "render_frozen_loop always outputs stereo");
    }

    #[test]
    fn request_sent_before_spawn_is_still_processed() {
        // Regression test for a real bug: a request sent through a
        // `RenderTrigger` created up front (e.g. `PrismPlugin::default()`)
        // must still get processed even if `RenderWorker::spawn` (which
        // reuses that same trigger's mailbox, see its doc comment) hasn't
        // been called yet - this is exactly what happens if a host creates
        // the editor before calling `initialize()` (which is what actually
        // spawns the worker). Before the trigger/worker were decoupled, the
        // editor could only ever capture a trigger from an already-`Some`
        // `self.worker`, so a request made this early was simply lost.
        let sample_rate = 48000.0;
        let trigger = RenderTrigger::new();
        trigger.request_render(RenderRequest {
            freeze_point_pct: 50.0,
            volume_pct: 100.0,
            formant_shift_semitones: 0.0,
            stereo_width_pct: 30.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams::default(),
        });

        let output = Arc::new(ArcSwap::new(Arc::new(LoopBufferData {
            channels: Vec::new(),
            sample_rate,
            root_note: DEFAULT_ROOT_NOTE,
        })));
        let _worker = RenderWorker::spawn(trigger, make_source(sample_rate, 1.0), make_empty_source(), sample_rate, DEFAULT_ROOT_NOTE, output.clone());

        assert!(wait_for_render(&output, Duration::from_secs(2)), "the pre-spawn request was never processed");
        assert_eq!(output.load().channels.len(), 2);
    }

    #[test]
    fn rapid_requests_collapse_to_the_latest_one() {
        let sample_rate = 48000.0;
        let output = Arc::new(ArcSwap::new(Arc::new(LoopBufferData {
            channels: Vec::new(),
            sample_rate,
            root_note: DEFAULT_ROOT_NOTE,
        })));

        let trigger = RenderTrigger::new();
        let _worker =
            RenderWorker::spawn(trigger.clone(), make_source(sample_rate, 1.0), make_empty_source(), sample_rate, DEFAULT_ROOT_NOTE, output.clone());
        // Fire a burst of superseding requests - the mailbox should collapse
        // these down rather than queueing every one of them.
        for freeze_point_pct in [10.0, 20.0, 30.0, 40.0, 50.0] {
            trigger.request_render(RenderRequest {
                freeze_point_pct,
                volume_pct: 100.0,
                formant_shift_semitones: 0.0,
                stereo_width_pct: 0.0,
                loop_length_seconds: DEFAULT_LOOP_SECONDS,
                fusion: FusionRenderParams::default(),
            });
        }

        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish a render in time");
        // Give the worker a moment to settle in case it were (incorrectly)
        // still working through a backlog of the earlier requests.
        thread::sleep(Duration::from_millis(200));
        assert_eq!(output.load().channels.len(), 2);
    }

    #[test]
    fn swapping_the_source_and_re_requesting_uses_the_new_source() {
        // Proves the whole point of ArcSwap<Vec<Vec<f32>>> over a fixed Arc
        // captured at spawn time: a source swapped in after the worker
        // starts must actually be used by the *next* render, not whatever
        // was current when the thread was spawned.
        let sample_rate = 48000.0;
        let source = make_source(sample_rate, 1.0);
        let output = Arc::new(ArcSwap::new(Arc::new(LoopBufferData {
            channels: Vec::new(),
            sample_rate,
            root_note: DEFAULT_ROOT_NOTE,
        })));

        let trigger = RenderTrigger::new();
        let _worker = RenderWorker::spawn(trigger.clone(), source.clone(), make_empty_source(), sample_rate, DEFAULT_ROOT_NOTE, output.clone());
        let request = RenderRequest {
            freeze_point_pct: 50.0,
            volume_pct: 100.0,
            formant_shift_semitones: 0.0,
            stereo_width_pct: 0.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams::default(),
        };
        trigger.request_render(request);
        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish the first render in time");
        let first_render = output.load().channels[0].clone();

        // Swap in a source with a different frequency and request again via
        // the same cloned trigger, mirroring how the GUI thread would use
        // it - the resulting audio must differ, proving the *new* source
        // was actually read rather than whatever was current at spawn time.
        let swapped_len = (sample_rate * 1.0) as usize;
        let swapped_source: Vec<f32> =
            (0..swapped_len).map(|i| (i as f32 / sample_rate * 880.0 * std::f32::consts::TAU).sin()).collect();
        source.store(Arc::new(vec![swapped_source]));

        // Force a fresh publish to detect: clear the output first so we can
        // tell a *new* render landed rather than reading the still-valid
        // previous one during the wait.
        output.store(Arc::new(LoopBufferData { channels: Vec::new(), sample_rate, root_note: DEFAULT_ROOT_NOTE }));
        trigger.request_render(request);
        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish the second render in time");

        let second_render = output.load().channels[0].clone();
        assert_eq!(first_render.len(), second_render.len(), "loop_length_seconds is unchanged, so both renders should be the same length");
        assert_ne!(first_render, second_render, "expected the swapped-in source to produce different audio");
    }

    #[test]
    fn worker_falls_back_to_off_when_b_requiring_mode_has_no_sample_b_loaded() {
        let sample_rate = 48000.0;
        let output = Arc::new(ArcSwap::new(Arc::new(LoopBufferData {
            channels: Vec::new(),
            sample_rate,
            root_note: DEFAULT_ROOT_NOTE,
        })));

        let source_a = make_source(sample_rate, 1.0);
        let trigger = RenderTrigger::new();
        let _worker = RenderWorker::spawn(trigger.clone(), source_a, make_empty_source(), sample_rate, DEFAULT_ROOT_NOTE, output.clone());
        // Audition needs Sample B, but none was ever loaded (make_empty_source) -
        // this must degrade to Off (plain Sample A) rather than panic on an
        // empty source.
        trigger.request_render(RenderRequest {
            freeze_point_pct: 50.0,
            volume_pct: 100.0,
            formant_shift_semitones: 0.0,
            stereo_width_pct: 30.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams { mode: FusionMode::Audition, ..FusionRenderParams::default() },
        });

        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish a render in time");
        assert_eq!(output.load().channels.len(), 2, "should still produce a normal stereo render, not panic");
    }

    #[test]
    fn worker_uses_source_b_when_present_for_audition_mode() {
        let sample_rate = 48000.0;
        let output = Arc::new(ArcSwap::new(Arc::new(LoopBufferData {
            channels: Vec::new(),
            sample_rate,
            root_note: DEFAULT_ROOT_NOTE,
        })));

        // Sample A and Sample B are different frequencies - Audition mode
        // must audibly use B, not silently fall back to A.
        let source_a = make_source(sample_rate, 1.0);
        let source_b = {
            let len = (sample_rate * 1.0) as usize;
            let tone: Vec<f32> = (0..len).map(|i| (i as f32 / sample_rate * 880.0 * std::f32::consts::TAU).sin()).collect();
            Arc::new(ArcSwap::new(Arc::new(vec![tone])))
        };

        let trigger_a_only = RenderTrigger::new();
        let _worker_a_only =
            RenderWorker::spawn(trigger_a_only.clone(), source_a.clone(), make_empty_source(), sample_rate, DEFAULT_ROOT_NOTE, output.clone());
        trigger_a_only.request_render(RenderRequest {
            freeze_point_pct: 50.0,
            volume_pct: 100.0,
            formant_shift_semitones: 0.0,
            stereo_width_pct: 0.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams::default(),
        });
        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish the A-alone render in time");
        let a_alone = output.load().channels[0].clone();

        output.store(Arc::new(LoopBufferData { channels: Vec::new(), sample_rate, root_note: DEFAULT_ROOT_NOTE }));
        let trigger_audition = RenderTrigger::new();
        let _worker_audition =
            RenderWorker::spawn(trigger_audition.clone(), source_a, source_b, sample_rate, DEFAULT_ROOT_NOTE, output.clone());
        trigger_audition.request_render(RenderRequest {
            freeze_point_pct: 50.0,
            volume_pct: 100.0,
            formant_shift_semitones: 0.0,
            stereo_width_pct: 0.0,
            loop_length_seconds: DEFAULT_LOOP_SECONDS,
            fusion: FusionRenderParams { mode: FusionMode::Audition, ..FusionRenderParams::default() },
        });
        assert!(wait_for_render(&output, Duration::from_secs(2)), "worker did not publish the Audition render in time");
        let audition = output.load().channels[0].clone();

        assert_ne!(audition, a_alone, "Audition mode with Sample B loaded should audibly differ from plain Sample A");
    }
}
