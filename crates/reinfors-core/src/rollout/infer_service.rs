//! Inference service for grouped collection: one thread owns the callback; workers block
//! on one-shot `Result` replies. Callback panics cancel the run and answer every pending
//! request with `Err` so no worker deadlocks.
//!
//! The service runs on either a per-collect scoped thread (the plain `collect_grouped`
//! path, where the callback is a per-call loan) or a [`ServiceHost`] — a resident
//! thread owning the callback for a session's lifetime, so callbacks that are
//! thread-affine (e.g. `torch.compile` cudagraphs, whose capture state is
//! thread-local) see one fixed thread across collects.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(crate) struct InferRequest {
    pub player: usize,
    pub obs: Vec<f32>,
    pub n: usize,
    pub reply: SyncSender<Result<Vec<f64>, ()>>,
}

#[derive(Default)]
pub(crate) struct ServiceStats {
    pub seconds: f64,
    pub calls: usize,
    pub rows: usize,
}

pub(crate) struct ServiceState {
    pub cancel: AtomicBool,
    pub error: Mutex<Option<String>>,
    pub stats: Mutex<ServiceStats>,
}

impl ServiceState {
    pub fn new() -> Self {
        ServiceState {
            cancel: AtomicBool::new(false),
            error: Mutex::new(None),
            stats: Mutex::new(ServiceStats::default()),
        }
    }
}

pub(crate) fn run_service<F>(infer: &mut F, rx: &Receiver<InferRequest>, state: &ServiceState)
where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    while let Ok(req) = rx.recv() {
        if state.cancel.load(Ordering::Relaxed) {
            let _ = req.reply.send(Err(()));
            continue;
        }
        let InferRequest {
            player,
            obs,
            n,
            reply,
        } = req;
        let started = Instant::now();
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| infer(player, obs, n)));
        match outcome {
            Ok(rows) => {
                let mut stats = state.stats.lock().expect("service stats poisoned");
                stats.seconds += started.elapsed().as_secs_f64();
                stats.calls += 1;
                stats.rows += n;
                drop(stats);
                let _ = reply.send(Ok(rows));
            }
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "infer callback panicked".to_string());
                *state.error.lock().expect("service error poisoned") = Some(msg);
                state.cancel.store(true, Ordering::Relaxed);
                let _ = reply.send(Err(()));
            }
        }
    }
}

struct Begin {
    rx: Receiver<InferRequest>,
    state: Arc<ServiceState>,
}

/// A resident inference-service thread owning the callback for a session's lifetime.
///
/// Every callback invocation across every collect happens on this one thread — the
/// affinity contract thread-affine callbacks need. Per collect, [`Self::begin`] hands
/// the thread that collect's request channel and state; the service loop exits when
/// the workers drop their senders, and [`Self::wait_done`] is the quiesce barrier.
/// Callback panics are handled inside the loop (cancel + `Err` replies), so the
/// resident thread survives them; dropping the host stops and joins the thread.
pub struct ServiceHost {
    tx: Option<Sender<Begin>>,
    done_rx: Receiver<()>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ServiceHost {
    pub fn spawn<F>(mut infer: F) -> Self
    where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64> + Send + 'static,
    {
        let (tx, ctrl_rx) = std::sync::mpsc::channel::<Begin>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            while let Ok(Begin { rx, state }) = ctrl_rx.recv() {
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    run_service(&mut infer, &rx, &state)
                }));
                if outcome.is_err() {
                    let mut err = state.error.lock().expect("service error poisoned");
                    err.get_or_insert_with(|| "inference service panicked".to_string());
                    drop(err);
                    state.cancel.store(true, Ordering::Relaxed);
                }
                if done_tx.send(()).is_err() {
                    return;
                }
            }
        });
        ServiceHost {
            tx: Some(tx),
            done_rx,
            handle: Some(handle),
        }
    }

    pub(crate) fn begin(&self, rx: Receiver<InferRequest>, state: Arc<ServiceState>) {
        self.tx
            .as_ref()
            .expect("service host stopped")
            .send(Begin { rx, state })
            .expect("inference service thread died");
    }

    pub(crate) fn wait_done(&self) {
        self.done_rx.recv().expect("inference service thread died");
    }
}

impl Drop for ServiceHost {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
