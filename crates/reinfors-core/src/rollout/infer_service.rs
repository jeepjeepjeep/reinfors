//! Inference service for grouped collection: one thread owns the callback; group workers
//! submit tagged requests over a bounded channel and block on one-shot `Result` replies.
//! On callback panic the service records the message, flips the shared cancel flag, and
//! answers every pending and future request with `Err` so no worker deadlocks.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Mutex;
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

pub(crate) fn run_service<F>(mut infer: F, rx: &Receiver<InferRequest>, state: &ServiceState)
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
