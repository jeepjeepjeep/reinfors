//! Collection telemetry, engine-neutral so policies can fold into it without
//! importing the engine.

/// Summary of one finished episode.
#[derive(Clone)]
pub struct EpisodeSummary {
    pub reward: Vec<f64>,
    pub length: usize,
    pub seeded: bool,
}

/// Telemetry for one collection call.
#[derive(Default, Clone)]
pub struct CollectStats {
    pub episodes: Vec<EpisodeSummary>,
    pub decisions: usize,
    pub max_depth: i32,
    pub sum_leaves: f64,
    pub sum_rounds: f64,
    pub sum_expansions: f64,
    pub sum_sigma: f64,
    pub sum_disagreement: f64,
    pub infer_seconds: f64,
    pub infer_calls: usize,
    pub infer_rows: usize,
    pub padded_rows: usize,
    pub cache_lookups: usize,
    pub cache_hits: usize,
    pub sum_terminal_sims: usize,
    pub sum_depthcap_sims: usize,
    pub sum_shared_rows: usize,
    pub sum_fresh_rows: usize,
    pub sum_hit_rows: usize,
    pub sum_extra_eval_rows: usize,
}
