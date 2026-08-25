//! Protocol invariants of the observation arena, tested independently of the
//! engine. Run under Miri with `cargo +nightly miri test -p reinfors-core --test
//! arena`; the CAS interleavings have a Loom model in `tests/loom_arena.rs`.

use std::sync::Arc;

use reinfors_core::rollout::arena::{AliasOutcome, Arena, Reserve, SealError};

fn full(arena: &Arena, rows: usize) -> reinfors_core::rollout::arena::SpanGuard<'_> {
    match arena.try_reserve(rows) {
        Reserve::Full { span, .. } => span,
        _ => panic!("expected full reservation"),
    }
}

fn row(v: f32, dim: usize) -> Vec<f32> {
    vec![v; dim]
}

#[test]
fn out_of_order_commits_seal_once() {
    let arena = Arena::new(6, 3, 16);
    let mut a = full(&arena, 2);
    let mut b = full(&arena, 2);
    let mut c = full(&arena, 2);
    for (guard, base) in [(&mut a, 0.0f32), (&mut b, 2.0), (&mut c, 4.0)] {
        for i in 0..2 {
            guard.push_row(&row(base + i as f32, 3));
        }
    }
    c.commit();
    a.commit();
    assert_eq!(arena.seal_state(), Err(SealError::CommitsPending));
    b.commit();
    let info = arena.seal_state().unwrap();
    assert_eq!((info.rows, info.aliases), (6, 0));
    let rows = arena.into_rows().map_err(|(_, e)| e).unwrap();
    assert_eq!(rows.len(), 18);
    let expect: Vec<f32> = (0..6).flat_map(|r| row(r as f32, 3)).collect();
    assert_eq!(rows, expect);
}

#[test]
fn dropped_guard_poisons_but_resolves() {
    let arena = Arena::new(6, 2, 16);
    let mut a = full(&arena, 2);
    a.push_row(&row(1.0, 2));
    a.push_row(&row(2.0, 2));
    a.commit();
    {
        let _abandoned = full(&arena, 2);
    }
    assert!(arena.is_poisoned());
    let (arena, err) = arena.into_rows().expect_err("seal must fail");
    assert_eq!(err, SealError::NotClosed);
    arena.close().unwrap();
    let (arena, err) = arena.into_rows().expect_err("seal must fail");
    assert_eq!(err, SealError::Poisoned);
    drop(arena);
}

#[test]
fn panicking_worker_poisons() {
    let arena = Arc::new(Arena::new(2, 1, 16));
    let inner = arena.clone();
    let joined = std::thread::spawn(move || {
        let mut span = full(&inner, 1);
        span.push_row(&[7.0]);
        panic!("worker dies before commit");
    })
    .join();
    assert!(joined.is_err());
    assert!(arena.is_poisoned());
    arena.close().unwrap();
    assert_eq!(arena.seal_state(), Err(SealError::Poisoned));
}

#[test]
fn filling_reservation_closes_and_splits() {
    let arena = Arena::new(4, 1, 16);
    let mut head = full(&arena, 1);
    head.push_row(&[0.0]);
    head.commit();
    let (mut span, closed) = match arena.try_reserve(5) {
        Reserve::Partial { span, closed } => (span, closed),
        _ => panic!("expected partial reservation"),
    };
    assert_eq!(span.rows(), 3);
    assert_eq!((closed.rows, closed.aliases), (4, 0));
    assert!(matches!(arena.try_reserve(1), Reserve::Closed));
    for i in 0..3 {
        span.push_row(&[i as f32 + 1.0]);
    }
    span.commit();
    assert_eq!(
        arena.into_rows().map_err(|(_, e)| e).unwrap(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn exact_fill_closes_in_same_cas() {
    let arena = Arena::new(2, 1, 16);
    match arena.try_reserve(2) {
        Reserve::Full { mut span, closed } => {
            assert_eq!(closed.unwrap().rows, 2);
            span.push_row(&[1.0]);
            span.push_row(&[2.0]);
            span.commit();
        }
        _ => panic!("expected full reservation"),
    }
    assert!(matches!(arena.try_reserve(1), Reserve::Closed));
}

#[test]
fn alias_close_and_saturation() {
    let arena = Arena::new(4, 1, 2);
    let t1 = match arena.try_alias() {
        AliasOutcome::Ticket(t) => t,
        _ => panic!("expected ticket"),
    };
    let _t2 = match arena.try_alias() {
        AliasOutcome::Ticket(t) => t,
        _ => panic!("expected ticket"),
    };
    assert!(matches!(arena.try_alias(), AliasOutcome::Saturated));
    t1.commit();
    let info = arena.close().unwrap();
    assert_eq!(info.aliases, 2);
    assert!(matches!(arena.try_alias(), AliasOutcome::Closed));
    assert_eq!(arena.aliases_abandoned(), 0);
}

#[test]
fn abandoned_alias_is_counted() {
    let arena = Arena::new(2, 1, 8);
    {
        let _ticket = match arena.try_alias() {
            AliasOutcome::Ticket(t) => t,
            _ => panic!("expected ticket"),
        };
    }
    assert_eq!(arena.aliases_abandoned(), 1);
}

#[test]
fn zeroed_span_underwrite_reads_zero() {
    let arena = Arena::new(2, 4, 16);
    let mut span = full(&arena, 2);
    let dst = span.zeroed();
    dst[0] = 9.0;
    dst[5] = 3.0;
    span.commit();
    assert!(arena.close_info().is_some(), "fill auto-closes");
    let rows = arena.into_rows().map_err(|(_, e)| e).unwrap();
    assert_eq!(rows, vec![9.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
}

#[test]
fn partial_close_truncates_to_prefix() {
    let arena = Arena::new(8, 2, 16);
    let mut span = full(&arena, 3);
    for i in 0..3 {
        span.push_row(&row(i as f32, 2));
    }
    span.commit();
    let info = arena.close().unwrap();
    assert_eq!(info.rows, 3);
    let rows = arena.into_rows().map_err(|(_, e)| e).unwrap();
    assert_eq!(rows.len(), 6);
    assert_eq!(rows, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn concurrent_spans_stay_disjoint() {
    const WORKERS: usize = 8;
    const SPANS: usize = 16;
    const ROWS: usize = 2;
    const DIM: usize = 5;
    let arena = Arc::new(Arena::new(WORKERS * SPANS * ROWS, DIM, 16));
    let workers: Vec<_> = (0..WORKERS)
        .map(|w| {
            let arena = arena.clone();
            std::thread::spawn(move || {
                for _ in 0..SPANS {
                    let mut span = match arena.try_reserve(ROWS) {
                        Reserve::Full { span, .. } => span,
                        _ => panic!("pool sized exactly"),
                    };
                    let first = span.row_range().start;
                    for r in 0..ROWS {
                        span.push_row(&row((first + r) as f32 + (w as f32) * 1e-3, DIM));
                    }
                    span.commit();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let arena = Arc::into_inner(arena).unwrap();
    let info = arena.close_info().unwrap();
    assert_eq!(info.rows, WORKERS * SPANS * ROWS);
    let rows = arena.into_rows().map_err(|(_, e)| e).unwrap();
    for r in 0..WORKERS * SPANS * ROWS {
        let cell = rows[r * DIM];
        assert_eq!(
            cell.floor() as usize,
            r,
            "row {r} overwritten by another span"
        );
        for d in 1..DIM {
            assert_eq!(rows[r * DIM + d], cell, "row {r} torn");
        }
    }
}

#[test]
fn unsealed_arena_drops_cleanly() {
    let arena = Arena::new(16, 8, 16);
    let mut span = full(&arena, 4);
    span.zeroed();
    span.commit();
    drop(arena);
}

#[test]
#[should_panic(expected = "commit with uninitialized rows")]
fn commit_requires_full_initialization() {
    let arena = Arena::new(2, 2, 16);
    let mut span = full(&arena, 2);
    span.push_row(&row(1.0, 2));
    span.commit();
}

#[test]
#[should_panic(expected = "capacity 0 out of range")]
fn zero_capacity_rejected() {
    let _ = Arena::new(0, 4, 16);
}
