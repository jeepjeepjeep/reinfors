//! Loom model of the reservation / close / alias CAS interleavings. Run with
//! `RUSTFLAGS="--cfg loom" cargo test -p reinfors-core --test loom_arena --release`.
#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::Arc;
use loom::thread;

use reinfors_core::rollout::arena::{AliasOutcome, Arena, Reserve};

#[test]
fn reserve_race_stays_disjoint_and_closes_once() {
    loom::model(|| {
        let arena = Arc::new(Arena::new(3, 1, 4));
        let contender = {
            let arena = arena.clone();
            thread::spawn(move || match arena.try_reserve(2) {
                Reserve::Full { mut span, closed } => {
                    let range = span.row_range();
                    span.push_row(&[range.start as f32]);
                    span.push_row(&[range.start as f32 + 1.0]);
                    span.commit();
                    (range, closed.is_some())
                }
                Reserve::Partial { mut span, .. } => {
                    let range = span.row_range();
                    span.push_row(&[range.start as f32]);
                    span.commit();
                    (range, true)
                }
                Reserve::Closed => (0..0, false),
            })
        };
        let mine = match arena.try_reserve(2) {
            Reserve::Full { mut span, closed } => {
                let range = span.row_range();
                span.push_row(&[range.start as f32]);
                span.push_row(&[range.start as f32 + 1.0]);
                span.commit();
                (range, closed.is_some())
            }
            Reserve::Partial { mut span, .. } => {
                let range = span.row_range();
                span.push_row(&[range.start as f32]);
                span.commit();
                (range, true)
            }
            Reserve::Closed => (0..0, false),
        };
        let theirs = contender.join().unwrap();
        assert!(
            mine.0.end <= theirs.0.start || theirs.0.end <= mine.0.start,
            "spans overlap: {mine:?} vs {theirs:?}"
        );
        assert_eq!(
            usize::from(mine.1) + usize::from(theirs.1),
            1,
            "exactly one close"
        );
        let arena = Arc::try_unwrap(arena).unwrap_or_else(|_| panic!("worker leaked arena"));
        let info = arena.close_info().expect("filled buffer must be closed");
        assert_eq!(info.rows, 3);
        let rows = arena.into_rows().map_err(|(_, e)| e).unwrap();
        for (i, v) in rows.iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
    });
}

#[test]
fn alias_vs_close_linearizes() {
    loom::model(|| {
        let arena = Arc::new(Arena::new(2, 1, 4));
        let closer = {
            let arena = arena.clone();
            thread::spawn(move || arena.close().map(|info| info.aliases))
        };
        let aliased = match arena.try_alias() {
            AliasOutcome::Ticket(t) => {
                t.commit();
                true
            }
            AliasOutcome::Closed => false,
            AliasOutcome::Saturated => unreachable!("limit is 4"),
        };
        let frozen = closer.join().unwrap().expect("only closer closes");
        assert_eq!(
            frozen,
            u64::from(aliased),
            "no interleaving orphans a claimant"
        );
        assert_eq!(arena.close_info().unwrap().aliases, frozen);
    });
}

#[test]
fn reserve_alias_and_close_race_three_ways() {
    loom::model(|| {
        let arena = Arc::new(Arena::new(2, 1, 4));
        let reserver = {
            let arena = arena.clone();
            thread::spawn(move || match arena.try_reserve(1) {
                Reserve::Full { mut span, .. } => {
                    span.push_row(&[7.0]);
                    span.commit();
                    true
                }
                Reserve::Partial { .. } => unreachable!("capacity 2, request 1"),
                Reserve::Closed => false,
            })
        };
        let aliaser = {
            let arena = arena.clone();
            thread::spawn(move || match arena.try_alias() {
                AliasOutcome::Ticket(t) => {
                    t.commit();
                    true
                }
                AliasOutcome::Closed => false,
                AliasOutcome::Saturated => unreachable!("limit is 4"),
            })
        };
        let frozen = arena.close().expect("only this thread closes");
        let reserved = reserver.join().unwrap();
        let aliased = aliaser.join().unwrap();
        // Losers of the close CAS must not appear in the frozen counts, winners must.
        assert_eq!(frozen.rows, usize::from(reserved));
        assert_eq!(frozen.aliases, u64::from(aliased));
        let final_info = arena.close_info().unwrap();
        assert_eq!(
            (final_info.rows, final_info.aliases),
            (frozen.rows, frozen.aliases)
        );
        let arena = Arc::try_unwrap(arena).unwrap_or_else(|_| panic!("worker leaked arena"));
        let rows = arena.into_rows().map_err(|(_, e)| e).unwrap();
        assert_eq!(rows, vec![7.0; usize::from(reserved)]);
    });
}

/// Loom cannot instrument the arena's raw buffer, so the commit/seal publication
/// is modeled directly: same shape (payload write → `Release` count → `Acquire`
/// check → read), race-checked through `loom::cell::UnsafeCell`. Weakening either
/// ordering makes this fail.
#[test]
fn commit_release_publication_model() {
    loom::model(|| {
        struct Model {
            cell: UnsafeCell<f32>,
            resolved: AtomicU64,
        }
        unsafe impl Sync for Model {}
        let m = Arc::new(Model {
            cell: UnsafeCell::new(0.0),
            resolved: AtomicU64::new(0),
        });
        let worker = {
            let m = m.clone();
            thread::spawn(move || {
                m.cell.with_mut(|p| unsafe { *p = 41.0 });
                m.resolved.fetch_add(1, Ordering::Release);
            })
        };
        while m.resolved.load(Ordering::Acquire) != 1 {
            thread::yield_now();
        }
        let seen = m.cell.with(|p| unsafe { *p });
        assert_eq!(seen, 41.0, "committed bytes must be visible at seal");
        worker.join().unwrap();
    });
}
