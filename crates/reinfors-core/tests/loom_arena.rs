//! Loom model of the reservation / close / alias CAS interleavings. Run with
//! `RUSTFLAGS="--cfg loom" cargo test -p reinfors-core --test loom_arena --release`.
#![cfg(loom)]

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
fn commit_release_visible_at_seal() {
    loom::model(|| {
        let arena = Arc::new(Arena::new(1, 2, 4));
        let worker = {
            let arena = arena.clone();
            thread::spawn(move || match arena.try_reserve(1) {
                Reserve::Full { mut span, .. } => {
                    span.push_row(&[41.0, 42.0]);
                    span.commit();
                }
                _ => unreachable!("sole worker"),
            })
        };
        worker.join().unwrap();
        let arena = Arc::try_unwrap(arena).unwrap_or_else(|_| panic!("worker leaked arena"));
        assert_eq!(
            arena.into_rows().map_err(|(_, e)| e).unwrap(),
            vec![41.0, 42.0]
        );
    });
}
