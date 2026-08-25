//! Fixed-capacity observation arena: concurrent workers reserve disjoint row spans,
//! write them in place, and commit; the sealed prefix converts to an owned `Vec<f32>`
//! with zero copies. This is the storage layer of the zero-copy scheduler proposal
//! (`docs/development/scheduler-zero-copy.md`); it contains the project's only
//! by-hand aliasing proof, so every invariant lives here, next to the `unsafe`.
//!
//! Invariants (tested independently of the engine, in `tests/arena.rs`):
//!
//! - **Bounded reservation.** Spans are claimed by CAS on a packed cursor that fails
//!   at capacity; a reservation that fills the buffer closes it in the same CAS.
//! - **Disjointness.** Every handed-out span is a strictly disjoint mutable range,
//!   guaranteed by non-overlapping ranges from the monotone cursor. Only
//!   disjointness makes the mutable references sound.
//! - **Initialize-before-reference.** Storage is uninitialized at birth. A row
//!   becomes referenced as `&mut [f32]` only after the span zeroes it
//!   ([`SpanGuard::zeroed`]) or is initialized by a full-row copy
//!   ([`SpanGuard::push_row`]). Uninitialized memory is never reachable through
//!   any reference.
//! - **Commit tracking.** Workers finish out of order; commits publish with
//!   `Release` and the sealing check `Acquire`s before any byte is read.
//! - **Atomic close, linearized with aliases.** One packed `AtomicU64` carries
//!   `(closed:1, row_cursor:31, alias_count:32)`; reservations and alias
//!   registrations both CAS it and both fail once closed. The closing CAS freezes
//!   the exact final counts.
//! - **Panic safety.** [`SpanGuard`] commits or poisons on drop; [`AliasTicket`]
//!   commits or counts an abandonment on drop. Neither can stall sealing forever.

use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::Range;
use std::ptr::NonNull;

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const CLOSED: u64 = 1 << 63;
const ROWS_SHIFT: u32 = 32;
const ROWS_MASK: u64 = 0x7FFF_FFFF;
const ALIAS_MASK: u64 = 0xFFFF_FFFF;
/// Row capacity must fit the 31-bit cursor field.
pub const MAX_CAPACITY: usize = ROWS_MASK as usize;

fn pack(closed: bool, rows: usize, aliases: u64) -> u64 {
    (if closed { CLOSED } else { 0 }) | ((rows as u64) << ROWS_SHIFT) | aliases
}

fn unpack(word: u64) -> (bool, usize, u64) {
    (
        word & CLOSED != 0,
        ((word >> ROWS_SHIFT) & ROWS_MASK) as usize,
        word & ALIAS_MASK,
    )
}

/// Final counts frozen by the closing CAS: sealing waits for exactly `rows` commit
/// resolutions, and the scheduler reconciles exactly `aliases` alias outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CloseInfo {
    pub rows: usize,
    pub aliases: u64,
}

/// Outcome of [`Arena::try_reserve`].
pub enum Reserve<'a> {
    /// The full request. `closed` is set if this reservation filled the buffer.
    Full {
        span: SpanGuard<'a>,
        closed: Option<CloseInfo>,
    },
    /// The remaining rows only; the reservation filled and closed the buffer.
    /// The caller re-requests the rest against the next buffer.
    Partial {
        span: SpanGuard<'a>,
        closed: CloseInfo,
    },
    /// Buffer already closed; nothing reserved.
    Closed,
}

/// Outcome of [`Arena::try_alias`]. `Closed` and `Saturated` both mean: fall back
/// to an ordinary reservation (a duplicate row rides to inference — compute cost,
/// not correctness).
pub enum AliasOutcome<'a> {
    Ticket(AliasTicket<'a>),
    Closed,
    Saturated,
}

/// Why [`Arena::into_rows`] is not yet possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealError {
    NotClosed,
    CommitsPending,
    Poisoned,
}

pub struct Arena {
    buf: NonNull<MaybeUninit<f32>>,
    elems: usize,
    state: AtomicU64,
    /// Rows resolved (committed or poisoned), `Release` on write, `Acquire` at seal.
    resolved: AtomicU64,
    poisoned: AtomicBool,
    aliases_abandoned: AtomicU64,
    capacity: usize,
    dim: usize,
    alias_limit: u64,
}

// SAFETY: the raw buffer is only mutated through disjoint spans handed out by the
// CAS protocol below, and only read after the Acquire-ordered seal check proves
// every span resolved. All other fields are atomics.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// `alias_limit` bounds the alias count below its 32-bit field; the caller sizes
    /// it to the staged map's fixed capacity.
    pub fn new(capacity: usize, dim: usize, alias_limit: u32) -> Self {
        assert!(
            capacity > 0 && capacity <= MAX_CAPACITY,
            "capacity {capacity} out of range"
        );
        assert!(dim > 0, "dim must be positive");
        let elems = capacity.checked_mul(dim).expect("capacity * dim overflows");
        let mut storage: Vec<MaybeUninit<f32>> = Vec::with_capacity(elems);
        let buf = NonNull::new(storage.as_mut_ptr()).expect("allocation failed");
        std::mem::forget(storage);
        Arena {
            buf,
            elems,
            state: AtomicU64::new(pack(false, 0, 0)),
            resolved: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
            aliases_abandoned: AtomicU64::new(0),
            capacity,
            dim,
            alias_limit: u64::from(alias_limit),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Claim up to `rows` rows. Fails once closed; fills-and-closes in one CAS when
    /// the request reaches capacity.
    pub fn try_reserve(&self, rows: usize) -> Reserve<'_> {
        assert!(rows > 0, "reserve zero rows");
        let mut cur = self.state.load(Ordering::Acquire);
        loop {
            let (closed, taken, aliases) = unpack(cur);
            if closed {
                return Reserve::Closed;
            }
            let remaining = self.capacity - taken;
            let take = rows.min(remaining);
            let fills = take == remaining;
            let next = pack(fills, taken + take, aliases);
            match self
                .state
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    let span = SpanGuard {
                        arena: self,
                        first_row: taken,
                        rows: take,
                        filled: 0,
                        committed: false,
                    };
                    let info = CloseInfo {
                        rows: taken + take,
                        aliases,
                    };
                    return if take == rows {
                        Reserve::Full {
                            span,
                            closed: fills.then_some(info),
                        }
                    } else {
                        Reserve::Partial { span, closed: info }
                    };
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Register an alias claim against this buffer's fire. Fails once closed or at
    /// `alias_limit`; every successful CAS is guaranteed a routing entry by the
    /// close linearization.
    pub fn try_alias(&self) -> AliasOutcome<'_> {
        let mut cur = self.state.load(Ordering::Acquire);
        loop {
            let (closed, rows, aliases) = unpack(cur);
            if closed {
                return AliasOutcome::Closed;
            }
            if aliases >= self.alias_limit {
                return AliasOutcome::Saturated;
            }
            let next = pack(false, rows, aliases + 1);
            match self
                .state
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    return AliasOutcome::Ticket(AliasTicket {
                        arena: self,
                        done: false,
                    })
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Force-close (scheduler ladder close: stall or quiescence). Returns the frozen
    /// counts if this call won the CAS, `None` if already closed.
    pub fn close(&self) -> Option<CloseInfo> {
        let mut cur = self.state.load(Ordering::Acquire);
        loop {
            let (closed, rows, aliases) = unpack(cur);
            if closed {
                return None;
            }
            let next = pack(true, rows, aliases);
            match self
                .state
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some(CloseInfo { rows, aliases }),
                Err(actual) => cur = actual,
            }
        }
    }

    /// Frozen counts, if closed.
    pub fn close_info(&self) -> Option<CloseInfo> {
        let (closed, rows, aliases) = unpack(self.state.load(Ordering::Acquire));
        closed.then_some(CloseInfo { rows, aliases })
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    pub fn aliases_abandoned(&self) -> u64 {
        self.aliases_abandoned.load(Ordering::Acquire)
    }

    /// Whether the sealed prefix is ready: closed, every reserved row resolved, no
    /// poison. The `Acquire` here pairs with each commit's `Release`.
    pub fn seal_state(&self) -> Result<CloseInfo, SealError> {
        let info = self.close_info().ok_or(SealError::NotClosed)?;
        if self.resolved.load(Ordering::Acquire) != info.rows as u64 {
            return Err(SealError::CommitsPending);
        }
        if self.is_poisoned() {
            return Err(SealError::Poisoned);
        }
        Ok(info)
    }

    /// Convert the committed prefix into an owned `Vec<f32>` (len `rows * dim`,
    /// capacity the full arena — the slack rides along until Python frees it).
    pub fn into_rows(self) -> Result<Vec<f32>, (Self, SealError)> {
        let info = match self.seal_state() {
            Ok(info) => info,
            Err(e) => return Err((self, e)),
        };
        let this = ManuallyDrop::new(self);
        // SAFETY: `seal_state` proved every reserved span committed with Release
        // ordering that the check Acquired, so rows 0..info.rows are initialized
        // f32s; `MaybeUninit<f32>` and `f32` share layout; the pointer and full
        // element capacity come from the original allocation, reclaimed exactly
        // once because `ManuallyDrop` suppresses `Arena::drop`.
        Ok(unsafe {
            Vec::from_raw_parts(
                this.buf.as_ptr().cast::<f32>(),
                info.rows * this.dim,
                this.elems,
            )
        })
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: reconstructs the forgotten Vec with len 0 (contents may be
        // uninitialized; f32 needs no drop) to release the allocation.
        unsafe { drop(Vec::from_raw_parts(self.buf.as_ptr(), 0, self.elems)) }
    }
}

/// RAII reservation: commits or poisons on drop. A worker panicking after reserving
/// must not leave an uncommitted hole that stalls sealing forever.
pub struct SpanGuard<'a> {
    arena: &'a Arena,
    first_row: usize,
    rows: usize,
    filled: usize,
    committed: bool,
}

impl SpanGuard<'_> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_range(&self) -> Range<usize> {
        self.first_row..self.first_row + self.rows
    }

    /// Stage-1 path: initialize the next row of the span by copying a complete
    /// source row. No preliminary zero — the copy itself initializes.
    pub fn push_row(&mut self, src: &[f32]) {
        assert_eq!(src.len(), self.arena.dim, "row dim mismatch");
        assert!(self.filled < self.rows, "span already full");
        let offset = (self.first_row + self.filled) * self.arena.dim;
        // SAFETY: offset addresses this span's next row; spans are disjoint by the
        // reservation arithmetic, so no other thread touches these elements.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.arena.buf.as_ptr().add(offset).cast::<f32>(),
                self.arena.dim,
            );
        }
        self.filled += 1;
    }

    /// Stage-2 path: zero the whole span, then hand it out as `&mut [f32]`.
    /// The sequence is reserve → zero → only now construct the reference — an
    /// encoder that underwrites produces zeros, not undefined behavior.
    pub fn zeroed(&mut self) -> &mut [f32] {
        let len = self.rows * self.arena.dim;
        let start = self.first_row * self.arena.dim;
        // SAFETY: this span's range is disjoint from every other span and from the
        // sealed prefix (seal waits for this guard to resolve); zeroing initializes
        // every byte before the reference exists.
        unsafe {
            let ptr = self.arena.buf.as_ptr().add(start).cast::<f32>();
            std::ptr::write_bytes(ptr, 0, len);
            self.filled = self.rows;
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Publish the span. Requires every row initialized.
    pub fn commit(mut self) {
        assert_eq!(self.filled, self.rows, "commit with uninitialized rows");
        self.committed = true;
        self.arena
            .resolved
            .fetch_add(self.rows as u64, Ordering::Release);
    }
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.arena.poisoned.store(true, Ordering::Release);
            self.arena
                .resolved
                .fetch_add(self.rows as u64, Ordering::Release);
        }
    }
}

/// RAII alias claim: commit when the claimant's metadata message is delivered, or
/// count an abandonment on drop so `(k, m)` reconciliation never waits on a
/// notification that cannot arrive.
pub struct AliasTicket<'a> {
    arena: &'a Arena,
    done: bool,
}

impl AliasTicket<'_> {
    pub fn commit(mut self) {
        self.done = true;
    }
}

impl Drop for AliasTicket<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.arena.aliases_abandoned.fetch_add(1, Ordering::Release);
        }
    }
}
