//! Observation arena: concurrent disjoint span reservation over one packed
//! `AtomicU64` `(closed:1, row_cursor:31, alias_count:32)`; the committed prefix
//! seals into an owned `Vec<f32>` with zero copies. Protocol and invariants:
//! `docs/concepts/engine-collection.md`, "The observation arena".

use std::mem::MaybeUninit;
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

/// Final counts frozen by the closing CAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CloseInfo {
    pub rows: usize,
    pub aliases: u64,
}

pub enum Reserve<'a> {
    /// The full request; `closed` is set if it filled the buffer.
    Full {
        span: SpanGuard<'a>,
        closed: Option<CloseInfo>,
    },
    /// The remaining rows only — the caller continues on the next buffer.
    Partial {
        span: SpanGuard<'a>,
        closed: CloseInfo,
    },
    Closed,
}

/// `Closed` and `Saturated` both mean: fall back to an ordinary reservation.
pub enum AliasOutcome<'a> {
    Ticket(AliasTicket<'a>),
    Closed,
    Saturated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealError {
    NotClosed,
    CommitsPending,
    Poisoned,
    Released,
}

pub struct Arena {
    buf: NonNull<MaybeUninit<f32>>,
    // the allocation's ACTUAL capacity: from_raw_parts with anything else is UB
    cap_elems: usize,
    state: AtomicU64,
    // committed or poisoned rows; Release on write, Acquire at seal
    resolved: AtomicU64,
    poisoned: AtomicBool,
    released: AtomicBool,
    aliases_abandoned: AtomicU64,
    capacity: usize,
    dim: usize,
    alias_limit: u64,
}

// SAFETY: the buffer is mutated only through disjoint spans and read only after
// the seal check proves every span resolved.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(capacity: usize, dim: usize, alias_limit: u32) -> Self {
        assert!(
            capacity > 0 && capacity <= MAX_CAPACITY,
            "capacity {capacity} out of range"
        );
        assert!(dim > 0, "dim must be positive");
        let elems = capacity.checked_mul(dim).expect("capacity * dim overflows");
        let mut storage: Vec<MaybeUninit<f32>> = Vec::with_capacity(elems);
        let cap_elems = storage.capacity();
        let buf = NonNull::new(storage.as_mut_ptr()).expect("allocation failed");
        std::mem::forget(storage);
        Arena {
            buf,
            cap_elems,
            state: AtomicU64::new(pack(false, 0, 0)),
            resolved: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
            released: AtomicBool::new(false),
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

    /// Claim up to `rows` rows; a fill closes the buffer in the same CAS.
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

    /// Force-close; `Some` iff this call won the CAS.
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

    /// The committed prefix as an owned `Vec<f32>` (capacity slack rides along).
    pub fn into_rows(self) -> Result<Vec<f32>, (Self, SealError)> {
        match self.take_rows() {
            Ok(rows) => Ok(rows),
            Err(e) => Err((self, e)),
        }
    }

    /// Seal through a shared handle; succeeds once, after which the storage
    /// belongs to the returned `Vec` and the arena's `Drop` is inert.
    pub fn take_rows(&self) -> Result<Vec<f32>, SealError> {
        let info = self.seal_state()?;
        if self.released.swap(true, Ordering::AcqRel) {
            return Err(SealError::Released);
        }
        // SAFETY: seal_state proved rows 0..info.rows initialized (Acquire pairing
        // each commit's Release); the released CAS keeps the reclaim single.
        Ok(unsafe {
            Vec::from_raw_parts(
                self.buf.as_ptr().cast::<f32>(),
                info.rows * self.dim,
                self.cap_elems,
            )
        })
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if self.released.load(Ordering::Acquire) {
            return;
        }
        // SAFETY: rebuilds the forgotten Vec at len 0 to free the allocation.
        unsafe { drop(Vec::from_raw_parts(self.buf.as_ptr(), 0, self.cap_elems)) }
    }
}

/// RAII reservation: commits or poisons on drop, so a panic never stalls sealing.
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

    /// Initialize the span's next row by copy — no preliminary zero needed.
    pub fn push_row(&mut self, src: &[f32]) {
        assert_eq!(src.len(), self.arena.dim, "row dim mismatch");
        assert!(self.filled < self.rows, "span already full");
        let offset = (self.first_row + self.filled) * self.arena.dim;
        // SAFETY: spans are disjoint, so no other thread touches these elements.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.arena.buf.as_ptr().add(offset).cast::<f32>(),
                self.arena.dim,
            );
        }
        self.filled += 1;
    }

    /// Zero the whole span, then hand it out as `&mut [f32]`.
    pub fn zeroed(&mut self) -> &mut [f32] {
        let len = self.rows * self.arena.dim;
        let start = self.first_row * self.arena.dim;
        // SAFETY: disjoint range, zeroed before the reference exists.
        unsafe {
            let ptr = self.arena.buf.as_ptr().add(start).cast::<f32>();
            std::ptr::write_bytes(ptr, 0, len);
            self.filled = self.rows;
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

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

/// RAII alias claim: uncommitted drops count as abandonments, so `(k, m)`
/// reconciliation never waits on a notification that cannot arrive.
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
