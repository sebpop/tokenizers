//! Lock-free batch work distribution with dynamic window sizing.
//!
//! Replaces rayon's parallel iteration for batch encode with a simpler
//! mechanism: an atomic counter distributes work in small windows to worker
//! threads running on rayon's persistent thread pool.
//!
//! The design avoids rayon's recursive `bridge_producer_consumer` splitting,
//! which shows up in profiles as a significant synchronization cost at higher
//! thread counts.
//!
//! Cache-line isolation: the work counter, each input slot, and each result
//! slot are accessed by at most one thread at a time, so there is no false
//! sharing.  The counter is aligned to a full cache line.
//!
//! Window sizing: the window size is computed dynamically to ensure at least
//! `WINDOWS_PER_THREAD` (4) windows per thread for good load balancing,
//! capped at `MAX_WINDOW_SIZE` (8) to limit the cost of each atomic
//! fetch_add.  For example, 100 items / 16 threads yields window_size=2
//! (50 windows), whereas 10000 items / 16 threads yields window_size=8.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Minimum number of windows each thread should get for load balancing.
const WINDOWS_PER_THREAD: usize = 4;

/// Maximum window size (items per atomic claim).  Larger values reduce
/// atomic contention but worsen tail-latency from uneven last windows.
const MAX_WINDOW_SIZE: usize = 8;

/// Cache-line-aligned atomic counter.
/// Ensures the counter does not share a cache line with any other data.
#[repr(C, align(64))]
struct AlignedCounter(AtomicUsize);

/// Lock-free work distributor.
///
/// Workers atomically claim non-overlapping windows of item indices.
/// The window size is chosen dynamically based on `total` and
/// `num_threads` so that every thread gets several windows of work.
/// The counter is on its own cache line so claiming work does not
/// contend with result writes.
pub struct BatchWorkQueue {
    next: AlignedCounter,
    total: usize,
    window_size: usize,
}

impl BatchWorkQueue {
    /// Create a new queue distributing `total` items across `num_threads`.
    ///
    /// The window size is chosen to give each thread at least
    /// `WINDOWS_PER_THREAD` windows, capped at `MAX_WINDOW_SIZE`.
    pub fn new(total: usize, num_threads: usize) -> Self {
        let target_windows = num_threads.saturating_mul(WINDOWS_PER_THREAD).max(1);
        let window_size = ((total + target_windows - 1) / target_windows)
            .max(1)
            .min(MAX_WINDOW_SIZE);
        Self {
            next: AlignedCounter(AtomicUsize::new(0)),
            total,
            window_size,
        }
    }

    /// Claim the next window of work items.
    /// Returns `Some((start, end))` half-open range, or `None` when all
    /// items have been claimed.
    pub fn claim_window(&self) -> Option<(usize, usize)> {
        let start = self.next.0.fetch_add(self.window_size, Ordering::Relaxed);
        if start >= self.total {
            return None;
        }
        Some((start, (start + self.window_size).min(self.total)))
    }
}

/// A `Vec` whose elements can each be *taken* exactly once from any thread.
///
/// The `BatchWorkQueue` guarantees that no two threads access the same index,
/// so no synchronization is needed beyond the queue itself.
pub struct TakeVec<T>(UnsafeCell<Vec<Option<T>>>);

// Safety: each index is accessed by exactly one thread at a time
// because `BatchWorkQueue::claim_window` returns non-overlapping ranges.
unsafe impl<T: Send> Sync for TakeVec<T> {}

impl<T> TakeVec<T> {
    /// Wrap a `Vec<T>` so items can be taken by index.
    pub fn new(items: Vec<T>) -> Self {
        Self(UnsafeCell::new(items.into_iter().map(Some).collect()))
    }

    /// Take the item at `index`, leaving `None` in its place.
    /// Panics if the item was already taken.
    pub fn take(&self, index: usize) -> T {
        // Safety: each index is only accessed by one thread.
        unsafe {
            (&mut (*self.0.get()))[index]
                .take()
                .expect("batch item already taken")
        }
    }
}

/// A `Vec<Option<T>>` where each slot is written exactly once from any thread.
///
/// The `BatchWorkQueue` guarantees non-overlapping index access.
pub struct ResultVec<T>(UnsafeCell<Vec<Option<T>>>);

// Safety: each index is written by exactly one thread at a time.
unsafe impl<T: Send> Sync for ResultVec<T> {}

impl<T> ResultVec<T> {
    /// Allocate `len` empty result slots.
    pub fn new(len: usize) -> Self {
        Self(UnsafeCell::new((0..len).map(|_| None).collect()))
    }

    /// Write a result to the slot at `index`.
    pub fn set(&self, index: usize, value: T) {
        // Safety: each index is only written by one thread.
        unsafe {
            (&mut (*self.0.get()))[index] = Some(value);
        }
    }

    /// Consume self and return the results in order.
    /// Panics if any slot was not written.
    pub fn into_vec(self) -> Vec<T> {
        self.0
            .into_inner()
            .into_iter()
            .map(|o| o.expect("result slot was never written"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_work_queue_single_thread() {
        // 20 items, 1 thread => target 4 windows => window_size = 5.
        let queue = BatchWorkQueue::new(20, 1);
        let mut ranges = Vec::new();
        while let Some(range) = queue.claim_window() {
            ranges.push(range);
        }
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], (0, 5));
        assert_eq!(ranges[1], (5, 10));
        assert_eq!(ranges[2], (10, 15));
        assert_eq!(ranges[3], (15, 20));
    }

    #[test]
    fn test_batch_work_queue_many_threads() {
        // 100 items, 16 threads => target 64 windows => window_size = 2.
        let queue = BatchWorkQueue::new(100, 16);
        let mut ranges = Vec::new();
        while let Some(range) = queue.claim_window() {
            ranges.push(range);
        }
        assert_eq!(ranges.len(), 50);
        assert_eq!(ranges[0], (0, 2));
        assert_eq!(ranges[49], (98, 100));
    }

    #[test]
    fn test_batch_work_queue_window_capped() {
        // 10000 items, 4 threads => target 16 windows => window_size = 625,
        // but capped at MAX_WINDOW_SIZE (8).
        let queue = BatchWorkQueue::new(10000, 4);
        let mut count = 0;
        while queue.claim_window().is_some() {
            count += 1;
        }
        // 10000 / 8 = 1250 windows.
        assert_eq!(count, 1250);
    }

    #[test]
    fn test_batch_work_queue_empty() {
        let queue = BatchWorkQueue::new(0, 4);
        assert!(queue.claim_window().is_none());
    }

    #[test]
    fn test_take_vec() {
        let tv = TakeVec::new(vec![10, 20, 30]);
        assert_eq!(tv.take(1), 20);
        assert_eq!(tv.take(0), 10);
        assert_eq!(tv.take(2), 30);
    }

    #[test]
    fn test_result_vec() {
        let rv = ResultVec::<i32>::new(3);
        rv.set(2, 30);
        rv.set(0, 10);
        rv.set(1, 20);
        assert_eq!(rv.into_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn test_parallel_distribution() {
        let n = 100;
        let num_threads = 4;
        let queue = BatchWorkQueue::new(n, num_threads);
        let results = ResultVec::new(n);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|| {
                    while let Some((start, end)) = queue.claim_window() {
                        for i in start..end {
                            results.set(i, i * 2);
                        }
                    }
                });
            }
        });

        let v = results.into_vec();
        for i in 0..n {
            assert_eq!(v[i], i * 2);
        }
    }
}
