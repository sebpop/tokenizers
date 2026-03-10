use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

/// Spin-wait barrier that stays entirely in userspace.
///
/// `std::sync::Barrier` uses `Mutex` + `Condvar` which call into the
/// kernel via `futex_wait`/`futex_wake`.  Each futex syscall pulls
/// dozens of kernel functions (`futex_hash`, `queued_spin_lock`,
/// `futex_q_lock`, `finish_task_switch`, …) into the L1 instruction
/// cache, evicting hot tokenizer code.  At 88 threads the kernel
/// futex path accounts for **34%** of all L1i cache misses.
///
/// This spin barrier replaces futex with a generation-based atomic
/// counter (~20 instructions total), keeping synchronization
/// entirely in userspace and eliminating the kernel icache pollution.
#[repr(align(128))]
struct SpinBarrier {
    count: AtomicUsize,
    generation: AtomicUsize,
    total: usize,
}

impl SpinBarrier {
    fn new(total: usize) -> Self {
        Self {
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            total,
        }
    }

    /// Spin briefly, then yield.  Pure spinning gives +29% at 88
    /// threads (1 thread per physical core) but regresses -72% at 176
    /// threads (SMT) because a spinning hyperthread steals execution
    /// resources from its sibling doing real work.  Yielding after a
    /// short spin window lets the OS scheduler run the sibling.
    #[inline]
    fn wait(&self) {
        let gen = self.generation.load(Ordering::Relaxed);
        if self.count.fetch_add(1, Ordering::AcqRel) == self.total - 1 {
            // Last thread to arrive: reset count and bump generation.
            self.count.store(0, Ordering::Relaxed);
            self.generation.store(gen.wrapping_add(1), Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.generation.load(Ordering::Acquire) == gen {
                if spins < 64 {
                    core::hint::spin_loop();
                    spins += 1;
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

/// Persistent thread pool with spin-barrier dispatch.
///
/// Unlike rayon, this pool uses no work-stealing deques and no
/// epoch-based garbage collection.  On high-core-count systems
/// (e.g. 88+ physical cores) rayon's crossbeam-epoch
/// `Global::try_advance` walks every thread's epoch record on
/// foreign pages, causing a TLB miss storm that dominates
/// scaling.  This pool avoids that entirely: threads park on a
/// spin barrier and are woken to run a caller-supplied closure.
struct Inner {
    start: SpinBarrier,
    done: SpinBarrier,
    shutdown: AtomicBool,
    /// Trait-object fat pointer stored as two `usize` words
    /// (data pointer + vtable pointer).  Written by `broadcast()`
    /// before `start.wait()`, read by workers after `start.wait()`.
    /// Barriers provide the necessary synchronization.
    work: UnsafeCell<[usize; 2]>,
}

// Safety: `work` is only accessed under barrier synchronization:
// the dispatcher writes before start.wait(), workers read between
// start.wait() and done.wait(), dispatcher reads nothing after.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct WorkPool {
    num_threads: usize,
    inner: Arc<Inner>,
    handles: Vec<JoinHandle<()>>,
}

impl WorkPool {
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0);

        let inner = Arc::new(Inner {
            start: SpinBarrier::new(num_threads + 1),
            done: SpinBarrier::new(num_threads + 1),
            shutdown: AtomicBool::new(false),
            work: UnsafeCell::new([0; 2]),
        });

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let inner = Arc::clone(&inner);
                thread::spawn(move || {
                    loop {
                        inner.start.wait();
                        if inner.shutdown.load(Ordering::Relaxed) {
                            return;
                        }
                        // Safety: broadcast() stored a valid fat pointer before
                        // start.wait(), and the referent lives until done.wait().
                        let work: &dyn Fn(usize) = unsafe {
                            let raw = *inner.work.get();
                            std::mem::transmute::<[usize; 2], &dyn Fn(usize)>(raw)
                        };
                        work(tid);
                        inner.done.wait();
                    }
                })
            })
            .collect();

        Self {
            num_threads,
            inner,
            handles,
        }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Run `f(thread_index)` on every thread in the pool and block
    /// until all threads complete.
    pub fn broadcast<F: Fn(usize) + Sync>(&self, f: F) {
        // Safety: `f` lives on our stack frame, which outlives the
        // worker access because we block on `done.wait()` below.
        unsafe {
            let trait_obj: &dyn Fn(usize) = &f;
            let raw = std::mem::transmute::<&dyn Fn(usize), [usize; 2]>(trait_obj);
            *self.inner.work.get() = raw;
        }
        self.inner.start.wait();
        self.inner.done.wait();
    }
}

impl Drop for WorkPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        self.inner.start.wait();
        for h in self.handles.drain(..) {
            h.join().unwrap();
        }
    }
}

static GLOBAL_POOL: OnceLock<WorkPool> = OnceLock::new();

/// Returns a reference to the process-wide persistent thread pool.
///
/// The pool is created lazily on first use.  Thread count is
/// determined by (in priority order):
/// 1. `RAYON_NUM_THREADS` env var (for backward compatibility)
/// 2. `std::thread::available_parallelism()` (number of CPUs)
pub fn global_pool() -> &'static WorkPool {
    GLOBAL_POOL.get_or_init(|| {
        let n = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        WorkPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_broadcast_runs_all_threads() {
        let pool = WorkPool::new(4);
        let counter = AtomicUsize::new(0);
        pool.broadcast(|_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_thread_indices_are_stable() {
        let pool = WorkPool::new(4);
        let seen = [
            AtomicBool::new(false),
            AtomicBool::new(false),
            AtomicBool::new(false),
            AtomicBool::new(false),
        ];
        pool.broadcast(|tid| {
            seen[tid].store(true, Ordering::Relaxed);
        });
        for s in &seen {
            assert!(s.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn test_multiple_dispatches() {
        let pool = WorkPool::new(4);
        let counter = AtomicUsize::new(0);
        for _ in 0..10 {
            pool.broadcast(|_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(counter.load(Ordering::Relaxed), 40);
    }
}
