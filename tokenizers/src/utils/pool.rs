use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, OnceLock};
use std::thread::{self, JoinHandle};

/// Persistent thread pool with barrier-based dispatch.
///
/// Unlike rayon, this pool uses no work-stealing deques and no
/// epoch-based garbage collection.  On high-core-count systems
/// (e.g. 88+ physical cores) rayon's crossbeam-epoch
/// `Global::try_advance` walks every thread's epoch record on
/// foreign pages, causing a TLB miss storm that dominates
/// scaling.  This pool avoids that entirely: threads park on a
/// barrier and are woken to run a caller-supplied closure.
struct Inner {
    start: Barrier,
    done: Barrier,
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
            start: Barrier::new(num_threads + 1),
            done: Barrier::new(num_threads + 1),
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
/// The pool is created lazily on first use, with one thread per
/// available CPU core.
pub fn global_pool() -> &'static WorkPool {
    GLOBAL_POOL.get_or_init(|| {
        let n = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
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
