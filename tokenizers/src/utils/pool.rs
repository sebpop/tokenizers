use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

/// Detect SMT at process init by checking CPU topology.
/// On Linux: reads thread_siblings_list for CPU 0.
/// Returns true if the CPU has SMT siblings (e.g., Vera: "0-1").
/// Returns false if no SMT (e.g., Grace: "0", or non-Linux).
fn detect_smt() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string(
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
        )
        .map(|s| s.trim().contains('-') || s.trim().contains(','))
        .unwrap_or(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

static HAS_SMT: OnceLock<bool> = OnceLock::new();

#[inline(always)]
fn has_smt() -> bool {
    *HAS_SMT.get().unwrap_or(&false)
}

/// Spin-wait barrier that stays entirely in userspace.
///
/// Uses ISB (spin_loop) for the fast path, then selects the slow path
/// based on SMT detection at process init:
///
/// - **No SMT** (Grace, Graviton): pure ISB spin.  No overhead from
///   WFI or sched_yield.
///
/// - **SMT** (Vera): WFI in the slow path.  On aarch64 Linux with
///   SMT (e.g., Vera), the ntwi_yield kernel module is required to
///   yield execution resources of a shared physical core to the
///   sibling thread that is not blocked on the spin_loop.  The module
///   traps WFI from EL0, sets TIF_NEED_RESCHED, and the kernel
///   return-to-EL0 path calls schedule() to yield to the sibling.
///   If the kernel module is NOT loaded, WFI executes natively as an
///   86 ns idle cycle without yielding SMT resources, causing severe
///   performance degradation at thread counts above the physical core
///   count.  Measured on Vera (88 physical cores, 176 logical):
///     - 176t with module:    11.40 GiB/s (full SMT utilization)
///     - 176t without module:  2.06 GiB/s (5.5x slower, SMT starved)
///     - 88t (no SMT needed): 11.26 GiB/s (unaffected)
///
/// - **x86_64 / macOS**: `core::hint::spin_loop()` which emits
///   `pause` (x86) or `isb` (aarch64-mac).
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
                } else if has_smt() {
                    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
                    unsafe {
                        core::arch::asm!("wfi", options(nomem, nostack));
                    }
                    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
                    core::hint::spin_loop();
                } else {
                    core::hint::spin_loop();
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
        HAS_SMT.get_or_init(detect_smt);

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

    #[test]
    fn test_smt_detection() {
        let result = detect_smt();
        println!("SMT detected: {result}");
    }
}
