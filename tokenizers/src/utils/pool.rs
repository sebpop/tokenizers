use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

/// Count physical cores by reading sysfs topology.
/// Returns (physical_cores, has_smt).
fn detect_topology() -> (usize, bool) {
    #[cfg(target_os = "linux")]
    {
        let has_smt = std::fs::read_to_string(
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
        )
        .map(|s| s.trim().contains('-') || s.trim().contains(','))
        .unwrap_or(false);

        // Count physical cores from unique core_id values across all CPUs.
        let physical = if has_smt {
            // With SMT, physical cores = online CPUs / siblings per core.
            // Read siblings count from the first core's thread_siblings_list.
            let siblings = std::fs::read_to_string(
                "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
            )
            .ok()
            .and_then(|s| {
                let parts: Vec<&str> = s.trim().split(&['-', ','][..]).collect();
                if parts.len() >= 2 {
                    let lo: usize = parts[0].parse().ok()?;
                    let hi: usize = parts[parts.len() - 1].parse().ok()?;
                    Some(hi - lo + 1)
                } else {
                    Some(1)
                }
            })
            .unwrap_or(2);
            thread::available_parallelism().map(|n| n.get()).unwrap_or(1) / siblings
        } else {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        };

        (physical, has_smt)
    }
    #[cfg(not(target_os = "linux"))]
    {
        (thread::available_parallelism().map(|n| n.get()).unwrap_or(1), false)
    }
}

static TOPOLOGY: OnceLock<(usize, bool)> = OnceLock::new();

fn get_topology() -> (usize, bool) {
    *TOPOLOGY.get_or_init(detect_topology)
}

/// Spin-wait barrier that stays entirely in userspace.
///
/// Uses ISB (spin_loop) for the fast path.  The slow path depends on
/// whether the pool is oversubscribed (more threads than physical cores):
///
/// - **Not oversubscribed** (e.g., 88 threads on 88 physical cores):
///   pure ISB spin.  No kernel entry, no WFI trap overhead.
///
/// - **Oversubscribed** (e.g., 176 threads on 88 physical cores):
///   WFI on aarch64 Linux.  The ntwi_yield kernel module traps WFI
///   from EL0, sets TIF_NEED_RESCHED, and yields to the SMT sibling.
///   Without the module, WFI is a ~86ns idle cycle that does NOT yield.
///
/// - **x86_64 / macOS**: `core::hint::spin_loop()` (pause / isb).
#[repr(align(128))]
struct SpinBarrier {
    count: AtomicUsize,
    generation: AtomicUsize,
    total: usize,
    /// True when pool threads > physical cores (SMT oversubscribed).
    use_wfi: bool,
}

impl SpinBarrier {
    fn new(total: usize) -> Self {
        let (physical_cores, has_smt) = get_topology();
        Self {
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            total,
            use_wfi: has_smt && total > physical_cores,
        }
    }

    #[inline]
    fn wait(&self) {
        let gen = self.generation.load(Ordering::Relaxed);
        if self.count.fetch_add(1, Ordering::AcqRel) == self.total - 1 {
            self.count.store(0, Ordering::Relaxed);
            self.generation.store(gen.wrapping_add(1), Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.generation.load(Ordering::Acquire) == gen {
                if spins < 64 {
                    core::hint::spin_loop();
                    spins += 1;
                } else if self.use_wfi {
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
        // Ensure topology is detected before constructing barriers.
        TOPOLOGY.get_or_init(detect_topology);

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
    fn test_topology_detection() {
        let (physical, has_smt) = detect_topology();
        println!("physical_cores: {physical}, has_smt: {has_smt}");
        assert!(physical > 0);
    }
}
