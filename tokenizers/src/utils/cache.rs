use ahash::AHashMap;
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// The default capacity for a `BPE`'s internal cache.
pub static DEFAULT_CACHE_CAPACITY: usize = 10_000;
/// Merge thread-local inserts into the global cache after this many buffered entries.
pub static DEFAULT_MERGE_AFTER_INSERTS: usize = 256;
/// Number of shards to reduce lock contention; reads/writes spread across shards by key hash.
const NUM_SHARDS: usize = 64;
/// The maximum length we should cache in a model
/// Strings that are too long have minimal chances to cache hit anyway
pub static MAX_LENGTH: usize = 256;

fn shard_index<Q: Hash + ?Sized>(key: &Q) -> usize {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    (h.finish() as usize) % NUM_SHARDS
}

/// Provides a simple multithread cache to speed up BPE tokenization that will try to read values
/// concurrently but won't block if another thread is writing.
/// The cache is sharded so different keys use different locks, improving scalability.
/// Inserts are buffered per-thread and merged into the global map after
/// `merge_after_inserts` entries or on explicit flush to reduce lock contention.
#[derive(Debug)]
pub(crate) struct Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    shards: Vec<RwLock<AHashMap<K, V>>>,
    pub capacity: usize,
    pub merge_after_inserts: usize,
}

// We dont really care about Cache comparison, so let's make them always equal
impl<K, V> PartialEq for Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn eq(&self, _other: &Cache<K, V>) -> bool {
        true
    }
}

impl<K, V> Default for Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_CAPACITY)
    }
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create new `Cache` with the given capacity.
    pub(crate) fn new(capacity: usize) -> Self {
        let cap_per_shard = (capacity + NUM_SHARDS - 1) / NUM_SHARDS;
        let shards = (0..NUM_SHARDS)
            .map(|_| RwLock::new(AHashMap::with_capacity(cap_per_shard)))
            .collect();
        Cache {
            shards,
            capacity,
            merge_after_inserts: DEFAULT_MERGE_AFTER_INSERTS,
        }
    }

    /// Create a fresh `Cache` with the same configuration.
    pub(crate) fn fresh(&self) -> Self {
        let cap_per_shard = (self.capacity + NUM_SHARDS - 1) / NUM_SHARDS;
        let shards = (0..NUM_SHARDS)
            .map(|_| RwLock::new(AHashMap::with_capacity(cap_per_shard)))
            .collect();
        Cache {
            shards,
            capacity: self.capacity,
            merge_after_inserts: self.merge_after_inserts,
        }
    }

    /// Merge buffer into global cache (per-shard) and clear the buffer.
    pub(crate) fn flush_buf(&self, buf: &mut AHashMap<K, V>) {
        if buf.is_empty() {
            return;
        }
        let cap_per_shard = (self.capacity + NUM_SHARDS - 1) / NUM_SHARDS;
        // Group by shard to take each lock once.
        let mut by_shard: Vec<Vec<(K, V)>> = (0..NUM_SHARDS).map(|_| Vec::new()).collect();
        for (k, v) in buf.drain() {
            let i = shard_index(&k);
            if by_shard[i].len() < cap_per_shard {
                by_shard[i].push((k, v));
            }
        }
        for (i, entries) in by_shard.into_iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            if let Ok(mut shard) = self.shards[i].try_write() {
                let free = cap_per_shard.saturating_sub(shard.len());
                if free > 0 {
                    shard.extend(entries.into_iter().take(free));
                }
            }
        }
    }

    /// Clear the cache.
    pub(crate) fn clear(&self) {
        for shard in &self.shards {
            shard.write().unwrap().clear();
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get_values<'a, I, Q>(&self, keys_iter: I) -> Option<Vec<Option<V>>>
    where
        I: Iterator<Item = &'a Q>,
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'a,
    {
        let mut out = Vec::new();
        for k in keys_iter {
            let i = shard_index(k);
            if let Ok(shard) = self.shards[i].try_read() {
                out.push(shard.get(k).cloned());
            } else {
                out.push(None);
            }
        }
        Some(out)
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let i = shard_index(key);
        if let Ok(shard) = self.shards[i].try_read() {
            return shard.get(key).cloned();
        }
        None
    }

    pub(crate) fn set_values<I>(&self, entries: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let cap_per_shard = (self.capacity + NUM_SHARDS - 1) / NUM_SHARDS;
        for (k, v) in entries {
            let i = shard_index(&k);
            if let Ok(mut shard) = self.shards[i].try_write() {
                if shard.len() < cap_per_shard {
                    shard.insert(k, v);
                }
            }
        }
    }

    pub(crate) fn set(&self, key: K, value: V) {
        self.set_values(std::iter::once((key, value)))
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
        let cap_per_shard = (capacity + NUM_SHARDS - 1) / NUM_SHARDS;
        for shard in &self.shards {
            if let Ok(mut s) = shard.try_write() {
                s.shrink_to(cap_per_shard);
            }
        }
    }
}
