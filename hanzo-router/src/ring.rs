//! Consistent-hash ring: a stable key -> ordered replica placement. Pure over
//! its member set. Adding or removing a replica moves only ~1/N of keys (the
//! invariant prefix-affinity needs, so a conversation keeps landing on the same
//! replica as the pool grows or shrinks). No hardware or clock access — a value.

const VNODES: u32 = 160;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const MAX_MEMBERS: usize = 128;

// FNV-1a then the murmur3 fmix64 finalizer: FNV alone clusters on the near-
// identical vnode keys (`r0#0`, `r0#1`, ...); fmix64 avalanches them so slots
// and keys spread evenly. Deterministic, std-only.
fn hash(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

/// A ring of `members` (replica ids), each placed at [`VNODES`] positions so
/// keys spread evenly and membership changes are local.
#[derive(Debug, Default, Clone)]
pub struct Ring {
    members: Vec<String>,
    slots: Vec<(u64, usize)>,
}

impl Ring {
    /// Build the ring. Panics above [`MAX_MEMBERS`] replicas (the placement-order
    /// iterator tracks visited members in a `u128` bitset); no real model fans out
    /// to that many engine instances.
    pub fn new(members: impl IntoIterator<Item = String>) -> Self {
        let members: Vec<String> = members.into_iter().collect();
        assert!(members.len() <= MAX_MEMBERS, "ring: >{MAX_MEMBERS} members");
        let mut slots = Vec::with_capacity(members.len() * VNODES as usize);
        for (i, m) in members.iter().enumerate() {
            for v in 0..VNODES {
                slots.push((hash(format!("{m}#{v}").as_bytes()), i));
            }
        }
        slots.sort_unstable_by_key(|&(h, _)| h);
        Self { members, slots }
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Replica ids in placement order for `key`: the slot owner first, then
    /// successors around the ring, each distinct id once. This single order gives
    /// both affinity (a stable primary) and spill (walk the successors).
    pub fn route<'a>(&'a self, key: &str) -> RouteIter<'a> {
        let h = hash(key.as_bytes());
        let start = self.slots.partition_point(|&(sh, _)| sh < h);
        RouteIter {
            ring: self,
            pos: if start == self.slots.len() { 0 } else { start },
            steps: 0,
            seen: 0,
            yielded: 0,
        }
    }

    /// The primary owner of `key` (first in [`Ring::route`]).
    pub fn primary(&self, key: &str) -> Option<&str> {
        self.route(key).next()
    }
}

/// Lazy placement-order walk over distinct members (see [`Ring::route`]).
pub struct RouteIter<'a> {
    ring: &'a Ring,
    pos: usize,
    steps: usize,
    seen: u128,
    yielded: usize,
}

impl<'a> Iterator for RouteIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let slots = &self.ring.slots;
        while self.yielded < self.ring.members.len() && self.steps < slots.len() {
            let (_, mi) = slots[self.pos % slots.len()];
            self.pos += 1;
            self.steps += 1;
            let bit = 1u128 << mi;
            if self.seen & bit == 0 {
                self.seen |= bit;
                self.yielded += 1;
                return Some(&self.ring.members[mi]);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ids(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}{i}")).collect()
    }

    #[test]
    fn same_key_same_primary_and_deterministic() {
        let a = Ring::new(ids("r", 4));
        let b = Ring::new(ids("r", 4));
        for k in ["conv-abc", "system+hello", "", "x-session-9"] {
            let pa = a.primary(k).unwrap();
            assert_eq!(pa, a.primary(k).unwrap(), "stable within a ring");
            assert_eq!(pa, b.primary(k).unwrap(), "deterministic across rings");
        }
    }

    #[test]
    fn route_yields_every_member_once_primary_first() {
        let ring = Ring::new(ids("r", 5));
        let order: Vec<&str> = ring.route("conv-42").collect();
        assert_eq!(order.len(), 5);
        assert_eq!(order[0], ring.primary("conv-42").unwrap());
        let uniq: std::collections::BTreeSet<&&str> = order.iter().collect();
        assert_eq!(uniq.len(), 5, "no duplicates");
    }

    #[test]
    fn removing_a_replica_moves_only_its_keys() {
        let full = Ring::new(ids("r", 5));
        let without = Ring::new(ids("r", 5).into_iter().filter(|m| m != "r2"));
        let keys: Vec<String> = (0..2000).map(|i| format!("conv-{i}")).collect();
        let mut moved = 0;
        for k in &keys {
            let before = full.primary(k).unwrap();
            let after = without.primary(k).unwrap();
            if before == "r2" {
                assert_ne!(after, "r2");
            } else {
                assert_eq!(before, after, "non-owned key {k} must not move");
            }
            if before != after {
                moved += 1;
            }
        }
        let share = moved as f64 / keys.len() as f64;
        assert!(share < 0.35, "moved {share:.3} of keys, want ~1/5");
    }

    #[test]
    fn keys_spread_across_members() {
        let ring = Ring::new(ids("r", 4));
        let mut hits: HashMap<&str, usize> = HashMap::new();
        for i in 0..4000 {
            *hits
                .entry(ring.primary(&format!("k{i}")).unwrap())
                .or_default() += 1;
        }
        assert_eq!(hits.len(), 4, "all members receive keys");
        for (_, c) in hits {
            assert!(c > 500, "each member gets a fair share");
        }
    }

    #[test]
    fn empty_ring_routes_nothing() {
        let ring = Ring::new(Vec::<String>::new());
        assert!(ring.is_empty());
        assert!(ring.primary("k").is_none());
        assert_eq!(ring.route("k").count(), 0);
    }
}
