use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Read, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use sha1::{Digest, Sha1};

const KVC_MAGIC: [u8; 3] = *b"KVC";
const KVC_VERSION: u8 = 1;
const KVC_HEADER_BYTES: usize = 48;
const FILE_SUFFIX: &str = ".kv";
const MIN_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveReason {
    Unknown = 0,
    Cold = 1,
    Continued = 2,
    Evict = 3,
    Shutdown = 4,
}

impl SaveReason {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Cold,
            2 => Self::Continued,
            3 => Self::Evict,
            4 => Self::Shutdown,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KvcHeader {
    pub quant_bits: u8,
    pub save_reason: SaveReason,
    pub ext_flags: u8,
    pub token_count: u32,
    pub hit_count: u32,
    pub ctx_size: u32,
    pub created_unix: u64,
    pub last_used_unix: u64,
    pub payload_bytes: u64,
}

impl KvcHeader {
    pub fn new(quant_bits: u8, save_reason: SaveReason, token_count: u32, ctx_size: u32) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            quant_bits,
            save_reason,
            ext_flags: 0,
            token_count,
            hit_count: 0,
            ctx_size,
            created_unix: now,
            last_used_unix: now,
            payload_bytes: 0,
        }
    }

    fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut buf = [0u8; KVC_HEADER_BYTES];
        buf[0..3].copy_from_slice(&KVC_MAGIC);
        buf[3] = KVC_VERSION;
        buf[4] = self.quant_bits;
        buf[5] = self.save_reason as u8;
        buf[6] = self.ext_flags;
        // buf[7] reserved
        buf[8..12].copy_from_slice(&self.token_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.hit_count.to_le_bytes());
        buf[16..20].copy_from_slice(&self.ctx_size.to_le_bytes());
        // buf[20..24] reserved
        buf[24..32].copy_from_slice(&self.created_unix.to_le_bytes());
        buf[32..40].copy_from_slice(&self.last_used_unix.to_le_bytes());
        buf[40..48].copy_from_slice(&self.payload_bytes.to_le_bytes());
        w.write_all(&buf)
    }

    fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; KVC_HEADER_BYTES];
        r.read_exact(&mut buf)?;
        if buf[0..3] != KVC_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad KVC magic"));
        }
        if buf[3] != KVC_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported KVC version {}", buf[3]),
            ));
        }
        Ok(Self {
            quant_bits: buf[4],
            save_reason: SaveReason::from_u8(buf[5]),
            ext_flags: buf[6],
            token_count: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            hit_count: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            ctx_size: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            created_unix: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            last_used_unix: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            payload_bytes: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
        })
    }
}

fn hex40(digest: &[u8]) -> String {
    let mut s = String::with_capacity(40);
    for b in digest {
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{b:02x}"));
    }
    s
}

/// Content-address a byte slice as a lowercase SHA-1 hex string — the disk-cache key.
///
/// This is the single key-derivation primitive. Callers hash whatever identifies the
/// prefix: rendered-text bytes (cross-restart session resume) or little-endian token
/// bytes (prefix-cache spill/restore, see `prefix_cacher`).
pub fn key_for_bytes(bytes: &[u8]) -> String {
    let mut h = Sha1::new();
    h.update(bytes);
    hex40(&h.finalize())
}

pub struct DiskKvCache {
    dir: PathBuf,
    budget_bytes: u64,
}

/// One `.kv` file as seen by a directory scan: its content key, on-disk size, and
/// last-used timestamp (read from the header). Shared by eviction and recency
/// listing so the on-disk layout is parsed in exactly one place.
struct DiskEntry {
    path: PathBuf,
    key: String,
    size: u64,
    last_used: u64,
}

#[derive(Clone, Debug)]
pub struct CacheHit {
    pub header: KvcHeader,
    pub rendered_text: Vec<u8>,
    pub payload: Vec<u8>,
}

impl DiskKvCache {
    pub fn new(dir: impl Into<PathBuf>, budget_mb: u64) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let budget_bytes = (budget_mb * 1024 * 1024).max(MIN_BUDGET_BYTES);
        Ok(Self { dir, budget_bytes })
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}{FILE_SUFFIX}"))
    }

    pub fn save(
        &self,
        key: &str,
        mut header: KvcHeader,
        rendered_text: &[u8],
        payload: &[u8],
    ) -> io::Result<()> {
        header.payload_bytes = payload.len() as u64;
        let path = self.path_for(key);
        let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}-{}.kv.tmp", std::process::id(), seq));
        {
            let f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            let mut w = BufWriter::new(f);
            header.write(&mut w)?;
            let text_len = u32::try_from(rendered_text.len())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "text too large"))?;
            w.write_all(&text_len.to_le_bytes())?;
            w.write_all(rendered_text)?;
            w.write_all(payload)?;
            w.flush()?;
        }
        fs::rename(tmp, path)
    }

    pub fn load(&self, key: &str) -> io::Result<Option<CacheHit>> {
        let path = self.path_for(key);
        match File::open(&path) {
            Ok(mut f) => {
                let mut header = KvcHeader::read(&mut f)?;
                let mut text_len_buf = [0u8; 4];
                f.read_exact(&mut text_len_buf)?;
                let text_len = u32::from_le_bytes(text_len_buf) as u64;
                if text_len > self.budget_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "kvc text_len exceeds cache budget",
                    ));
                }
                let text_len = usize::try_from(text_len).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "kvc text_len too large")
                })?;
                let mut rendered_text = vec![0u8; text_len];
                f.read_exact(&mut rendered_text)?;
                if header.payload_bytes > self.budget_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "kvc payload_bytes exceeds cache budget",
                    ));
                }
                let payload_len = usize::try_from(header.payload_bytes).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "kvc payload_bytes too large")
                })?;
                let mut payload = vec![0u8; payload_len];
                f.read_exact(&mut payload)?;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(header.last_used_unix);
                header.last_used_unix = now;
                header.hit_count = header.hit_count.saturating_add(1);
                let _ = self.touch_header(key, &header);
                Ok(Some(CacheHit {
                    header,
                    rendered_text,
                    payload,
                }))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn touch_header(&self, key: &str, header: &KvcHeader) -> io::Result<()> {
        let path = self.path_for(key);
        let mut f = OpenOptions::new().write(true).open(path)?;
        header.write(&mut f)
    }

    /// Scan the cache directory for `.kv` entries, reading each header for its
    /// last-used time. The single place that maps the on-disk layout to entries.
    fn scan_entries(&self) -> io::Result<Vec<DiskEntry>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("kv") {
                continue;
            }
            let Some(key) = p.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
                continue;
            };
            let size = entry.metadata()?.len();
            let last_used = match File::open(&p).and_then(|mut f| KvcHeader::read(&mut f)) {
                Ok(h) => h.last_used_unix,
                Err(_) => 0,
            };
            entries.push(DiskEntry {
                path: p,
                key,
                size,
                last_used,
            });
        }
        Ok(entries)
    }

    pub fn evict_to_budget(&self) -> io::Result<usize> {
        let mut entries = self.scan_entries()?;
        let total: u64 = entries.iter().map(|e| e.size).sum();
        if total <= self.budget_bytes {
            return Ok(0);
        }
        // Oldest first: drop least-recently-used entries until under budget.
        entries.sort_by_key(|e| e.last_used);
        let mut removed = 0usize;
        let mut current = total;
        for entry in entries {
            if current <= self.budget_bytes {
                break;
            }
            if fs::remove_file(&entry.path).is_ok() {
                current = current.saturating_sub(entry.size);
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Content keys present on disk, most-recently-used first. The prefix cacher
    /// uses this to repopulate its in-memory map on a cold start, restoring the
    /// hottest prefixes within a bounded budget.
    pub fn keys_by_recency(&self) -> io::Result<Vec<String>> {
        let mut entries = self.scan_entries()?;
        entries.sort_by(|a, b| b.last_used.cmp(&a.last_used));
        Ok(entries.into_iter().map(|e| e.key).collect())
    }

    /// Remove orphaned partial writes (`*.kv.tmp`) left behind by an interrupted or
    /// crashed `save`. Safe to call at cold start before any new writes; returns the
    /// number of files removed.
    pub fn sweep_partial(&self) -> io::Result<usize> {
        let mut removed = 0;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry
                .file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".kv.tmp"))
                && fs::remove_file(entry.path()).is_ok()
            {
                removed += 1;
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_sha1_hex() {
        let k = key_for_bytes(b"hello world");
        assert_eq!(k.len(), 40);
        assert_eq!(k, "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed");
    }

    #[test]
    fn header_round_trip() {
        let h = KvcHeader::new(2, SaveReason::Cold, 12345, 32768);
        let mut buf: Vec<u8> = Vec::new();
        h.write(&mut buf).unwrap();
        assert_eq!(buf.len(), KVC_HEADER_BYTES);
        let r = KvcHeader::read(&mut buf.as_slice()).unwrap();
        assert_eq!(r.quant_bits, 2);
        assert_eq!(r.save_reason, SaveReason::Cold);
        assert_eq!(r.token_count, 12345);
        assert_eq!(r.ctx_size, 32768);
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 64).unwrap();
        let text = b"<|user|>hi<|assistant|>";
        let payload = b"binary session payload bytes";
        let key = key_for_bytes(text);
        let header = KvcHeader::new(2, SaveReason::Cold, 17, 8192);
        cache.save(&key, header, text, payload).unwrap();

        let hit = cache.load(&key).unwrap().expect("hit");
        assert_eq!(hit.rendered_text, text);
        assert_eq!(hit.payload, payload);
        assert_eq!(hit.header.token_count, 17);
        assert_eq!(hit.header.ctx_size, 8192);
        assert_eq!(hit.header.quant_bits, 2);
        assert_eq!(hit.header.hit_count, 1);
    }

    #[test]
    fn load_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 64).unwrap();
        assert!(cache
            .load("0000000000000000000000000000000000000000")
            .unwrap()
            .is_none());
    }

    #[test]
    fn evict_to_budget_keeps_recent() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 1).unwrap();
        let big = vec![0u8; 256 * 1024];
        for i in 0..400 {
            let key = key_for_bytes(format!("prefix{i}").as_bytes());
            let mut header = KvcHeader::new(2, SaveReason::Cold, i as u32, 8192);
            header.last_used_unix = 1_000_000 + i as u64;
            cache.save(&key, header, b"", &big).unwrap();
        }
        let removed = cache.evict_to_budget().unwrap();
        assert!(removed > 0, "expected eviction beyond budget");
    }

    #[test]
    fn save_is_atomic_no_tmp_left_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 64).unwrap();
        let key = key_for_bytes(b"atomic");
        cache
            .save(&key, KvcHeader::new(0, SaveReason::Cold, 1, 0), b"", b"payload")
            .unwrap();
        // Only the final `.kv` file should remain; the temp file is renamed in.
        let mut kv = 0;
        let mut partials = 0;
        for entry in fs::read_dir(tmp.path()).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            if name.ends_with(FILE_SUFFIX) {
                kv += 1;
            }
            if name.contains(".tmp") {
                partials += 1;
            }
        }
        assert_eq!(kv, 1);
        assert_eq!(partials, 0);
    }

    #[test]
    fn resave_overwrites_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 64).unwrap();
        let key = key_for_bytes(b"k");
        cache
            .save(&key, KvcHeader::new(0, SaveReason::Cold, 1, 0), b"", b"v1")
            .unwrap();
        cache
            .save(
                &key,
                KvcHeader::new(0, SaveReason::Continued, 2, 0),
                b"",
                b"v2-longer",
            )
            .unwrap();
        let hit = cache.load(&key).unwrap().unwrap();
        assert_eq!(hit.payload, b"v2-longer");
        assert_eq!(hit.header.token_count, 2);
    }

    #[test]
    fn load_persists_hit_count_across_handles() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 64).unwrap();
        let key = key_for_bytes(b"hot");
        cache
            .save(&key, KvcHeader::new(0, SaveReason::Cold, 1, 0), b"", b"p")
            .unwrap();
        assert_eq!(cache.load(&key).unwrap().unwrap().header.hit_count, 1);
        assert_eq!(cache.load(&key).unwrap().unwrap().header.hit_count, 2);
        // hit_count is written back to disk, so a fresh handle keeps counting.
        let cache2 = DiskKvCache::new(tmp.path(), 64).unwrap();
        assert_eq!(cache2.load(&key).unwrap().unwrap().header.hit_count, 3);
    }

    #[test]
    fn keys_by_recency_orders_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = DiskKvCache::new(tmp.path(), 64).unwrap();
        for (i, name) in ["old", "mid", "new"].iter().enumerate() {
            let key = key_for_bytes(name.as_bytes());
            let mut header = KvcHeader::new(0, SaveReason::Cold, i as u32, 0);
            header.last_used_unix = 1_000 + i as u64;
            cache.save(&key, header, b"", b"x").unwrap();
        }
        let keys = cache.keys_by_recency().unwrap();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], key_for_bytes(b"new"));
        assert_eq!(keys[1], key_for_bytes(b"mid"));
        assert_eq!(keys[2], key_for_bytes(b"old"));
    }
}
