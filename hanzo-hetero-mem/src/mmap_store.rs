//! Pure-Rust safe (`#![forbid(unsafe_code)]`) paged weight container (`WeightFile`)
//! with 4 KiB page alignment, packed INT4/INT8/BF16 tiles, SHA-256 integrity
//! verification, real 4 KiB page-fault accounting, demand paging, asynchronous
//! background lookahead prefetch, and zero-copy unified memory views.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::corpus::SplitMix64;
use crate::model::WeightFormat;
use crate::sha256::{to_hex, Sha256};

/// Canonical OS virtual memory page size (`4,096` bytes).
pub const PAGE_SIZE: usize = 4096;
/// Magic header bytes identifying a Hanzo paged weight container (`HZWT`).
pub const WEIGHT_FILE_MAGIC: [u8; 4] = *b"HZWT";
/// Container format version.
pub const WEIGHT_FILE_VERSION: u32 = 1;
/// Default quantization group size (`128` elements per scale/zero group).
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Rounds `n` up to the nearest multiple of `PAGE_SIZE` (`4096` bytes).
pub const fn align_up_page(n: usize) -> usize {
    if n == 0 {
        PAGE_SIZE
    } else {
        n.div_ceil(PAGE_SIZE) * PAGE_SIZE
    }
}

/// Exact packed byte length for `tile_elems` weights in `format` with `group_size`.
pub fn packed_tile_payload_bytes(
    format: WeightFormat,
    tile_elems: usize,
    group_size: usize,
) -> usize {
    let gs = group_size.max(2);
    let groups = tile_elems.div_ceil(gs);
    match format {
        WeightFormat::Bf16 => tile_elems * 2,
        // Per group: 4B f32 scale + 4B f32 min + gs bytes of u8 codes
        WeightFormat::Int8Packed => groups * (8 + gs),
        // Per group: 4B f32 scale + 4B f32 min + ceil(gs/2) bytes of 4-bit nibbles
        WeightFormat::Int4Packed => groups * (8 + gs.div_ceil(2)),
    }
}

/// Encodes `values` into packed tile bytes using `format` and `group_size`.
pub fn encode_tile(format: WeightFormat, values: &[f32], group_size: usize) -> Vec<u8> {
    let gs = group_size.max(2);
    let payload_len = packed_tile_payload_bytes(format, values.len(), gs);
    let mut out = Vec::with_capacity(payload_len);

    match format {
        WeightFormat::Bf16 => {
            for &v in values {
                let bits = (v.to_bits() >> 16) as u16;
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        WeightFormat::Int8Packed => {
            for chunk in values.chunks(gs) {
                let (min_v, max_v) = chunk
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
                        (lo.min(x), hi.max(x))
                    });
                let range = (max_v - min_v).max(1e-8);
                let scale = range / 255.0;
                out.extend_from_slice(&scale.to_le_bytes());
                out.extend_from_slice(&min_v.to_le_bytes());
                for &v in chunk {
                    let q = ((v - min_v) / scale).round().clamp(0.0, 255.0) as u8;
                    out.push(q);
                }
                out.extend(std::iter::repeat_n(0, gs - chunk.len()));
            }
        }
        WeightFormat::Int4Packed => {
            let nibble_bytes = gs.div_ceil(2);
            for chunk in values.chunks(gs) {
                let (min_v, max_v) = chunk
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
                        (lo.min(x), hi.max(x))
                    });
                let range = (max_v - min_v).max(1e-8);
                let scale = range / 15.0;
                out.extend_from_slice(&scale.to_le_bytes());
                out.extend_from_slice(&min_v.to_le_bytes());
                let mut packed = vec![0u8; nibble_bytes];
                for (i, &v) in chunk.iter().enumerate() {
                    let q = ((v - min_v) / scale).round().clamp(0.0, 15.0) as u8;
                    if i % 2 == 0 {
                        packed[i / 2] |= q & 0x0f;
                    } else {
                        packed[i / 2] |= (q & 0x0f) << 4;
                    }
                }
                out.extend_from_slice(&packed);
            }
        }
    }
    out
}

/// Decodes packed tile bytes back into `tile_elems` `f32` values.
pub fn decode_tile(
    format: WeightFormat,
    bytes: &[u8],
    tile_elems: usize,
    group_size: usize,
) -> io::Result<Vec<f32>> {
    let gs = group_size.max(2);
    let expected = packed_tile_payload_bytes(format, tile_elems, gs);
    if bytes.len() < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tile buffer too short: got {} bytes, need {}",
                bytes.len(),
                expected
            ),
        ));
    }

    let mut out = Vec::with_capacity(tile_elems);
    match format {
        WeightFormat::Bf16 => {
            for i in 0..tile_elems {
                let off = i * 2;
                let hi = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                out.push(f32::from_bits((hi as u32) << 16));
            }
        }
        WeightFormat::Int8Packed => {
            let group_bytes = 8 + gs;
            let groups = tile_elems.div_ceil(gs);
            for g in 0..groups {
                let base = g * group_bytes;
                let scale = f32::from_le_bytes([
                    bytes[base],
                    bytes[base + 1],
                    bytes[base + 2],
                    bytes[base + 3],
                ]);
                let min_v = f32::from_le_bytes([
                    bytes[base + 4],
                    bytes[base + 5],
                    bytes[base + 6],
                    bytes[base + 7],
                ]);
                let remain = (tile_elems - out.len()).min(gs);
                for i in 0..remain {
                    let q = bytes[base + 8 + i] as f32;
                    out.push(min_v + q * scale);
                }
            }
        }
        WeightFormat::Int4Packed => {
            let nibble_bytes = gs.div_ceil(2);
            let group_bytes = 8 + nibble_bytes;
            let groups = tile_elems.div_ceil(gs);
            for g in 0..groups {
                let base = g * group_bytes;
                let scale = f32::from_le_bytes([
                    bytes[base],
                    bytes[base + 1],
                    bytes[base + 2],
                    bytes[base + 3],
                ]);
                let min_v = f32::from_le_bytes([
                    bytes[base + 4],
                    bytes[base + 5],
                    bytes[base + 6],
                    bytes[base + 7],
                ]);
                let remain = (tile_elems - out.len()).min(gs);
                for i in 0..remain {
                    let byte = bytes[base + 8 + i / 2];
                    let q = if i % 2 == 0 {
                        (byte & 0x0f) as f32
                    } else {
                        ((byte >> 4) & 0x0f) as f32
                    };
                    out.push(min_v + q * scale);
                }
            }
        }
    }
    Ok(out)
}

/// Parsed 4 KiB-aligned `WeightFile` header enabling `O(1)` tile lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightFileHeader {
    /// Format version (`WEIGHT_FILE_VERSION`).
    pub version: u32,
    /// Packed weight format.
    pub format: WeightFormat,
    /// Quantization group size.
    pub group_size: usize,
    /// Number of tiles in the file.
    pub num_tiles: usize,
    /// Number of logical `f32` elements per tile.
    pub tile_elems: usize,
    /// 4 KiB-aligned on-disk stride per tile in bytes.
    pub tile_stride_bytes: usize,
    /// Unpadded packed payload bytes per tile.
    pub tile_payload_bytes: usize,
    /// File byte offset where tile 0 begins (`PAGE_SIZE`).
    pub data_offset: u64,
    /// Expected SHA-256 digest over all payload bytes (`data_offset..EOF`).
    pub payload_sha256: [u8; 32],
}

impl WeightFileHeader {
    /// Serialises the header into a 4 KiB-aligned page buffer.
    pub fn to_page_bytes(&self) -> [u8; PAGE_SIZE] {
        let mut buf = [0u8; PAGE_SIZE];
        buf[0..4].copy_from_slice(&WEIGHT_FILE_MAGIC);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        let fmt_tag: u32 = match self.format {
            WeightFormat::Bf16 => 0,
            WeightFormat::Int8Packed => 1,
            WeightFormat::Int4Packed => 2,
        };
        buf[8..12].copy_from_slice(&fmt_tag.to_le_bytes());
        buf[12..16].copy_from_slice(&(self.group_size as u32).to_le_bytes());
        buf[16..24].copy_from_slice(&(self.num_tiles as u64).to_le_bytes());
        buf[24..32].copy_from_slice(&(self.tile_elems as u64).to_le_bytes());
        buf[32..40].copy_from_slice(&(self.tile_stride_bytes as u64).to_le_bytes());
        buf[40..48].copy_from_slice(&(self.tile_payload_bytes as u64).to_le_bytes());
        buf[48..56].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[56..88].copy_from_slice(&self.payload_sha256);
        buf
    }

    /// Parses a 4 KiB header page.
    pub fn from_page_bytes(buf: &[u8; PAGE_SIZE]) -> io::Result<Self> {
        if buf[0..4] != WEIGHT_FILE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid WeightFile magic header",
            ));
        }
        let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        if version != WEIGHT_FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported WeightFile version {version}"),
            ));
        }
        let fmt_tag = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let format = match fmt_tag {
            0 => WeightFormat::Bf16,
            1 => WeightFormat::Int8Packed,
            2 => WeightFormat::Int4Packed,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown weight format tag {other}"),
                ))
            }
        };
        let group_size = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
        let num_tiles = u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize;
        let tile_elems = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;
        let tile_stride_bytes = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        let tile_payload_bytes = u64::from_le_bytes(buf[40..48].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        let mut payload_sha256 = [0u8; 32];
        payload_sha256.copy_from_slice(&buf[56..88]);

        if tile_stride_bytes == 0
            || !tile_stride_bytes.is_multiple_of(PAGE_SIZE)
            || tile_payload_bytes > tile_stride_bytes
            || data_offset != PAGE_SIZE as u64
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "malformed WeightFile page alignment metadata",
            ));
        }
        Ok(Self {
            version,
            format,
            group_size,
            num_tiles,
            tile_elems,
            tile_stride_bytes,
            tile_payload_bytes,
            data_offset,
            payload_sha256,
        })
    }

    /// Number of 4 KiB pages occupied by a single tile on disk.
    pub fn pages_per_tile(&self) -> usize {
        self.tile_stride_bytes / PAGE_SIZE
    }

    /// Exact byte offset of `tile_id` in the file (`O(1)` direct calculation).
    pub fn tile_offset(&self, tile_id: usize) -> Option<u64> {
        if tile_id >= self.num_tiles {
            None
        } else {
            Some(self.data_offset + (tile_id as u64) * (self.tile_stride_bytes as u64))
        }
    }

    /// Lowercase hexadecimal SHA-256 digest of the file payload.
    pub fn sha256_hex(&self) -> String {
        to_hex(&self.payload_sha256)
    }
}

/// On-disk paged weight container with 4 KiB alignment and SHA-256 integrity verification.
pub struct WeightFile {
    path: PathBuf,
    header: WeightFileHeader,
    delete_on_drop: bool,
}

impl Drop for WeightFile {
    fn drop(&mut self) {
        if self.delete_on_drop {
            let _ = fs::remove_file(&self.path);
        }
    }
}

impl WeightFile {
    /// Creates a new 4 KiB-aligned `WeightFile` at `path` from raw `f32` tiles.
    pub fn create(
        path: impl AsRef<Path>,
        format: WeightFormat,
        group_size: usize,
        tiles: &[Vec<f32>],
        delete_on_drop: bool,
    ) -> io::Result<Self> {
        if tiles.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot create WeightFile with zero tiles",
            ));
        }
        let tile_elems = tiles[0].len();
        if tile_elems == 0 || tiles.iter().any(|t| t.len() != tile_elems) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "all tiles must have identical non-zero element counts",
            ));
        }
        let tile_payload_bytes = packed_tile_payload_bytes(format, tile_elems, group_size);
        let tile_stride_bytes = align_up_page(tile_payload_bytes);
        let mut hasher = Sha256::new();
        let mut padded_tiles = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let mut buf = encode_tile(format, tile, group_size);
            buf.resize(tile_stride_bytes, 0u8);
            hasher.update(&buf);
            padded_tiles.push(buf);
        }
        let payload_sha256 = hasher.finalize();
        let header = WeightFileHeader {
            version: WEIGHT_FILE_VERSION,
            format,
            group_size,
            num_tiles: tiles.len(),
            tile_elems,
            tile_stride_bytes,
            tile_payload_bytes,
            data_offset: PAGE_SIZE as u64,
            payload_sha256,
        };

        let path_buf = path.as_ref().to_path_buf();
        let mut file = File::create(&path_buf)?;
        file.write_all(&header.to_page_bytes())?;
        for buf in &padded_tiles {
            file.write_all(buf)?;
        }
        file.flush()?;
        Ok(Self {
            path: path_buf,
            header,
            delete_on_drop,
        })
    }

    /// Opens an existing `WeightFile` in `O(1)` time by reading only its 4 KiB header page.
    pub fn open_header_only(path: impl AsRef<Path>, delete_on_drop: bool) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = File::open(&path_buf)?;
        let mut page = [0u8; PAGE_SIZE];
        file.read_exact(&mut page)?;
        let header = WeightFileHeader::from_page_bytes(&page)?;
        Ok(Self {
            path: path_buf,
            header,
            delete_on_drop,
        })
    }

    /// Verifies the full payload SHA-256 digest against the header.
    pub fn verify_sha256(&self) -> io::Result<bool> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.header.data_offset))?;
        let mut hasher = Sha256::new();
        let mut chunk = vec![0u8; 64 * 1024];
        let mut remaining = (self.header.num_tiles as u64) * (self.header.tile_stride_bytes as u64);
        while remaining > 0 {
            let want = (remaining as usize).min(chunk.len());
            file.read_exact(&mut chunk[..want])?;
            hasher.update(&chunk[..want]);
            remaining -= want as u64;
        }
        Ok(hasher.finalize() == self.header.payload_sha256)
    }

    /// Returns the parsed `WeightFileHeader`.
    pub fn header(&self) -> &WeightFileHeader {
        &self.header
    }

    /// Returns the filesystem path backing this `WeightFile`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Exact total file size on disk in bytes (header page + aligned tiles).
    pub fn total_file_bytes(&self) -> u64 {
        self.header.data_offset
            + (self.header.num_tiles as u64) * (self.header.tile_stride_bytes as u64)
    }
}

/// Real page-fault and I/O telemetry counters for `PagedWeightStore`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PageFaultStats {
    /// 4 KiB pages faulted synchronously from disk (`seek` + `read_exact`).
    pub major_faults: u64,
    /// 4 KiB pages bound from resident page cache / completed prefetch buffer without blocking disk I/O.
    pub minor_faults: u64,
    /// Tile fetches served immediately from background async prefetch buffer.
    pub prefetch_hits: u64,
    /// Tile fetches that missed the async prefetch buffer and required synchronous read.
    pub prefetch_misses: u64,
    /// Number of tiles evicted from bounded resident cache.
    pub evictions: u64,
    /// Total bytes read from the backing file.
    pub bytes_read_from_disk: u64,
    /// Total bytes copied across CPU<->GPU buffers (`0` for unified zero-copy views).
    pub copy_bytes: u64,
    /// Peak simultaneously resident 4 KiB pages.
    pub peak_resident_pages: u64,
}

impl PageFaultStats {
    /// Prefetch hit rate in `[0.0, 1.0]` (`0.0` if no prefetch lookups occurred).
    pub fn prefetch_hit_rate(&self) -> f64 {
        let total = self.prefetch_hits + self.prefetch_misses;
        if total == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / total as f64
        }
    }
}

/// Loading & paging strategy for [`PagedWeightStore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMode {
    /// Eagerly reads the entire weight file into memory at open time before serving tile 0.
    EagerFullRead,
    /// Opens header in `O(1)` and loads tiles strictly on demand into a bounded LRU page cache.
    DemandPaged {
        /// Maximum tiles kept resident simultaneously.
        max_resident_tiles: usize,
    },
    /// Opens header in `O(1)` and spawns a real background reader thread prefetching `depth`
    /// lookahead tiles ahead of the active consumer.
    AsyncLookaheadPrefetch {
        /// Lookahead tile prefetch depth.
        depth: usize,
        /// Maximum tiles kept resident simultaneously.
        max_resident_tiles: usize,
    },
    /// Zero-copy unified memory view sharing the exact same immutable `Arc<[u8]>` pages
    /// between CPU and GPU logical executors (`0` copy bytes, `0` duplicate residency).
    UnifiedZeroCopyView {
        /// Maximum tiles kept resident simultaneously.
        max_resident_tiles: usize,
    },
}

/// Shared zero-copy tile view exposing identical underlying pages to both CPU and GPU.
#[derive(Clone, Debug)]
pub struct UnifiedTileView {
    /// Tile index.
    pub tile_id: usize,
    /// Logical CPU slice handle.
    pub cpu_view: Arc<[u8]>,
    /// Logical GPU slice handle.
    pub gpu_view: Arc<[u8]>,
    /// Bytes copied to construct this view (`0` for true zero-copy).
    pub bytes_copied: usize,
}

impl UnifiedTileView {
    /// Verifies that CPU and GPU views point to the exact same underlying memory allocation.
    pub fn is_zero_copy(&self) -> bool {
        Arc::ptr_eq(&self.cpu_view, &self.gpu_view) && self.bytes_copied == 0
    }
}

enum PrefetchCommand {
    Prefetch(usize),
    Shutdown,
}

/// Pure-Rust paged weight store supporting eager read, demand paging, background
/// async lookahead prefetch, and zero-copy unified views.
pub struct PagedWeightStore {
    header: WeightFileHeader,
    mode: LoadMode,
    file: File,
    max_resident_tiles: usize,
    resident: HashMap<usize, Arc<[u8]>>,
    lru: VecDeque<usize>,
    hinted_tiles: std::collections::HashSet<usize>,
    prefetch_tx: Option<Sender<PrefetchCommand>>,
    prefetch_ready: Arc<Mutex<HashMap<usize, Arc<[u8]>>>>,
    prefetch_worker: Option<JoinHandle<()>>,
    stats: PageFaultStats,
}

impl Drop for PagedWeightStore {
    fn drop(&mut self) {
        if let Some(tx) = self.prefetch_tx.take() {
            let _ = tx.send(PrefetchCommand::Shutdown);
        }
        if let Some(handle) = self.prefetch_worker.take() {
            let _ = handle.join();
        }
    }
}

impl PagedWeightStore {
    /// Opens a `WeightFile` under `mode`, recording real startup major page faults.
    pub fn open(weight_file: &WeightFile, mode: LoadMode) -> io::Result<Self> {
        let path = weight_file.path().to_path_buf();
        let mut file = File::open(&path)?;
        // Read header page (1 major fault).
        let mut header_page = [0u8; PAGE_SIZE];
        file.read_exact(&mut header_page)?;
        let header = WeightFileHeader::from_page_bytes(&header_page)?;

        let mut stats = PageFaultStats {
            major_faults: 1,
            bytes_read_from_disk: PAGE_SIZE as u64,
            peak_resident_pages: 1,
            ..PageFaultStats::default()
        };

        let max_resident_tiles = match mode {
            LoadMode::EagerFullRead => header.num_tiles.max(1),
            LoadMode::DemandPaged { max_resident_tiles }
            | LoadMode::AsyncLookaheadPrefetch {
                max_resident_tiles, ..
            }
            | LoadMode::UnifiedZeroCopyView { max_resident_tiles } => max_resident_tiles.max(1),
        };

        let mut resident = HashMap::new();
        let mut lru = VecDeque::new();

        if matches!(mode, LoadMode::EagerFullRead) {
            let pages_per_tile = header.pages_per_tile() as u64;
            for tid in 0..header.num_tiles {
                let mut buf = vec![0u8; header.tile_stride_bytes];
                file.read_exact(&mut buf)?;
                resident.insert(tid, Arc::from(buf.into_boxed_slice()));
                lru.push_back(tid);
                stats.major_faults += pages_per_tile;
                stats.bytes_read_from_disk += header.tile_stride_bytes as u64;
            }
            stats.peak_resident_pages =
                1 + (resident.len() as u64) * (header.pages_per_tile() as u64);
        }

        let prefetch_ready = Arc::new(Mutex::new(HashMap::new()));
        let (prefetch_tx, prefetch_worker) =
            if let LoadMode::AsyncLookaheadPrefetch { depth, .. } = mode {
                let (tx, rx) = mpsc::channel::<PrefetchCommand>();
                let (ready_tx, ready_rx) = mpsc::channel::<()>();
                let ready_clone = Arc::clone(&prefetch_ready);
                let worker_header = header.clone();
                let worker_path = path.clone();
                let cap = (depth * 4).max(8);
                let handle = thread::spawn(move || {
                    let Ok(mut bg_file) = File::open(&worker_path) else {
                        let _ = ready_tx.send(());
                        return;
                    };
                    let _ = ready_tx.send(());
                    while let Ok(cmd) = rx.recv() {
                        match cmd {
                            PrefetchCommand::Shutdown => break,
                            PrefetchCommand::Prefetch(tid) => {
                                {
                                    let guard = ready_clone.lock().unwrap();
                                    if guard.contains_key(&tid) || guard.len() >= cap {
                                        continue;
                                    }
                                }
                                let Some(off) = worker_header.tile_offset(tid) else {
                                    continue;
                                };
                                if bg_file.seek(SeekFrom::Start(off)).is_err() {
                                    continue;
                                }
                                let mut buf = vec![0u8; worker_header.tile_stride_bytes];
                                if bg_file.read_exact(&mut buf).is_ok() {
                                    let mut guard = ready_clone.lock().unwrap();
                                    guard.insert(tid, Arc::from(buf.into_boxed_slice()));
                                }
                            }
                        }
                    }
                });
                let _ = ready_rx.recv();
                (Some(tx), Some(handle))
            } else {
                (None, None)
            };

        Ok(Self {
            header,
            mode,
            file,
            max_resident_tiles,
            resident,
            lru,
            hinted_tiles: std::collections::HashSet::new(),
            prefetch_tx,
            prefetch_ready,
            prefetch_worker,
            stats,
        })
    }

    /// Hints the background async reader to prefetch `tile_id` (deduplicated).
    pub fn hint_prefetch(&mut self, tile_id: usize) {
        if tile_id >= self.header.num_tiles
            || self.resident.contains_key(&tile_id)
            || !self.hinted_tiles.insert(tile_id)
        {
            return;
        }
        if let Some(tx) = &self.prefetch_tx {
            let _ = tx.send(PrefetchCommand::Prefetch(tile_id));
        }
    }

    /// Fetches packed tile bytes for `tile_id`, updating real 4 KiB page-fault stats.
    pub fn fetch_tile_bytes(&mut self, tile_id: usize) -> io::Result<Arc<[u8]>> {
        if tile_id >= self.header.num_tiles {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("tile_id {tile_id} out of range {}", self.header.num_tiles),
            ));
        }
        let pages = self.header.pages_per_tile() as u64;

        // 1. Check resident cache first.
        if let Some(arc) = self.resident.get(&tile_id) {
            self.stats.minor_faults += pages;
            return Ok(Arc::clone(arc));
        }

        // 2. If async lookahead prefetch is active, issue deduplicated hints and check ready table.
        if let LoadMode::AsyncLookaheadPrefetch { depth, .. } = self.mode {
            for ahead in 1..=depth {
                self.hint_prefetch(tile_id + ahead);
            }
            let mut prefetched = {
                let mut guard = self.prefetch_ready.lock().unwrap();
                guard.remove(&tile_id)
            };
            // If this tile was already hinted and is in flight on the background reader,
            // yield once so the OS scheduler can complete the in-flight read without starving it.
            if prefetched.is_none() && self.hinted_tiles.contains(&tile_id) {
                thread::yield_now();
                let mut guard = self.prefetch_ready.lock().unwrap();
                prefetched = guard.remove(&tile_id);
            }
            if let Some(arc) = prefetched {
                self.stats.prefetch_hits += 1;
                self.stats.minor_faults += pages;
                self.stats.bytes_read_from_disk += self.header.tile_stride_bytes as u64;
                self.insert_resident(tile_id, Arc::clone(&arc));
                return Ok(arc);
            } else {
                self.stats.prefetch_misses += 1;
            }
        }

        // 3. Synchronous 4 KiB page-aligned read from disk (major fault).
        let off = self.header.tile_offset(tile_id).unwrap();
        self.file.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; self.header.tile_stride_bytes];
        self.file.read_exact(&mut buf)?;
        let arc: Arc<[u8]> = Arc::from(buf.into_boxed_slice());
        self.stats.major_faults += pages;
        self.stats.bytes_read_from_disk += self.header.tile_stride_bytes as u64;
        self.insert_resident(tile_id, Arc::clone(&arc));
        Ok(arc)
    }

    /// Fetches and dequantizes `tile_id` into `f32` values.
    pub fn fetch_tile_f32(&mut self, tile_id: usize) -> io::Result<Vec<f32>> {
        let bytes = self.fetch_tile_bytes(tile_id)?;
        decode_tile(
            self.header.format,
            &bytes[..self.header.tile_payload_bytes],
            self.header.tile_elems,
            self.header.group_size,
        )
    }

    /// Returns a true zero-copy unified view sharing identical underlying bytes
    /// between CPU and GPU (`0` bytes copied, `0` duplicate residency).
    pub fn fetch_unified_zero_copy_view(&mut self, tile_id: usize) -> io::Result<UnifiedTileView> {
        let shared = self.fetch_tile_bytes(tile_id)?;
        Ok(UnifiedTileView {
            tile_id,
            cpu_view: Arc::clone(&shared),
            gpu_view: shared,
            bytes_copied: 0,
        })
    }

    /// Negative-control helper: simulates discrete-style copy-bounce on unified memory
    /// by allocating and copying a duplicate buffer for the GPU view.
    pub fn fetch_naive_copy_bounce_view(&mut self, tile_id: usize) -> io::Result<UnifiedTileView> {
        let host_buf = self.fetch_tile_bytes(tile_id)?;
        let dup: Vec<u8> = host_buf.as_ref().to_vec();
        let copy_len = dup.len();
        self.stats.copy_bytes += copy_len as u64;
        // Account for duplicate resident pages in peak tracking.
        let extra_pages = self.header.pages_per_tile() as u64;
        let current_pages = 1 + (self.resident.len() as u64) * extra_pages + extra_pages;
        if current_pages > self.stats.peak_resident_pages {
            self.stats.peak_resident_pages = current_pages;
        }
        Ok(UnifiedTileView {
            tile_id,
            cpu_view: host_buf,
            gpu_view: Arc::from(dup.into_boxed_slice()),
            bytes_copied: copy_len,
        })
    }

    /// Returns current page-fault and I/O telemetry counters.
    pub fn stats(&self) -> &PageFaultStats {
        &self.stats
    }

    /// Returns the parsed `WeightFileHeader`.
    pub fn header(&self) -> &WeightFileHeader {
        &self.header
    }

    fn insert_resident(&mut self, tile_id: usize, data: Arc<[u8]>) {
        if self.resident.len() >= self.max_resident_tiles {
            if let Some(evict_id) = self.lru.pop_front() {
                self.resident.remove(&evict_id);
                self.stats.evictions += 1;
            }
        }
        self.resident.insert(tile_id, data);
        self.lru.push_back(tile_id);
        let resident_pages =
            1 + (self.resident.len() as u64) * (self.header.pages_per_tile() as u64);
        if resident_pages > self.stats.peak_resident_pages {
            self.stats.peak_resident_pages = resident_pages;
        }
    }
}

/// Directly measured Linux host I/O result for a single `LoadMode`.
#[derive(Clone, Debug)]
pub struct RealIoModeMeasurement {
    /// Mode label (`eager_full_read`, `demand_paged`, `async_lookahead_prefetch`,
    /// `unified_zero_copy_view`, `unified_naive_copy_bounce`).
    pub mode_label: &'static str,
    /// Directly measured wall-clock cold-start latency to first decoded tile (ms).
    pub cold_first_tile_ms: f64,
    /// Directly measured wall-clock warm access latency to an already-resident tile (ms).
    pub warm_first_tile_ms: f64,
    /// Total wall-clock time to stream & decode all tiles with dot-product compute (ms).
    pub full_stream_ms: f64,
    /// Sustained compressed-to-f32 tile decode throughput (MiB/s of on-disk bytes).
    pub sustained_decode_mibs: f64,
    /// Major 4 KiB page faults during cold start (up to first tile).
    pub cold_start_major_faults: u64,
    /// Total major 4 KiB page faults across the full sweep.
    pub total_major_faults: u64,
    /// Total minor 4 KiB page faults across the full sweep.
    pub total_minor_faults: u64,
    /// Async lookahead prefetch hit rate (`0.0..=1.0`).
    pub prefetch_hit_rate: f64,
    /// Total copied bytes (`0` for zero-copy unified view).
    pub copy_bytes: u64,
    /// Peak simultaneously resident 4 KiB pages.
    pub peak_resident_pages: u64,
    /// Checksum accumulator proving identical decoded values across all modes.
    pub checksum_acc: f64,
}

/// Complete real-file I/O benchmark report directly measured on the host OS.
#[derive(Clone, Debug)]
pub struct RealHostIoReport {
    /// Total file size on disk in bytes.
    pub file_bytes: u64,
    /// Number of tiles in the container.
    pub num_tiles: usize,
    /// Number of 4 KiB pages per tile.
    pub pages_per_tile: usize,
    /// Weight format used on disk.
    pub format: WeightFormat,
    /// SHA-256 digest of the weight file payload.
    pub payload_sha256_hex: String,
    /// Result of full-file SHA-256 integrity verification.
    pub sha256_verified: bool,
    /// Directly measured host timings and fault counts per load mode.
    pub modes: Vec<RealIoModeMeasurement>,
}

/// Executes a real on-disk `WeightFile` benchmark on the local filesystem, comparing
/// eager full-file read, demand paging, async background lookahead prefetch, zero-copy
/// unified view, and naive unified copy-bounce.
pub fn run_real_host_io_benchmark(
    seed: u64,
    num_tiles: usize,
    tile_elems: usize,
) -> io::Result<RealHostIoReport> {
    let mut rng = SplitMix64::new(seed ^ 0x5745_4947_4854_0057);
    let mut tiles = Vec::with_capacity(num_tiles);
    for _ in 0..num_tiles {
        let mut t = Vec::with_capacity(tile_elems);
        for _ in 0..tile_elems {
            t.push(rng.next_signed_f32() * 0.5);
        }
        tiles.push(t);
    }

    static TEMP_FILE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = TEMP_FILE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = std::env::temp_dir().join(format!(
        "hanzo_hetero_mem_{}_{}_{}_{}_{}.hzwt",
        std::process::id(),
        seq,
        seed,
        num_tiles,
        tile_elems
    ));
    let wf = WeightFile::create(
        &tmp_path,
        WeightFormat::Int4Packed,
        DEFAULT_GROUP_SIZE,
        &tiles,
        true,
    )?;
    let sha256_verified = wf.verify_sha256()?;
    let file_bytes = wf.total_file_bytes();
    let payload_sha256_hex = wf.header().sha256_hex();
    let pages_per_tile = wf.header().pages_per_tile();

    // Deterministic activation vector for real CPU dot-product work between tile fetches.
    let probe: Vec<f32> = (0..tile_elems)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let dot = |w: &[f32]| -> f64 {
        let mut acc = 0.0f64;
        for (&a, &b) in w.iter().zip(probe.iter()) {
            acc += (a as f64) * (b as f64);
        }
        acc
    };

    let mut modes = Vec::with_capacity(5);

    // 1. Eager full read
    {
        let t0 = Instant::now();
        let mut store = PagedWeightStore::open(&wf, LoadMode::EagerFullRead)?;
        let first = store.fetch_tile_f32(0)?;
        let cold_first_tile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cold_start_major_faults = store.stats().major_faults;

        let t_warm = Instant::now();
        let _ = store.fetch_tile_f32(0)?;
        let warm_first_tile_ms = t_warm.elapsed().as_secs_f64() * 1000.0;

        let mut acc = dot(&first);
        for tid in 1..num_tiles {
            let w = store.fetch_tile_f32(tid)?;
            acc += dot(&w);
        }
        let full_stream_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sustained_decode_mibs =
            (file_bytes as f64 / (1024.0 * 1024.0)) / (full_stream_ms / 1000.0).max(1e-6);
        let st = store.stats().clone();
        modes.push(RealIoModeMeasurement {
            mode_label: "eager_full_read",
            cold_first_tile_ms,
            warm_first_tile_ms,
            full_stream_ms,
            sustained_decode_mibs,
            cold_start_major_faults,
            total_major_faults: st.major_faults,
            total_minor_faults: st.minor_faults,
            prefetch_hit_rate: st.prefetch_hit_rate(),
            copy_bytes: st.copy_bytes,
            peak_resident_pages: st.peak_resident_pages,
            checksum_acc: acc,
        });
    }

    // 2. Demand-paged cold start
    {
        let t0 = Instant::now();
        let mut store = PagedWeightStore::open(
            &wf,
            LoadMode::DemandPaged {
                max_resident_tiles: (num_tiles / 4).max(8),
            },
        )?;
        let first = store.fetch_tile_f32(0)?;
        let cold_first_tile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cold_start_major_faults = store.stats().major_faults;

        let t_warm = Instant::now();
        let _ = store.fetch_tile_f32(0)?;
        let warm_first_tile_ms = t_warm.elapsed().as_secs_f64() * 1000.0;

        let mut acc = dot(&first);
        for tid in 1..num_tiles {
            let w = store.fetch_tile_f32(tid)?;
            acc += dot(&w);
        }
        let full_stream_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sustained_decode_mibs =
            (file_bytes as f64 / (1024.0 * 1024.0)) / (full_stream_ms / 1000.0).max(1e-6);
        let st = store.stats().clone();
        modes.push(RealIoModeMeasurement {
            mode_label: "demand_paged",
            cold_first_tile_ms,
            warm_first_tile_ms,
            full_stream_ms,
            sustained_decode_mibs,
            cold_start_major_faults,
            total_major_faults: st.major_faults,
            total_minor_faults: st.minor_faults,
            prefetch_hit_rate: st.prefetch_hit_rate(),
            copy_bytes: st.copy_bytes,
            peak_resident_pages: st.peak_resident_pages,
            checksum_acc: acc,
        });
    }

    // 3. Async lookahead prefetch (real background thread overlapping I/O with decode+dot)
    {
        let t0 = Instant::now();
        let mut store = PagedWeightStore::open(
            &wf,
            LoadMode::AsyncLookaheadPrefetch {
                depth: 4,
                max_resident_tiles: (num_tiles / 4).max(8),
            },
        )?;
        // Prime first lookahead window after opening header.
        for ahead in 1..=4 {
            store.hint_prefetch(ahead);
        }
        let first = store.fetch_tile_f32(0)?;
        let cold_first_tile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cold_start_major_faults = store.stats().major_faults;

        let t_warm = Instant::now();
        let _ = store.fetch_tile_f32(0)?;
        let warm_first_tile_ms = t_warm.elapsed().as_secs_f64() * 1000.0;

        let mut acc = dot(&first);
        for tid in 1..num_tiles {
            let w = store.fetch_tile_f32(tid)?;
            acc += dot(&w);
        }
        let full_stream_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sustained_decode_mibs =
            (file_bytes as f64 / (1024.0 * 1024.0)) / (full_stream_ms / 1000.0).max(1e-6);
        let st = store.stats().clone();
        modes.push(RealIoModeMeasurement {
            mode_label: "async_lookahead_prefetch",
            cold_first_tile_ms,
            warm_first_tile_ms,
            full_stream_ms,
            sustained_decode_mibs,
            cold_start_major_faults,
            total_major_faults: st.major_faults,
            total_minor_faults: st.minor_faults,
            prefetch_hit_rate: st.prefetch_hit_rate(),
            copy_bytes: st.copy_bytes,
            peak_resident_pages: st.peak_resident_pages,
            checksum_acc: acc,
        });
    }

    // 4. Zero-copy unified memory view
    {
        let t0 = Instant::now();
        let mut store = PagedWeightStore::open(
            &wf,
            LoadMode::UnifiedZeroCopyView {
                max_resident_tiles: num_tiles,
            },
        )?;
        let v0 = store.fetch_unified_zero_copy_view(0)?;
        debug_assert!(v0.is_zero_copy());
        let first = decode_tile(
            wf.header().format,
            &v0.gpu_view[..wf.header().tile_payload_bytes],
            wf.header().tile_elems,
            wf.header().group_size,
        )?;
        let cold_first_tile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cold_start_major_faults = store.stats().major_faults;

        let t_warm = Instant::now();
        let _ = store.fetch_unified_zero_copy_view(0)?;
        let warm_first_tile_ms = t_warm.elapsed().as_secs_f64() * 1000.0;

        let mut acc = dot(&first);
        for tid in 1..num_tiles {
            let view = store.fetch_unified_zero_copy_view(tid)?;
            let w = decode_tile(
                wf.header().format,
                &view.gpu_view[..wf.header().tile_payload_bytes],
                wf.header().tile_elems,
                wf.header().group_size,
            )?;
            acc += dot(&w);
        }
        let full_stream_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sustained_decode_mibs =
            (file_bytes as f64 / (1024.0 * 1024.0)) / (full_stream_ms / 1000.0).max(1e-6);
        let st = store.stats().clone();
        modes.push(RealIoModeMeasurement {
            mode_label: "unified_zero_copy_view",
            cold_first_tile_ms,
            warm_first_tile_ms,
            full_stream_ms,
            sustained_decode_mibs,
            cold_start_major_faults,
            total_major_faults: st.major_faults,
            total_minor_faults: st.minor_faults,
            prefetch_hit_rate: st.prefetch_hit_rate(),
            copy_bytes: st.copy_bytes,
            peak_resident_pages: st.peak_resident_pages,
            checksum_acc: acc,
        });
    }

    // 5. Naive discrete-style copy-bounce on unified memory (negative control)
    {
        let t0 = Instant::now();
        let mut store = PagedWeightStore::open(
            &wf,
            LoadMode::UnifiedZeroCopyView {
                max_resident_tiles: num_tiles,
            },
        )?;
        let v0 = store.fetch_naive_copy_bounce_view(0)?;
        debug_assert!(!v0.is_zero_copy());
        let first = decode_tile(
            wf.header().format,
            &v0.gpu_view[..wf.header().tile_payload_bytes],
            wf.header().tile_elems,
            wf.header().group_size,
        )?;
        let cold_first_tile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cold_start_major_faults = store.stats().major_faults;

        let t_warm = Instant::now();
        let _ = store.fetch_naive_copy_bounce_view(0)?;
        let warm_first_tile_ms = t_warm.elapsed().as_secs_f64() * 1000.0;

        let mut acc = dot(&first);
        for tid in 1..num_tiles {
            let view = store.fetch_naive_copy_bounce_view(tid)?;
            let w = decode_tile(
                wf.header().format,
                &view.gpu_view[..wf.header().tile_payload_bytes],
                wf.header().tile_elems,
                wf.header().group_size,
            )?;
            acc += dot(&w);
        }
        let full_stream_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sustained_decode_mibs =
            (file_bytes as f64 / (1024.0 * 1024.0)) / (full_stream_ms / 1000.0).max(1e-6);
        let st = store.stats().clone();
        modes.push(RealIoModeMeasurement {
            mode_label: "unified_naive_copy_bounce",
            cold_first_tile_ms,
            warm_first_tile_ms,
            full_stream_ms,
            sustained_decode_mibs,
            cold_start_major_faults,
            total_major_faults: st.major_faults,
            total_minor_faults: st.minor_faults,
            prefetch_hit_rate: st.prefetch_hit_rate(),
            copy_bytes: st.copy_bytes,
            peak_resident_pages: st.peak_resident_pages,
            checksum_acc: acc,
        });
    }

    Ok(RealHostIoReport {
        file_bytes,
        num_tiles,
        pages_per_tile,
        format: WeightFormat::Int4Packed,
        payload_sha256_hex,
        sha256_verified,
        modes,
    })
}
