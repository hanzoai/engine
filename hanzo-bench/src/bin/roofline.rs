//! roofline — measure the two memory-hierarchy ceilings on THIS box, so a decode
//! roofline is computed against real bandwidth, not a spec sheet.
//!
//! Two self-contained probes (the box has neither mbw/sysbench nor fio — only dd):
//!   * RAM: a STREAM triad `a = b + s*c` over buffers far larger than the LLC,
//!     single- and all-thread. On a unified-memory GPU the accelerator shares
//!     this pool, so this is the decode ceiling's denominator.
//!   * NVMe: an O_DIRECT sequential read of a real file (default: the GGUF), page
//!     cache bypassed — the miss-path bandwidth of an off-RAM expert stream.
//!
//! Emits one JSON object on stdout. The arithmetic that turns these ceilings into
//! a per-token roofline (bytes/token, predicted tok/s, %-of-ceiling) lives in the
//! report assembler where the per-row model data is — not here (one concern each).

use std::alloc::{alloc, dealloc, Layout};
use std::fs::OpenOptions;
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde_json::json;

#[derive(Parser)]
#[command(about = "Measure RAM (STREAM triad) and NVMe (O_DIRECT) bandwidth on this box")]
struct Args {
    /// File read for the NVMe probe (defaults to the bench target GGUF).
    #[arg(
        long,
        default_value = "/home/z/models/Qwen3-30B-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
    )]
    file: PathBuf,

    /// Per-array buffer size for the triad, in GiB. Three arrays; keep >> LLC.
    #[arg(long, default_value_t = 2.0)]
    ram_buffer_gb: f64,

    /// Triad repetitions (timed together).
    #[arg(long, default_value_t = 5)]
    ram_reps: usize,

    /// Bytes to read for the NVMe probe, in GiB (or the whole file, whichever is smaller).
    #[arg(long, default_value_t = 4.0)]
    nvme_read_gb: f64,

    /// NVMe read chunk in MiB (multiple of 4 KiB for O_DIRECT alignment).
    #[arg(long, default_value_t = 4)]
    nvme_chunk_mb: usize,

    /// Thread count for the all-thread triad (defaults to all logical cores).
    #[arg(long)]
    threads: Option<usize>,
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const DIRECT_ALIGN: usize = 4096;

/// A heap buffer aligned for O_DIRECT (offset, length, and buffer all 4 KiB-aligned).
struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, DIRECT_ALIGN).expect("valid layout");
        // SAFETY: layout has non-zero size (len is a positive multiple of the align).
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "aligned alloc failed");
        Self { ptr, layout, len }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is a valid allocation of self.len bytes owned by self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: ptr/layout are the exact pair returned by alloc above.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

/// STREAM triad over one contiguous slab: `a[i] = b[i] + scalar * c[i]`.
/// No in-loop reduction — a loop-carried accumulate would throttle the store
/// stream and understate bandwidth. Elision is prevented by black-boxing the
/// slices at the call site instead.
fn triad(a: &mut [f64], b: &[f64], c: &[f64], scalar: f64) {
    for ((ai, &bi), &ci) in a.iter_mut().zip(b.iter()).zip(c.iter()) {
        *ai = bi + scalar * ci;
    }
}

/// GB/s for `reps` triads over `n` f64 elements: 3 arrays × 8 bytes each pass.
fn triad_gbs(n: usize, reps: usize, secs: f64) -> f64 {
    (3.0 * 8.0 * n as f64 * reps as f64) / secs / 1e9
}

fn measure_ram(buffer_gb: f64, reps: usize, threads: usize) -> serde_json::Value {
    let n = ((buffer_gb * GIB) / 8.0) as usize;
    let scalar = 3.0_f64;
    let mut a = vec![0.0_f64; n];
    let b = vec![1.0_f64; n];
    let c = vec![2.0_f64; n];

    // Single thread.
    std::hint::black_box((&a, &b, &c));
    let start = Instant::now();
    for _ in 0..reps {
        triad(&mut a, &b, &c, scalar);
    }
    let single = triad_gbs(n, reps, start.elapsed().as_secs_f64());
    std::hint::black_box(&a);

    // All threads: disjoint chunks, timed together.
    let chunk = n.div_ceil(threads.max(1));
    let start = Instant::now();
    std::thread::scope(|s| {
        for ((ac, bc), cc) in a
            .chunks_mut(chunk)
            .zip(b.chunks(chunk))
            .zip(c.chunks(chunk))
        {
            s.spawn(move || {
                for _ in 0..reps {
                    triad(ac, bc, cc, scalar);
                }
                std::hint::black_box(&*ac);
            });
        }
    });
    let all = triad_gbs(n, reps, start.elapsed().as_secs_f64());

    json!({
        "single_thread": single,
        "all_thread": all,
        "threads": threads,
        "buffer_gb": buffer_gb,
        "reps": reps,
    })
}

/// Open `path` read-only with the OS's page-cache bypass, so the NVMe probe reflects the disk and
/// not RAM. Linux: O_DIRECT, falling back to a buffered read where the filesystem rejects it (e.g.
/// tmpfs). macOS: a normal open plus fcntl(F_NOCACHE) -- there is no O_DIRECT. Other platforms: a
/// plain buffered read. The bool reports whether cache-bypass is actually in effect.
#[cfg(target_os = "linux")]
fn open_uncached(path: &PathBuf) -> std::io::Result<(std::fs::File, bool)> {
    match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Ok(f) => Ok((f, true)),
        Err(_) => Ok((OpenOptions::new().read(true).open(path)?, false)),
    }
}
#[cfg(target_os = "macos")]
fn open_uncached(path: &PathBuf) -> std::io::Result<(std::fs::File, bool)> {
    let file = OpenOptions::new().read(true).open(path)?;
    // SAFETY: fd is a live descriptor owned by `file`; F_NOCACHE bypasses the unified buffer cache.
    let bypassed = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == 0;
    Ok((file, bypassed))
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_uncached(path: &PathBuf) -> std::io::Result<(std::fs::File, bool)> {
    Ok((OpenOptions::new().read(true).open(path)?, false))
}

/// Drop this file's already-cached pages so the read reflects the disk. Linux has posix_fadvise; on
/// macOS the descriptor is opened F_NOCACHE (see `open_uncached`), so there is nothing to drop and
/// this is a no-op (as it is on any platform without posix_fadvise).
#[cfg(target_os = "linux")]
fn fadvise_dontneed(fd: i32) {
    // SAFETY: fd is a live descriptor; a 0 length means "to end of file".
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}
#[cfg(not(target_os = "linux"))]
fn fadvise_dontneed(_fd: i32) {}

fn measure_nvme(
    path: &PathBuf,
    read_gb: f64,
    chunk_mb: usize,
) -> std::io::Result<serde_json::Value> {
    let chunk = chunk_mb * 1024 * 1024;
    let target = (read_gb * GIB) as u64;

    // Bypass the page cache so the read reflects the disk, not RAM (O_DIRECT on Linux, F_NOCACHE on
    // macOS; a buffered fallback where neither applies).
    let (mut file, direct) = open_uncached(path)?;
    fadvise_dontneed(file.as_raw_fd());

    let mut buf = AlignedBuf::new(chunk);
    let mut read: u64 = 0;
    let start = Instant::now();
    loop {
        if read >= target {
            break;
        }
        let n = file.read(buf.as_mut_slice())?;
        if n == 0 {
            break; // EOF
        }
        read += n as u64;
    }
    let secs = start.elapsed().as_secs_f64();
    let gbs = (read as f64) / secs / 1e9;
    if !direct {
        fadvise_dontneed(file.as_raw_fd());
    }

    Ok(json!({
        "value": gbs,
        "direct": direct,
        "bytes_read": read,
        "chunk_mb": chunk_mb,
        "file": path.to_string_lossy(),
    }))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    let ram = measure_ram(args.ram_buffer_gb, args.ram_reps, threads);
    let nvme = match measure_nvme(&args.file, args.nvme_read_gb, args.nvme_chunk_mb) {
        Ok(v) => v,
        Err(e) => json!({ "error": e.to_string(), "file": args.file.to_string_lossy() }),
    };

    let out = json!({ "ram_bw_gbs": ram, "nvme_bw_gbs": nvme });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
