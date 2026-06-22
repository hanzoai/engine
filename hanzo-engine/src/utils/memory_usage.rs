use hanzo_ml::{Device, Result};
use sysinfo::System;
#[cfg(feature = "metal")]
use tracing::warn;

#[cfg(feature = "metal")]
const SIZE_IN_MB: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub enum DeviceMemory {
    Discrete { total: usize, free: usize },
    Unified { budget: usize, allocated: usize },
}

impl DeviceMemory {
    pub fn total(&self) -> usize {
        match *self {
            Self::Discrete { total, .. } => total,
            Self::Unified { budget, .. } => budget,
        }
    }

    pub fn available(&self) -> usize {
        match *self {
            Self::Discrete { free, .. } => free,
            Self::Unified { budget, allocated } => budget.saturating_sub(allocated),
        }
    }

    pub fn is_unified(&self) -> bool {
        matches!(self, Self::Unified { .. })
    }
}

/// `(total, available)` system RAM in bytes via a memory-only `sysinfo` refresh.
/// `System::new_all()` also scans every process/CPU (~110ms on GB10); we only need memory.
#[cfg(feature = "cuda")]
fn system_memory_bytes() -> Result<(usize, usize)> {
    use sysinfo::{MemoryRefreshKind, RefreshKind};
    let sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    Ok((
        usize::try_from(sys.total_memory())?,
        usize::try_from(sys.available_memory())?,
    ))
}

pub struct MemoryUsage;

/// macOS total physical RAM via the `hw.memsize` sysctl. Used as a fallback when
/// sysinfo reports 0 on macOS 26 (see `MemoryUsage::query` `Device::Cpu` arm).
#[cfg(target_os = "macos")]
fn macos_total_memory() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse::<usize>().ok()
}

impl MemoryUsage {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn query(&self, device: &Device) -> Result<DeviceMemory> {
        match device {
            Device::Cpu => {
                let sys = System::new_all();
                #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
                let mut total = usize::try_from(sys.total_memory())?;
                #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
                let mut free = usize::try_from(sys.available_memory())?;
                // sysinfo 0.36 reports 0 for system RAM on macOS 26 (the mach
                // host_statistics path it uses returns nothing there), which makes
                // auto device mapping think nothing fits. `total` and `available`
                // can each independently read 0, so patch them separately: total
                // from the `hw.memsize` sysctl (always works), and available as
                // ~90% of total (a safe device-fit budget on a 64GB box).
                #[cfg(target_os = "macos")]
                {
                    if total == 0 {
                        total = macos_total_memory().unwrap_or(0);
                    }
                    if free == 0 {
                        free = total - total / 10;
                    }
                }
                Ok(DeviceMemory::Discrete { total, free })
            }
            #[cfg(feature = "vulkan")]
            Device::Vulkan(_) => {
                // 8060S APU: unified memory shared with system RAM
                let sys = System::new_all();
                Ok(DeviceMemory::Discrete {
                    total: usize::try_from(sys.total_memory())?,
                    free: usize::try_from(sys.available_memory())?,
                })
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                // gfx1151 APU: unified memory shared with system RAM
                let sys = System::new_all();
                Ok(DeviceMemory::Discrete {
                    total: usize::try_from(sys.total_memory())?,
                    free: usize::try_from(sys.available_memory())?,
                })
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                if super::normal::is_integrated_gpu(device) {
                    let (total_bytes, avail_bytes) = system_memory_bytes()?;
                    let fraction = igpu_memory_fraction();
                    let budget = (total_bytes as f64 * fraction) as usize;
                    let free = (avail_bytes as f64 * fraction) as usize;
                    Ok(DeviceMemory::Unified {
                        budget,
                        allocated: budget.saturating_sub(free),
                    })
                } else {
                    use hanzo_ml::cuda::cudarc::driver::result;
                    use hanzo_ml::cuda_backend::WrapErr;

                    dev.cuda_stream().context().bind_to_thread().w()?;
                    let (free, total) = result::mem_get_info().w()?;
                    Ok(DeviceMemory::Discrete { total, free })
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                hanzo_ml::bail!("Cannot query memory for CUDA device")
            }
            #[cfg(feature = "metal")]
            Device::Metal(dev) => {
                let sysctl_floor = metal_sysctl_floor_bytes()?;
                let device_max = dev.device().recommended_max_working_set_size();
                let budget = sysctl_floor.max(device_max);
                let allocated = dev.current_allocated_size();

                // recommendedMaxWorkingSetSize is dynamic and can underreport on small/pressured Apple Silicon.
                // Dividing by 2 here is a heuristic to indicate that we are now below an expected value.
                // See: https://github.com/hanzoai/engine/issues/2127
                if device_max < sysctl_floor / 2 {
                    warn!(
                        "Metal recommendedMaxWorkingSetSize ({} MB) is much smaller than the system-RAM floor ({} MB); currentAllocatedSize = {} MB. Using the floor.",
                        device_max / SIZE_IN_MB,
                        sysctl_floor / SIZE_IN_MB,
                        allocated / SIZE_IN_MB,
                    );
                }

                Ok(DeviceMemory::Unified { budget, allocated })
            }
            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => {
                hanzo_ml::bail!("Cannot query memory for Metal device")
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn igpu_memory_fraction() -> f64 {
    std::env::var("IGPU_MEMORY_FRACTION")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .and_then(|f| {
            if (0.0..=1.0).contains(&f) {
                Some(f)
            } else {
                None
            }
        })
        .unwrap_or(0.75)
}

#[cfg(feature = "metal")]
fn metal_sysctl_floor_bytes() -> Result<usize> {
    let sys = System::new_all();
    let system_ram_mb = usize::try_from(sys.total_memory())? / SIZE_IN_MB;

    let sysctl_mb = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("iogpu.wired_limit_mb")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok());

    let default_cap_mb = match system_ram_mb {
        x if x <= 36 * 1024 => (system_ram_mb * 2) / 3,
        x if x > 36 * 1024 => (system_ram_mb * 3) / 4,
        x => {
            return Err(hanzo_ml::Error::Msg(format!(
                "Invalid system ram mb value {x}."
            )))
        }
    };

    let floor_mb = match sysctl_mb {
        Some(0) | None => default_cap_mb,
        Some(x) => x,
    };
    Ok(floor_mb * SIZE_IN_MB)
}
