use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::ptr;
use libc::{madvise, MADV_WILLNEED};

pub struct ModelPrewarmer;

impl ModelPrewarmer {
    /// Pre-warms the model file pages into the kernel cache using madvise.
    /// This follows the PPC/MAIO approach for reducing cold-load latency.
    pub fn prewarm_file(file: &File, size: usize) -> Result<(), String> {
        let fd = file.as_raw_fd();
        let addr = ptr::null_mut(); // Let the kernel decide the mapping
        
        unsafe {
            let result = madvise(addr, size, MADV_WILLNEED);
            if result != 0 {
                return Err(format!("madvise failed with code: {}", result));
            }
        }
        Ok(())
    }
}
