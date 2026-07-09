use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use hanzo_ml::{CpuStorage, Result};

use super::RingConfig;

pub(crate) type SharedTcpStream = Arc<Mutex<TcpStream>>;
pub(crate) type LeftRight = (SharedTcpStream, SharedTcpStream);

pub(crate) const CHUNK_SIZE: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(120);

// Lazily-initialized pair of TCP streams shared by every ring collective and the pipeline
// transport. left = accepted from the left neighbour, right = dialed to the right neighbour.
static LEFT_RIGHT_STREAMS: OnceLock<LeftRight> = OnceLock::new();

pub(crate) fn get_ring_streams(config: &RingConfig) -> LeftRight {
    LEFT_RIGHT_STREAMS
        .get_or_init(|| {
            let cur_port = config.port;
            let right_ip = config.right_ip();
            let right_port = config.right_port;

            let left_listener =
                TcpListener::bind(format!("0.0.0.0:{cur_port}")).expect("bind left");

            let start = Instant::now();
            let right = loop {
                match TcpStream::connect(format!("{right_ip}:{right_port}")) {
                    Ok(s) => break s,
                    Err(_) if start.elapsed() > CONNECT_TIMEOUT => {
                        panic!("Failed to connect to right node due to 120-second timeout");
                    }
                    Err(_) => continue,
                }
            };

            let (left, _) = left_listener.accept().expect("accept left neighbour");

            left.set_nodelay(true).unwrap();
            left.set_nonblocking(false).unwrap();
            right.set_nodelay(true).unwrap();
            right.set_nonblocking(false).unwrap();

            (Arc::new(Mutex::new(left)), Arc::new(Mutex::new(right)))
        })
        .clone()
}

// Canonical wire dtype is f32. Heterogeneous ranks auto-pick different compute dtypes
// (CUDA->bf16, ROCm->f16); f32 is the lossless superset so raw bytes are never misread,
// and reductions accumulate in higher precision.
pub(crate) fn cpu_to_f32(cpu: &CpuStorage) -> Result<Vec<f32>> {
    Ok(match cpu {
        CpuStorage::F32(x) => x.clone(),
        CpuStorage::F16(x) => x.iter().map(|v| v.to_f32()).collect(),
        CpuStorage::BF16(x) => x.iter().map(|v| v.to_f32()).collect(),
        _ => hanzo_ml::bail!("Unsupported dtype for ring backend"),
    })
}

// One symmetric ring step: stream `send` to the right neighbour and read the same-sized payload
// from the left, returning the left neighbour's f32 slice. Used by the collectives.
pub(crate) fn ring_exchange(
    left: &SharedTcpStream,
    right: &SharedTcpStream,
    buffers: &Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    send: &[f32],
) -> Result<Vec<f32>> {
    let nbytes = std::mem::size_of_val(send);
    let data_bytes = unsafe { std::slice::from_raw_parts(send.as_ptr() as *const u8, nbytes) };

    let mut buffers_guard = buffers
        .lock()
        .map_err(|e| hanzo_ml::Error::msg(format!("lock buffers: {e:?}")))?;
    let recv_buf = buffers_guard.entry(nbytes).or_insert_with(|| vec![0u8; nbytes]);

    let mut right_guard = right
        .lock()
        .map_err(|e| hanzo_ml::Error::msg(format!("lock right: {e:?}")))?;
    let mut left_guard = left
        .lock()
        .map_err(|e| hanzo_ml::Error::msg(format!("lock left: {e:?}")))?;

    let mut offset = 0;
    while offset < nbytes {
        let len = std::cmp::min(CHUNK_SIZE, nbytes - offset);
        right_guard
            .write_all(&data_bytes[offset..offset + len])
            .map_err(|e| hanzo_ml::Error::msg(format!("write: {e:?}")))?;
        left_guard
            .read_exact(&mut recv_buf[offset..offset + len])
            .map_err(|e| hanzo_ml::Error::msg(format!("read: {e:?}")))?;
        offset += len;
    }
    drop(left_guard);
    drop(right_guard);

    let peer = unsafe { std::slice::from_raw_parts(recv_buf.as_ptr() as *const f32, send.len()) };
    Ok(peer.to_vec())
}

// Write a byte slice to a stream in CHUNK_SIZE pieces (mirrors the collective framing).
pub(crate) fn write_bytes(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    let mut offset = 0;
    while offset < bytes.len() {
        let len = std::cmp::min(CHUNK_SIZE, bytes.len() - offset);
        stream
            .write_all(&bytes[offset..offset + len])
            .map_err(|e| hanzo_ml::Error::msg(format!("write: {e:?}")))?;
        offset += len;
    }
    Ok(())
}

pub(crate) fn read_bytes(stream: &mut TcpStream, buf: &mut [u8]) -> Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        let len = std::cmp::min(CHUNK_SIZE, buf.len() - offset);
        stream
            .read_exact(&mut buf[offset..offset + len])
            .map_err(|e| hanzo_ml::Error::msg(format!("read: {e:?}")))?;
        offset += len;
    }
    Ok(())
}
