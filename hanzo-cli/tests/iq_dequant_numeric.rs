// Bit-correctness gate for the 11 newly-added GGUF quant dequant routines in hanzo-ml
// (src/quantized/iq_quants.rs). For each type we read a reference dump produced by the
// real ggml library (C:\dev\qref\<TYPE>.bin = {u32 n_bytes, bytes, u32 n_floats, f32[]}),
// reinterpret the raw bytes as the matching Block* struct, run <Block as GgmlType>::to_float,
// and compare against ggml's floats. Any |diff| > 1e-3 is a failure.
//
// CPU-only: NOT gated on any GPU feature. Block structs + GgmlType are reachable via the
// public `hanzo_ml::quantized::iq_quants` module (same crate path used elsewhere).
use hanzo_ml::quantized::iq_quants::{
    BlockIQ1m, BlockIQ1s, BlockIQ2s, BlockIQ2xs, BlockIQ2xxs, BlockIQ3s, BlockIQ3xxs, BlockNVFP4,
    BlockQ1_0, BlockTQ1_0, BlockTQ2_0,
};
use hanzo_ml::quantized::GgmlType;
use std::io::Write;

const QREF_DIR: &str = "C:\\dev\\qref";
const RESULT_PATH: &str = "C:\\dev\\qref-result.txt";
const TOL: f32 = 1e-3;

// Parse {u32 n_bytes, bytes[n_bytes], u32 n_floats, f32[n_floats]} (all little-endian).
fn read_ref(name: &str) -> (Vec<u8>, Vec<f32>) {
    let path = format!("{QREF_DIR}\\{name}.bin");
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut p = 0usize;
    let rd_u32 = |raw: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([raw[*p], raw[*p + 1], raw[*p + 2], raw[*p + 3]]);
        *p += 4;
        v
    };
    let n_bytes = rd_u32(&raw, &mut p) as usize;
    let bytes = raw[p..p + n_bytes].to_vec();
    p += n_bytes;
    let n_floats = rd_u32(&raw, &mut p) as usize;
    let mut floats = Vec::with_capacity(n_floats);
    for _ in 0..n_floats {
        floats.push(f32::from_le_bytes([
            raw[p],
            raw[p + 1],
            raw[p + 2],
            raw[p + 3],
        ]));
        p += 4;
    }
    (bytes, floats)
}

// Copy raw GGUF bytes into a properly-aligned Vec<T> of whole blocks. (fs::read returns a
// 1-byte-aligned buffer; several Block* structs contain u16 fields and need align >= 2, so a
// from_raw_parts reinterpret of the byte buffer would be UB. Copying into an allocated Vec<T>
// gives correct alignment without changing the bytes.)
fn as_blocks<T: Clone>(bytes: &[u8]) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    assert_eq!(
        bytes.len() % sz,
        0,
        "byte length {} not a multiple of block size {}",
        bytes.len(),
        sz
    );
    let n = bytes.len() / sz;
    let mut v: Vec<T> = Vec::with_capacity(n);
    // SAFETY: T is #[repr(C)] POD (plain quant block), v has capacity for n*sz bytes, and the
    // src/dst ranges are non-overlapping. We then set_len after the bytes are populated.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * sz);
        v.set_len(n);
    }
    v
}

// Dequant via hanzo's GgmlType::to_float, compare to the ggml reference, return (max_abs_err, nbad).
fn check<T: GgmlType + Clone>(name: &str) -> (f32, usize, usize) {
    let (bytes, reff) = read_ref(name);
    let blocks = as_blocks::<T>(&bytes);
    let mut out = vec![0f32; reff.len()];
    T::to_float(&blocks, &mut out);

    let mut max_err = 0f32;
    let mut nbad = 0usize;
    for i in 0..reff.len() {
        let d = (out[i] - reff[i]).abs();
        if d > max_err {
            max_err = d;
        }
        if d > TOL || (out[i].is_nan() != reff[i].is_nan()) {
            nbad += 1;
        }
    }
    (max_err, nbad, reff.len())
}

#[test]
fn iq_dequant_numeric() {
    // (display name, closure running the typed check)
    type CheckFn = fn(&str) -> (f32, usize, usize);
    let cases: &[(&str, CheckFn)] = &[
        ("Q1_0", check::<BlockQ1_0>),
        ("TQ1_0", check::<BlockTQ1_0>),
        ("TQ2_0", check::<BlockTQ2_0>),
        ("NVFP4", check::<BlockNVFP4>),
        ("IQ2_XXS", check::<BlockIQ2xxs>),
        ("IQ2_XS", check::<BlockIQ2xs>),
        ("IQ2_S", check::<BlockIQ2s>),
        ("IQ3_XXS", check::<BlockIQ3xxs>),
        ("IQ3_S", check::<BlockIQ3s>),
        ("IQ1_S", check::<BlockIQ1s>),
        ("IQ1_M", check::<BlockIQ1m>),
    ];

    let mut summary = String::new();
    let mut total_bad = 0usize;
    for (name, f) in cases {
        let (max_err, nbad, total) = f(name);
        total_bad += nbad;
        let line = format!("{name:<9} max_err={max_err:.6e} nbad={nbad}/{total}\n");
        summary.push_str(&line);
        eprint!("{line}");
    }

    let _ = std::fs::File::create(RESULT_PATH).and_then(|mut fh| fh.write_all(summary.as_bytes()));
    eprintln!("--- wrote {RESULT_PATH} ---\n{summary}");

    assert!(
        total_bad == 0,
        "IQ dequant mismatch vs ggml: {total_bad} bad values total\n{summary}"
    );
}
