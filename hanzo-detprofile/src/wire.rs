// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Canonical little-endian length-prefixed binary wire encoding & decoding.
//!
//! Every profile descriptor, operator graph, tensor payload, and execution
//! receipt serializes to a single canonical byte sequence so independent
//! verifiers across architectures compute identical SHA-256 commitments.

use crate::error::{DetError, DetResult};

/// Magic 4-byte prefix for `hanzo-detprofile` canonical wire envelopes (`DZKP`).
pub const WIRE_MAGIC: [u8; 4] = *b"DZKP";

/// Maximum permitted vector length when decoding untrusted wire payloads
/// (16 Mi elements) to prevent OOM DoS during verification.
pub const MAX_WIRE_ELEMENTS: usize = 16 * 1024 * 1024;

/// Canonical little-endian binary writer.
#[derive(Debug, Default, Clone)]
pub struct WireWriter {
    buf: Vec<u8>,
}

impl WireWriter {
    /// Create a new empty writer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Create a writer with preallocated byte capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    /// Consume the writer and return the canonical byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the currently written bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Append raw fixed-size bytes (without length prefix).
    #[inline]
    pub fn write_raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Append a single byte.
    #[inline]
    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Append a signed 8-bit integer.
    #[inline]
    pub fn write_i8(&mut self, v: i8) {
        self.buf.push(v as u8);
    }

    /// Append a signed 16-bit little-endian integer.
    #[inline]
    pub fn write_i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append an unsigned 32-bit little-endian integer.
    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a signed 32-bit little-endian integer.
    #[inline]
    pub fn write_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append an unsigned 64-bit little-endian integer.
    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a signed 64-bit little-endian integer.
    #[inline]
    pub fn write_i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a `u32`-length-prefixed byte slice.
    pub fn write_bytes(&mut self, data: &[u8]) {
        let len = u32::try_from(data.len()).expect("wire byte slice length exceeds u32::MAX");
        self.write_u32(len);
        self.buf.extend_from_slice(data);
    }

    /// Append a `u32`-length-prefixed UTF-8 string.
    pub fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    /// Append a `u32`-length-prefixed slice of `i8`.
    pub fn write_i8_slice(&mut self, values: &[i8]) {
        let len = u32::try_from(values.len()).expect("wire slice length exceeds u32::MAX");
        self.write_u32(len);
        self.buf.reserve(values.len());
        for &v in values {
            self.buf.push(v as u8);
        }
    }

    /// Append a `u32`-length-prefixed slice of `i32` in little-endian order.
    pub fn write_i32_slice(&mut self, values: &[i32]) {
        let len = u32::try_from(values.len()).expect("wire slice length exceeds u32::MAX");
        self.write_u32(len);
        self.buf.reserve(values.len() * 4);
        for &v in values {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Canonical little-endian binary reader with strict bounds and trailing-byte
/// validation.
#[derive(Debug, Clone)]
pub struct WireReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> WireReader<'a> {
    /// Wrap a byte slice for canonical decoding.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    /// Current byte cursor offset.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Remaining unread bytes.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    /// Assert that all bytes have been consumed (rejects trailing garbage).
    pub fn ensure_eof(&self) -> DetResult<()> {
        if self.offset != self.data.len() {
            return Err(DetError::WireDecodeError {
                offset: self.offset,
                detail: "unexpected trailing bytes after canonical payload",
            });
        }
        Ok(())
    }

    /// Read a fixed-length slice of `n` bytes.
    pub fn read_exact(&mut self, n: usize) -> DetResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(n)
            .ok_or(DetError::WireDecodeError {
                offset: self.offset,
                detail: "offset overflow",
            })?;
        if end > self.data.len() {
            return Err(DetError::WireDecodeError {
                offset: self.offset,
                detail: "unexpected EOF while reading canonical field",
            });
        }
        let slice = &self.data[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    /// Read a fixed 4-byte array.
    pub fn read_array<const N: usize>(&mut self) -> DetResult<[u8; N]> {
        let slice = self.read_exact(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    /// Read a single `u8`.
    pub fn read_u8(&mut self) -> DetResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    /// Read a little-endian `u32`.
    pub fn read_u32(&mut self) -> DetResult<u32> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    /// Read a little-endian `i32`.
    pub fn read_i32(&mut self) -> DetResult<i32> {
        Ok(i32::from_le_bytes(self.read_array::<4>()?))
    }

    /// Read a little-endian `u64`.
    pub fn read_u64(&mut self) -> DetResult<u64> {
        Ok(u64::from_le_bytes(self.read_array::<8>()?))
    }

    /// Read a `u32`-length-prefixed byte slice.
    pub fn read_bytes(&mut self) -> DetResult<&'a [u8]> {
        let start = self.offset;
        let len = self.read_u32()? as usize;
        if len > MAX_WIRE_ELEMENTS * 4 {
            return Err(DetError::WireDecodeError {
                offset: start,
                detail: "length prefix exceeds safety limit",
            });
        }
        self.read_exact(len)
    }

    /// Read a `u32`-length-prefixed UTF-8 string.
    pub fn read_str(&mut self) -> DetResult<String> {
        let start = self.offset;
        let bytes = self.read_bytes()?;
        std::str::from_utf8(bytes)
            .map(ToOwned::to_owned)
            .map_err(|_| DetError::WireDecodeError {
                offset: start,
                detail: "invalid UTF-8 string in canonical wire payload",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_roundtrip_and_eof_enforcement() {
        let mut w = WireWriter::new();
        w.write_raw(&WIRE_MAGIC);
        w.write_u32(42);
        w.write_i32(-123456);
        w.write_u64(0x0123_4567_89AB_CDEF);
        w.write_str("det.zen-kai.v1");
        let bytes = w.into_bytes();

        let mut r = WireReader::new(&bytes);
        assert_eq!(r.read_array::<4>().unwrap(), WIRE_MAGIC);
        assert_eq!(r.read_u32().unwrap(), 42);
        assert_eq!(r.read_i32().unwrap(), -123456);
        assert_eq!(r.read_u64().unwrap(), 0x0123_4567_89AB_CDEF);
        assert_eq!(r.read_str().unwrap(), "det.zen-kai.v1");
        r.ensure_eof().unwrap();

        // Trailing byte must fail.
        let mut extended = bytes.clone();
        extended.push(0xFF);
        let mut r2 = WireReader::new(&extended);
        let _ = r2.read_array::<4>().unwrap();
        let _ = r2.read_u32().unwrap();
        let _ = r2.read_i32().unwrap();
        let _ = r2.read_u64().unwrap();
        let _ = r2.read_str().unwrap();
        assert!(matches!(
            r2.ensure_eof(),
            Err(DetError::WireDecodeError { .. })
        ));
    }
}
