# Copyright 2026 Hanzo AI Inc.
# Licensed under the Apache License, Version 2.0.
"""Clean-room Python 3.12 stdlib cryptographic primitives for hanzo-attest.

Zero external pip dependencies. Implements:
- FIPS 180-4 SHA-256 and SHA-512
- RFC 2104 HMAC-SHA256 and RFC 5869 HKDF-SHA256
- RFC 8032 Ed25519 with strict canonical point (y < p) and scalar (S < L) checks
- RFC 7748 X25519 Diffie-Hellman with all-zero low-order peer key rejection
- RFC 8439 ChaCha20-Poly1305 AEAD
"""

from __future__ import annotations

import hashlib
import hmac
import struct

P = (1 << 255) - 19
L = (1 << 252) + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, P - 2, P)) % P
D2 = (2 * D) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)

_BY = (4 * pow(5, P - 2, P)) % P
_BX_SQ = ((_BY * _BY - 1) * pow(D * _BY * _BY + 1, P - 2, P)) % P
_BX = pow(_BX_SQ, (P + 3) // 8, P)
if (_BX * _BX - _BX_SQ) % P != 0:
    _BX = (_BX * SQRT_M1) % P
if _BX & 1:
    _BX = P - _BX

ED_BASE = (_BX, _BY, 1, (_BX * _BY) % P)
ED_IDENTITY = (0, 1, 1, 0)


class CryptoError(Exception):
    """Low-level cryptographic failure carrying a canonical hanzo-attest error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def ct_eq(a: bytes, b: bytes) -> bool:
    """Constant-time byte comparison."""
    return hmac.compare_digest(a, b)


def sha256(data: bytes) -> bytes:
    """FIPS 180-4 SHA-256 digest."""
    return hashlib.sha256(data).digest()


def sha512(data: bytes) -> bytes:
    """FIPS 180-4 SHA-512 digest."""
    return hashlib.sha512(data).digest()


def hmac_sha256(key: bytes, data: bytes) -> bytes:
    """RFC 2104 HMAC-SHA256."""
    return hmac.new(key, data, hashlib.sha256).digest()


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """RFC 5869 HKDF-Extract with SHA-256."""
    if not salt:
        salt = b"\x00" * 32
    return hmac_sha256(salt, ikm)


def hkdf_expand(prk: bytes, info: bytes, okm_len: int) -> bytes:
    """RFC 5869 HKDF-Expand with SHA-256."""
    if okm_len > 255 * 32:
        raise CryptoError("INVALID_ENCODING", "HKDF-SHA256 length exceeds 255 * 32")
    n = (okm_len + 31) // 32
    out = bytearray()
    prev = b""
    for counter in range(1, n + 1):
        prev = hmac_sha256(prk, prev + info + bytes([counter]))
        out.extend(prev)
    return bytes(out[:okm_len])


def hkdf_sha256_32(salt: bytes, ikm: bytes, info: bytes) -> bytes:
    """One-shot 32-byte RFC 5869 HKDF-SHA256."""
    prk = hkdf_extract(salt, ikm)
    return hmac_sha256(prk, info + b"\x01")


# ============================================================================
# RFC 7748 X25519
# ============================================================================


def x25519(scalar: bytes, u_coord: bytes) -> bytes:
    """Raw RFC 7748 X25519 scalar multiplication."""
    if len(scalar) != 32 or len(u_coord) != 32:
        raise CryptoError("INVALID_ENCODING", "X25519 inputs must be 32 bytes")

    k_bytes = bytearray(scalar)
    k_bytes[0] &= 248
    k_bytes[31] &= 127
    k_bytes[31] |= 64
    k = int.from_bytes(k_bytes, "little")

    u_bytes = bytearray(u_coord)
    u_bytes[31] &= 127
    x_1 = int.from_bytes(u_bytes, "little") % P

    x_2, z_2 = 1, 0
    x_3, z_3 = x_1, 1
    swap = 0

    for t in range(254, -1, -1):
        k_t = (k >> t) & 1
        swap ^= k_t
        if swap:
            x_2, x_3 = x_3, x_2
            z_2, z_3 = z_3, z_2
        swap = k_t

        a = (x_2 + z_2) % P
        aa = (a * a) % P
        b = (x_2 - z_2) % P
        bb = (b * b) % P
        e = (aa - bb) % P
        c = (x_3 + z_3) % P
        d = (x_3 - z_3) % P
        da = (d * a) % P
        cb = (c * b) % P
        x_3 = pow(da + cb, 2, P)
        z_3 = (x_1 * pow(da - cb, 2, P)) % P
        x_2 = (aa * bb) % P
        z_2 = (e * (aa + 121665 * e)) % P

    if swap:
        x_2, x_3 = x_3, x_2
        z_2, z_3 = z_3, z_2

    res = (x_2 * pow(z_2, P - 2, P)) % P
    return res.to_bytes(32, "little")


def x25519_public_key(secret: bytes) -> bytes:
    """Derives the 32-byte X25519 public key from a 32-byte secret key."""
    basepoint = b"\x09" + b"\x00" * 31
    return x25519(secret, basepoint)


def x25519_diffie_hellman(secret: bytes, peer_public: bytes) -> bytes:
    """Computes an X25519 shared secret and rejects all-zero low-order outputs."""
    shared = x25519(secret, peer_public)
    if ct_eq(shared, b"\x00" * 32):
        raise CryptoError(
            "INVALID_EPHEMERAL_KEY",
            "X25519 shared secret is all-zero (low-order or invalid peer public key)",
        )
    return shared


# ============================================================================
# RFC 8032 Ed25519
# ============================================================================


def _ed_add(
    p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % P
    b = ((y1 + x1) * (y2 + x2)) % P
    c = (t1 * D2 * t2) % P
    d = (2 * z1 * z2) % P
    e = (b - a) % P
    f = (d - c) % P
    g = (d + c) % P
    h = (b + a) % P
    return ((e * f) % P, (g * h) % P, (f * g) % P, (e * h) % P)


def _ed_double(pt: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, z1, _ = pt
    a = (x1 * x1) % P
    b = (y1 * y1) % P
    c = (2 * z1 * z1) % P
    h = (a + b) % P
    e = (h - (x1 + y1) * (x1 + y1)) % P
    g = (a - b) % P
    f = (c + g) % P
    return ((e * f) % P, (g * h) % P, (f * g) % P, (e * h) % P)


def _ed_scalar_mul(
    pt: tuple[int, int, int, int], scalar: int
) -> tuple[int, int, int, int]:
    acc = ED_IDENTITY
    cur = pt
    s = scalar
    while s > 0:
        if s & 1:
            acc = _ed_add(acc, cur)
        cur = _ed_double(cur)
        s >>= 1
    return acc


def _ed_compress(pt: tuple[int, int, int, int]) -> bytes:
    x, y, z, _ = pt
    z_inv = pow(z, P - 2, P)
    ax = (x * z_inv) % P
    ay = (y * z_inv) % P
    out = bytearray(ay.to_bytes(32, "little"))
    out[31] |= (ax & 1) << 7
    return bytes(out)


def _ed_decompress(encoded: bytes) -> tuple[int, int, int, int] | None:
    if len(encoded) != 32:
        return None
    sign = (encoded[31] >> 7) & 1
    y_bytes = bytearray(encoded)
    y_bytes[31] &= 0x7F
    y = int.from_bytes(y_bytes, "little")
    # Strict canonical y < p check.
    if y >= P:
        return None
    y2 = (y * y) % P
    u = (y2 - 1) % P
    v = (D * y2 + 1) % P
    v3 = (v * v * v) % P
    v7 = (v3 * v3 * v) % P
    x = (u * v3 * pow((u * v7) % P, (P - 5) // 8, P)) % P
    vx2 = (v * x * x) % P
    if vx2 != u:
        if vx2 == (-u) % P:
            x = (x * SQRT_M1) % P
        else:
            return None
    if x == 0:
        if sign == 1:
            return None
    elif (x & 1) != sign:
        x = P - x
    return (x, y, 1, (x * y) % P)


def _ed_eq(p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]) -> bool:
    x1, y1, z1, _ = p1
    x2, y2, z2, _ = p2
    return (x1 * z2 - x2 * z1) % P == 0 and (y1 * z2 - y2 * z1) % P == 0


def ed25519_public_key(seed: bytes) -> bytes:
    """Derives the 32-byte RFC 8032 Ed25519 public key from a 32-byte seed."""
    if len(seed) != 32:
        raise CryptoError("INVALID_ENCODING", "Ed25519 seed must be 32 bytes")
    h = sha512(seed)
    s_bytes = bytearray(h[:32])
    s_bytes[0] &= 248
    s_bytes[31] &= 63
    s_bytes[31] |= 64
    s = int.from_bytes(s_bytes, "little")
    return _ed_compress(_ed_scalar_mul(ED_BASE, s))


def ed25519_sign(seed: bytes, message: bytes) -> bytes:
    """Produces a deterministic 64-byte RFC 8032 Ed25519 signature."""
    if len(seed) != 32:
        raise CryptoError("INVALID_ENCODING", "Ed25519 seed must be 32 bytes")
    h = sha512(seed)
    s_bytes = bytearray(h[:32])
    s_bytes[0] &= 248
    s_bytes[31] &= 63
    s_bytes[31] |= 64
    s = int.from_bytes(s_bytes, "little")
    pubkey = _ed_compress(_ed_scalar_mul(ED_BASE, s))

    r = int.from_bytes(sha512(h[32:] + message), "little") % L
    r_point = _ed_compress(_ed_scalar_mul(ED_BASE, r))
    k = int.from_bytes(sha512(r_point + pubkey + message), "little") % L
    sig_s = (r + k * s) % L
    return r_point + sig_s.to_bytes(32, "little")


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Strictly verifies an RFC 8032 Ed25519 signature (rejects S >= L and y >= p)."""
    if len(public_key) != 32 or len(signature) != 64:
        return False
    r_bytes = signature[:32]
    s = int.from_bytes(signature[32:], "little")
    if s >= L:
        return False
    a_point = _ed_decompress(public_key)
    r_point = _ed_decompress(r_bytes)
    if a_point is None or r_point is None:
        return False
    k = int.from_bytes(sha512(r_bytes + public_key + message), "little") % L
    lhs = _ed_scalar_mul(ED_BASE, s)
    rhs = _ed_add(r_point, _ed_scalar_mul(a_point, k))
    return _ed_eq(lhs, rhs)


# ============================================================================
# RFC 8439 ChaCha20-Poly1305 AEAD
# ============================================================================

_MASK32 = 0xFFFFFFFF


def _rotl32(v: int, n: int) -> int:
    return ((v << n) & _MASK32) | (v >> (32 - n))


def _qround(st: list[int], a: int, b: int, c: int, d: int) -> None:
    st[a] = (st[a] + st[b]) & _MASK32
    st[d] = _rotl32(st[d] ^ st[a], 16)
    st[c] = (st[c] + st[d]) & _MASK32
    st[b] = _rotl32(st[b] ^ st[c], 12)
    st[a] = (st[a] + st[b]) & _MASK32
    st[d] = _rotl32(st[d] ^ st[a], 8)
    st[c] = (st[c] + st[d]) & _MASK32
    st[b] = _rotl32(st[b] ^ st[c], 7)


def chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """Single 64-byte RFC 8439 ChaCha20 keystream block."""
    if len(key) != 32 or len(nonce) != 12:
        raise CryptoError("INVALID_ENCODING", "invalid ChaCha20 key or nonce length")
    k = struct.unpack("<8I", key)
    n = struct.unpack("<3I", nonce)
    state = [
        0x61707865,
        0x3320646E,
        0x79622D32,
        0x6B206574,
        k[0],
        k[1],
        k[2],
        k[3],
        k[4],
        k[5],
        k[6],
        k[7],
        counter & _MASK32,
        n[0],
        n[1],
        n[2],
    ]
    w = list(state)
    for _ in range(10):
        _qround(w, 0, 4, 8, 12)
        _qround(w, 1, 5, 9, 13)
        _qround(w, 2, 6, 10, 14)
        _qround(w, 3, 7, 11, 15)
        _qround(w, 0, 5, 10, 15)
        _qround(w, 1, 6, 11, 12)
        _qround(w, 2, 7, 8, 13)
        _qround(w, 3, 4, 9, 14)
    out = [(w[i] + state[i]) & _MASK32 for i in range(16)]
    return struct.pack("<16I", *out)


def chacha20_xor(key: bytes, initial_counter: int, nonce: bytes, data: bytes) -> bytes:
    """Applies the RFC 8439 ChaCha20 stream cipher."""
    out = bytearray(len(data))
    counter = initial_counter
    for offset in range(0, len(data), 64):
        chunk = data[offset : offset + 64]
        ks = chacha20_block(key, counter, nonce)[: len(chunk)]
        xored = int.from_bytes(chunk, "little") ^ int.from_bytes(ks, "little")
        out[offset : offset + len(chunk)] = xored.to_bytes(len(chunk), "little")
        counter = (counter + 1) & _MASK32
    return bytes(out)


def poly1305_mac(one_time_key: bytes, msg: bytes) -> bytes:
    """Computes an exact RFC 8439 §2.5 Poly1305 16-byte MAC."""
    if len(one_time_key) != 32:
        raise CryptoError("INVALID_ENCODING", "Poly1305 key must be 32 bytes")
    r_bytes = bytearray(one_time_key[:16])
    r_bytes[3] &= 15
    r_bytes[7] &= 15
    r_bytes[11] &= 15
    r_bytes[15] &= 15
    r_bytes[4] &= 252
    r_bytes[8] &= 252
    r_bytes[12] &= 252
    r = int.from_bytes(r_bytes, "little")
    s = int.from_bytes(one_time_key[16:32], "little")
    p1305 = (1 << 130) - 5

    acc = 0
    for offset in range(0, len(msg), 16):
        chunk = msg[offset : offset + 16]
        n = int.from_bytes(chunk + b"\x01", "little")
        acc = ((acc + n) * r) % p1305
    tag = (acc + s) % (1 << 128)
    return tag.to_bytes(16, "little")


def _poly1305_aead_tag(
    key: bytes, nonce: bytes, aad: bytes, ciphertext: bytes
) -> bytes:
    otk = chacha20_block(key, 0, nonce)[:32]
    aad_pad = b"\x00" * ((16 - (len(aad) % 16)) % 16)
    ct_pad = b"\x00" * ((16 - (len(ciphertext) % 16)) % 16)
    mac_data = (
        aad
        + aad_pad
        + ciphertext
        + ct_pad
        + struct.pack("<QQ", len(aad), len(ciphertext))
    )
    return poly1305_mac(otk, mac_data)


def chacha20_poly1305_encrypt(
    key: bytes, nonce: bytes, aad: bytes, plaintext: bytes
) -> tuple[bytes, bytes]:
    """Encrypts plaintext with RFC 8439 ChaCha20-Poly1305, returning (ciphertext, tag)."""
    ciphertext = chacha20_xor(key, 1, nonce, plaintext)
    tag = _poly1305_aead_tag(key, nonce, aad, ciphertext)
    return ciphertext, tag


def chacha20_poly1305_decrypt(
    key: bytes, nonce: bytes, aad: bytes, ciphertext: bytes, tag: bytes
) -> bytes:
    """Verifies tag in constant time and decrypts RFC 8439 ChaCha20-Poly1305 ciphertext."""
    if len(tag) != 16:
        raise CryptoError("AEAD_AUTH_FAILED", "Poly1305 tag must be 16 bytes")
    expected = _poly1305_aead_tag(key, nonce, aad, ciphertext)
    if not ct_eq(expected, tag):
        raise CryptoError("AEAD_AUTH_FAILED", "AEAD tag verification failed")
    return chacha20_xor(key, 1, nonce, ciphertext)
