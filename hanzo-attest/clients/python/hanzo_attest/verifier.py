# Copyright 2026 Hanzo AI Inc.
# Licensed under the Apache License, Version 2.0.
"""Independent clean-room Python 3.12 provider-independent attestation verifier
and authenticated request binder (HIP-1350, hanzoai/engine#69).

Uses zero external pip dependencies and shares canonical JSON schemas, signing
preimages, AEAD binding headers, and exact error codes with the Rust `hanzo-attest`
crate.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from typing import Any

from .crypto import (
    CryptoError,
    chacha20_poly1305_decrypt,
    chacha20_poly1305_encrypt,
    ct_eq,
    ed25519_sign,
    ed25519_verify,
    hkdf_sha256_32,
    sha256,
    x25519_diffie_hellman,
    x25519_public_key,
)

VALID_PLATFORMS = frozenset({"nvidia_cc_h100", "amd_sev_snp", "intel_tdx"})
BOUND_REQUEST_PROTOCOL_VERSION = "hanzo-attest/bound-req/v1"


class AttestError(Exception):
    """Structured attestation or channel error with canonical cross-language error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _hex_decode_exact(s: Any, expected_bytes: int) -> bytes:
    if not isinstance(s, str):
        raise AttestError("INVALID_JSON", "expected hex string")
    try:
        raw = bytes.fromhex(s)
    except ValueError as exc:
        raise AttestError("INVALID_ENCODING", f"invalid hex encoding: {exc}") from exc
    if len(raw) != expected_bytes:
        raise AttestError(
            "INVALID_ENCODING",
            f"expected {expected_bytes} bytes, got {len(raw)}",
        )
    return raw


def _req(obj: dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise AttestError("INVALID_JSON", f"missing required JSON field '{key}'")
    return obj[key]


def _put_len_prefixed(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack(">I", len(encoded)))
    buf.extend(encoded)


@dataclasses.dataclass(frozen=True)
class HardwareRootCert:
    """Client-pinned hardware root-of-trust anchor."""

    root_id: str
    platform: str
    ed25519_public_key: bytes
    not_before_unix: int
    not_after_unix: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ed25519_public_key": self.ed25519_public_key.hex(),
            "not_after_unix": self.not_after_unix,
            "not_before_unix": self.not_before_unix,
            "platform": self.platform,
            "root_id": self.root_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareRootCert:
        platform = _req(d, "platform")
        if platform not in VALID_PLATFORMS:
            raise AttestError("PLATFORM_MISMATCH", f"unknown TEE platform '{platform}'")
        return cls(
            root_id=str(_req(d, "root_id")),
            platform=platform,
            ed25519_public_key=_hex_decode_exact(_req(d, "ed25519_public_key"), 32),
            not_before_unix=int(_req(d, "not_before_unix")),
            not_after_unix=int(_req(d, "not_after_unix")),
        )


@dataclasses.dataclass
class TrustStore:
    """Sovereign offline collection of pinned hardware roots of trust."""

    roots: dict[str, HardwareRootCert] = dataclasses.field(default_factory=dict)

    def insert(self, root: HardwareRootCert) -> None:
        self.roots[root.root_id] = root

    def get(self, root_id: str) -> HardwareRootCert | None:
        return self.roots.get(root_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": [self.roots[k].to_dict() for k in sorted(self.roots.keys())]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrustStore:
        store = cls()
        for item in _req(d, "roots"):
            store.insert(HardwareRootCert.from_dict(item))
        return store

    def export_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def import_json(cls, raw: str) -> TrustStore:
        return cls.from_dict(json.loads(raw))


@dataclasses.dataclass
class RevocationList:
    """Offline revocation state checked locally with zero control-plane calls."""

    min_revocation_epoch: int = 0
    revoked_root_ids: set[str] = dataclasses.field(default_factory=set)
    revoked_platform_cert_ids: set[str] = dataclasses.field(default_factory=set)
    revoked_device_ids: set[str] = dataclasses.field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_revocation_epoch": self.min_revocation_epoch,
            "revoked_device_ids": sorted(self.revoked_device_ids),
            "revoked_platform_cert_ids": sorted(self.revoked_platform_cert_ids),
            "revoked_root_ids": sorted(self.revoked_root_ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RevocationList:
        return cls(
            min_revocation_epoch=int(_req(d, "min_revocation_epoch")),
            revoked_root_ids=set(_req(d, "revoked_root_ids")),
            revoked_platform_cert_ids=set(_req(d, "revoked_platform_cert_ids")),
            revoked_device_ids=set(_req(d, "revoked_device_ids")),
        )

    def export_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def import_json(cls, raw: str) -> RevocationList:
        return cls.from_dict(json.loads(raw))


@dataclasses.dataclass(frozen=True)
class SoftwareMeasurement:
    """Measured TEE kernel, runtime, and enclave binary digests."""

    rtmr_kernel: bytes
    rtmr_runtime: bytes
    enclave_code_hash: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "enclave_code_hash": self.enclave_code_hash.hex(),
            "rtmr_kernel": self.rtmr_kernel.hex(),
            "rtmr_runtime": self.rtmr_runtime.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SoftwareMeasurement:
        return cls(
            rtmr_kernel=_hex_decode_exact(_req(d, "rtmr_kernel"), 32),
            rtmr_runtime=_hex_decode_exact(_req(d, "rtmr_runtime"), 32),
            enclave_code_hash=_hex_decode_exact(_req(d, "enclave_code_hash"), 32),
        )


@dataclasses.dataclass(frozen=True)
class ModelCommitment:
    """Cryptographic commitment to the intended model, tokenizer, and workload policy."""

    model_id: str
    model_weights_sha256: bytes
    tokenizer_sha256: bytes
    workload_policy_hash: bytes

    def model_commitment_digest(self) -> bytes:
        buf = bytearray(b"hanzo-attest/model-commit/v1")
        _put_len_prefixed(buf, self.model_id)
        buf.extend(self.model_weights_sha256)
        buf.extend(self.tokenizer_sha256)
        return sha256(bytes(buf))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_weights_sha256": self.model_weights_sha256.hex(),
            "tokenizer_sha256": self.tokenizer_sha256.hex(),
            "workload_policy_hash": self.workload_policy_hash.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelCommitment:
        return cls(
            model_id=str(_req(d, "model_id")),
            model_weights_sha256=_hex_decode_exact(_req(d, "model_weights_sha256"), 32),
            tokenizer_sha256=_hex_decode_exact(_req(d, "tokenizer_sha256"), 32),
            workload_policy_hash=_hex_decode_exact(_req(d, "workload_policy_hash"), 32),
        )


@dataclasses.dataclass
class VerificationPolicy:
    """Provider-independent local verification policy."""

    allowed_platforms: set[str]
    min_tcb_versions: dict[str, int]
    require_debug_disabled: bool
    require_secure_boot: bool
    require_memory_encryption: bool
    allowed_measurements: list[SoftwareMeasurement]
    expected_model: ModelCommitment
    max_quote_age_secs: int
    max_clock_skew_secs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_measurements": [m.to_dict() for m in self.allowed_measurements],
            "allowed_platforms": sorted(self.allowed_platforms),
            "expected_model": self.expected_model.to_dict(),
            "max_clock_skew_secs": self.max_clock_skew_secs,
            "max_quote_age_secs": self.max_quote_age_secs,
            "min_tcb_versions": {
                k: self.min_tcb_versions[k] for k in sorted(self.min_tcb_versions.keys())
            },
            "require_debug_disabled": self.require_debug_disabled,
            "require_memory_encryption": self.require_memory_encryption,
            "require_secure_boot": self.require_secure_boot,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerificationPolicy:
        allowed_platforms = set(_req(d, "allowed_platforms"))
        for p in allowed_platforms:
            if p not in VALID_PLATFORMS:
                raise AttestError("PLATFORM_MISMATCH", f"unknown TEE platform '{p}'")
        min_tcb: dict[str, int] = {}
        for k, v in _req(d, "min_tcb_versions").items():
            if k not in VALID_PLATFORMS:
                raise AttestError("PLATFORM_MISMATCH", f"unknown TEE platform '{k}'")
            min_tcb[k] = int(v)
        return cls(
            allowed_platforms=allowed_platforms,
            min_tcb_versions=min_tcb,
            require_debug_disabled=bool(_req(d, "require_debug_disabled")),
            require_secure_boot=bool(_req(d, "require_secure_boot")),
            require_memory_encryption=bool(_req(d, "require_memory_encryption")),
            allowed_measurements=[
                SoftwareMeasurement.from_dict(m) for m in _req(d, "allowed_measurements")
            ],
            expected_model=ModelCommitment.from_dict(_req(d, "expected_model")),
            max_quote_age_secs=int(_req(d, "max_quote_age_secs")),
            max_clock_skew_secs=int(_req(d, "max_clock_skew_secs")),
        )

    def export_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def import_json(cls, raw: str) -> VerificationPolicy:
        return cls.from_dict(json.loads(raw))


@dataclasses.dataclass
class SovereignBundle:
    """Portable bundle containing TrustStore, VerificationPolicy, and RevocationList."""

    trust_store: TrustStore
    policy: VerificationPolicy
    revocations: RevocationList

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "revocations": self.revocations.to_dict(),
            "schema_version": "hanzo-attest/sovereign-bundle/v1",
            "trust_store": self.trust_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SovereignBundle:
        ver = _req(d, "schema_version")
        if ver != "hanzo-attest/sovereign-bundle/v1":
            raise AttestError(
                "INVALID_JSON", f"unsupported sovereign bundle schema_version '{ver}'"
            )
        return cls(
            trust_store=TrustStore.from_dict(_req(d, "trust_store")),
            policy=VerificationPolicy.from_dict(_req(d, "policy")),
            revocations=RevocationList.from_dict(_req(d, "revocations")),
        )

    def export_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def import_json(cls, raw: str) -> SovereignBundle:
        return cls.from_dict(json.loads(raw))


class NonceReplayCache:
    """Replay-defense cache tracking consumed 32-byte client nonces until expiry."""

    def __init__(self) -> None:
        self._seen: dict[bytes, int] = {}

    def prune_expired(self, now_unix: int) -> None:
        expired = [k for k, exp in self._seen.items() if exp < now_unix]
        for k in expired:
            del self._seen[k]

    def check_and_record(
        self, nonce: bytes, expires_at_unix: int, now_unix: int
    ) -> None:
        self.prune_expired(now_unix)
        if nonce in self._seen:
            raise AttestError(
                "NONCE_REPLAYED",
                "client nonce has already been consumed in replay cache",
            )
        self._seen[nonce] = expires_at_unix

    def __len__(self) -> int:
        return len(self._seen)


def compute_report_data(
    client_nonce: bytes,
    enclave_x25519_pubkey: bytes,
    model: ModelCommitment,
) -> bytes:
    """Computes the canonical 32-byte report_data bound inside a signed quote."""
    return sha256(
        b"hanzo-attest/report-data/v1"
        + client_nonce
        + enclave_x25519_pubkey
        + model.model_commitment_digest()
        + model.workload_policy_hash
    )


@dataclasses.dataclass
class PlatformCert:
    """Platform endorsement certificate signed by a hardware root of trust."""

    cert_id: str
    root_id: str
    platform: str
    device_id: str
    platform_ed25519_pubkey: bytes
    tcb_version: int
    issued_at_unix: int
    expires_at_unix: int
    root_signature: bytes

    def signing_preimage(self) -> bytes:
        buf = bytearray(b"hanzo-attest/platform-cert/v1\x00")
        _put_len_prefixed(buf, self.cert_id)
        _put_len_prefixed(buf, self.root_id)
        _put_len_prefixed(buf, self.platform)
        _put_len_prefixed(buf, self.device_id)
        buf.extend(self.platform_ed25519_pubkey)
        buf.extend(
            struct.pack(">QQQ", self.tcb_version, self.issued_at_unix, self.expires_at_unix)
        )
        return bytes(buf)

    @classmethod
    def issue(
        cls,
        cert_id: str,
        root_id: str,
        platform: str,
        device_id: str,
        platform_ed25519_pubkey: bytes,
        tcb_version: int,
        issued_at_unix: int,
        expires_at_unix: int,
        root_ed25519_seed: bytes,
    ) -> PlatformCert:
        cert = cls(
            cert_id=cert_id,
            root_id=root_id,
            platform=platform,
            device_id=device_id,
            platform_ed25519_pubkey=platform_ed25519_pubkey,
            tcb_version=tcb_version,
            issued_at_unix=issued_at_unix,
            expires_at_unix=expires_at_unix,
            root_signature=b"\x00" * 64,
        )
        cert.root_signature = ed25519_sign(root_ed25519_seed, cert.signing_preimage())
        return cert

    def to_dict(self) -> dict[str, Any]:
        return {
            "cert_id": self.cert_id,
            "device_id": self.device_id,
            "expires_at_unix": self.expires_at_unix,
            "issued_at_unix": self.issued_at_unix,
            "platform": self.platform,
            "platform_ed25519_pubkey": self.platform_ed25519_pubkey.hex(),
            "root_id": self.root_id,
            "root_signature": self.root_signature.hex(),
            "tcb_version": self.tcb_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlatformCert:
        platform = _req(d, "platform")
        if platform not in VALID_PLATFORMS:
            raise AttestError("PLATFORM_MISMATCH", f"unknown TEE platform '{platform}'")
        return cls(
            cert_id=str(_req(d, "cert_id")),
            root_id=str(_req(d, "root_id")),
            platform=platform,
            device_id=str(_req(d, "device_id")),
            platform_ed25519_pubkey=_hex_decode_exact(
                _req(d, "platform_ed25519_pubkey"), 32
            ),
            tcb_version=int(_req(d, "tcb_version")),
            issued_at_unix=int(_req(d, "issued_at_unix")),
            expires_at_unix=int(_req(d, "expires_at_unix")),
            root_signature=_hex_decode_exact(_req(d, "root_signature"), 64),
        )


@dataclasses.dataclass(frozen=True)
class EnvironmentState:
    """Hardware-reported TEE security flags and TCB level."""

    debug_disabled: bool
    secure_boot: bool
    memory_encryption: bool
    tcb_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "debug_disabled": self.debug_disabled,
            "memory_encryption": self.memory_encryption,
            "secure_boot": self.secure_boot,
            "tcb_version": self.tcb_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvironmentState:
        return cls(
            debug_disabled=bool(_req(d, "debug_disabled")),
            secure_boot=bool(_req(d, "secure_boot")),
            memory_encryption=bool(_req(d, "memory_encryption")),
            tcb_version=int(_req(d, "tcb_version")),
        )


@dataclasses.dataclass
class AttestationQuote:
    """Hardware-signed remote attestation quote."""

    quote_id: str
    platform: str
    platform_cert_id: str
    environment: EnvironmentState
    measurement: SoftwareMeasurement
    model: ModelCommitment
    client_nonce: bytes
    issued_at_unix: int
    expires_at_unix: int
    revocation_epoch: int
    enclave_x25519_pubkey: bytes
    report_data: bytes
    platform_signature: bytes

    def signing_preimage(self) -> bytes:
        buf = bytearray(b"hanzo-attest/quote/v1\x00")
        _put_len_prefixed(buf, self.quote_id)
        _put_len_prefixed(buf, self.platform)
        _put_len_prefixed(buf, self.platform_cert_id)
        buf.append(1 if self.environment.debug_disabled else 0)
        buf.append(1 if self.environment.secure_boot else 0)
        buf.append(1 if self.environment.memory_encryption else 0)
        buf.extend(struct.pack(">Q", self.environment.tcb_version))
        buf.extend(self.measurement.rtmr_kernel)
        buf.extend(self.measurement.rtmr_runtime)
        buf.extend(self.measurement.enclave_code_hash)
        _put_len_prefixed(buf, self.model.model_id)
        buf.extend(self.model.model_weights_sha256)
        buf.extend(self.model.tokenizer_sha256)
        buf.extend(self.model.workload_policy_hash)
        buf.extend(self.client_nonce)
        buf.extend(
            struct.pack(
                ">QQQ",
                self.issued_at_unix,
                self.expires_at_unix,
                self.revocation_epoch,
            )
        )
        buf.extend(self.enclave_x25519_pubkey)
        buf.extend(self.report_data)
        return bytes(buf)

    def quote_hash(self) -> bytes:
        return sha256(self.signing_preimage())

    @classmethod
    def issue(
        cls,
        quote_id: str,
        platform: str,
        platform_cert_id: str,
        environment: EnvironmentState,
        measurement: SoftwareMeasurement,
        model: ModelCommitment,
        client_nonce: bytes,
        issued_at_unix: int,
        expires_at_unix: int,
        revocation_epoch: int,
        enclave_x25519_pubkey: bytes,
        platform_ed25519_seed: bytes,
    ) -> AttestationQuote:
        report_data = compute_report_data(client_nonce, enclave_x25519_pubkey, model)
        q = cls(
            quote_id=quote_id,
            platform=platform,
            platform_cert_id=platform_cert_id,
            environment=environment,
            measurement=measurement,
            model=model,
            client_nonce=client_nonce,
            issued_at_unix=issued_at_unix,
            expires_at_unix=expires_at_unix,
            revocation_epoch=revocation_epoch,
            enclave_x25519_pubkey=enclave_x25519_pubkey,
            report_data=report_data,
            platform_signature=b"\x00" * 64,
        )
        q.platform_signature = ed25519_sign(platform_ed25519_seed, q.signing_preimage())
        return q

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_nonce": self.client_nonce.hex(),
            "enclave_x25519_pubkey": self.enclave_x25519_pubkey.hex(),
            "environment": self.environment.to_dict(),
            "expires_at_unix": self.expires_at_unix,
            "issued_at_unix": self.issued_at_unix,
            "measurement": self.measurement.to_dict(),
            "model": self.model.to_dict(),
            "platform": self.platform,
            "platform_cert_id": self.platform_cert_id,
            "platform_signature": self.platform_signature.hex(),
            "quote_id": self.quote_id,
            "report_data": self.report_data.hex(),
            "revocation_epoch": self.revocation_epoch,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttestationQuote:
        platform = _req(d, "platform")
        if platform not in VALID_PLATFORMS:
            raise AttestError("PLATFORM_MISMATCH", f"unknown TEE platform '{platform}'")
        return cls(
            quote_id=str(_req(d, "quote_id")),
            platform=platform,
            platform_cert_id=str(_req(d, "platform_cert_id")),
            environment=EnvironmentState.from_dict(_req(d, "environment")),
            measurement=SoftwareMeasurement.from_dict(_req(d, "measurement")),
            model=ModelCommitment.from_dict(_req(d, "model")),
            client_nonce=_hex_decode_exact(_req(d, "client_nonce"), 32),
            issued_at_unix=int(_req(d, "issued_at_unix")),
            expires_at_unix=int(_req(d, "expires_at_unix")),
            revocation_epoch=int(_req(d, "revocation_epoch")),
            enclave_x25519_pubkey=_hex_decode_exact(
                _req(d, "enclave_x25519_pubkey"), 32
            ),
            report_data=_hex_decode_exact(_req(d, "report_data"), 32),
            platform_signature=_hex_decode_exact(_req(d, "platform_signature"), 64),
        )


@dataclasses.dataclass
class AttestationBundle:
    """Wire attestation bundle containing PlatformCert and AttestationQuote."""

    platform_cert: PlatformCert
    quote: AttestationQuote

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_cert": self.platform_cert.to_dict(),
            "quote": self.quote.to_dict(),
            "schema_version": "hanzo-attest/bundle/v1",
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttestationBundle:
        ver = _req(d, "schema_version")
        if ver != "hanzo-attest/bundle/v1":
            raise AttestError(
                "INVALID_JSON", f"unsupported attestation bundle schema_version '{ver}'"
            )
        return cls(
            platform_cert=PlatformCert.from_dict(_req(d, "platform_cert")),
            quote=AttestationQuote.from_dict(_req(d, "quote")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> AttestationBundle:
        return cls.from_dict(json.loads(raw))


_VERIFIED_TOKEN_SENTINEL = object()


@dataclasses.dataclass(frozen=True)
class VerifiedAttestation:
    """Proof-carrying attestation result issued exclusively by verify_attestation."""

    quote_hash: bytes
    platform: str
    root_id: str
    platform_cert_id: str
    device_id: str
    tcb_version: int
    measurement: SoftwareMeasurement
    model: ModelCommitment
    client_nonce: bytes
    enclave_x25519_pubkey: bytes
    issued_at_unix: int
    expires_at_unix: int
    revocation_epoch: int
    _token: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED_TOKEN_SENTINEL:
            raise AttestError(
                "INVALID_ENCODING",
                "VerifiedAttestation must be created by verify_attestation()",
            )


def verify_attestation(
    bundle: AttestationBundle,
    trust_store: TrustStore,
    policy: VerificationPolicy,
    revocations: RevocationList,
    expected_client_nonce: bytes,
    now_unix: int,
    replay_cache: NonceReplayCache | None = None,
) -> VerifiedAttestation:
    """Strictly verifies an AttestationBundle using the exact same check order and
    error codes as the Rust `hanzo-attest` crate.
    """
    cert = bundle.platform_cert
    quote = bundle.quote

    # 1. Hardware root trust & revocation
    root = trust_store.get(cert.root_id)
    if root is None:
        raise AttestError(
            "UNTRUSTED_ROOT",
            f"hardware root '{cert.root_id}' is not in client trust store",
        )
    if root.root_id in revocations.revoked_root_ids:
        raise AttestError("ROOT_REVOKED", f"hardware root '{root.root_id}' is revoked")
    if now_unix < root.not_before_unix or now_unix > root.not_after_unix:
        raise AttestError("ROOT_EXPIRED", f"hardware root '{root.root_id}' expired")

    # 2. Platform consistency and allowlist
    if (
        root.platform != cert.platform
        or cert.platform != quote.platform
        or quote.platform not in policy.allowed_platforms
    ):
        raise AttestError("PLATFORM_MISMATCH", "TEE platform mismatch or disallowed")

    # 3. Platform cert revocation, validity, and root signature
    if cert.cert_id in revocations.revoked_platform_cert_ids:
        raise AttestError(
            "PLATFORM_CERT_REVOKED", f"platform cert '{cert.cert_id}' is revoked"
        )
    if cert.device_id in revocations.revoked_device_ids:
        raise AttestError(
            "DEVICE_REVOKED", f"hardware device '{cert.device_id}' is revoked"
        )
    if now_unix > cert.expires_at_unix:
        raise AttestError(
            "PLATFORM_CERT_EXPIRED", f"platform cert '{cert.cert_id}' expired"
        )
    if cert.issued_at_unix > now_unix + policy.max_clock_skew_secs:
        raise AttestError(
            "PLATFORM_CERT_NOT_YET_VALID",
            f"platform cert '{cert.cert_id}' issued_at in future",
        )
    cert_preimage = cert.signing_preimage()
    if not ed25519_verify(root.ed25519_public_key, cert_preimage, cert.root_signature):
        raise AttestError(
            "INVALID_PLATFORM_CERT_SIGNATURE",
            "hardware root Ed25519 signature invalid for platform cert",
        )

    # 4. Quote linkage and platform signature
    if quote.platform_cert_id != cert.cert_id:
        raise AttestError(
            "QUOTE_CERT_ID_MISMATCH",
            "quote platform_cert_id does not match platform cert",
        )
    quote_preimage = quote.signing_preimage()
    if not ed25519_verify(
        cert.platform_ed25519_pubkey, quote_preimage, quote.platform_signature
    ):
        raise AttestError(
            "INVALID_QUOTE_SIGNATURE",
            "platform Ed25519 signature invalid for quote",
        )

    # 5. Attested environment flags and TCB floor
    if policy.require_debug_disabled and not quote.environment.debug_disabled:
        raise AttestError(
            "DEBUG_MODE_ENABLED", "attested TEE environment reports debug mode enabled"
        )
    if policy.require_secure_boot and not quote.environment.secure_boot:
        raise AttestError(
            "SECURE_BOOT_DISABLED",
            "attested TEE environment reports secure boot disabled",
        )
    if policy.require_memory_encryption and not quote.environment.memory_encryption:
        raise AttestError(
            "MEMORY_ENCRYPTION_DISABLED",
            "attested TEE environment reports memory encryption disabled",
        )
    if quote.environment.tcb_version != cert.tcb_version:
        raise AttestError(
            "TCB_CERT_MISMATCH",
            "quote tcb_version disagrees with platform cert tcb_version",
        )
    min_tcb = policy.min_tcb_versions.get(quote.platform, 0)
    if quote.environment.tcb_version < min_tcb:
        raise AttestError(
            "TCB_VERSION_TOO_LOW",
            f"reported TCB version {quote.environment.tcb_version} < floor {min_tcb}",
        )

    # 6. Software measurement allowlist
    if quote.measurement not in policy.allowed_measurements:
        raise AttestError(
            "MEASUREMENT_MISMATCH",
            "attested RTMR/enclave measurement is not allowlisted by policy",
        )

    # 7. Model & workload commitment
    if quote.model.model_id != policy.expected_model.model_id:
        raise AttestError("MODEL_ID_MISMATCH", "attested model_id mismatch")
    if not ct_eq(
        quote.model.model_weights_sha256, policy.expected_model.model_weights_sha256
    ):
        raise AttestError(
            "MODEL_WEIGHTS_MISMATCH", "attested model_weights_sha256 mismatch"
        )
    if not ct_eq(quote.model.tokenizer_sha256, policy.expected_model.tokenizer_sha256):
        raise AttestError("TOKENIZER_MISMATCH", "attested tokenizer_sha256 mismatch")
    if not ct_eq(
        quote.model.workload_policy_hash, policy.expected_model.workload_policy_hash
    ):
        raise AttestError(
            "WORKLOAD_POLICY_MISMATCH", "attested workload_policy_hash mismatch"
        )

    # 8. Freshness, nonce matching, and revocation epoch
    if not ct_eq(quote.client_nonce, expected_client_nonce):
        raise AttestError(
            "NONCE_MISMATCH",
            "attested quote client_nonce does not match expected challenge nonce",
        )
    if now_unix > quote.expires_at_unix:
        raise AttestError("QUOTE_EXPIRED", "attestation quote expired")
    if quote.issued_at_unix > now_unix + policy.max_clock_skew_secs:
        raise AttestError("QUOTE_NOT_YET_VALID", "attestation quote issued_at in future")
    if max(0, now_unix - quote.issued_at_unix) > policy.max_quote_age_secs:
        raise AttestError("QUOTE_TOO_OLD", "attestation quote exceeds max_quote_age_secs")
    if quote.revocation_epoch < revocations.min_revocation_epoch:
        raise AttestError(
            "REVOCATION_EPOCH_STALE",
            "quote revocation_epoch below client minimum",
        )

    # 9. Secure-channel ephemeral key & report_data binding
    expected_report_data = compute_report_data(
        quote.client_nonce, quote.enclave_x25519_pubkey, quote.model
    )
    if not ct_eq(quote.report_data, expected_report_data):
        raise AttestError(
            "REPORT_DATA_MISMATCH",
            "quote report_data disagrees with bound (nonce, x25519_pub, model, workload)",
        )
    try:
        x25519_diffie_hellman(b"\x42" * 32, quote.enclave_x25519_pubkey)
    except CryptoError as exc:
        raise AttestError(exc.code, exc.message) from exc

    # 10. Nonce replay defense
    if replay_cache is not None:
        replay_cache.check_and_record(quote.client_nonce, quote.expires_at_unix, now_unix)

    return VerifiedAttestation(
        quote_hash=sha256(quote_preimage),
        platform=quote.platform,
        root_id=root.root_id,
        platform_cert_id=cert.cert_id,
        device_id=cert.device_id,
        tcb_version=quote.environment.tcb_version,
        measurement=quote.measurement,
        model=quote.model,
        client_nonce=quote.client_nonce,
        enclave_x25519_pubkey=quote.enclave_x25519_pubkey,
        issued_at_unix=quote.issued_at_unix,
        expires_at_unix=quote.expires_at_unix,
        revocation_epoch=quote.revocation_epoch,
        _token=_VERIFIED_TOKEN_SENTINEL,
    )


# ============================================================================
# Authenticated Request Binding & Enclave Channel
# ============================================================================


def canonical_binding_header(
    quote_hash: bytes,
    model_weights_sha256: bytes,
    workload_policy_hash: bytes,
    client_nonce: bytes,
) -> bytes:
    """Returns the canonical 152-byte AEAD Associated Data (AAD) binding header."""
    return (
        b"hanzo-attest/req-aad/v1\x00"
        + quote_hash
        + model_weights_sha256
        + workload_policy_hash
        + client_nonce
    )


def derive_session_key(
    shared_secret: bytes,
    quote_hash: bytes,
    client_nonce: bytes,
    client_x25519_pubkey: bytes,
    enclave_x25519_pubkey: bytes,
    binding_aad: bytes,
) -> bytes:
    """Derives the 32-byte ChaCha20-Poly1305 session key via HKDF-SHA256."""
    salt = quote_hash + client_nonce
    info = (
        b"hanzo-attest/session-key/v1"
        + client_x25519_pubkey
        + enclave_x25519_pubkey
        + binding_aad
    )
    return hkdf_sha256_32(salt, shared_secret, info)


@dataclasses.dataclass
class EncryptedBoundRequest:
    """Authenticated encrypted request bound to an attested enclave, model, and workload."""

    protocol_version: str
    quote_hash: bytes
    model_id: str
    model_weights_sha256: bytes
    workload_policy_hash: bytes
    client_nonce: bytes
    client_x25519_pubkey: bytes
    enclave_x25519_pubkey: bytes
    aead_nonce: bytes
    ciphertext: bytes
    tag: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "aead_nonce": self.aead_nonce.hex(),
            "ciphertext_hex": self.ciphertext.hex(),
            "client_nonce": self.client_nonce.hex(),
            "client_x25519_pubkey": self.client_x25519_pubkey.hex(),
            "enclave_x25519_pubkey": self.enclave_x25519_pubkey.hex(),
            "model_id": self.model_id,
            "model_weights_sha256": self.model_weights_sha256.hex(),
            "protocol_version": self.protocol_version,
            "quote_hash": self.quote_hash.hex(),
            "tag": self.tag.hex(),
            "workload_policy_hash": self.workload_policy_hash.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EncryptedBoundRequest:
        ver = str(_req(d, "protocol_version"))
        if ver != BOUND_REQUEST_PROTOCOL_VERSION:
            raise AttestError(
                "INVALID_JSON", f"unsupported bound request protocol_version '{ver}'"
            )
        ct_hex = _req(d, "ciphertext_hex")
        if not isinstance(ct_hex, str):
            raise AttestError("INVALID_JSON", "expected ciphertext_hex string")
        try:
            ct = bytes.fromhex(ct_hex)
        except ValueError as exc:
            raise AttestError("INVALID_ENCODING", f"invalid ciphertext hex: {exc}") from exc
        return cls(
            protocol_version=ver,
            quote_hash=_hex_decode_exact(_req(d, "quote_hash"), 32),
            model_id=str(_req(d, "model_id")),
            model_weights_sha256=_hex_decode_exact(_req(d, "model_weights_sha256"), 32),
            workload_policy_hash=_hex_decode_exact(_req(d, "workload_policy_hash"), 32),
            client_nonce=_hex_decode_exact(_req(d, "client_nonce"), 32),
            client_x25519_pubkey=_hex_decode_exact(_req(d, "client_x25519_pubkey"), 32),
            enclave_x25519_pubkey=_hex_decode_exact(_req(d, "enclave_x25519_pubkey"), 32),
            aead_nonce=_hex_decode_exact(_req(d, "aead_nonce"), 12),
            ciphertext=ct,
            tag=_hex_decode_exact(_req(d, "tag"), 16),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> EncryptedBoundRequest:
        return cls.from_dict(json.loads(raw))


def bind_and_encrypt_request(
    verified: VerifiedAttestation,
    client_x25519_secret: bytes,
    aead_nonce: bytes,
    plaintext: bytes,
) -> EncryptedBoundRequest:
    """Encrypts plaintext bound to a VerifiedAttestation via X25519 + HKDF-SHA256 + ChaCha20-Poly1305."""
    if not isinstance(verified, VerifiedAttestation):
        raise AttestError("INVALID_ENCODING", "verified must be a VerifiedAttestation")
    client_pub = x25519_public_key(client_x25519_secret)
    try:
        shared = x25519_diffie_hellman(client_x25519_secret, verified.enclave_x25519_pubkey)
    except CryptoError as exc:
        raise AttestError(exc.code, exc.message) from exc
    aad = canonical_binding_header(
        verified.quote_hash,
        verified.model.model_weights_sha256,
        verified.model.workload_policy_hash,
        verified.client_nonce,
    )
    session_key = derive_session_key(
        shared,
        verified.quote_hash,
        verified.client_nonce,
        client_pub,
        verified.enclave_x25519_pubkey,
        aad,
    )
    ct, tag = chacha20_poly1305_encrypt(session_key, aead_nonce, aad, plaintext)
    return EncryptedBoundRequest(
        protocol_version=BOUND_REQUEST_PROTOCOL_VERSION,
        quote_hash=verified.quote_hash,
        model_id=verified.model.model_id,
        model_weights_sha256=verified.model.model_weights_sha256,
        workload_policy_hash=verified.model.workload_policy_hash,
        client_nonce=verified.client_nonce,
        client_x25519_pubkey=client_pub,
        enclave_x25519_pubkey=verified.enclave_x25519_pubkey,
        aead_nonce=aead_nonce,
        ciphertext=ct,
        tag=tag,
    )


class AttestedSession:
    """Amortized attested session caching the derived X25519/HKDF-SHA256 key."""

    def __init__(
        self,
        verified: VerifiedAttestation,
        client_x25519_secret: bytes,
        base_nonce: bytes,
        max_requests: int,
    ) -> None:
        self._quote_hash = verified.quote_hash
        self._model = verified.model
        self._client_nonce = verified.client_nonce
        self._client_pub = x25519_public_key(client_x25519_secret)
        self._enclave_pub = verified.enclave_x25519_pubkey
        self._aad = canonical_binding_header(
            verified.quote_hash,
            verified.model.model_weights_sha256,
            verified.model.workload_policy_hash,
            verified.client_nonce,
        )
        try:
            shared = x25519_diffie_hellman(
                client_x25519_secret, verified.enclave_x25519_pubkey
            )
        except CryptoError as exc:
            raise AttestError(exc.code, exc.message) from exc
        self._session_key = derive_session_key(
            shared,
            verified.quote_hash,
            verified.client_nonce,
            self._client_pub,
            self._enclave_pub,
            self._aad,
        )
        self._base_nonce = base_nonce
        self._expires_at_unix = verified.expires_at_unix
        self._max_requests = max_requests
        self._seq = 0

    def encrypt_request(self, now_unix: int, plaintext: bytes) -> EncryptedBoundRequest:
        if now_unix > self._expires_at_unix or self._seq >= self._max_requests:
            raise AttestError(
                "SESSION_EXPIRED",
                "attested session expired or reached max_requests",
            )
        nonce = bytearray(self._base_nonce)
        seq_bytes = struct.pack(">Q", self._seq)
        for i in range(8):
            nonce[4 + i] ^= seq_bytes[i]
        self._seq += 1
        nonce_bytes = bytes(nonce)
        ct, tag = chacha20_poly1305_encrypt(
            self._session_key, nonce_bytes, self._aad, plaintext
        )
        return EncryptedBoundRequest(
            protocol_version=BOUND_REQUEST_PROTOCOL_VERSION,
            quote_hash=self._quote_hash,
            model_id=self._model.model_id,
            model_weights_sha256=self._model.model_weights_sha256,
            workload_policy_hash=self._model.workload_policy_hash,
            client_nonce=self._client_nonce,
            client_x25519_pubkey=self._client_pub,
            enclave_x25519_pubkey=self._enclave_pub,
            aead_nonce=nonce_bytes,
            ciphertext=ct,
            tag=tag,
        )


class EnclaveChannel:
    """Enclave-side endpoint verifying binding headers and decrypting bound requests."""

    def __init__(
        self,
        enclave_x25519_secret: bytes,
        quote_hash: bytes,
        model: ModelCommitment,
        client_nonce: bytes,
    ) -> None:
        self._enclave_secret = enclave_x25519_secret
        self.public_key = x25519_public_key(enclave_x25519_secret)
        self._quote_hash = quote_hash
        self._model = model
        self._client_nonce = client_nonce

    def decrypt_bound_request(self, req: EncryptedBoundRequest) -> bytes:
        if (
            not ct_eq(req.quote_hash, self._quote_hash)
            or req.model_id != self._model.model_id
            or not ct_eq(req.model_weights_sha256, self._model.model_weights_sha256)
            or not ct_eq(req.workload_policy_hash, self._model.workload_policy_hash)
            or not ct_eq(req.client_nonce, self._client_nonce)
            or not ct_eq(req.enclave_x25519_pubkey, self.public_key)
        ):
            raise AttestError(
                "BINDING_HEADER_MISMATCH",
                "encrypted request binding header disagrees with attested enclave state",
            )
        try:
            shared = x25519_diffie_hellman(self._enclave_secret, req.client_x25519_pubkey)
        except CryptoError as exc:
            raise AttestError(exc.code, exc.message) from exc
        aad = canonical_binding_header(
            self._quote_hash,
            self._model.model_weights_sha256,
            self._model.workload_policy_hash,
            self._client_nonce,
        )
        session_key = derive_session_key(
            shared,
            self._quote_hash,
            self._client_nonce,
            req.client_x25519_pubkey,
            self.public_key,
            aad,
        )
        try:
            return chacha20_poly1305_decrypt(
                session_key, req.aead_nonce, aad, req.ciphertext, req.tag
            )
        except CryptoError as exc:
            raise AttestError(exc.code, exc.message) from exc
