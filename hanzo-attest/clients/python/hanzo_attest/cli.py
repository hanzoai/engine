# Copyright 2026 Hanzo AI Inc.
# Licensed under the Apache License, Version 2.0.
"""Command-line interface for the independent Python 3.12 hanzo-attest client.

Supports verifying attestation fixtures, encrypting/decrypting bound requests,
cross-language Rust<->Python differential testing, and reproducible benchmarking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .crypto import sha256
from .verifier import (
    AttestationBundle,
    AttestedSession,
    AttestError,
    EnclaveChannel,
    EncryptedBoundRequest,
    ModelCommitment,
    NonceReplayCache,
    SovereignBundle,
    bind_and_encrypt_request,
    verify_attestation,
)


def _cmd_verify_batch(payload: dict[str, Any]) -> dict[str, Any]:
    results = []
    for case in payload["cases"]:
        name = case["name"]
        try:
            sov = SovereignBundle.from_dict(case["sovereign_bundle"])
            att = AttestationBundle.from_dict(case["attestation_bundle"])
            nonce = bytes.fromhex(case["expected_client_nonce_hex"])
            now_unix = int(case["now_unix"])
            cache = NonceReplayCache()
            verified = verify_attestation(
                att,
                sov.trust_store,
                sov.policy,
                sov.revocations,
                nonce,
                now_unix,
                replay_cache=cache,
            )
            if case.get("replay_second_pass", False):
                verify_attestation(
                    att,
                    sov.trust_store,
                    sov.policy,
                    sov.revocations,
                    nonce,
                    now_unix,
                    replay_cache=cache,
                )
            if "encrypted_request" in case and "enclave_x25519_secret_hex" in case:
                enclave = EnclaveChannel(
                    bytes.fromhex(case["enclave_x25519_secret_hex"]),
                    verified.quote_hash,
                    verified.model,
                    verified.client_nonce,
                )
                req = EncryptedBoundRequest.from_dict(case["encrypted_request"])
                pt = enclave.decrypt_bound_request(req)
                results.append(
                    {
                        "name": name,
                        "ok": True,
                        "error_code": "",
                        "quote_hash": verified.quote_hash.hex(),
                        "decrypted_sha256": sha256(pt).hex(),
                    }
                )
            else:
                results.append(
                    {
                        "name": name,
                        "ok": True,
                        "error_code": "",
                        "quote_hash": verified.quote_hash.hex(),
                        "decrypted_sha256": "",
                    }
                )
        except AttestError as exc:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "error_code": exc.code,
                    "quote_hash": "",
                    "decrypted_sha256": "",
                }
            )
    return {"results": results}


def _cmd_encrypt(payload: dict[str, Any]) -> dict[str, Any]:
    sov = SovereignBundle.from_dict(payload["sovereign_bundle"])
    att = AttestationBundle.from_dict(payload["attestation_bundle"])
    nonce = bytes.fromhex(payload["expected_client_nonce_hex"])
    now_unix = int(payload["now_unix"])
    client_sk = bytes.fromhex(payload["client_x25519_secret_hex"])
    aead_nonce = bytes.fromhex(payload["aead_nonce_hex"])
    plaintext = bytes.fromhex(payload["plaintext_hex"])
    verified = verify_attestation(
        att,
        sov.trust_store,
        sov.policy,
        sov.revocations,
        nonce,
        now_unix,
    )
    req = bind_and_encrypt_request(verified, client_sk, aead_nonce, plaintext)
    return {"ok": True, "encrypted_request": req.to_dict()}


def _cmd_decrypt(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        enclave = EnclaveChannel(
            bytes.fromhex(payload["enclave_x25519_secret_hex"]),
            bytes.fromhex(payload["quote_hash_hex"]),
            ModelCommitment.from_dict(payload["model"]),
            bytes.fromhex(payload["client_nonce_hex"]),
        )
        req = EncryptedBoundRequest.from_dict(payload["encrypted_request"])
        pt = enclave.decrypt_bound_request(req)
        return {"ok": True, "plaintext_hex": pt.hex(), "error_code": ""}
    except AttestError as exc:
        return {"ok": False, "plaintext_hex": "", "error_code": exc.code}


def _cmd_bench(payload: dict[str, Any]) -> dict[str, Any]:
    sov = SovereignBundle.from_dict(payload["sovereign_bundle"])
    att = AttestationBundle.from_dict(payload["attestation_bundle"])
    nonce = bytes.fromhex(payload["expected_client_nonce_hex"])
    now_unix = int(payload["now_unix"])
    client_sk = bytes.fromhex(payload["client_x25519_secret_hex"])
    enclave_sk = bytes.fromhex(payload["enclave_x25519_secret_hex"])
    aead_nonce = bytes.fromhex(payload["aead_nonce_hex"])
    prompt = bytes.fromhex(payload["prompt_hex"])
    cold_iters = int(payload.get("cold_iters", 25))
    session_iters = int(payload.get("session_iters", 200))

    # Warmup
    v = verify_attestation(att, sov.trust_store, sov.policy, sov.revocations, nonce, now_unix)
    req = bind_and_encrypt_request(v, client_sk, aead_nonce, prompt)
    enclave = EnclaveChannel(enclave_sk, v.quote_hash, v.model, v.client_nonce)
    _ = enclave.decrypt_bound_request(req)

    # Cold attestation verify only
    t0 = time.perf_counter()
    for _ in range(cold_iters):
        verify_attestation(att, sov.trust_store, sov.policy, sov.revocations, nonce, now_unix)
    verify_us = (time.perf_counter() - t0) * 1e6 / cold_iters

    # Cold verify + X25519 + HKDF + AEAD encrypt
    t0 = time.perf_counter()
    for _ in range(cold_iters):
        verified = verify_attestation(
            att, sov.trust_store, sov.policy, sov.revocations, nonce, now_unix
        )
        bind_and_encrypt_request(verified, client_sk, aead_nonce, prompt)
    cold_us = (time.perf_counter() - t0) * 1e6 / cold_iters

    # Session-cached encrypt only
    sess = AttestedSession(v, client_sk, aead_nonce, max_requests=session_iters + 10)
    t0 = time.perf_counter()
    for _ in range(session_iters):
        sess.encrypt_request(now_unix, prompt)
    session_us = (time.perf_counter() - t0) * 1e6 / session_iters

    return {
        "cold_verify_us": verify_us,
        "cold_verify_and_encrypt_us": cold_us,
        "session_encrypt_us": session_us,
        "cold_verifications_per_sec": 1e6 / verify_us if verify_us > 0 else 0.0,
        "cold_iters": cold_iters,
        "session_iters": session_iters,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hanzo_attest",
        description="Independent Python 3.12 confidential GPU attestation verifier CLI.",
    )
    parser.add_argument(
        "command",
        choices=["verify-batch", "encrypt", "decrypt", "bench"],
        help="Subcommand to execute on JSON read from stdin.",
    )
    args = parser.parse_args(argv)
    payload = json.loads(sys.stdin.read())
    if args.command == "verify-batch":
        out = _cmd_verify_batch(payload)
    elif args.command == "encrypt":
        out = _cmd_encrypt(payload)
    elif args.command == "decrypt":
        out = _cmd_decrypt(payload)
    else:
        out = _cmd_bench(payload)
    sys.stdout.write(json.dumps(out, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
