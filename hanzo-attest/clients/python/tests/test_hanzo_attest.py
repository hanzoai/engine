# Copyright 2026 Hanzo AI Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit and adversarial test suite for the independent Python 3.12 hanzo_attest client."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from hanzo_attest import (  # noqa: E402
    AttestationBundle,
    AttestationQuote,
    AttestedSession,
    AttestError,
    CryptoError,
    EnclaveChannel,
    EnvironmentState,
    HardwareRootCert,
    ModelCommitment,
    NonceReplayCache,
    PlatformCert,
    RevocationList,
    SoftwareMeasurement,
    SovereignBundle,
    TrustStore,
    VerificationPolicy,
    bind_and_encrypt_request,
    chacha20_block,
    chacha20_poly1305_decrypt,
    chacha20_poly1305_encrypt,
    compute_report_data,
    ed25519_public_key,
    ed25519_sign,
    ed25519_verify,
    hkdf_expand,
    hkdf_extract,
    poly1305_mac,
    sha256,
    verify_attestation,
    x25519,
    x25519_diffie_hellman,
    x25519_public_key,
)
from hanzo_attest.crypto import L  # noqa: E402


def _make_fixture(platform: str = "nvidia_cc_h100") -> dict:
    now_unix = 1_760_000_000
    root_seed = sha256(f"py-root-{platform}".encode())
    plat_seed = sha256(f"py-plat-{platform}".encode())
    enclave_sk = sha256(f"py-enclave-{platform}".encode())
    client_sk = sha256(b"py-client-x25519")
    nonce = sha256(f"py-nonce-{platform}".encode())
    aead_nonce = bytes(range(1, 13))

    root_id = f"root-{platform}"
    cert_id = f"cert-{platform}"
    device_id = f"dev-{platform}"

    store = TrustStore()
    store.insert(
        HardwareRootCert(
            root_id=root_id,
            platform=platform,
            ed25519_public_key=ed25519_public_key(root_seed),
            not_before_unix=now_unix - 10_000,
            not_after_unix=now_unix + 100_000,
        )
    )
    meas = SoftwareMeasurement(
        rtmr_kernel=sha256(b"kernel-rtmr-v1"),
        rtmr_runtime=sha256(b"modelwrap-runtime-v1"),
        enclave_code_hash=sha256(b"hanzo-engine-enclave-bin-v1"),
    )
    model = ModelCommitment(
        model_id="hanzo/zen-9b-instruct-q4",
        model_weights_sha256=sha256(b"zen-9b-weights"),
        tokenizer_sha256=sha256(b"zen-9b-tokenizer"),
        workload_policy_hash=sha256(b"hip-1350-policy"),
    )
    policy = VerificationPolicy(
        allowed_platforms={platform},
        min_tcb_versions={platform: 5},
        require_debug_disabled=True,
        require_secure_boot=True,
        require_memory_encryption=True,
        allowed_measurements=[meas],
        expected_model=model,
        max_quote_age_secs=300,
        max_clock_skew_secs=15,
    )
    revocations = RevocationList(min_revocation_epoch=10)
    cert = PlatformCert.issue(
        cert_id=cert_id,
        root_id=root_id,
        platform=platform,
        device_id=device_id,
        platform_ed25519_pubkey=ed25519_public_key(plat_seed),
        tcb_version=7,
        issued_at_unix=now_unix - 100,
        expires_at_unix=now_unix + 10_000,
        root_ed25519_seed=root_seed,
    )
    quote = AttestationQuote.issue(
        quote_id=f"quote-{platform}",
        platform=platform,
        platform_cert_id=cert_id,
        environment=EnvironmentState(
            debug_disabled=True,
            secure_boot=True,
            memory_encryption=True,
            tcb_version=7,
        ),
        measurement=meas,
        model=model,
        client_nonce=nonce,
        issued_at_unix=now_unix - 5,
        expires_at_unix=now_unix + 120,
        revocation_epoch=12,
        enclave_x25519_pubkey=x25519_public_key(enclave_sk),
        platform_ed25519_seed=plat_seed,
    )
    return {
        "sovereign": SovereignBundle(store, policy, revocations),
        "attestation": AttestationBundle(cert, quote),
        "root_seed": root_seed,
        "plat_seed": plat_seed,
        "enclave_sk": enclave_sk,
        "client_sk": client_sk,
        "nonce": nonce,
        "aead_nonce": aead_nonce,
        "now_unix": now_unix,
    }


def _resign_quote(fx: dict, recompute_rd: bool = False) -> None:
    q = fx["attestation"].quote
    if recompute_rd:
        q.report_data = compute_report_data(
            q.client_nonce, q.enclave_x25519_pubkey, q.model
        )
    q.platform_signature = ed25519_sign(fx["plat_seed"], q.signing_preimage())


class TestCryptoRFCVectors(unittest.TestCase):
    def test_rfc5869_hkdf_sha256_case1(self) -> None:
        ikm = b"\x0b" * 22
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        prk = hkdf_extract(salt, ikm)
        self.assertEqual(
            prk.hex(),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
        )
        okm = hkdf_expand(prk, info, 42)
        self.assertEqual(
            okm.hex(),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865",
        )

    def test_rfc8032_section_7_1_vectors_and_malleability(self) -> None:
        vectors = [
            (
                "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
                "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
                b"",
                "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
            ),
            (
                "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
                "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
                b"\x72",
                "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
            ),
            (
                "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
                "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
                b"\xaf\x82",
                "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
            ),
        ]
        for sk_hex, pk_hex, msg, sig_hex in vectors:
            sk = bytes.fromhex(sk_hex)
            pk = ed25519_public_key(sk)
            self.assertEqual(pk.hex(), pk_hex)
            sig = ed25519_sign(sk, msg)
            self.assertEqual(sig.hex(), sig_hex)
            self.assertTrue(ed25519_verify(pk, msg, sig))
            # Malleable S + L must be rejected
            s_val = int.from_bytes(sig[32:], "little") + L
            malleable = sig[:32] + s_val.to_bytes(32, "little")
            self.assertFalse(ed25519_verify(pk, msg, malleable))

        # Non-canonical y == p must be rejected
        non_canon = bytes([0xED] + [0xFF] * 30 + [0x7F])
        self.assertFalse(ed25519_verify(non_canon, b"msg", b"\x00" * 64))

    def test_rfc7748_x25519_vectors(self) -> None:
        self.assertEqual(
            x25519(
                bytes.fromhex(
                    "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"
                ),
                bytes.fromhex(
                    "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"
                ),
            ).hex(),
            "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552",
        )
        alice_sk = bytes.fromhex(
            "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"
        )
        bob_sk = bytes.fromhex(
            "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb"
        )
        alice_pk = x25519_public_key(alice_sk)
        bob_pk = x25519_public_key(bob_sk)
        self.assertEqual(
            alice_pk.hex(),
            "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
        )
        self.assertEqual(
            bob_pk.hex(),
            "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
        )
        ss_a = x25519_diffie_hellman(alice_sk, bob_pk)
        ss_b = x25519_diffie_hellman(bob_sk, alice_pk)
        self.assertEqual(ss_a, ss_b)
        self.assertEqual(
            ss_a.hex(),
            "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742",
        )
        with self.assertRaises(CryptoError) as ctx:
            x25519_diffie_hellman(alice_sk, b"\x00" * 32)
        self.assertEqual(ctx.exception.code, "INVALID_EPHEMERAL_KEY")

    def test_rfc8439_chacha20_poly1305_vectors(self) -> None:
        blk = chacha20_block(
            bytes.fromhex(
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
            ),
            1,
            bytes.fromhex("000000090000004a00000000"),
        )
        self.assertEqual(blk[:16].hex(), "10f1e7e4d13b5915500fdd1fa32071c4")
        mac = poly1305_mac(
            bytes.fromhex(
                "85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b"
            ),
            b"Cryptographic Forum Research Group",
        )
        self.assertEqual(mac.hex(), "a8061dc1305136c6c22b8baf0c0127a9")

        key = bytes.fromhex(
            "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
        )
        nonce = bytes.fromhex("070000004041424344454647")
        aad = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
        pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it."
        ct, tag = chacha20_poly1305_encrypt(key, nonce, aad, pt)
        self.assertEqual(
            ct.hex(),
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116",
        )
        self.assertEqual(tag.hex(), "1ae10b594f09e26a7e902ecbd0600691")
        self.assertEqual(chacha20_poly1305_decrypt(key, nonce, aad, ct, tag), pt)


class TestAttestationAndChannel(unittest.TestCase):
    def test_multi_tee_verify_and_channel_binding(self) -> None:
        for platform in ("nvidia_cc_h100", "amd_sev_snp", "intel_tdx"):
            fx = _make_fixture(platform)
            sov = fx["sovereign"]
            cache = NonceReplayCache()
            verified = verify_attestation(
                fx["attestation"],
                sov.trust_store,
                sov.policy,
                sov.revocations,
                fx["nonce"],
                fx["now_unix"],
                replay_cache=cache,
            )
            self.assertEqual(verified.platform, platform)
            req = bind_and_encrypt_request(
                verified, fx["client_sk"], fx["aead_nonce"], b"sensitive prompt"
            )
            enclave = EnclaveChannel(
                fx["enclave_sk"], verified.quote_hash, verified.model, fx["nonce"]
            )
            self.assertEqual(enclave.decrypt_bound_request(req), b"sensitive prompt")

    def test_adversarial_matrix_all_rejected(self) -> None:
        attacks = []

        # 1. Forged root sig
        fx = _make_fixture()
        sig = bytearray(fx["attestation"].platform_cert.root_signature)
        sig[0] ^= 1
        fx["attestation"].platform_cert.root_signature = bytes(sig)
        attacks.append(("forged_root_sig", fx, "INVALID_PLATFORM_CERT_SIGNATURE"))

        # 2. Untrusted root
        fx = _make_fixture()
        fx["attestation"].platform_cert.root_id = "untrusted-root"
        attacks.append(("untrusted_root", fx, "UNTRUSTED_ROOT"))

        # 3. Revoked platform cert
        fx = _make_fixture()
        fx["sovereign"].revocations.revoked_platform_cert_ids.add(
            fx["attestation"].platform_cert.cert_id
        )
        attacks.append(("revoked_plat_cert", fx, "PLATFORM_CERT_REVOKED"))

        # 4. Debug mode enabled
        fx = _make_fixture()
        fx["attestation"].quote.environment = EnvironmentState(False, True, True, 7)
        _resign_quote(fx)
        attacks.append(("debug_enabled", fx, "DEBUG_MODE_ENABLED"))

        # 5. Stale expired quote
        fx = _make_fixture()
        fx["attestation"].quote.expires_at_unix = fx["now_unix"] - 1
        _resign_quote(fx)
        attacks.append(("quote_expired", fx, "QUOTE_EXPIRED"))

        # 6. Wrong nonce
        fx = _make_fixture()
        fx["attestation"].quote.client_nonce = b"\xaa" * 32
        _resign_quote(fx, recompute_rd=True)
        attacks.append(("nonce_mismatch", fx, "NONCE_MISMATCH"))

        # 7. TCB downgrade
        fx = _make_fixture()
        fx["attestation"].platform_cert.tcb_version = 2
        fx["attestation"].platform_cert.root_signature = ed25519_sign(
            fx["root_seed"], fx["attestation"].platform_cert.signing_preimage()
        )
        fx["attestation"].quote.environment = EnvironmentState(True, True, True, 2)
        _resign_quote(fx)
        attacks.append(("tcb_downgrade", fx, "TCB_VERSION_TOO_LOW"))

        # 8. Software measurement mismatch
        fx = _make_fixture()
        fx["attestation"].quote.measurement = SoftwareMeasurement(
            rtmr_kernel=sha256(b"kernel-rtmr-v1"),
            rtmr_runtime=sha256(b"evil-runtime"),
            enclave_code_hash=sha256(b"hanzo-engine-enclave-bin-v1"),
        )
        _resign_quote(fx)
        attacks.append(("measurement_mismatch", fx, "MEASUREMENT_MISMATCH"))

        # 9. Swapped model weights
        fx = _make_fixture()
        old = fx["attestation"].quote.model
        fx["attestation"].quote.model = ModelCommitment(
            model_id=old.model_id,
            model_weights_sha256=sha256(b"backdoored-weights"),
            tokenizer_sha256=old.tokenizer_sha256,
            workload_policy_hash=old.workload_policy_hash,
        )
        _resign_quote(fx, recompute_rd=True)
        attacks.append(("swapped_model", fx, "MODEL_WEIGHTS_MISMATCH"))

        # 10. MITM swapped ephemeral X25519 key (report_data mismatch)
        fx = _make_fixture()
        fx["attestation"].quote.enclave_x25519_pubkey = x25519_public_key(
            sha256(b"attacker-x25519")
        )
        _resign_quote(fx, recompute_rd=False)
        attacks.append(("mitm_x25519", fx, "REPORT_DATA_MISMATCH"))

        # 11. Low-order all-zero X25519 key
        fx = _make_fixture()
        fx["attestation"].quote.enclave_x25519_pubkey = b"\x00" * 32
        _resign_quote(fx, recompute_rd=True)
        attacks.append(("low_order_x25519", fx, "INVALID_EPHEMERAL_KEY"))

        for name, fx, expected_code in attacks:
            with self.subTest(attack=name):
                sov = fx["sovereign"]
                with self.assertRaises(AttestError) as ctx:
                    verify_attestation(
                        fx["attestation"],
                        sov.trust_store,
                        sov.policy,
                        sov.revocations,
                        fx["nonce"],
                        fx["now_unix"],
                    )
                self.assertEqual(ctx.exception.code, expected_code)

        # 12. Tampered AEAD ciphertext & swapped AAD header
        fx = _make_fixture()
        sov = fx["sovereign"]
        verified = verify_attestation(
            fx["attestation"],
            sov.trust_store,
            sov.policy,
            sov.revocations,
            fx["nonce"],
            fx["now_unix"],
        )
        req = bind_and_encrypt_request(
            verified, fx["client_sk"], fx["aead_nonce"], b"top secret"
        )
        enclave = EnclaveChannel(
            fx["enclave_sk"], verified.quote_hash, verified.model, fx["nonce"]
        )
        bad_ct = copy.deepcopy(req)
        bad_ct.ciphertext = bytes([bad_ct.ciphertext[0] ^ 1]) + bad_ct.ciphertext[1:]
        with self.assertRaises(AttestError) as ctx:
            enclave.decrypt_bound_request(bad_ct)
        self.assertEqual(ctx.exception.code, "AEAD_AUTH_FAILED")

        bad_aad = copy.deepcopy(req)
        bad_aad.model_weights_sha256 = sha256(b"other-model")
        with self.assertRaises(AttestError) as ctx:
            enclave.decrypt_bound_request(bad_aad)
        self.assertEqual(ctx.exception.code, "BINDING_HEADER_MISMATCH")

    def test_sovereign_bundle_and_session_lifecycle(self) -> None:
        fx = _make_fixture("amd_sev_snp")
        raw = fx["sovereign"].export_json()
        reloaded = SovereignBundle.import_json(raw)
        self.assertEqual(reloaded.export_json(), raw)

        verified = verify_attestation(
            fx["attestation"],
            reloaded.trust_store,
            reloaded.policy,
            reloaded.revocations,
            fx["nonce"],
            fx["now_unix"],
        )
        sess = AttestedSession(verified, fx["client_sk"], fx["aead_nonce"], max_requests=2)
        r1 = sess.encrypt_request(fx["now_unix"], b"one")
        r2 = sess.encrypt_request(fx["now_unix"], b"two")
        self.assertNotEqual(r1.aead_nonce, r2.aead_nonce)
        with self.assertRaises(AttestError) as ctx:
            sess.encrypt_request(fx["now_unix"], b"three")
        self.assertEqual(ctx.exception.code, "SESSION_EXPIRED")


if __name__ == "__main__":
    unittest.main()
