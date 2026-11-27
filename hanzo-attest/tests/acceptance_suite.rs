//! Acceptance and cross-language interoperability test suite for `hanzo-attest`
//! (`hanzoai/engine#69`, HIP-1350).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use hanzo_attest::crypto::{
    chacha20_block, chacha20_poly1305_decrypt, chacha20_poly1305_encrypt, ed25519_public_key,
    ed25519_sign, ed25519_verify, hex_decode, hex_decode_array, hex_encode, poly1305_mac, sha256,
    x25519, x25519_diffie_hellman, x25519_public_key,
};
use hanzo_attest::json::{parse_json, JsonValue};
use hanzo_attest::{
    bind_and_encrypt_request, compute_report_data, verify_attestation, AttestationBundle,
    AttestationQuote, AttestedSession, EnclaveChannel, EncryptedBoundRequest, EnvironmentState,
    ErrorCode, HardwareRootCert, ModelCommitment, NonceReplayCache, PlatformCert, RevocationList,
    SoftwareMeasurement, SovereignBundle, TeePlatform, TrustStore, VerificationPolicy,
};

struct TestEnv {
    sovereign: SovereignBundle,
    attestation: AttestationBundle,
    root_seed: [u8; 32],
    platform_seed: [u8; 32],
    enclave_x25519_secret: [u8; 32],
    client_x25519_secret: [u8; 32],
    client_nonce: [u8; 32],
    aead_nonce: [u8; 12],
    now_unix: u64,
}

fn make_env(platform: TeePlatform) -> TestEnv {
    let now_unix = 1_760_000_000u64;
    let root_seed = sha256(format!("test-root-{}", platform.as_str()).as_bytes());
    let platform_seed = sha256(format!("test-plat-{}", platform.as_str()).as_bytes());
    let enclave_x25519_secret = sha256(format!("test-enclave-{}", platform.as_str()).as_bytes());
    let client_x25519_secret = sha256(b"test-client-x25519");
    let client_nonce = sha256(format!("test-nonce-{}", platform.as_str()).as_bytes());
    let aead_nonce = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    let root_id = format!("root-{}", platform.as_str());
    let cert_id = format!("cert-{}", platform.as_str());
    let device_id = format!("dev-{}", platform.as_str());

    let mut trust_store = TrustStore::new();
    trust_store.insert(HardwareRootCert {
        root_id: root_id.clone(),
        platform,
        ed25519_public_key: ed25519_public_key(&root_seed),
        not_before_unix: now_unix - 10_000,
        not_after_unix: now_unix + 100_000,
    });

    let measurement = SoftwareMeasurement {
        rtmr_kernel: sha256(b"kernel-rtmr-v1"),
        rtmr_runtime: sha256(b"modelwrap-runtime-v1"),
        enclave_code_hash: sha256(b"hanzo-engine-enclave-bin-v1"),
    };
    let model = ModelCommitment {
        model_id: "hanzo/zen-9b-instruct-q4".to_string(),
        model_weights_sha256: sha256(b"zen-9b-weights-manifest"),
        tokenizer_sha256: sha256(b"zen-9b-tokenizer-json"),
        workload_policy_hash: sha256(b"hip-1350-confidential-policy-v1"),
    };

    let mut allowed_platforms = BTreeSet::new();
    allowed_platforms.insert(platform);
    let mut min_tcb_versions = BTreeMap::new();
    min_tcb_versions.insert(platform, 5);

    let policy = VerificationPolicy {
        allowed_platforms,
        min_tcb_versions,
        require_debug_disabled: true,
        require_secure_boot: true,
        require_memory_encryption: true,
        allowed_measurements: vec![measurement.clone()],
        expected_model: model.clone(),
        max_quote_age_secs: 300,
        max_clock_skew_secs: 15,
    };

    let revocations = RevocationList {
        min_revocation_epoch: 10,
        revoked_root_ids: BTreeSet::new(),
        revoked_platform_cert_ids: BTreeSet::new(),
        revoked_device_ids: BTreeSet::new(),
    };

    let platform_cert = PlatformCert::issue(
        cert_id.clone(),
        root_id,
        platform,
        device_id,
        ed25519_public_key(&platform_seed),
        7,
        now_unix - 100,
        now_unix + 10_000,
        &root_seed,
    );

    let enclave_pub = x25519_public_key(&enclave_x25519_secret);
    let quote = AttestationQuote::issue(
        format!("quote-{}", platform.as_str()),
        platform,
        cert_id,
        EnvironmentState {
            debug_disabled: true,
            secure_boot: true,
            memory_encryption: true,
            tcb_version: 7,
        },
        measurement,
        model,
        client_nonce,
        now_unix - 5,
        now_unix + 120,
        12,
        enclave_pub,
        &platform_seed,
    );

    TestEnv {
        sovereign: SovereignBundle {
            trust_store,
            policy,
            revocations,
        },
        attestation: AttestationBundle {
            platform_cert,
            quote,
        },
        root_seed,
        platform_seed,
        enclave_x25519_secret,
        client_x25519_secret,
        client_nonce,
        aead_nonce,
        now_unix,
    }
}

fn resign_quote(env: &mut TestEnv, recompute_rd: bool) {
    if recompute_rd {
        env.attestation.quote.report_data = compute_report_data(
            &env.attestation.quote.client_nonce,
            &env.attestation.quote.enclave_x25519_pubkey,
            &env.attestation.quote.model,
        );
    }
    let pre = env.attestation.quote.signing_preimage();
    env.attestation.quote.platform_signature = ed25519_sign(&env.platform_seed, &pre);
}

#[test]
fn test_rfc_crypto_vectors_comprehensive() {
    // RFC 8439 §2.3.2 ChaCha20 block test vector
    let key: [u8; 32] =
        hex_decode_array("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            .unwrap();
    let nonce: [u8; 12] = hex_decode_array("000000090000004a00000000").unwrap();
    let block = chacha20_block(&key, 1, &nonce);
    assert_eq!(hex_encode(&block[..16]), "10f1e7e4d13b5915500fdd1fa32071c4");

    // RFC 8439 §2.5.2 Poly1305 test vector
    let otk: [u8; 32] =
        hex_decode_array("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
            .unwrap();
    let mac = poly1305_mac(&otk, b"Cryptographic Forum Research Group");
    assert_eq!(hex_encode(&mac), "a8061dc1305136c6c22b8baf0c0127a9");

    // RFC 7748 §5.2 iterated X25519 (1 and 1,000 iterations)
    let mut k = [0u8; 32];
    k[0] = 9;
    let mut u = k;
    let mut next = x25519(&k, &u);
    assert_eq!(
        hex_encode(&next),
        "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079"
    );
    for _ in 1..1000 {
        u = k;
        k = next;
        next = x25519(&k, &u);
    }
    assert_eq!(
        hex_encode(&next),
        "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51"
    );

    // Non-canonical Ed25519 point encoding y = p (0xed..0x7f) must be rejected
    let non_canonical_pk: [u8; 32] = [
        0xed, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0x7f,
    ];
    assert!(!ed25519_verify(&non_canonical_pk, b"msg", &[0u8; 64]));

    // ChaCha20-Poly1305 AAD mutation rejection
    let (ct, tag) = chacha20_poly1305_encrypt(&key, &nonce, b"aad-good", b"secret prompt");
    assert_eq!(
        chacha20_poly1305_decrypt(&key, &nonce, b"aad-evil", &ct, &tag)
            .unwrap_err()
            .code,
        ErrorCode::AeadAuthFailed
    );
    assert_eq!(
        x25519_diffie_hellman(&key, &[0u8; 32]).unwrap_err().code,
        ErrorCode::InvalidEphemeralKey
    );
}

#[test]
fn test_multi_tee_platforms_verify_bind_and_decrypt() {
    for platform in [
        TeePlatform::NvidiaCcH100,
        TeePlatform::AmdSevSnp,
        TeePlatform::IntelTdx,
    ] {
        let env = make_env(platform);
        let mut cache = NonceReplayCache::new();
        let verified = verify_attestation(
            &env.attestation,
            &env.sovereign.trust_store,
            &env.sovereign.policy,
            &env.sovereign.revocations,
            &env.client_nonce,
            env.now_unix,
            Some(&mut cache),
        )
        .expect("genuine attestation must pass");
        assert_eq!(verified.platform(), platform);
        assert_eq!(cache.len(), 1);

        let prompt = b"Confidential clinical summary for HIP-1350 attested inference";
        let req =
            bind_and_encrypt_request(&verified, &env.client_x25519_secret, env.aead_nonce, prompt)
                .expect("encrypt bound request");

        let enclave = EnclaveChannel::new(
            env.enclave_x25519_secret,
            *verified.quote_hash(),
            verified.model().clone(),
            env.client_nonce,
        );
        let recovered = enclave
            .decrypt_bound_request(&req)
            .expect("enclave decrypt");
        assert_eq!(recovered, prompt);
    }
}

#[test]
fn test_all_adversarial_attacks_rejected_without_poisoning_replay_cache() {
    // Verify that an invalid quote carrying a legitimate nonce does NOT poison the replay cache.
    let mut env = make_env(TeePlatform::NvidiaCcH100);
    let mut cache = NonceReplayCache::new();
    let mut forged = env.attestation.clone();
    forged.quote.platform_signature[0] ^= 0xff;
    let err = verify_attestation(
        &forged,
        &env.sovereign.trust_store,
        &env.sovereign.policy,
        &env.sovereign.revocations,
        &env.client_nonce,
        env.now_unix,
        Some(&mut cache),
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidQuoteSignature);
    assert!(
        cache.is_empty(),
        "invalid quote must not poison replay cache"
    );

    // Genuine quote still succeeds, then replay fails.
    verify_attestation(
        &env.attestation,
        &env.sovereign.trust_store,
        &env.sovereign.policy,
        &env.sovereign.revocations,
        &env.client_nonce,
        env.now_unix,
        Some(&mut cache),
    )
    .expect("genuine quote succeeds");
    let replay_err = verify_attestation(
        &env.attestation,
        &env.sovereign.trust_store,
        &env.sovereign.policy,
        &env.sovereign.revocations,
        &env.client_nonce,
        env.now_unix,
        Some(&mut cache),
    )
    .unwrap_err();
    assert_eq!(replay_err.code, ErrorCode::NonceReplayed);

    // Debug mode enabled
    env.attestation.quote.environment.debug_disabled = false;
    resign_quote(&mut env, false);
    assert_eq!(
        verify_attestation(
            &env.attestation,
            &env.sovereign.trust_store,
            &env.sovereign.policy,
            &env.sovereign.revocations,
            &env.client_nonce,
            env.now_unix,
            None,
        )
        .unwrap_err()
        .code,
        ErrorCode::DebugModeEnabled
    );

    // TCB downgrade
    let mut env_tcb = make_env(TeePlatform::NvidiaCcH100);
    env_tcb.attestation.platform_cert.tcb_version = 4;
    env_tcb.attestation.quote.environment.tcb_version = 4;
    let cpre = env_tcb.attestation.platform_cert.signing_preimage();
    env_tcb.attestation.platform_cert.root_signature = ed25519_sign(&env_tcb.root_seed, &cpre);
    resign_quote(&mut env_tcb, false);
    assert_eq!(
        verify_attestation(
            &env_tcb.attestation,
            &env_tcb.sovereign.trust_store,
            &env_tcb.sovereign.policy,
            &env_tcb.sovereign.revocations,
            &env_tcb.client_nonce,
            env_tcb.now_unix,
            None,
        )
        .unwrap_err()
        .code,
        ErrorCode::TcbVersionTooLow
    );

    // MITM swapped ephemeral X25519 pubkey -> report_data mismatch
    let mut env_mitm = make_env(TeePlatform::NvidiaCcH100);
    env_mitm.attestation.quote.enclave_x25519_pubkey =
        x25519_public_key(&sha256(b"mitm-ephemeral-key"));
    resign_quote(&mut env_mitm, false);
    assert_eq!(
        verify_attestation(
            &env_mitm.attestation,
            &env_mitm.sovereign.trust_store,
            &env_mitm.sovereign.policy,
            &env_mitm.sovereign.revocations,
            &env_mitm.client_nonce,
            env_mitm.now_unix,
            None,
        )
        .unwrap_err()
        .code,
        ErrorCode::ReportDataMismatch
    );
}

#[test]
fn test_attested_session_nonce_uniqueness_and_expiry() {
    let env = make_env(TeePlatform::NvidiaCcH100);
    let verified = verify_attestation(
        &env.attestation,
        &env.sovereign.trust_store,
        &env.sovereign.policy,
        &env.sovereign.revocations,
        &env.client_nonce,
        env.now_unix,
        None,
    )
    .unwrap();
    let enclave = EnclaveChannel::new(
        env.enclave_x25519_secret,
        *verified.quote_hash(),
        verified.model().clone(),
        env.client_nonce,
    );

    let mut session =
        AttestedSession::open(&verified, &env.client_x25519_secret, env.aead_nonce, 2).unwrap();
    let r1 = session.encrypt_request(env.now_unix, b"prompt-1").unwrap();
    let r2 = session.encrypt_request(env.now_unix, b"prompt-2").unwrap();
    assert_ne!(r1.aead_nonce, r2.aead_nonce);
    assert_eq!(enclave.decrypt_bound_request(&r1).unwrap(), b"prompt-1");
    assert_eq!(enclave.decrypt_bound_request(&r2).unwrap(), b"prompt-2");
    assert_eq!(session.requests_sent(), 2);

    // 3rd request exceeds max_requests -> must fail with SessionExpired
    assert_eq!(
        session
            .encrypt_request(env.now_unix, b"prompt-3")
            .unwrap_err()
            .code,
        ErrorCode::SessionExpired
    );

    // Past quote expires_at_unix -> must fail with SessionExpired
    let mut session2 =
        AttestedSession::open(&verified, &env.client_x25519_secret, env.aead_nonce, 100).unwrap();
    assert_eq!(
        session2
            .encrypt_request(verified.expires_at_unix() + 1, b"late")
            .unwrap_err()
            .code,
        ErrorCode::SessionExpired
    );
}

#[test]
fn test_sovereign_bundle_json_roundtrip() {
    let env = make_env(TeePlatform::AmdSevSnp);
    let json = env.sovereign.export_json();
    let parsed = SovereignBundle::import_json(&json).unwrap();
    assert_eq!(parsed, env.sovereign);
    assert_eq!(parsed.export_json(), json);
}

#[test]
fn test_cross_language_rust_python_bidirectional_interop() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_root = manifest_dir.join("clients/python");

    let env = make_env(TeePlatform::NvidiaCcH100);
    let prompt = b"Cross-language confidential prompt verified by Rust and Python!";

    let verified = verify_attestation(
        &env.attestation,
        &env.sovereign.trust_store,
        &env.sovereign.policy,
        &env.sovereign.revocations,
        &env.client_nonce,
        env.now_unix,
        None,
    )
    .unwrap();
    let rust_req =
        bind_and_encrypt_request(&verified, &env.client_x25519_secret, env.aead_nonce, prompt)
            .unwrap();

    // 1. Rust encrypt -> Python decrypt
    let mut dec_in = BTreeMap::new();
    dec_in.insert(
        "enclave_x25519_secret_hex".to_string(),
        JsonValue::Str(hex_encode(&env.enclave_x25519_secret)),
    );
    dec_in.insert(
        "quote_hash_hex".to_string(),
        JsonValue::Str(hex_encode(verified.quote_hash())),
    );
    dec_in.insert("model".to_string(), verified.model().to_json_value());
    dec_in.insert(
        "client_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&env.client_nonce)),
    );
    dec_in.insert("encrypted_request".to_string(), rust_req.to_json_value());

    let mut child = Command::new("python3")
        .args(["-m", "hanzo_attest.cli", "decrypt"])
        .env("PYTHONPATH", &python_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 decrypt");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(JsonValue::Object(dec_in).to_canonical_string().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "python3 decrypt failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let dec_json = parse_json(&String::from_utf8(out.stdout).unwrap()).unwrap();
    let dec_obj = dec_json.as_object().unwrap();
    assert!(JsonValue::req(dec_obj, "ok").unwrap().as_bool().unwrap());
    let pt_hex = JsonValue::req(dec_obj, "plaintext_hex")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(hex_decode(pt_hex).unwrap(), prompt);

    // 2. Python verify + encrypt -> Rust decrypt & byte-for-byte wire parity
    let mut enc_in = BTreeMap::new();
    enc_in.insert(
        "sovereign_bundle".to_string(),
        env.sovereign.to_json_value(),
    );
    enc_in.insert(
        "attestation_bundle".to_string(),
        env.attestation.to_json_value(),
    );
    enc_in.insert(
        "expected_client_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&env.client_nonce)),
    );
    enc_in.insert("now_unix".to_string(), JsonValue::Uint(env.now_unix));
    enc_in.insert(
        "client_x25519_secret_hex".to_string(),
        JsonValue::Str(hex_encode(&env.client_x25519_secret)),
    );
    enc_in.insert(
        "aead_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&env.aead_nonce)),
    );
    enc_in.insert(
        "plaintext_hex".to_string(),
        JsonValue::Str(hex_encode(prompt)),
    );

    let mut child2 = Command::new("python3")
        .args(["-m", "hanzo_attest.cli", "encrypt"])
        .env("PYTHONPATH", &python_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 encrypt");
    child2
        .stdin
        .as_mut()
        .unwrap()
        .write_all(JsonValue::Object(enc_in).to_canonical_string().as_bytes())
        .unwrap();
    let out2 = child2.wait_with_output().unwrap();
    assert!(
        out2.status.success(),
        "python3 encrypt failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );
    let enc_json = parse_json(&String::from_utf8(out2.stdout).unwrap()).unwrap();
    let py_req = EncryptedBoundRequest::from_json_value(
        JsonValue::req(enc_json.as_object().unwrap(), "encrypted_request").unwrap(),
    )
    .unwrap();
    assert_eq!(
        py_req, rust_req,
        "Rust and Python wire envelopes must match identically"
    );

    let enclave = EnclaveChannel::new(
        env.enclave_x25519_secret,
        *verified.quote_hash(),
        verified.model().clone(),
        env.client_nonce,
    );
    assert_eq!(enclave.decrypt_bound_request(&py_req).unwrap(), prompt);
}
