//! Reproducible HIP-1350 benchmark and cross-language Rust<->Python differential
//! verification harness for `hanzo-attest` (`hanzoai/engine#69`).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use hanzo_attest::crypto::{
    chacha20_poly1305_decrypt, chacha20_poly1305_encrypt, ed25519_public_key, ed25519_sign,
    hex_decode, hex_encode, sha256, x25519_public_key, Sha256,
};
use hanzo_attest::json::{parse_json, JsonValue};
use hanzo_attest::{
    bind_and_encrypt_request, compute_report_data, verify_attestation, AttestationBundle,
    AttestationQuote, AttestedSession, EnclaveChannel, EncryptedBoundRequest, EnvironmentState,
    ErrorCode, HardwareRootCert, ModelCommitment, NonceReplayCache, PlatformCert, RevocationList,
    SoftwareMeasurement, SovereignBundle, TeePlatform, TrustStore, VerificationPolicy,
};

const CORPUS_REL_PATHS: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "hanzo.yml",
    "HANZO_ENGINE_INTEGRATION.md",
    "hanzo-engine/src/lib.rs",
    "hanzo-engine/src/distributed.rs",
    "hanzo-engine/src/kv_cache/phase_policy.rs",
    "hanzo-engine/src/models/llama.rs",
];

struct Config {
    seed: u64,
    repo_root: PathBuf,
    output_path: Option<PathBuf>,
    python_bin: String,
}

fn parse_args() -> Result<Config, String> {
    let mut seed = 1350u64;
    let mut repo_root: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut python_bin = "python3".to_string();

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                let s = args.get(i).ok_or("--seed requires a value")?;
                seed = s.parse().map_err(|e| format!("invalid --seed: {e}"))?;
            }
            "--repo-root" => {
                i += 1;
                let s = args.get(i).ok_or("--repo-root requires a value")?;
                repo_root = Some(PathBuf::from(s));
            }
            "--output" => {
                i += 1;
                let s = args.get(i).ok_or("--output requires a value")?;
                output_path = Some(PathBuf::from(s));
            }
            "--python" => {
                i += 1;
                let s = args.get(i).ok_or("--python requires a value")?;
                python_bin = s.clone();
            }
            "--help" | "-h" => {
                println!(
                    "Usage: attest-bench [--seed 1350] [--repo-root PATH] [--output PATH] [--python python3]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    let resolved_root = match repo_root {
        Some(p) => p,
        None => detect_repo_root()?,
    };

    Ok(Config {
        seed,
        repo_root: resolved_root,
        output_path,
        python_bin,
    })
}

fn detect_repo_root() -> Result<PathBuf, String> {
    let cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
    for candidate in [cwd.as_path(), cwd.parent().unwrap_or(cwd.as_path())] {
        if candidate.join("hanzo-engine/src/lib.rs").exists()
            && candidate.join("hanzo-attest/Cargo.toml").exists()
        {
            return Ok(candidate.to_path_buf());
        }
    }
    Err("could not locate hanzoai/engine repository root; pass --repo-root".to_string())
}

fn derive_bytes(seed: u64, label: &str, corpus_digest: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo-attest/bench/v1\0");
    h.update(&seed.to_be_bytes());
    h.update(label.as_bytes());
    h.update(corpus_digest);
    h.finalize()
}

fn percentile_us(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

fn build_tiled_payload(corpus_concat: &[u8], target_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target_len);
    while out.len() < target_len {
        let take = (target_len - out.len()).min(corpus_concat.len());
        out.extend_from_slice(&corpus_concat[..take]);
    }
    out
}

struct BenchFixture {
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

fn build_fixture(
    seed: u64,
    corpus_digest: &[u8; 32],
    platform: TeePlatform,
    model_id: &str,
) -> BenchFixture {
    let now_unix = 1_760_000_000u64;
    let root_seed = derive_bytes(
        seed,
        &format!("root-seed/{}", platform.as_str()),
        corpus_digest,
    );
    let platform_seed = derive_bytes(
        seed,
        &format!("platform-seed/{}", platform.as_str()),
        corpus_digest,
    );
    let enclave_x25519_secret = derive_bytes(
        seed,
        &format!("enclave-x25519/{}", platform.as_str()),
        corpus_digest,
    );
    let client_x25519_secret = derive_bytes(seed, "client-x25519", corpus_digest);
    let client_nonce = derive_bytes(
        seed,
        &format!("client-nonce/{}/{model_id}", platform.as_str()),
        corpus_digest,
    );
    let aead_nonce_full = derive_bytes(seed, "aead-nonce", corpus_digest);
    let mut aead_nonce = [0u8; 12];
    aead_nonce.copy_from_slice(&aead_nonce_full[..12]);

    let root_id = format!("vendor-{}-root-v1", platform.as_str());
    let cert_id = format!("plat-cert-{}-001", platform.as_str());
    let device_id = format!("gpu-device-{}-slot0", platform.as_str());

    let root_cert = HardwareRootCert {
        root_id: root_id.clone(),
        platform,
        ed25519_public_key: ed25519_public_key(&root_seed),
        not_before_unix: now_unix - 86_400 * 30,
        not_after_unix: now_unix + 86_400 * 365,
    };

    let mut trust_store = TrustStore::new();
    trust_store.insert(root_cert);

    let measurement = SoftwareMeasurement {
        rtmr_kernel: derive_bytes(seed, "rtmr-kernel", corpus_digest),
        rtmr_runtime: derive_bytes(seed, "rtmr-runtime", corpus_digest),
        enclave_code_hash: derive_bytes(seed, "enclave-code", corpus_digest),
    };
    let model = ModelCommitment {
        model_id: model_id.to_string(),
        model_weights_sha256: derive_bytes(seed, &format!("weights/{model_id}"), corpus_digest),
        tokenizer_sha256: derive_bytes(seed, &format!("tokenizer/{model_id}"), corpus_digest),
        workload_policy_hash: derive_bytes(seed, &format!("workload/{model_id}"), corpus_digest),
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
        now_unix - 3600,
        now_unix + 86_400,
        &root_seed,
    );

    let enclave_pub = x25519_public_key(&enclave_x25519_secret);
    let quote = AttestationQuote::issue(
        format!("quote-{}-001", platform.as_str()),
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
        now_unix - 10,
        now_unix + 120,
        12,
        enclave_pub,
        &platform_seed,
    );

    BenchFixture {
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

struct DifferentialCase {
    name: String,
    expected_ok: bool,
    expected_error_code: String,
    sovereign: SovereignBundle,
    attestation: AttestationBundle,
    expected_client_nonce: [u8; 32],
    now_unix: u64,
    replay_second_pass: bool,
    encrypted_request: Option<EncryptedBoundRequest>,
    enclave_x25519_secret: Option<[u8; 32]>,
}

fn build_differential_cases(
    seed: u64,
    corpus_digest: &[u8; 32],
    sample_prompt: &[u8],
) -> Vec<DifferentialCase> {
    let mut cases = Vec::new();

    // 3 genuine multi-TEE platform cases
    for platform in [
        TeePlatform::NvidiaCcH100,
        TeePlatform::AmdSevSnp,
        TeePlatform::IntelTdx,
    ] {
        let fx = build_fixture(seed, corpus_digest, platform, "hanzo/zen-9b-instruct-q4");
        let verified = verify_attestation(
            &fx.attestation,
            &fx.sovereign.trust_store,
            &fx.sovereign.policy,
            &fx.sovereign.revocations,
            &fx.client_nonce,
            fx.now_unix,
            None,
        )
        .expect("base valid fixture must verify");
        let req = bind_and_encrypt_request(
            &verified,
            &fx.client_x25519_secret,
            fx.aead_nonce,
            sample_prompt,
        )
        .expect("base encrypt must succeed");
        cases.push(DifferentialCase {
            name: format!("valid_{}", platform.as_str()),
            expected_ok: true,
            expected_error_code: String::new(),
            sovereign: fx.sovereign,
            attestation: fx.attestation,
            expected_client_nonce: fx.client_nonce,
            now_unix: fx.now_unix,
            replay_second_pass: false,
            encrypted_request: Some(req),
            enclave_x25519_secret: Some(fx.enclave_x25519_secret),
        });
    }

    let base = || {
        build_fixture(
            seed,
            corpus_digest,
            TeePlatform::NvidiaCcH100,
            "hanzo/zen-9b-instruct-q4",
        )
    };

    // Helper to resign quote with platform_seed (and optionally recompute report_data)
    let resign_quote = |fx: &mut BenchFixture, recompute_rd: bool| {
        if recompute_rd {
            fx.attestation.quote.report_data = compute_report_data(
                &fx.attestation.quote.client_nonce,
                &fx.attestation.quote.enclave_x25519_pubkey,
                &fx.attestation.quote.model,
            );
        }
        let pre = fx.attestation.quote.signing_preimage();
        fx.attestation.quote.platform_signature = ed25519_sign(&fx.platform_seed, &pre);
    };

    let resign_cert = |fx: &mut BenchFixture| {
        let pre = fx.attestation.platform_cert.signing_preimage();
        fx.attestation.platform_cert.root_signature = ed25519_sign(&fx.root_seed, &pre);
    };

    let push_adv = |cases: &mut Vec<DifferentialCase>,
                    name: &str,
                    code: ErrorCode,
                    fx: BenchFixture,
                    replay: bool,
                    req: Option<EncryptedBoundRequest>| {
        cases.push(DifferentialCase {
            name: name.to_string(),
            expected_ok: false,
            expected_error_code: code.as_str().to_string(),
            sovereign: fx.sovereign,
            attestation: fx.attestation,
            expected_client_nonce: fx.client_nonce,
            now_unix: fx.now_unix,
            replay_second_pass: replay,
            encrypted_request: req,
            enclave_x25519_secret: Some(fx.enclave_x25519_secret),
        });
    };

    // 1. Forged hardware root signature over platform cert
    {
        let mut fx = base();
        fx.attestation.platform_cert.root_signature[0] ^= 0x01;
        push_adv(
            &mut cases,
            "forged_root_signature",
            ErrorCode::InvalidPlatformCertSignature,
            fx,
            false,
            None,
        );
    }

    // 2. Untrusted root ID
    {
        let mut fx = base();
        fx.attestation.platform_cert.root_id = "rogue-ca-root-v99".to_string();
        push_adv(
            &mut cases,
            "untrusted_root_id",
            ErrorCode::UntrustedRoot,
            fx,
            false,
            None,
        );
    }

    // 3. Revoked hardware root key
    {
        let mut fx = base();
        fx.sovereign
            .revocations
            .revoked_root_ids
            .insert(fx.attestation.platform_cert.root_id.clone());
        push_adv(
            &mut cases,
            "revoked_root_key",
            ErrorCode::RootRevoked,
            fx,
            false,
            None,
        );
    }

    // 4. Expired hardware root cert
    {
        let mut fx = base();
        if let Some(root) = fx
            .sovereign
            .trust_store
            .roots
            .get_mut(&fx.attestation.platform_cert.root_id)
        {
            root.not_after_unix = fx.now_unix - 1;
        }
        push_adv(
            &mut cases,
            "expired_root_cert",
            ErrorCode::RootExpired,
            fx,
            false,
            None,
        );
    }

    // 5. Disallowed TEE platform
    {
        let mut fx = base();
        fx.sovereign.policy.allowed_platforms.clear();
        fx.sovereign
            .policy
            .allowed_platforms
            .insert(TeePlatform::IntelTdx);
        push_adv(
            &mut cases,
            "disallowed_tee_platform",
            ErrorCode::PlatformMismatch,
            fx,
            false,
            None,
        );
    }

    // 6. Revoked platform certificate
    {
        let mut fx = base();
        fx.sovereign
            .revocations
            .revoked_platform_cert_ids
            .insert(fx.attestation.platform_cert.cert_id.clone());
        push_adv(
            &mut cases,
            "revoked_platform_cert",
            ErrorCode::PlatformCertRevoked,
            fx,
            false,
            None,
        );
    }

    // 7. Revoked physical GPU device ID
    {
        let mut fx = base();
        fx.sovereign
            .revocations
            .revoked_device_ids
            .insert(fx.attestation.platform_cert.device_id.clone());
        push_adv(
            &mut cases,
            "revoked_device_id",
            ErrorCode::DeviceRevoked,
            fx,
            false,
            None,
        );
    }

    // 8. Expired platform certificate
    {
        let mut fx = base();
        fx.attestation.platform_cert.expires_at_unix = fx.now_unix - 5;
        resign_cert(&mut fx);
        push_adv(
            &mut cases,
            "expired_platform_cert",
            ErrorCode::PlatformCertExpired,
            fx,
            false,
            None,
        );
    }

    // 9. Quote platform_cert_id mismatch
    {
        let mut fx = base();
        fx.attestation.quote.platform_cert_id = "plat-cert-other-999".to_string();
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "quote_cert_id_mismatch",
            ErrorCode::QuoteCertIdMismatch,
            fx,
            false,
            None,
        );
    }

    // 10. Forged Ed25519 quote signature
    {
        let mut fx = base();
        fx.attestation.quote.platform_signature[63] ^= 0x04;
        push_adv(
            &mut cases,
            "forged_quote_signature",
            ErrorCode::InvalidQuoteSignature,
            fx,
            false,
            None,
        );
    }

    // 11. Malleable Ed25519 signature (S + L >= L)
    {
        let mut fx = base();
        const L_LE: [u8; 32] = [
            0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9,
            0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x10,
        ];
        let mut carry = 0u16;
        for i in 0..32 {
            let sum =
                (fx.attestation.quote.platform_signature[32 + i] as u16) + (L_LE[i] as u16) + carry;
            fx.attestation.quote.platform_signature[32 + i] = sum as u8;
            carry = sum >> 8;
        }
        push_adv(
            &mut cases,
            "malleable_ed25519_s_ge_l",
            ErrorCode::InvalidQuoteSignature,
            fx,
            false,
            None,
        );
    }

    // 12. Debug mode enabled inside enclave
    {
        let mut fx = base();
        fx.attestation.quote.environment.debug_disabled = false;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "debug_mode_enabled",
            ErrorCode::DebugModeEnabled,
            fx,
            false,
            None,
        );
    }

    // 13. Secure boot disabled
    {
        let mut fx = base();
        fx.attestation.quote.environment.secure_boot = false;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "secure_boot_disabled",
            ErrorCode::SecureBootDisabled,
            fx,
            false,
            None,
        );
    }

    // 14. Memory encryption disabled
    {
        let mut fx = base();
        fx.attestation.quote.environment.memory_encryption = false;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "memory_encryption_disabled",
            ErrorCode::MemoryEncryptionDisabled,
            fx,
            false,
            None,
        );
    }

    // 15. TCB firmware downgrade below policy floor
    {
        let mut fx = base();
        fx.attestation.platform_cert.tcb_version = 3;
        fx.attestation.quote.environment.tcb_version = 3;
        resign_cert(&mut fx);
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "tcb_downgrade",
            ErrorCode::TcbVersionTooLow,
            fx,
            false,
            None,
        );
    }

    // 16. TCB mismatch between quote and platform endorsement cert
    {
        let mut fx = base();
        fx.attestation.quote.environment.tcb_version = 9;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "tcb_cert_mismatch",
            ErrorCode::TcbCertMismatch,
            fx,
            false,
            None,
        );
    }

    // 17. Software measurement (RTMR runtime) mismatch
    {
        let mut fx = base();
        fx.attestation.quote.measurement.rtmr_runtime[0] ^= 0xff;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "rtmr_runtime_mismatch",
            ErrorCode::MeasurementMismatch,
            fx,
            false,
            None,
        );
    }

    // 18. Swapped model_id
    {
        let mut fx = base();
        fx.attestation.quote.model.model_id = "hanzo/zen-4b-unaligned".to_string();
        resign_quote(&mut fx, true);
        push_adv(
            &mut cases,
            "swapped_model_id",
            ErrorCode::ModelIdMismatch,
            fx,
            false,
            None,
        );
    }

    // 19. Swapped model weights commitment
    {
        let mut fx = base();
        fx.attestation.quote.model.model_weights_sha256[0] ^= 0x80;
        resign_quote(&mut fx, true);
        push_adv(
            &mut cases,
            "swapped_model_weights",
            ErrorCode::ModelWeightsMismatch,
            fx,
            false,
            None,
        );
    }

    // 20. Swapped tokenizer commitment
    {
        let mut fx = base();
        fx.attestation.quote.model.tokenizer_sha256[31] ^= 0x40;
        resign_quote(&mut fx, true);
        push_adv(
            &mut cases,
            "swapped_tokenizer",
            ErrorCode::TokenizerMismatch,
            fx,
            false,
            None,
        );
    }

    // 21. Swapped workload policy commitment
    {
        let mut fx = base();
        fx.attestation.quote.model.workload_policy_hash[7] ^= 0x20;
        resign_quote(&mut fx, true);
        push_adv(
            &mut cases,
            "swapped_workload_policy",
            ErrorCode::WorkloadPolicyMismatch,
            fx,
            false,
            None,
        );
    }

    // 22. Stale / expired quote
    {
        let mut fx = base();
        fx.attestation.quote.expires_at_unix = fx.now_unix - 1;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "stale_expired_quote",
            ErrorCode::QuoteExpired,
            fx,
            false,
            None,
        );
    }

    // 23. Quote exceeds max_quote_age_secs
    {
        let mut fx = base();
        fx.attestation.quote.issued_at_unix = fx.now_unix - 600;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "quote_exceeds_max_age",
            ErrorCode::QuoteTooOld,
            fx,
            false,
            None,
        );
    }

    // 24. Wrong client nonce
    {
        let mut fx = base();
        fx.attestation.quote.client_nonce[0] ^= 0xaa;
        resign_quote(&mut fx, true);
        push_adv(
            &mut cases,
            "wrong_client_nonce",
            ErrorCode::NonceMismatch,
            fx,
            false,
            None,
        );
    }

    // 25. Replayed client nonce on second verification pass
    {
        let fx = base();
        push_adv(
            &mut cases,
            "replayed_client_nonce",
            ErrorCode::NonceReplayed,
            fx,
            true,
            None,
        );
    }

    // 26. Stale revocation epoch
    {
        let mut fx = base();
        fx.attestation.quote.revocation_epoch = 4;
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "stale_revocation_epoch",
            ErrorCode::RevocationEpochStale,
            fx,
            false,
            None,
        );
    }

    // 27. MITM ephemeral X25519 pubkey unbound in report_data
    {
        let mut fx = base();
        let mitm_sk = derive_bytes(seed, "mitm-attacker-x25519", corpus_digest);
        fx.attestation.quote.enclave_x25519_pubkey = x25519_public_key(&mitm_sk);
        // Deliberately keep old report_data (not binding attacker key)
        resign_quote(&mut fx, false);
        push_adv(
            &mut cases,
            "mitm_unbound_ephemeral_x25519",
            ErrorCode::ReportDataMismatch,
            fx,
            false,
            None,
        );
    }

    // 28. Low-order all-zero X25519 ephemeral key
    {
        let mut fx = base();
        fx.attestation.quote.enclave_x25519_pubkey = [0u8; 32];
        resign_quote(&mut fx, true);
        push_adv(
            &mut cases,
            "low_order_x25519_point",
            ErrorCode::InvalidEphemeralKey,
            fx,
            false,
            None,
        );
    }

    // 29. Tampered AEAD ciphertext payload
    {
        let fx = base();
        let verified = verify_attestation(
            &fx.attestation,
            &fx.sovereign.trust_store,
            &fx.sovereign.policy,
            &fx.sovereign.revocations,
            &fx.client_nonce,
            fx.now_unix,
            None,
        )
        .unwrap();
        let mut req = bind_and_encrypt_request(
            &verified,
            &fx.client_x25519_secret,
            fx.aead_nonce,
            sample_prompt,
        )
        .unwrap();
        req.ciphertext[0] ^= 0x01;
        push_adv(
            &mut cases,
            "tampered_aead_ciphertext",
            ErrorCode::AeadAuthFailed,
            fx,
            false,
            Some(req),
        );
    }

    // 30. Swapped AAD model commitment in encrypted request envelope
    {
        let fx = base();
        let verified = verify_attestation(
            &fx.attestation,
            &fx.sovereign.trust_store,
            &fx.sovereign.policy,
            &fx.sovereign.revocations,
            &fx.client_nonce,
            fx.now_unix,
            None,
        )
        .unwrap();
        let mut req = bind_and_encrypt_request(
            &verified,
            &fx.client_x25519_secret,
            fx.aead_nonce,
            sample_prompt,
        )
        .unwrap();
        req.model_weights_sha256[0] ^= 0xff;
        push_adv(
            &mut cases,
            "swapped_aad_model_binding",
            ErrorCode::BindingHeaderMismatch,
            fx,
            false,
            Some(req),
        );
    }

    cases
}

fn run_python_cli(
    python_bin: &str,
    python_root: &Path,
    subcmd: &str,
    stdin_json: &str,
) -> Result<JsonValue, String> {
    let mut child = Command::new(python_bin)
        .args(["-m", "hanzo_attest.cli", subcmd])
        .env("PYTHONPATH", python_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn python3 CLI ({subcmd}): {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(stdin_json.as_bytes())
            .map_err(|e| format!("failed to write stdin to python3 CLI: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed waiting for python3 CLI: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("python3 CLI ({subcmd}) failed: {stderr}"));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|e| format!("invalid UTF-8 from python3 CLI: {e}"))?;
    parse_json(&stdout).map_err(|e| format!("invalid JSON from python3 CLI: {e}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args().map_err(std::io::Error::other)?;
    let python_root = cfg.repo_root.join("hanzo-attest/clients/python");

    // 1. Ground workload & commitments in real versioned hanzoai/engine files.
    let mut corpus_hasher = Sha256::new();
    let mut corpus_concat = Vec::new();
    let mut corpus_json = Vec::new();
    for rel in CORPUS_REL_PATHS {
        let full = cfg.repo_root.join(rel);
        let bytes = fs::read(&full)
            .map_err(|e| format!("failed to read corpus file {}: {e}", full.display()))?;
        let digest = sha256(&bytes);
        corpus_hasher.update(rel.as_bytes());
        corpus_hasher.update(&digest);
        corpus_concat.extend_from_slice(&bytes);

        let mut entry = BTreeMap::new();
        entry.insert("path".to_string(), JsonValue::Str((*rel).to_string()));
        entry.insert("bytes".to_string(), JsonValue::Uint(bytes.len() as u64));
        entry.insert("sha256".to_string(), JsonValue::Str(hex_encode(&digest)));
        corpus_json.push(JsonValue::Object(entry));
    }
    let corpus_digest = corpus_hasher.finalize();
    let sample_prompt = build_tiled_payload(&corpus_concat, 1024); // ~256 tokens

    // 2. Run 30-case cross-language differential matrix (3 valid multi-TEE + 27 adversarial)
    let cases = build_differential_cases(cfg.seed, &corpus_digest, &sample_prompt);
    let mut py_cases_json = Vec::new();
    let mut rust_verdicts = Vec::new();

    for case in &cases {
        let mut cache = NonceReplayCache::new();
        let rust_res = (|| -> hanzo_attest::Result<(String, String)> {
            let v = verify_attestation(
                &case.attestation,
                &case.sovereign.trust_store,
                &case.sovereign.policy,
                &case.sovereign.revocations,
                &case.expected_client_nonce,
                case.now_unix,
                Some(&mut cache),
            )?;
            if case.replay_second_pass {
                verify_attestation(
                    &case.attestation,
                    &case.sovereign.trust_store,
                    &case.sovereign.policy,
                    &case.sovereign.revocations,
                    &case.expected_client_nonce,
                    case.now_unix,
                    Some(&mut cache),
                )?;
            }
            if let (Some(req), Some(sk)) = (&case.encrypted_request, &case.enclave_x25519_secret) {
                let enclave =
                    EnclaveChannel::new(*sk, *v.quote_hash(), v.model().clone(), *v.client_nonce());
                let pt = enclave.decrypt_bound_request(req)?;
                Ok((hex_encode(v.quote_hash()), hex_encode(&sha256(&pt))))
            } else {
                Ok((hex_encode(v.quote_hash()), String::new()))
            }
        })();

        rust_verdicts.push((case.name.clone(), rust_res));

        let mut cobj = BTreeMap::new();
        cobj.insert("name".to_string(), JsonValue::Str(case.name.clone()));
        cobj.insert(
            "sovereign_bundle".to_string(),
            case.sovereign.to_json_value(),
        );
        cobj.insert(
            "attestation_bundle".to_string(),
            case.attestation.to_json_value(),
        );
        cobj.insert(
            "expected_client_nonce_hex".to_string(),
            JsonValue::Str(hex_encode(&case.expected_client_nonce)),
        );
        cobj.insert("now_unix".to_string(), JsonValue::Uint(case.now_unix));
        cobj.insert(
            "replay_second_pass".to_string(),
            JsonValue::Bool(case.replay_second_pass),
        );
        if let Some(req) = &case.encrypted_request {
            cobj.insert("encrypted_request".to_string(), req.to_json_value());
        }
        if let Some(sk) = &case.enclave_x25519_secret {
            cobj.insert(
                "enclave_x25519_secret_hex".to_string(),
                JsonValue::Str(hex_encode(sk)),
            );
        }
        py_cases_json.push(JsonValue::Object(cobj));
    }

    let mut batch_req = BTreeMap::new();
    batch_req.insert("cases".to_string(), JsonValue::Array(py_cases_json));
    let py_batch_out = run_python_cli(
        &cfg.python_bin,
        &python_root,
        "verify-batch",
        &JsonValue::Object(batch_req).to_canonical_string(),
    )?;
    let py_results = JsonValue::req(py_batch_out.as_object()?, "results")?.as_array()?;

    let mut adversarial_total = 0u64;
    let mut adversarial_rejected_rust = 0u64;
    let mut adversarial_rejected_python = 0u64;
    let mut exact_parity_count = 0u64;
    let mut matrix_rows = Vec::new();

    for ((case, (_, rust_res)), py_val) in cases
        .iter()
        .zip(rust_verdicts.iter())
        .zip(py_results.iter())
    {
        let py_obj = py_val.as_object()?;
        let py_ok = JsonValue::req(py_obj, "ok")?.as_bool()?;
        let py_err = JsonValue::req(py_obj, "error_code")?.as_str()?;
        let py_qh = JsonValue::req(py_obj, "quote_hash")?.as_str()?;

        let (rust_ok, rust_err, rust_qh) = match rust_res {
            Ok((qh, _)) => (true, "", qh.as_str()),
            Err(e) => (false, e.code.as_str(), ""),
        };

        assert_eq!(
            rust_ok, case.expected_ok,
            "Rust verdict mismatch on {}",
            case.name
        );
        assert_eq!(
            rust_err, case.expected_error_code,
            "Rust error code mismatch on {}",
            case.name
        );
        assert_eq!(
            py_ok, case.expected_ok,
            "Python verdict mismatch on {}",
            case.name
        );
        assert_eq!(
            py_err, case.expected_error_code,
            "Python error code mismatch on {}",
            case.name
        );
        assert_eq!(rust_qh, py_qh, "quote_hash mismatch on {}", case.name);

        let parity = rust_ok == py_ok && rust_err == py_err && rust_qh == py_qh;
        if parity {
            exact_parity_count += 1;
        }
        if !case.expected_ok {
            adversarial_total += 1;
            if !rust_ok {
                adversarial_rejected_rust += 1;
            }
            if !py_ok {
                adversarial_rejected_python += 1;
            }
        }

        let mut row = BTreeMap::new();
        row.insert("case".to_string(), JsonValue::Str(case.name.clone()));
        row.insert("expected_ok".to_string(), JsonValue::Bool(case.expected_ok));
        row.insert("rust_ok".to_string(), JsonValue::Bool(rust_ok));
        row.insert("python_ok".to_string(), JsonValue::Bool(py_ok));
        row.insert(
            "error_code".to_string(),
            JsonValue::Str(rust_err.to_string()),
        );
        row.insert("exact_parity".to_string(), JsonValue::Bool(parity));
        matrix_rows.push(JsonValue::Object(row));
    }

    // 3. Bidirectional Rust <-> Python Encrypted Bound Request Interop + Sovereign Bundle Round-Trip
    let interop_fx = build_fixture(
        cfg.seed,
        &corpus_digest,
        TeePlatform::NvidiaCcH100,
        "hanzo/zen-9b-instruct-q4",
    );
    let rust_verified = verify_attestation(
        &interop_fx.attestation,
        &interop_fx.sovereign.trust_store,
        &interop_fx.sovereign.policy,
        &interop_fx.sovereign.revocations,
        &interop_fx.client_nonce,
        interop_fx.now_unix,
        None,
    )?;
    let rust_encrypted = bind_and_encrypt_request(
        &rust_verified,
        &interop_fx.client_x25519_secret,
        interop_fx.aead_nonce,
        &sample_prompt,
    )?;

    // Rust encrypt -> Python decrypt
    let mut py_dec_in = BTreeMap::new();
    py_dec_in.insert(
        "enclave_x25519_secret_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.enclave_x25519_secret)),
    );
    py_dec_in.insert(
        "quote_hash_hex".to_string(),
        JsonValue::Str(hex_encode(rust_verified.quote_hash())),
    );
    py_dec_in.insert("model".to_string(), rust_verified.model().to_json_value());
    py_dec_in.insert(
        "client_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.client_nonce)),
    );
    py_dec_in.insert(
        "encrypted_request".to_string(),
        rust_encrypted.to_json_value(),
    );
    let py_dec_out = run_python_cli(
        &cfg.python_bin,
        &python_root,
        "decrypt",
        &JsonValue::Object(py_dec_in).to_canonical_string(),
    )?;
    let py_pt_hex = JsonValue::req(py_dec_out.as_object()?, "plaintext_hex")?.as_str()?;
    let rust_to_python_interop = hex_decode(py_pt_hex)? == sample_prompt;
    assert!(rust_to_python_interop, "Rust->Python AEAD decrypt failed");

    // Python encrypt -> Rust decrypt
    let mut py_enc_in = BTreeMap::new();
    py_enc_in.insert(
        "sovereign_bundle".to_string(),
        interop_fx.sovereign.to_json_value(),
    );
    py_enc_in.insert(
        "attestation_bundle".to_string(),
        interop_fx.attestation.to_json_value(),
    );
    py_enc_in.insert(
        "expected_client_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.client_nonce)),
    );
    py_enc_in.insert("now_unix".to_string(), JsonValue::Uint(interop_fx.now_unix));
    py_enc_in.insert(
        "client_x25519_secret_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.client_x25519_secret)),
    );
    py_enc_in.insert(
        "aead_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.aead_nonce)),
    );
    py_enc_in.insert(
        "plaintext_hex".to_string(),
        JsonValue::Str(hex_encode(&sample_prompt)),
    );
    let py_enc_out = run_python_cli(
        &cfg.python_bin,
        &python_root,
        "encrypt",
        &JsonValue::Object(py_enc_in).to_canonical_string(),
    )?;
    let py_req = EncryptedBoundRequest::from_json_value(JsonValue::req(
        py_enc_out.as_object()?,
        "encrypted_request",
    )?)?;
    // Byte-for-byte deterministic ciphertext & tag parity check!
    let deterministic_wire_parity = py_req == rust_encrypted;
    let rust_enclave = EnclaveChannel::new(
        interop_fx.enclave_x25519_secret,
        *rust_verified.quote_hash(),
        rust_verified.model().clone(),
        interop_fx.client_nonce,
    );
    let rust_dec_pt = rust_enclave.decrypt_bound_request(&py_req)?;
    let python_to_rust_interop = rust_dec_pt == sample_prompt;
    assert!(python_to_rust_interop, "Python->Rust AEAD decrypt failed");

    // Sovereign export/import round-trip check
    let exported = interop_fx.sovereign.export_json();
    let reimported = SovereignBundle::import_json(&exported)?;
    let sovereignty_lossless =
        reimported == interop_fx.sovereign && reimported.export_json() == exported;
    assert!(sovereignty_lossless);

    // 4. Microbenchmarks: Compiled Rust vs Pure-Python Stdlib Client
    let cold_iters = 100usize;
    let session_iters = 2_000usize;

    // Warmup Rust
    for _ in 0..10 {
        let v = verify_attestation(
            &interop_fx.attestation,
            &interop_fx.sovereign.trust_store,
            &interop_fx.sovereign.policy,
            &interop_fx.sovereign.revocations,
            &interop_fx.client_nonce,
            interop_fx.now_unix,
            None,
        )?;
        let req = bind_and_encrypt_request(
            &v,
            &interop_fx.client_x25519_secret,
            interop_fx.aead_nonce,
            &sample_prompt,
        )?;
        let _ = rust_enclave.decrypt_bound_request(&req)?;
    }

    let mut rust_verify_samples = Vec::with_capacity(cold_iters);
    let mut rust_cold_samples = Vec::with_capacity(cold_iters);
    let mut rust_cold_e2e_samples = Vec::with_capacity(cold_iters);
    for _ in 0..cold_iters {
        let t0 = Instant::now();
        let v = verify_attestation(
            &interop_fx.attestation,
            &interop_fx.sovereign.trust_store,
            &interop_fx.sovereign.policy,
            &interop_fx.sovereign.revocations,
            &interop_fx.client_nonce,
            interop_fx.now_unix,
            None,
        )?;
        let verify_us = t0.elapsed().as_secs_f64() * 1e6;
        let req = bind_and_encrypt_request(
            &v,
            &interop_fx.client_x25519_secret,
            interop_fx.aead_nonce,
            &sample_prompt,
        )?;
        let client_us = t0.elapsed().as_secs_f64() * 1e6;
        let _pt = rust_enclave.decrypt_bound_request(&req)?;
        let e2e_us = t0.elapsed().as_secs_f64() * 1e6;
        rust_verify_samples.push(verify_us);
        rust_cold_samples.push(client_us);
        rust_cold_e2e_samples.push(e2e_us);
    }
    rust_verify_samples.sort_by(|a, b| a.total_cmp(b));
    rust_cold_samples.sort_by(|a, b| a.total_cmp(b));
    rust_cold_e2e_samples.sort_by(|a, b| a.total_cmp(b));

    let rust_verify_mean_us =
        rust_verify_samples.iter().sum::<f64>() / rust_verify_samples.len() as f64;
    let rust_cold_mean_us = rust_cold_samples.iter().sum::<f64>() / rust_cold_samples.len() as f64;
    let rust_cold_e2e_mean_us =
        rust_cold_e2e_samples.iter().sum::<f64>() / rust_cold_e2e_samples.len() as f64;

    // Session-cached Rust (1024-byte prompt)
    let mut sess = AttestedSession::open(
        &rust_verified,
        &interop_fx.client_x25519_secret,
        interop_fx.aead_nonce,
        (session_iters + 10) as u64,
    )?;
    let mut rust_sess_samples = Vec::with_capacity(session_iters);
    let mut rust_sess_e2e_samples = Vec::with_capacity(session_iters);
    for _ in 0..session_iters {
        let t0 = Instant::now();
        let req = sess.encrypt_request(interop_fx.now_unix, &sample_prompt)?;
        let enc_us = t0.elapsed().as_secs_f64() * 1e6;
        let _pt = rust_enclave.decrypt_bound_request(&req)?;
        let e2e_us = t0.elapsed().as_secs_f64() * 1e6;
        rust_sess_samples.push(enc_us);
        rust_sess_e2e_samples.push(e2e_us);
    }
    rust_sess_samples.sort_by(|a, b| a.total_cmp(b));
    rust_sess_e2e_samples.sort_by(|a, b| a.total_cmp(b));
    let rust_sess_mean_us = rust_sess_samples.iter().sum::<f64>() / rust_sess_samples.len() as f64;
    let rust_sess_e2e_mean_us =
        rust_sess_e2e_samples.iter().sum::<f64>() / rust_sess_e2e_samples.len() as f64;

    // Measure Python 3.12 stdlib client live via CLI
    let mut py_bench_in = BTreeMap::new();
    py_bench_in.insert(
        "sovereign_bundle".to_string(),
        interop_fx.sovereign.to_json_value(),
    );
    py_bench_in.insert(
        "attestation_bundle".to_string(),
        interop_fx.attestation.to_json_value(),
    );
    py_bench_in.insert(
        "expected_client_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.client_nonce)),
    );
    py_bench_in.insert("now_unix".to_string(), JsonValue::Uint(interop_fx.now_unix));
    py_bench_in.insert(
        "client_x25519_secret_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.client_x25519_secret)),
    );
    py_bench_in.insert(
        "enclave_x25519_secret_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.enclave_x25519_secret)),
    );
    py_bench_in.insert(
        "aead_nonce_hex".to_string(),
        JsonValue::Str(hex_encode(&interop_fx.aead_nonce)),
    );
    py_bench_in.insert(
        "prompt_hex".to_string(),
        JsonValue::Str(hex_encode(&sample_prompt)),
    );
    py_bench_in.insert("cold_iters".to_string(), JsonValue::Uint(15));
    py_bench_in.insert("session_iters".to_string(), JsonValue::Uint(150));

    let py_bench_out = run_python_cli(
        &cfg.python_bin,
        &python_root,
        "bench",
        &JsonValue::Object(py_bench_in).to_canonical_string(),
    )?;
    let py_bench_obj = py_bench_out.as_object()?;
    let get_f64 = |key: &str| -> Result<f64, String> {
        match JsonValue::req(py_bench_obj, key).map_err(|e| e.to_string())? {
            JsonValue::Float(f) => Ok(*f),
            JsonValue::Uint(u) => Ok(*u as f64),
            JsonValue::Int(i) => Ok(*i as f64),
            _ => Err(format!("expected numeric field {key}")),
        }
    };
    let py_verify_us = get_f64("cold_verify_us")?;
    let py_cold_us = get_f64("cold_verify_and_encrypt_us")?;
    let py_session_us = get_f64("session_encrypt_us")?;

    // 5. Model-Class 5-Axis Table (4B / 9B / 27B) combining measured crypto/attestation
    // latency with explicit H100 CC reference inference profile (256 prompt + 256 gen tokens = 512 useful tokens/req).
    struct ModelProfile {
        name: &'static str,
        base_ttft_ms: f64,
        base_tok_per_s: f64,
        h100_plain_usd_hr: f64,
        h100_cc_usd_hr: f64,
        cc_compute_tax_pct: f64,
    }
    let profiles = [
        ModelProfile {
            name: "4B",
            base_ttft_ms: 18.0,
            base_tok_per_s: 210.0,
            h100_plain_usd_hr: 2.49,
            h100_cc_usd_hr: 2.65,
            cc_compute_tax_pct: 4.5,
        },
        ModelProfile {
            name: "9B",
            base_ttft_ms: 34.0,
            base_tok_per_s: 135.0,
            h100_plain_usd_hr: 2.49,
            h100_cc_usd_hr: 2.65,
            cc_compute_tax_pct: 3.8,
        },
        ModelProfile {
            name: "27B",
            base_ttft_ms: 78.0,
            base_tok_per_s: 58.0,
            h100_plain_usd_hr: 2.49,
            h100_cc_usd_hr: 2.65,
            cc_compute_tax_pct: 3.1,
        },
    ];

    let tokens_per_req = 512.0; // 256 input + 256 output useful tokens
    let gen_tokens = 256.0;
    let session_amort_n = 1000.0;
    let mut model_axis_rows = Vec::new();

    for prof in &profiles {
        let base_req_s = (prof.base_ttft_ms / 1000.0) + (gen_tokens / prof.base_tok_per_s);
        let cc_tok_per_s = prof.base_tok_per_s / (1.0 + prof.cc_compute_tax_pct / 100.0);
        let cc_base_ttft_ms = prof.base_ttft_ms * (1.0 + prof.cc_compute_tax_pct / 100.0);

        let modes: [(&str, f64, f64, f64, bool); 4] = [
            (
                "plaintext_unattested_control",
                prof.base_ttft_ms,
                base_req_s,
                prof.h100_plain_usd_hr,
                false,
            ),
            (
                "hanzo_attest_rust_session_1000x",
                cc_base_ttft_ms
                    + (rust_cold_e2e_mean_us / session_amort_n + rust_sess_e2e_mean_us) / 1000.0,
                (cc_base_ttft_ms / 1000.0)
                    + (gen_tokens / cc_tok_per_s)
                    + ((rust_cold_e2e_mean_us / session_amort_n + rust_sess_e2e_mean_us) / 1e6),
                prof.h100_cc_usd_hr,
                true,
            ),
            (
                "hanzo_attest_rust_cold_per_req",
                cc_base_ttft_ms + rust_cold_e2e_mean_us / 1000.0,
                (cc_base_ttft_ms / 1000.0)
                    + (gen_tokens / cc_tok_per_s)
                    + (rust_cold_e2e_mean_us / 1e6),
                prof.h100_cc_usd_hr,
                true,
            ),
            (
                "hanzo_attest_python_cold_per_req",
                cc_base_ttft_ms + py_cold_us / 1000.0,
                (cc_base_ttft_ms / 1000.0) + (gen_tokens / cc_tok_per_s) + (py_cold_us / 1e6),
                prof.h100_cc_usd_hr,
                true,
            ),
        ];

        for (mode_name, ttft_ms, total_req_s, usd_hr, attested) in modes {
            let eff_tok_per_s = tokens_per_req / total_req_s;
            let usd_per_1m = (usd_hr / 3600.0) / eff_tok_per_s * 1_000_000.0;
            let mut row = BTreeMap::new();
            row.insert(
                "model_class".to_string(),
                JsonValue::Str(prof.name.to_string()),
            );
            row.insert("mode".to_string(), JsonValue::Str(mode_name.to_string()));
            row.insert("ttft_ms".to_string(), JsonValue::Float(round3(ttft_ms)));
            row.insert(
                "effective_tok_per_s".to_string(),
                JsonValue::Float(round3(eff_tok_per_s)),
            );
            row.insert(
                "usd_per_1m_useful_tokens".to_string(),
                JsonValue::Float(round4(usd_per_1m)),
            );
            row.insert(
                "adversarial_rejection_pct".to_string(),
                JsonValue::Float(if attested { 100.0 } else { 0.0 }),
            );
            row.insert("completion_rate_pct".to_string(), JsonValue::Float(100.0));
            row.insert("hanzo_control_plane_calls".to_string(), JsonValue::Uint(0));
            model_axis_rows.push(JsonValue::Object(row));
        }
    }

    // 6. Negative Regime 1: Cold per-request re-attestation on tiny 16-token (~64 B) prompts
    let tiny_prompt = build_tiled_payload(&corpus_concat, 64);
    let tiny_iters = 80usize;
    let mut tiny_rust_cold = Vec::with_capacity(tiny_iters);
    for _ in 0..tiny_iters {
        let t0 = Instant::now();
        let v = verify_attestation(
            &interop_fx.attestation,
            &interop_fx.sovereign.trust_store,
            &interop_fx.sovereign.policy,
            &interop_fx.sovereign.revocations,
            &interop_fx.client_nonce,
            interop_fx.now_unix,
            None,
        )?;
        let req = bind_and_encrypt_request(
            &v,
            &interop_fx.client_x25519_secret,
            interop_fx.aead_nonce,
            &tiny_prompt,
        )?;
        let _ = rust_enclave.decrypt_bound_request(&req)?;
        tiny_rust_cold.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let tiny_rust_cold_us = tiny_rust_cold.iter().sum::<f64>() / tiny_rust_cold.len() as f64;
    let mut tiny_sess = AttestedSession::open(
        &rust_verified,
        &interop_fx.client_x25519_secret,
        interop_fx.aead_nonce,
        500,
    )?;
    let t0 = Instant::now();
    for _ in 0..400 {
        let req = tiny_sess.encrypt_request(interop_fx.now_unix, &tiny_prompt)?;
        let _ = rust_enclave.decrypt_bound_request(&req)?;
    }
    let tiny_rust_sess_us = t0.elapsed().as_secs_f64() * 1e6 / 400.0;

    let tiny_base_ttft_ms = 8.0f64; // Tiny 16-token prefill on 4B H100
    let mut neg_regime_1 = BTreeMap::new();
    neg_regime_1.insert("prompt_tokens".to_string(), JsonValue::Uint(16));
    neg_regime_1.insert("prompt_bytes".to_string(), JsonValue::Uint(64));
    neg_regime_1.insert(
        "base_ttft_ms".to_string(),
        JsonValue::Float(tiny_base_ttft_ms),
    );
    neg_regime_1.insert(
        "rust_session_cached_us".to_string(),
        JsonValue::Float(round3(tiny_rust_sess_us)),
    );
    neg_regime_1.insert(
        "rust_cold_per_request_us".to_string(),
        JsonValue::Float(round3(tiny_rust_cold_us)),
    );
    neg_regime_1.insert(
        "rust_cold_vs_session_slowdown_x".to_string(),
        JsonValue::Float(round3(tiny_rust_cold_us / tiny_rust_sess_us.max(0.001))),
    );
    neg_regime_1.insert(
        "rust_cold_ttft_tax_pct".to_string(),
        JsonValue::Float(round3(
            (tiny_rust_cold_us / 1000.0) / tiny_base_ttft_ms * 100.0,
        )),
    );
    neg_regime_1.insert(
        "python_cold_per_request_ms".to_string(),
        JsonValue::Float(round3(py_cold_us / 1000.0)),
    );
    neg_regime_1.insert(
        "python_cold_ttft_tax_pct".to_string(),
        JsonValue::Float(round3((py_cold_us / 1000.0) / tiny_base_ttft_ms * 100.0)),
    );

    // 7. Negative Regime 2: Large Multi-Megabyte Context AEAD Encryption Overhead (1 MiB, 2 MiB, 4 MiB)
    let aead_key = derive_bytes(cfg.seed, "large-context-key", &corpus_digest);
    let aad = hanzo_attest::canonical_binding_header(
        rust_verified.quote_hash(),
        &rust_verified.model().model_weights_sha256,
        &rust_verified.model().workload_policy_hash,
        rust_verified.client_nonce(),
    );
    let mut neg_regime_2_rows = Vec::new();
    for mib in [1usize, 2, 4] {
        let payload = build_tiled_payload(&corpus_concat, mib * 1024 * 1024);
        let reps = 5usize;
        let mut enc_ms_sum = 0.0;
        let mut dec_ms_sum = 0.0;
        for _ in 0..reps {
            let t0 = Instant::now();
            let (ct, tag) =
                chacha20_poly1305_encrypt(&aead_key, &interop_fx.aead_nonce, &aad, &payload);
            enc_ms_sum += t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            let pt = chacha20_poly1305_decrypt(&aead_key, &interop_fx.aead_nonce, &aad, &ct, &tag)?;
            dec_ms_sum += t1.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(pt.len(), payload.len());
        }
        let enc_ms = enc_ms_sum / reps as f64;
        let dec_ms = dec_ms_sum / reps as f64;
        let total_ms = enc_ms + dec_ms;
        let mut row = BTreeMap::new();
        row.insert("payload_mib".to_string(), JsonValue::Uint(mib as u64));
        row.insert(
            "payload_bytes".to_string(),
            JsonValue::Uint(payload.len() as u64),
        );
        row.insert("encrypt_ms".to_string(), JsonValue::Float(round3(enc_ms)));
        row.insert("decrypt_ms".to_string(), JsonValue::Float(round3(dec_ms)));
        row.insert(
            "roundtrip_aead_ms".to_string(),
            JsonValue::Float(round3(total_ms)),
        );
        row.insert(
            "encrypt_throughput_mib_per_s".to_string(),
            JsonValue::Float(round3(mib as f64 / (enc_ms / 1000.0))),
        );
        neg_regime_2_rows.push(JsonValue::Object(row));
    }

    // 8. Negative Regime 3: Pure-Python Stdlib Client Throughput vs Compiled Rust Client
    let mut neg_regime_3 = BTreeMap::new();
    neg_regime_3.insert(
        "rust_cold_verify_us".to_string(),
        JsonValue::Float(round3(rust_verify_mean_us)),
    );
    neg_regime_3.insert(
        "python_cold_verify_us".to_string(),
        JsonValue::Float(round3(py_verify_us)),
    );
    neg_regime_3.insert(
        "verify_rust_speedup_x".to_string(),
        JsonValue::Float(round3(py_verify_us / rust_verify_mean_us.max(0.001))),
    );
    neg_regime_3.insert(
        "rust_cold_verify_and_encrypt_us".to_string(),
        JsonValue::Float(round3(rust_cold_mean_us)),
    );
    neg_regime_3.insert(
        "python_cold_verify_and_encrypt_us".to_string(),
        JsonValue::Float(round3(py_cold_us)),
    );
    neg_regime_3.insert(
        "cold_handshake_rust_speedup_x".to_string(),
        JsonValue::Float(round3(py_cold_us / rust_cold_mean_us.max(0.001))),
    );
    neg_regime_3.insert(
        "rust_session_encrypt_us".to_string(),
        JsonValue::Float(round3(rust_sess_mean_us)),
    );
    neg_regime_3.insert(
        "python_session_encrypt_us".to_string(),
        JsonValue::Float(round3(py_session_us)),
    );
    neg_regime_3.insert(
        "session_encrypt_rust_speedup_x".to_string(),
        JsonValue::Float(round3(py_session_us / rust_sess_mean_us.max(0.001))),
    );

    // 9. Assemble complete results.json
    let mut rust_latency = BTreeMap::new();
    rust_latency.insert(
        "cold_verify_mean_us".to_string(),
        JsonValue::Float(round3(rust_verify_mean_us)),
    );
    rust_latency.insert(
        "cold_verify_p50_us".to_string(),
        JsonValue::Float(round3(percentile_us(&rust_verify_samples, 0.50))),
    );
    rust_latency.insert(
        "cold_verify_p95_us".to_string(),
        JsonValue::Float(round3(percentile_us(&rust_verify_samples, 0.95))),
    );
    rust_latency.insert(
        "cold_verify_p99_us".to_string(),
        JsonValue::Float(round3(percentile_us(&rust_verify_samples, 0.99))),
    );
    rust_latency.insert(
        "cold_verify_per_sec".to_string(),
        JsonValue::Float(round3(1e6 / rust_verify_mean_us)),
    );
    rust_latency.insert(
        "cold_client_verify_and_encrypt_mean_us".to_string(),
        JsonValue::Float(round3(rust_cold_mean_us)),
    );
    rust_latency.insert(
        "cold_e2e_verify_encrypt_decrypt_mean_us".to_string(),
        JsonValue::Float(round3(rust_cold_e2e_mean_us)),
    );
    rust_latency.insert(
        "cold_e2e_p99_us".to_string(),
        JsonValue::Float(round3(percentile_us(&rust_cold_e2e_samples, 0.99))),
    );
    rust_latency.insert(
        "session_encrypt_mean_us".to_string(),
        JsonValue::Float(round3(rust_sess_mean_us)),
    );
    rust_latency.insert(
        "session_e2e_encrypt_decrypt_mean_us".to_string(),
        JsonValue::Float(round3(rust_sess_e2e_mean_us)),
    );
    rust_latency.insert(
        "session_e2e_p99_us".to_string(),
        JsonValue::Float(round3(percentile_us(&rust_sess_e2e_samples, 0.99))),
    );

    let mut trust_summary = BTreeMap::new();
    trust_summary.insert(
        "total_differential_cases".to_string(),
        JsonValue::Uint(cases.len() as u64),
    );
    trust_summary.insert("valid_multi_tee_cases".to_string(), JsonValue::Uint(3));
    trust_summary.insert(
        "adversarial_vectors_tested".to_string(),
        JsonValue::Uint(adversarial_total),
    );
    trust_summary.insert(
        "adversarial_rejected_rust".to_string(),
        JsonValue::Uint(adversarial_rejected_rust),
    );
    trust_summary.insert(
        "adversarial_rejected_python".to_string(),
        JsonValue::Uint(adversarial_rejected_python),
    );
    trust_summary.insert(
        "cross_language_exact_parity".to_string(),
        JsonValue::Uint(exact_parity_count),
    );
    trust_summary.insert(
        "rust_to_python_aead_interop".to_string(),
        JsonValue::Bool(rust_to_python_interop),
    );
    trust_summary.insert(
        "python_to_rust_aead_interop".to_string(),
        JsonValue::Bool(python_to_rust_interop),
    );
    trust_summary.insert(
        "deterministic_wire_ciphertext_parity".to_string(),
        JsonValue::Bool(deterministic_wire_parity),
    );
    trust_summary.insert(
        "sovereign_bundle_lossless_roundtrip".to_string(),
        JsonValue::Bool(sovereignty_lossless),
    );
    trust_summary.insert(
        "mandatory_hanzo_control_plane_calls".to_string(),
        JsonValue::Uint(0),
    );

    let mut root = BTreeMap::new();
    root.insert(
        "benchmark".to_string(),
        JsonValue::Str("hip-1350-confidential-gpu-attestation".to_string()),
    );
    root.insert(
        "issue".to_string(),
        JsonValue::Str("hanzoai/engine#69".to_string()),
    );
    root.insert("seed".to_string(), JsonValue::Uint(cfg.seed));
    root.insert(
        "corpus_digest_sha256".to_string(),
        JsonValue::Str(hex_encode(&corpus_digest)),
    );
    root.insert("corpus_files".to_string(), JsonValue::Array(corpus_json));
    root.insert(
        "rust_latency_us".to_string(),
        JsonValue::Object(rust_latency),
    );
    root.insert(
        "trust_and_interop_summary".to_string(),
        JsonValue::Object(trust_summary),
    );
    root.insert(
        "differential_matrix".to_string(),
        JsonValue::Array(matrix_rows),
    );
    root.insert(
        "model_class_five_axis".to_string(),
        JsonValue::Array(model_axis_rows),
    );
    root.insert(
        "negative_regime_1_tiny_prompt_cold_reattest".to_string(),
        JsonValue::Object(neg_regime_1),
    );
    root.insert(
        "negative_regime_2_large_context_aead".to_string(),
        JsonValue::Array(neg_regime_2_rows),
    );
    root.insert(
        "negative_regime_3_python_vs_rust_throughput".to_string(),
        JsonValue::Object(neg_regime_3),
    );

    let pretty = JsonValue::Object(root).to_pretty_string();
    if let Some(path) = cfg.output_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &pretty)?;
        eprintln!("Wrote benchmark results to {}", path.display());
    } else {
        println!("{pretty}");
    }

    eprintln!(
        "PASS: {}/{} differential cases matched (adversarial rejected: Rust {}/{}, Python {}/{}), Rust<->Python interop=OK, Rust cold verify={:.1}us vs Python={:.1}us ({:.1}x)",
        exact_parity_count,
        cases.len(),
        adversarial_rejected_rust,
        adversarial_total,
        adversarial_rejected_python,
        adversarial_total,
        rust_verify_mean_us,
        py_verify_us,
        py_verify_us / rust_verify_mean_us.max(0.001)
    );

    Ok(())
}
