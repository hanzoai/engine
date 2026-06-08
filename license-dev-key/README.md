# License DEV signing key (development only)

This directory holds the **development** Ed25519 keypair used to mint and verify license tokens in
unit tests and via the `license_issuer` example. It is NOT the production key.

- `dev_signing_key.b64` -- **PRIVATE** 32-byte Ed25519 seed, base64url (no padding). **gitignored, never committed.**
- `dev_public_key.b64` -- public verification key, base64url (committed, harmless).
- `dev_public_key.rs.txt` -- the same public key as a Rust `[u8; 32]` array literal; this is what is
  embedded as `HANZO_LICENSE_PUBKEY` in `hanzo-engine/src/license.rs`.

The matching PUBLIC key is the compile-time const `HANZO_LICENSE_PUBKEY` in
`hanzo-engine/src/license.rs` (clearly marked `DEV KEY -- replace with prod pubkey from KMS for release`).

## Regenerating the private seed on a fresh clone

`hanzo-engine/src/license.rs` tests read `dev_signing_key.b64`. Because it is gitignored, a fresh
clone must recreate it before `cargo test -p hanzo-engine license`. The committed
`dev_public_key.rs.txt` pins the public half, so any regenerated seed MUST reproduce that pubkey --
i.e. you cannot just generate a fresh random key, you must use the seed below (kept out of git on
purpose; this is a throwaway dev value, never used for releases):

```bash
# Recreate the dev signing seed locally (throwaway dev value; matches the embedded pubkey const).
printf '%s\n' 'jUY--qNv2H-bT1xLTtC2LBEgnLWRBBTYAm286BLJ43w' > license-dev-key/dev_signing_key.b64
```

For a real release, generate a fresh keypair on a private host, store the private key in the KMS / CI
secret, and replace `HANZO_LICENSE_PUBKEY` with the new public key.
