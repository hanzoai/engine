#!/usr/bin/env bash
# Install the spark-local nightly retrain as a systemd USER timer. Idempotent.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
ENV_DIR="$HOME/.config/router-retrain"

export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release -p hanzo-router-retrain --manifest-path "$ROOT/Cargo.toml"

# Linger keeps the user manager (and Persistent= catch-up) running without a login.
loginctl enable-linger "$USER" || true

mkdir -p "$UNIT_DIR" "$ENV_DIR"
install -m 0644 "$SCRIPT_DIR/router-retrain.service" "$UNIT_DIR/router-retrain.service"
install -m 0644 "$SCRIPT_DIR/router-retrain.timer" "$UNIT_DIR/router-retrain.timer"
if [[ ! -f "$ENV_DIR/env" ]]; then
  cat > "$ENV_DIR/env" <<'EOF'
# router-retrain environment (chmod 600). Fill ROUTER_ADMIN_TOKEN from KMS:
#   ROUTER_ADMIN_TOKEN=$(hanzo kms get hanzo/prod/router-admin-token)   # super-admin (IAM admin org)
ROUTER_ADMIN_TOKEN=
# Opted-in orgs for the per-org fits (object.ListTrainingContributorOrgs), comma-separated.
CONTRIB_ORGS=
# Flip to 1 only after the engine mounts heads-{scope}.safetensors + CI RBAC allows the restart.
DO_RELOAD=0
EOF
  chmod 600 "$ENV_DIR/env"
fi

systemctl --user daemon-reload
systemctl --user enable --now router-retrain.timer
echo "installed. next run:"; systemctl --user list-timers router-retrain.timer --no-pager || true
