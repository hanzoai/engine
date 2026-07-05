#!/bin/sh
# Hanzo Engine installer — downloads a prebuilt, signed `hanzoai` binary.
#
#   curl -fsSL https://raw.githubusercontent.com/hanzoai/engine/main/install.sh | sh
#
# Detects your OS + CPU, fetches the matching bundle from the latest GitHub
# release, verifies it (cosign signature if `cosign` is present, else SHA256SUMS),
# and installs `hanzoai` onto your PATH. No Rust, no CUDA, no compiler required.
#
# Environment overrides:
#   HANZOAI_VERSION=v1.7.6      install a specific tag (default: latest)
#   HANZOAI_INSTALL_DIR=/path   install location (default: first writable of
#                               /usr/local/bin, ~/.local/bin, ~/.hanzo/bin)
#   HANZOAI_NO_VERIFY=1         skip signature/checksum verification
#   HANZOAI_BASE_URL=https://…  release mirror base (air-gapped / self-hosted;
#                               expects <base>/<asset>, <base>/<asset>.sig, …)
set -eu

REPO="hanzoai/engine"
BIN="hanzoai"

# ---- pretty output (to stderr so `| sh` stays clean) ------------------------
if [ -t 2 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi
info()    { printf "${BLUE}==>${NC} %s\n" "$1" >&2; }
success() { printf "${GREEN}✓${NC} %s\n" "$1" >&2; }
warn()    { printf "${YELLOW}warning:${NC} %s\n" "$1" >&2; }
error()   { printf "${RED}error:${NC} %s\n" "$1" >&2; exit 1; }

# ---- platform detection -----------------------------------------------------
detect_os() {
  case "$(uname -s)" in
    Linux*)                 echo linux ;;
    Darwin*)                echo macos ;;
    MINGW*|MSYS*|CYGWIN*)   echo windows ;;
    *) error "unsupported operating system: $(uname -s)" ;;
  esac
}
detect_arch() {
  # On Windows/git-bash uname -m may report the emulation arch; PROCESSOR_ARCHITECTURE is authoritative.
  arch="$(uname -m)"
  case "${PROCESSOR_ARCHITECTURE:-}${PROCESSOR_ARCHITEW6432:-}" in
    *ARM64*|*arm64*) echo arm64; return ;;
  esac
  case "$arch" in
    x86_64|amd64|x64)          echo amd64 ;;
    aarch64|arm64|armv8*)      echo arm64 ;;
    *) error "unsupported CPU architecture: $arch" ;;
  esac
}

# ---- http helpers -----------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }
fetch() { # fetch <url> <dest>
  if have curl; then curl -fSL --retry 3 -o "$2" "$1"
  elif have wget; then wget -q -O "$2" "$1"
  else error "need curl or wget to download"; fi
}
fetch_ok() { # url exists? (HEAD)
  if have curl; then curl -fsIL -o /dev/null "$1" 2>/dev/null
  elif have wget; then wget -q --spider "$1" 2>/dev/null
  else return 1; fi
}

# ---- install-dir resolution -------------------------------------------------
choose_dir() {
  if [ -n "${HANZOAI_INSTALL_DIR:-}" ]; then echo "$HANZOAI_INSTALL_DIR"; return; fi
  for d in /usr/local/bin "$HOME/.local/bin" "$HOME/.hanzo/bin"; do
    if [ -d "$d" ] && [ -w "$d" ]; then echo "$d"; return; fi
    if [ ! -d "$d" ] && mkdir -p "$d" 2>/dev/null; then echo "$d"; return; fi
  done
  echo "$HOME/.hanzo/bin"
}

main() {
  OS="$(detect_os)"; ARCH="$(detect_arch)"
  case "$OS" in windows) EXT=zip; BINF="${BIN}.exe" ;; *) EXT=tar.gz; BINF="$BIN" ;; esac
  ASSET="${BIN}-${OS}-${ARCH}.${EXT}"

  if [ -n "${HANZOAI_BASE_URL:-}" ]; then
    BASE="${HANZOAI_BASE_URL%/}"
    TAG="${HANZOAI_VERSION:-mirror}"
  elif [ -n "${HANZOAI_VERSION:-}" ]; then
    BASE="https://github.com/${REPO}/releases/download/${HANZOAI_VERSION}"
    TAG="${HANZOAI_VERSION}"
  else
    BASE="https://github.com/${REPO}/releases/latest/download"
    TAG="latest"
  fi
  URL="${BASE}/${ASSET}"

  info "Hanzo Engine — installing ${BOLD}${ASSET}${NC} (${TAG})"
  fetch_ok "$URL" || error "no prebuilt binary for ${OS}/${ARCH} in release ${TAG}.
       Available targets: linux amd64/arm64, macos arm64, windows amd64/arm64.
       See https://github.com/${REPO}/releases — or build from source (see README)."

  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT INT TERM
  info "Downloading $URL"
  fetch "$URL" "$TMP/$ASSET"

  # ---- verify ---------------------------------------------------------------
  if [ "${HANZOAI_NO_VERIFY:-0}" != "1" ]; then
    if have cosign && fetch_ok "${URL}.sig" && fetch_ok "${URL}.pem"; then
      fetch "${URL}.sig" "$TMP/$ASSET.sig"
      fetch "${URL}.pem" "$TMP/$ASSET.pem"
      if cosign verify-blob \
            --certificate "$TMP/$ASSET.pem" \
            --signature   "$TMP/$ASSET.sig" \
            --certificate-identity-regexp "https://github.com/${REPO%%/*}/.*" \
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
            "$TMP/$ASSET" >/dev/null 2>&1; then
        success "cosign signature verified"
      else
        error "cosign signature verification FAILED for $ASSET (set HANZOAI_NO_VERIFY=1 to override)"
      fi
    elif fetch_ok "${BASE}/SHA256SUMS" && have sha256sum; then
      fetch "${BASE}/SHA256SUMS" "$TMP/SHA256SUMS"
      want="$(grep " $ASSET\$" "$TMP/SHA256SUMS" 2>/dev/null | awk '{print $1}')"
      got="$(sha256sum "$TMP/$ASSET" | awk '{print $1}')"
      if [ -n "$want" ] && [ "$want" = "$got" ]; then
        success "sha256 checksum verified"
      else
        error "sha256 mismatch for $ASSET (want=$want got=$got)"
      fi
    else
      warn "no cosign and no SHA256SUMS available — skipping verification (install cosign for signed installs)"
    fi
  fi

  # ---- extract --------------------------------------------------------------
  info "Extracting"
  case "$EXT" in
    tar.gz) tar -xzf "$TMP/$ASSET" -C "$TMP" ;;
    zip)    have unzip || error "need 'unzip' to extract $ASSET"; unzip -qo "$TMP/$ASSET" -d "$TMP" ;;
  esac
  [ -f "$TMP/$BINF" ] || error "archive did not contain $BINF"
  chmod +x "$TMP/$BINF"

  # ---- install --------------------------------------------------------------
  DIR="$(choose_dir)"
  DEST="$DIR/$BINF"
  if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR" 2>/dev/null || { have sudo && sudo mkdir -p "$DIR"; } \
      || error "cannot create install dir $DIR"
  fi
  if [ -e "$DEST" ]; then info "Upgrading existing install at $DEST"; fi
  if mv "$TMP/$BINF" "$DEST" 2>/dev/null; then :
  elif have sudo; then
    warn "$DIR needs elevated permissions — using sudo"
    sudo mv "$TMP/$BINF" "$DEST"
  else
    error "cannot write to $DIR (no sudo). Re-run with HANZOAI_INSTALL_DIR=\$HOME/.hanzo/bin"
  fi
  success "installed $BINF -> $DEST"

  # ---- verify run + PATH hint ----------------------------------------------
  if "$DEST" --version >/dev/null 2>&1; then
    success "$("$DEST" --version 2>&1 | head -1)"
  else
    warn "installed, but '$DEST --version' did not run cleanly"
  fi
  case ":$PATH:" in
    *":$DIR:"*) : ;;
    *) warn "$DIR is not on your PATH. Add it:"
       printf "     ${BOLD}export PATH=\"%s:\$PATH\"${NC}\n" "$DIR" >&2 ;;
  esac

  printf "\n${BOLD}Next:${NC} serve an OpenAI + Anthropic compatible endpoint on :1234\n" >&2
  printf "  ${GREEN}%s --port 1234 run -m Qwen/Qwen3-4B${NC}\n\n" "$BIN" >&2
  printf "  # then hit it (OpenAI /v1/chat/completions + Anthropic /v1/messages):\n" >&2
  printf "  curl localhost:1234/v1/chat/completions -d '{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'\n\n" >&2
}

main "$@"
