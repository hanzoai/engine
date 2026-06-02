#!/bin/sh
set -e

# Hanzo Engine Installation Script
# Downloads pre-built binaries or builds from source with automatic hardware detection.

REPO="hanzoai/engine"
INSTALL_DIR="$HOME/.hanzo/bin"
BINARY="hanzo-engine"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { printf "${BLUE}info:${NC} %s\n" "$1" >&2; }
success() { printf "${GREEN}ok:${NC} %s\n" "$1" >&2; }
warn()    { printf "${YELLOW}warn:${NC} %s\n" "$1" >&2; }
error()   { printf "${RED}error:${NC} %s\n" "$1" >&2; exit 1; }

# Check if we can prompt the user
can_prompt() {
    [ -t 0 ] || [ -e /dev/tty ]
}

read_input() {
    if [ -t 0 ]; then
        read -r REPLY
    else
        read -r REPLY </dev/tty
    fi
}

print_banner() {
    printf "${BOLD}"
    cat <<'BANNER'
  _   _                         _____             _
 | | | | __ _ _ __  _______   | ____|_ __   __ _(_)_ __   ___
 | |_| |/ _` | '_ \|_  / _ \  |  _| | '_ \ / _` | | '_ \ / _ \
 |  _  | (_| | | | |/ / (_) | | |___| | | | (_| | | | | |  __/
 |_| |_|\__,_|_| |_/___\___/  |_____|_| |_|\__, |_|_| |_|\___|
                                            |___/
BANNER
    printf "${NC}"
    printf "${BLUE}High-performance AI inference for production.${NC}\n\n"
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       error "Unsupported OS: $(uname -s). Hanzo Engine supports Linux and macOS." ;;
    esac
}

# Detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *)             error "Unsupported architecture: $(uname -m)" ;;
    esac
}

# Get latest release tag from GitHub
get_latest_version() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
            | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p'
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
            | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p'
    else
        echo ""
    fi
}

# Download a file
download() {
    url="$1"
    dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$dest" "$url"
    else
        error "Neither curl nor wget found. Install one and retry."
    fi
}

# Try to install from pre-built binary release
try_binary_install() {
    os="$1"
    arch="$2"

    version=$(get_latest_version)
    if [ -z "$version" ]; then
        warn "Could not determine latest release version."
        return 1
    fi

    info "Latest version: $version"

    # Build expected asset name
    case "$os" in
        macos) os_tag="apple-darwin" ;;
        linux) os_tag="unknown-linux-gnu" ;;
    esac

    asset="${BINARY}-${arch}-${os_tag}.tar.gz"
    url="https://github.com/${REPO}/releases/download/${version}/${asset}"

    info "Downloading ${asset}..."
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    if download "$url" "$tmpdir/$asset" 2>/dev/null; then
        info "Extracting..."
        tar -xzf "$tmpdir/$asset" -C "$tmpdir"

        # Find the binary
        bin=$(find "$tmpdir" -name "$BINARY" -type f | head -1)
        if [ -z "$bin" ]; then
            warn "Binary not found in archive."
            return 1
        fi

        mkdir -p "$INSTALL_DIR"
        cp "$bin" "$INSTALL_DIR/$BINARY"
        chmod +x "$INSTALL_DIR/$BINARY"
        success "Installed $BINARY $version to $INSTALL_DIR/$BINARY"
        return 0
    else
        warn "Pre-built binary not available for ${os}/${arch}."
        return 1
    fi
}

# Minimum required Rust version
REQUIRED_RUST_VERSION="1.88"

check_rust() {
    command -v cargo >/dev/null 2>&1
}

get_rust_version() {
    rustc --version 2>/dev/null | sed -n 's/rustc \([0-9]*\.[0-9]*\).*/\1/p'
}

version_gte() {
    v1_major=$(echo "$1" | cut -d. -f1)
    v1_minor=$(echo "$1" | cut -d. -f2)
    v2_major=$(echo "$2" | cut -d. -f1)
    v2_minor=$(echo "$2" | cut -d. -f2)
    if [ "$v1_major" -gt "$v2_major" ] 2>/dev/null; then
        return 0
    elif [ "$v1_major" -eq "$v2_major" ] 2>/dev/null && [ "$v1_minor" -ge "$v2_minor" ] 2>/dev/null; then
        return 0
    fi
    return 1
}

install_rust() {
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    success "Rust installed."
}

update_rust() {
    info "Updating Rust..."
    rustup update stable
    success "Rust updated."
}

# Detect CUDA compute capability
detect_cuda_compute_cap() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$cc" ]; then
        echo "$cc"
    fi
}

detect_mkl() {
    if [ -n "$MKLROOT" ] && [ -d "$MKLROOT" ]; then return 0; fi
    for path in /opt/intel/oneapi/mkl/latest /opt/intel/mkl /opt/intel/oneapi/mkl; do
        if [ -d "$path" ]; then return 0; fi
    done
    return 1
}

is_intel_cpu() {
    if [ -f /proc/cpuinfo ]; then
        grep -qi "intel" /proc/cpuinfo && return 0
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "intel" && return 0
    fi
    return 1
}

detect_cudnn() {
    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
        if [ -f "$path/libcudnn.so" ] || ls "$path"/libcudnn.so.* >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

check_xcode_cli_tools() {
    if ! xcrun --version >/dev/null 2>&1; then
        warn "Xcode Command Line Tools not installed."
        if can_prompt; then
            printf "Install them now? [Y/n] "
            read_input
            case "$REPLY" in
                [Nn]*) error "Xcode CLT required for Metal support." ;;
            esac
        fi
        info "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "Complete the installation dialog, then press Enter."
        read_input
        sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    fi
}

check_metal_toolchain() {
    if ! xcrun metal --version >/dev/null 2>&1; then
        warn "Metal Toolchain not installed."
        if can_prompt; then
            printf "Install it now? [Y/n] "
            read_input
            case "$REPLY" in
                [Nn]*) error "Metal Toolchain required for Metal support." ;;
            esac
        fi
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain
    fi
}

# Build cargo feature flags based on detected hardware
build_features() {
    os="$1"
    features=""

    if [ "$os" = "macos" ]; then
        check_xcode_cli_tools
        check_metal_toolchain
        features="metal"
        info "macOS detected -- enabling Metal"
    else
        cuda_cc=$(detect_cuda_compute_cap)
        if [ -n "$cuda_cc" ]; then
            features="cuda"
            cc_major=$(echo "$cuda_cc" | cut -c1)
            cc_minor=$(echo "$cuda_cc" | cut -c2-)
            info "CUDA detected (compute capability: ${cc_major}.${cc_minor})"

            if detect_cudnn; then
                features="$features cudnn"
                info "cuDNN detected -- enabling"
            fi

            if [ "$cuda_cc" = "90" ]; then
                features="$features flash-attn-v3"
                info "Hopper GPU -- enabling flash-attn-v3"
            elif [ "$cuda_cc" -ge 80 ] 2>/dev/null; then
                features="$features flash-attn"
                info "Ampere+ GPU -- enabling flash-attn"
            fi
        else
            info "No NVIDIA GPU detected."
        fi
    fi

    if is_intel_cpu && detect_mkl; then
        features="$features mkl"
        info "Intel MKL detected -- enabling"
    fi

    echo "$features" | xargs
}

# Build from source via cargo
source_install() {
    os="$1"

    # Ensure Rust is available
    if check_rust; then
        rust_version=$(get_rust_version)
        info "Rust installed: $(rustc --version 2>/dev/null)"
        if [ -n "$rust_version" ] && ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
            warn "Rust $rust_version < required $REQUIRED_RUST_VERSION"
            if can_prompt; then
                printf "Update Rust? [Y/n] "
                read_input
                case "$REPLY" in
                    [Nn]*) error "Rust $REQUIRED_RUST_VERSION+ required." ;;
                esac
            fi
            update_rust
        fi
    else
        warn "Rust not installed."
        if can_prompt; then
            printf "Install Rust? [Y/n] "
            read_input
            case "$REPLY" in
                [Nn]*) error "Rust required for source install." ;;
            esac
        fi
        install_rust
    fi

    info "Detecting hardware..."
    features=$(build_features "$os")

    echo ""
    printf "${BOLD}Build Configuration${NC}\n"
    echo "==================="
    if [ -n "$features" ]; then
        printf "Features: ${GREEN}%s${NC}\n" "$features"
    else
        printf "Features: ${YELLOW}(none -- CPU only)${NC}\n"
    fi
    echo ""

    if can_prompt; then
        printf "Proceed? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*) info "Cancelled."; exit 0 ;;
        esac
    fi

    if [ -n "$features" ]; then
        info "Building hanzo-engine with features: $features"
        cargo install hanzo-engine --features "$features"
    else
        info "Building hanzo-engine (CPU only)"
        cargo install hanzo-engine
    fi

    success "hanzo-engine built and installed via cargo."
}

# Add install dir to PATH
setup_path() {
    case "$SHELL" in
        */zsh)  rc="$HOME/.zshrc" ;;
        */bash) rc="$HOME/.bashrc" ;;
        *)      rc="$HOME/.profile" ;;
    esac

    path_line="export PATH=\"$INSTALL_DIR:\$PATH\""

    if [ -f "$rc" ] && grep -qF "$INSTALL_DIR" "$rc" 2>/dev/null; then
        return
    fi

    echo "" >> "$rc"
    echo "# Hanzo Engine" >> "$rc"
    echo "$path_line" >> "$rc"
    info "Added $INSTALL_DIR to PATH in $rc"
}

# Print version
print_version() {
    if [ -x "$INSTALL_DIR/$BINARY" ]; then
        version=$("$INSTALL_DIR/$BINARY" --version 2>/dev/null || echo "installed")
        success "$BINARY $version"
    elif command -v "$BINARY" >/dev/null 2>&1; then
        version=$("$BINARY" --version 2>/dev/null || echo "installed")
        success "$BINARY $version"
    fi
}

main() {
    print_banner

    os=$(detect_os)
    arch=$(detect_arch)
    info "Platform: $os/$arch"

    # Try pre-built binary first
    info "Checking for pre-built binary..."
    if try_binary_install "$os" "$arch"; then
        setup_path
        export PATH="$INSTALL_DIR:$PATH"
        print_version
    else
        info "Falling back to source build..."
        source_install "$os"

        # cargo install puts binary in ~/.cargo/bin; also copy to our dir
        if [ -x "$HOME/.cargo/bin/$BINARY" ]; then
            mkdir -p "$INSTALL_DIR"
            cp "$HOME/.cargo/bin/$BINARY" "$INSTALL_DIR/$BINARY"
            setup_path
        fi
    fi

    echo ""
    printf "${BOLD}Quick Start${NC}\n"
    echo "==========="
    echo ""
    echo "  hanzo-engine serve --model zenlm/zen4 --port 8000"
    echo ""
    echo "  hanzo-engine serve --model zenlm/zen4-coder --port 8000 --features cuda"
    echo ""
    echo "Docs: https://engine.hanzo.ai"
    echo "Source: https://github.com/hanzoai/engine"
    echo ""

    if [ -f "$HOME/.cargo/env" ]; then
        printf "${YELLOW}Note:${NC} Run ${BOLD}. \"\$HOME/.cargo/env\"${NC} or restart your terminal.\n"
    fi
}

main "$@"
