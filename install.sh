#!/bin/sh
set -e

# hanzo Installation Script
# Cross-platform installer for Linux and macOS with automatic hardware detection

# Check if we can prompt the user (stdin is a tty or we have /dev/tty)
can_prompt() {
    [ -t 0 ] || [ -e /dev/tty ]
}

# Read user input, using /dev/tty if stdin is not a terminal (e.g., piped from curl)
read_input() {
    if [ -t 0 ]; then
        read -r REPLY
    else
        read -r REPLY </dev/tty
    fi
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions (output to stderr so they don't get captured in command substitution)
info() { printf "${BLUE}info:${NC} %s\n" "$1" >&2; }
success() { printf "${GREEN}success:${NC} %s\n" "$1" >&2; }
warn() { printf "${YELLOW}warning:${NC} %s\n" "$1" >&2; }
error() { printf "${RED}error:${NC} %s\n" "$1" >&2; exit 1; }

# Banner
print_banner() {
    printf "${BOLD}Hanzo Engine${NC} ${BLUE}- fast, flexible LLM inference.${NC}\n\n"
}

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        *)
            error "Unsupported operating system: $(uname -s)"
            ;;
    esac
}

# Minimum required Rust version
REQUIRED_RUST_VERSION="1.88"
HANZO_REPO_URL="https://github.com/hanzoai/engine"
HANZO_BRANCH="main"
HANZO_CLI_PACKAGE="hanzo-cli"

# Check if Rust is installed
check_rust() {
    command -v cargo >/dev/null 2>&1
}

# Get installed Rust version (major.minor)
get_rust_version() {
    rustc --version 2>/dev/null | sed -n 's/rustc \([0-9]*\.[0-9]*\).*/\1/p'
}

# Compare two version strings (returns 0 if $1 >= $2, 1 otherwise)
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

# Install Rust via rustup
install_rust() {
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    success "Rust installed successfully"
}

# Update Rust to latest version
update_rust() {
    info "Updating Rust to latest version..."
    rustup update stable
    success "Rust updated successfully"
}

# Detect CUDA compute capability
detect_cuda_compute_cap() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi

    # Try direct query
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

    if [ -n "$cc" ]; then
        echo "$cc"
    fi
}

# Detect CUDA toolkit major version from nvcc (e.g. 13). Empty if nvcc is unavailable.
# CUDA toolkit version as major*100+minor (e.g. 13.1 -> 1301), empty if nvcc absent.
detect_cuda_version_code() {
    if command -v nvcc >/dev/null 2>&1; then
        ver=$(nvcc --version 2>/dev/null | grep -oE "release [0-9]+\.[0-9]+" | head -1 | grep -oE "[0-9]+\.[0-9]+")
        if [ -n "$ver" ]; then
            echo $(( ${ver%%.*} * 100 + ${ver#*.} ))
        fi
    fi
}

# Check if MKL is installed
detect_mkl() {
    # Check MKLROOT environment variable
    if [ -n "$MKLROOT" ] && [ -d "$MKLROOT" ]; then
        return 0
    fi

    # Check common installation paths
    for path in /opt/intel/oneapi/mkl/latest /opt/intel/mkl /opt/intel/oneapi/mkl; do
        if [ -d "$path" ]; then
            return 0
        fi
    done

    return 1
}

# Check if CPU is Intel
is_intel_cpu() {
    if [ -f /proc/cpuinfo ]; then
        grep -qi "intel" /proc/cpuinfo && return 0
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "intel" && return 0
    fi
    return 1
}

# Check/install Xcode Command Line Tools (macOS)
check_xcode_cli_tools() {
    if ! xcrun --version >/dev/null 2>&1; then
        warn "Xcode Command Line Tools are not installed"
        echo ""
        printf "Would you like to install them now? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*)
                error "Xcode Command Line Tools are required for Metal support"
                ;;
        esac
        info "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "Please complete the installation in the dialog, then press Enter to continue..."
        read_input
        sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    fi
}

# Check/install Metal Toolchain (macOS)
check_metal_toolchain() {
    if ! xcrun metal --version >/dev/null 2>&1; then
        warn "Metal Toolchain is not installed"
        echo ""
        printf "Would you like to install it now? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*)
                error "Metal Toolchain is required for Metal support"
                ;;
        esac
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain
    fi
}

# Check if cuDNN is installed
detect_cudnn() {
    # Check common cuDNN library paths
    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
        if [ -f "$path/libcudnn.so" ] || ls "$path"/libcudnn.so.* >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

# Check if NCCL is installed
detect_nccl() {
    for root in "$NCCL_ROOT" "$NCCL_HOME" "$CUDA_HOME" "$CUDA_PATH" /usr/local/cuda; do
        [ -n "$root" ] || continue
        for subdir in lib lib64 lib/x86_64-linux-gnu; do
            if ls "$root/$subdir"/libnccl.so* >/dev/null 2>&1; then
                return 0
            fi
        done
    done

    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libnccl\\.so"; then
        return 0
    fi

    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/lib64 /usr/lib64; do
        if ls "$path"/libnccl.so* >/dev/null 2>&1; then
            return 0
        fi
    done

    return 1
}

# Native target of the AMD GPU (e.g. gfx1151). Empty when there is no ROCm GPU.
detect_rocm_arch() {
    if command -v offload-arch >/dev/null 2>&1; then
        arch=$(offload-arch 2>/dev/null | head -1)
        if [ -n "$arch" ]; then
            echo "$arch"
            return
        fi
    fi
    for bin in rocminfo "${ROCM_PATH:-/opt/rocm}/bin/rocminfo"; do
        command -v "$bin" >/dev/null 2>&1 || continue
        arch=$("$bin" 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)
        if [ -n "$arch" ]; then
            echo "$arch"
            return
        fi
    done
}

# The rocm feature compiles hanzo-rocm-kernels with hipcc and links rocBLAS.
detect_hipcc() {
    command -v hipcc >/dev/null 2>&1 || [ -x "${ROCM_PATH:-/opt/rocm}/bin/hipcc" ]
}

# Build feature string based on detected hardware
build_features() {
    os="$1"
    features=""

    if [ "$os" = "macos" ]; then
        features="metal"
        info "macOS detected - enabling metal"
    else
        # Check for CUDA
        cuda_cc=$(detect_cuda_compute_cap)
        if [ -n "$cuda_cc" ]; then
            features="cuda"
            # cuda_cc is the dot-stripped cap, so the minor is always the last digit:
            # 89 -> 8.9, 121 -> 12.1 (string surgery printed 1.21 for a 3-digit cap).
            cc_major=$(( cuda_cc / 10 ))
            cc_minor=$(( cuda_cc % 10 ))
            info "CUDA detected (compute capability: ${cc_major}.${cc_minor})"

            if [ "${HANZO_INSTALL_NO_NCCL:-}" = "1" ]; then
                info "HANZO_INSTALL_NO_NCCL=1 set - skipping nccl"
            elif detect_nccl; then
                features="$features nccl"
                info "NCCL detected - enabling nccl for CUDA multi-GPU tensor parallelism"
            elif [ "${HANZO_INSTALL_NCCL:-}" = "1" ]; then
                features="$features nccl"
                warn "HANZO_INSTALL_NCCL=1 set but NCCL was not detected; the build may fail unless libnccl is on the linker path"
            else
                warn "NCCL not found - skipping nccl. Install NCCL or set HANZO_INSTALL_NCCL=1 to force it; NCCL is the preferred CUDA multi-GPU path."
            fi

            # Check for cuDNN
            if detect_cudnn; then
                features="$features cudnn"
                info "cuDNN detected - enabling cudnn"
            else
                info "cuDNN not found - skipping cudnn feature"
            fi

            # Add flash attention based on compute capability
            if [ "$cuda_cc" = "90" ]; then
                features="$features flash-attn-v3"
                info "Hopper GPU detected - enabling flash-attn-v3"
            elif [ "$cuda_cc" -ge 80 ] 2>/dev/null; then
                features="$features flash-attn"
                info "Ampere+ GPU detected - enabling flash-attn"
            fi
            
            # cuTile: optimized CUDA kernels. Needs CUDA >= 13.1 (its JIT tool tileiras ships with 13.1+);
            # runs on Ampere (80-89) or Blackwell+ (>=100), not Hopper (90-99).
            cuda_ver_code=$(detect_cuda_version_code)
            if [ -n "$cuda_ver_code" ] && [ "$cuda_ver_code" -ge 1301 ] 2>/dev/null; then
                if { [ "$cuda_cc" -ge 80 ] && [ "$cuda_cc" -lt 90 ]; } || [ "$cuda_cc" -ge 100 ] 2>/dev/null; then
                    features="$features cutile"
                    info "CUDA >= 13.1 and supported arch - enabling cutile (optimized kernels)"
                fi
            fi
        else
            rocm_arch=$(detect_rocm_arch)
            if [ -z "$rocm_arch" ]; then
                warn "No accelerator detected (no NVIDIA, no AMD/ROCm) - installing a CPU-only engine"
            elif detect_hipcc; then
                features="rocm"
                ROCM_GFX_ARCH="${ROCM_GFX_ARCH:-$rocm_arch}"
                export ROCM_GFX_ARCH
                info "AMD GPU detected ($rocm_arch) - enabling rocm"
            else
                warn "AMD GPU detected ($rocm_arch) but hipcc is missing - install the ROCm toolchain (hip + rocblas), else this is a CPU-only engine"
            fi
        fi
    fi

    # Check for MKL on Intel
    if is_intel_cpu && detect_mkl; then
        features="$features mkl"
        info "Intel MKL detected - enabling mkl"
    fi

    # Trim leading/trailing whitespace
    echo "$features" | xargs
}

# Check if ffmpeg is installed
check_ffmpeg() {
    command -v ffmpeg >/dev/null 2>&1
}

# Install ffmpeg using the system package manager
install_ffmpeg() {
    os="$1"
    if [ "$os" = "macos" ]; then
        if command -v brew >/dev/null 2>&1; then
            info "Installing FFmpeg via Homebrew..."
            brew install ffmpeg
        else
            warn "Homebrew not found. Install FFmpeg manually: https://ffmpeg.org/download.html"
            return 1
        fi
    else
        if command -v apt-get >/dev/null 2>&1; then
            info "Installing FFmpeg via apt..."
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v dnf >/dev/null 2>&1; then
            info "Installing FFmpeg via dnf..."
            sudo dnf install -y ffmpeg
        else
            warn "Could not detect package manager. Install FFmpeg manually: https://ffmpeg.org/download.html"
            return 1
        fi
    fi
}

# Install the engine from the repository. There is no registry release of
# HANZO_CLI_PACKAGE, so the repository is the only place the binary exists.
# HANZOAI_VERSION pins a release tag; HANZOAI_INSTALL_DIR redirects the bin dir.
install_hanzo() {
    features="$1"

    set -- install --git "$HANZO_REPO_URL" --locked "$HANZO_CLI_PACKAGE"
    if [ -n "${HANZOAI_VERSION:-}" ]; then
        set -- "$@" --tag "$HANZOAI_VERSION"
        info "Pinning release tag $HANZOAI_VERSION"
    else
        set -- "$@" --branch "$HANZO_BRANCH"
    fi
    # cargo --root R writes R/bin, but callers name a BIN dir, so trim one /bin.
    if [ -n "${HANZOAI_INSTALL_DIR:-}" ]; then
        root=${HANZOAI_INSTALL_DIR%/}
        root=${root%/bin}
        set -- "$@" --root "$root"
        info "Installing into $root/bin"
    fi
    if [ -n "$features" ]; then
        set -- "$@" --features "$features"
        info "Installing $HANZO_CLI_PACKAGE with features: $features"
    else
        info "Installing $HANZO_CLI_PACKAGE with default features"
    fi
    cargo "$@"
}

# Main installation flow
main() {
    print_banner

    # Detect OS
    os=$(detect_os)
    info "Detected OS: $os"

    # Check for Rust
    if check_rust; then
        rust_version_full=$(rustc --version 2>/dev/null || echo "unknown")
        rust_version=$(get_rust_version)
        info "Rust is installed: $rust_version_full"

        # Check if version meets minimum requirement
        if [ -n "$rust_version" ] && ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
            warn "Rust $rust_version is below the required version $REQUIRED_RUST_VERSION"
            echo ""
            printf "Would you like to update Rust now? [Y/n] "
            read_input
            case "$REPLY" in
                [Nn]*)
                    error "Rust $REQUIRED_RUST_VERSION or newer is required to install hanzo"
                    ;;
            esac
            update_rust
            # Re-check version after update
            rust_version=$(get_rust_version)
            if ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
                error "Failed to update Rust to required version $REQUIRED_RUST_VERSION"
            fi
        fi
    else
        warn "Rust is not installed"
        echo ""
        printf "Would you like to install Rust now? [Y/n] "
        read_input
        case "$REPLY" in
            [Nn]*)
                error "Rust is required to install hanzo"
                ;;
        esac
        install_rust
    fi

    # Run prereq installers outside any $() so xcodebuild stdout (asset paths with slashes) can't leak into the captured feature string.
    if [ "$os" = "macos" ]; then
        check_xcode_cli_tools
        check_metal_toolchain
    fi

    echo ""
    info "Detecting hardware capabilities..."

    # Build features
    features=$(build_features "$os")

    # Check for FFmpeg (optional, needed for video input)
    FFMPEG_SKIPPED=""
    if check_ffmpeg; then
        info "FFmpeg is installed (enables video input support)"
    else
        echo ""
        printf "${YELLOW}(Optional)${NC} FFmpeg is required for video input support.\n"
        printf "Would you like to install FFmpeg? [y/N] "
        read_input
        case "$REPLY" in
            [Yy]*)
                install_ffmpeg "$os"
                if check_ffmpeg; then
                    success "FFmpeg installed successfully"
                else
                    warn "FFmpeg installation failed - you can install it manually later"
                    FFMPEG_SKIPPED=1
                fi
                ;;
            *)
                info "Skipping FFmpeg installation"
                FFMPEG_SKIPPED=1
                ;;
        esac
    fi

    echo ""
    printf "${BOLD}Installation Summary${NC}\n"
    echo "===================="
    if [ -n "$features" ]; then
        printf "Features: ${GREEN}%s${NC}\n" "$features"
    else
        printf "Features: ${YELLOW}(none - CPU only)${NC}\n"
    fi
    echo ""

    # Confirm installation
    printf "Proceed with installation? [Y/n] "
    read_input
    case "$REPLY" in
        [Nn]*)
            info "Installation cancelled"
            exit 0
            ;;
    esac

    echo ""
    install_hanzo "$features"

    # Ensure cargo bin is in PATH for this session
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi

    echo ""
    success "hanzo installed successfully!"
    echo ""
    printf "${BOLD}Quick Start${NC}\n"
    echo "==========="
    echo ""
    echo "  hanzo run -m Qwen/Qwen3-4B"
    echo ""
    echo "  hanzo serve --agent -m google/gemma-4-E4B-it"
    echo ""
    echo "For more information, visit: https://github.com/hanzoai/engine"
    echo ""
    if [ -n "$FFMPEG_SKIPPED" ]; then
        printf "${YELLOW}Note:${NC} FFmpeg was not installed. To enable video input support later, see:\n"
        printf "      https://github.com/hanzoai/engine/blob/master/docs/VIDEO.md\n"
        echo ""
    fi
    printf "${YELLOW}Note:${NC} To use 'hanzo' now, run: ${BOLD}. \"\$HOME/.cargo/env\"${NC}\n"
    printf "      Or restart your terminal.\n"
}

main "$@"
