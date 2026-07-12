# syntax=docker/dockerfile:1

# Stage 1: Build environment
FROM rust:latest AS builder

# Set working directory and copy files
WORKDIR /hanzo
COPY . .

# Portable, memory-bounded release build (see Dockerfile.cuda for the rationale):
# RUSTFLAGS="" strips .cargo/config.toml `target-cpu=native` so the image runs on
# any x86-64 host; CARGO_BUILD_JOBS=2 caps rustc so the ARC pod does not OOM.
ENV RUSTFLAGS="" \
    CARGO_INCREMENTAL=0 \
    CARGO_NET_RETRY=5 \
    CARGO_BUILD_JOBS=2
# Only the two binaries the runtime stage copies (hanzo-server, hanzo-bench) —
# a full --workspace build (incl. tests/examples) OOM-killed the runner.
RUN cargo build --release -p hanzo-server -p hanzo-bench


# Stage 2: Minimal runtime environment
FROM debian:bookworm-slim AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Install only essential runtime dependencies and clean up
ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    for i in 1 2 3 4 5; do apt-get -o Acquire::Retries=3 update && break; echo "apt-get update failed (attempt $i/5), mirror may be syncing; retrying in 15s"; sleep 15; done
    apt-get install -y --no-install-recommends \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        curl

    rm -rf /var/lib/apt/lists/*
HEREDOC

# Copy the built binaries from the builder stage
COPY --chmod=755 --from=builder /hanzo/target/release/hanzo-bench /usr/local/bin/
COPY --chmod=755 --from=builder /hanzo/target/release/hanzo-server /usr/local/bin/
# Copy chat templates for users running models which may not include them
COPY --from=builder /hanzo/chat_templates /chat_templates

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80
