# syntax=docker/dockerfile:1

# Stage 1: Build environment
# Builder and runtime MUST share one Debian release: the binary links the
# builder's glibc (rust:latest drifted to trixie/glibc-2.39 while the runtime
# was bookworm/2.36 -> "GLIBC_2.39 not found" at startup). Bump both together.
FROM rust:1-trixie AS builder

# Set working directory and copy files
WORKDIR /hanzo
COPY . .

# Portable release build: RUSTFLAGS="" strips .cargo/config.toml `target-cpu=native`
# so the image runs on any x86-64 host. Cargo takes every CPU the build Job has (8):
# cold, the server builds in under 3m50s at 7.0 GiB peak on eight Zen 5
# cores in a 16 GiB cgroup (2026-10-08), inside the door's 16 GiB limit. Capped
# at the two jobs the retired ARC pods needed, it takes about three times as long
# (24.5 CPU-minutes over two), against the door's 30-minute deadline, which also
# counts the time the Job waits for a node.
ENV RUSTFLAGS="" \
    CARGO_INCREMENTAL=0 \
    CARGO_NET_RETRY=5
# Only the binary the runtime stage copies: a full --workspace build (tests,
# examples) OOM-killed the runner. hanzo-bench has had no binary since b2832f1c5
# (the bench is `hanzo-engine bench`), so copying one fails the image.
RUN cargo build --release -p hanzo-server


# Stage 2: Minimal runtime environment (same Debian release as builder — see above)
FROM debian:trixie-slim AS runtime
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

# Copy the built binary from the builder stage
COPY --chmod=755 --from=builder /hanzo/target/release/hanzo-server /usr/local/bin/
# Copy chat templates for users running models which may not include them
COPY --from=builder /hanzo/chat_templates /chat_templates

ENV HUGGINGFACE_HUB_CACHE=/data

# Self-runnable: `docker run <image> --port 36900 plain -m <model>`.
# hanzo-server takes its port and model from args (it does not read $PORT).
ENTRYPOINT ["/usr/local/bin/hanzo-server"]
