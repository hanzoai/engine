# syntax=docker/dockerfile:1

# Stage 1: Build environment
FROM rust:latest AS builder

# Set working directory
WORKDIR /build

# Copy ml dependency (must be present at ../ml relative to engine)
COPY ml/ /build/ml/

# Copy node dependency (hanzo-zap crate)
COPY node/hanzo-libs/hanzo-zap/ /build/node/hanzo-libs/hanzo-zap/

# Copy engine source
COPY engine/ /build/engine/

# Build the project in release mode, excluding the specified workspace
WORKDIR /build/engine
RUN cargo build --release --workspace --exclude mistralrs-pyo3 --no-default-features


# Stage 2: Minimal runtime environment
FROM debian:bookworm-slim AS runtime
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Install only essential runtime dependencies and clean up
ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update
    apt-get install -y --no-install-recommends \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        curl

    rm -rf /var/lib/apt/lists/*
HEREDOC

# Copy the built binaries from the builder stage
COPY --chmod=755 --from=builder /build/engine/target/release/hanzo-engine /usr/local/bin/
COPY --chmod=755 --from=builder /build/engine/target/release/mistralrs-server /usr/local/bin/
COPY --chmod=755 --from=builder /build/engine/target/release/mistralrs-bench /usr/local/bin/
# Copy chat templates for users running models which may not include them
COPY --from=builder /build/engine/chat_templates /chat_templates

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=36900 \
    RUST_LOG=info

EXPOSE 36900

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:36900/health || exit 1

ENTRYPOINT ["hanzo-engine"]
