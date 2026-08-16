# Multi-stage build for microvllm.
#
# The model is NOT baked into the image. At 469 MB it would dominate the layer, make every
# rebuild push half a gigabyte, and tie the image to one model -- so it is mounted at
# runtime instead (a volume in Kubernetes, a bind mount locally). The image stays a few tens
# of megabytes and is model-agnostic.

# ---------------------------------------------------------------------------
# Builder: compiles microvllm and llama.cpp.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy the build definition and vendored llama.cpp first. These change far less often than
# our own sources, so Docker's layer cache keeps the expensive llama.cpp compile across
# edits to src/ -- which is most of the build time.
COPY CMakeLists.txt ./
COPY third_party/ third_party/

COPY include/ include/
COPY src/ src/
COPY tests/ tests/
COPY benchmarks/ benchmarks/

# GGML_NATIVE=OFF is the important one: the default tunes for the *build* machine's CPU,
# which produces an image that dies with SIGILL on any host with a smaller instruction set.
# Portability beats the last few percent for something meant to be scheduled anywhere.
#
# Tests and benchmarks are off: CI already runs them, and building them here would pull
# googletest and google/benchmark into an image that will never execute them.
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DMICROVLLM_BUILD_TESTS=OFF \
        -DMICROVLLM_BUILD_BENCHMARKS=OFF \
    && cmake --build build -j "$(nproc)" --target microvllm \
    && strip build/src/microvllm

# ---------------------------------------------------------------------------
# Runtime: just the binary and the shared libraries it actually needs.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# libgomp is required: ggml is built with OpenMP. Everything else the binary needs is in
# libc/libstdc++, which the base image already carries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin microvllm

COPY --from=builder /src/build/src/microvllm /usr/local/bin/microvllm

# Mount point for the model. Left empty in the image; Kubernetes mounts a volume here.
RUN mkdir -p /models && chown microvllm:microvllm /models

USER microvllm
EXPOSE 8080

# No shell form and no init wrapper: the process must receive SIGTERM directly as PID 1 so
# its graceful drain runs. A shell wrapper would swallow the signal and Kubernetes would
# SIGKILL the pod after the grace period, dropping every in-flight request on each rollout.
ENTRYPOINT ["/usr/local/bin/microvllm"]
CMD ["--model", "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--quiet", \
     "--log-requests"]
