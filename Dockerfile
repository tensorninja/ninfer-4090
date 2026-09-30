# syntax=docker/dockerfile:1

# The dashboard is a static bundle with no runtime dependency on the engine image, so it builds in
# its own stage. Debian rather than Alpine: the lockfile resolves the glibc lightningcss binary.
FROM oven/bun:1.3.5 AS web

WORKDIR /web
# Dependencies first, so editing a panel does not re-resolve the tree.
COPY apps/web/package.json apps/web/bun.lock ./
RUN bun install --frozen-lockfile
COPY apps/web/ ./
RUN bun run build

FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
# Only the C++ applications. apps/web is not part of the CMake build, and copying it here would
# invalidate this cached CUDA layer on every dashboard edit.
COPY apps/CMakeLists.txt apps/
COPY apps/cli/ apps/cli/
COPY apps/serve/ apps/serve/
COPY include/ include/
COPY src/ src/
COPY third_party/ third_party/

RUN --mount=type=cache,id=ninfer-sm89-cuda13.2-release,target=/build,sharing=locked \
    cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
        -DNINFER_BUILD_QWEN3_8_27B=ON \
        -DNINFER_BUILD_QWEN3_6_35B_A3B=OFF \
    && cmake --build /build --parallel 16 --target ninfer ninfer-serve \
    && install -D /build/apps/ninfer /opt/ninfer/bin/ninfer \
    && install -D /build/apps/ninfer-serve /opt/ninfer/bin/ninfer-serve

FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

ARG MODEL_URL=https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer
# Keep immutable model bytes below the frequently changing application layers.
RUN --mount=type=bind,source=.,target=/context,readonly \
    mkdir -p /opt/ninfer/models \
    && if [ -f /context/models/qwen3_8_27b.ninfer ]; then \
         cp /context/models/qwen3_8_27b.ninfer /opt/ninfer/models/qwen3_8_27b.ninfer; \
       else \
         curl --fail --location --retry 5 \
           --output /opt/ninfer/models/qwen3_8_27b.ninfer \
           "$MODEL_URL"; \
       fi \
    && printf '%s  %s\n' \
         eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e \
         /opt/ninfer/models/qwen3_8_27b.ninfer \
       | sha256sum --check

COPY --from=build /opt/ninfer/bin/ninfer /usr/local/bin/ninfer
COPY --from=build /opt/ninfer/bin/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=web /web/dist /opt/ninfer/web

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM
VOLUME ["/var/cache/ninfer"]

CMD ["ninfer-serve", "/opt/ninfer/models/qwen3_8_27b.ninfer", "--model-id", "qwen3.8-27b", "--host", "0.0.0.0", "--port", "8080", "--max-context", "262144", "--kv-capacity", "262144", "--max-concurrency", "4", "--max-pending-requests", "16", "--prefill-chunk", "1024", "--kv-dtype", "rk4v4", "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft", "--prefix-checkpoint-policy", "rolling-tool", "--continuation-cache", "l1-l2-l3", "--continuation-cache-dir", "/var/cache/ninfer", "--continuation-cache-namespace", "local", "--continuation-cache-l1-mib", "768", "--continuation-cache-l2-mib", "16384", "--continuation-cache-l3-mib", "49152", "--continuation-cache-filesystem-reserve-mib", "4096", "--preserve-thinking", "--web-dir", "/opt/ninfer/web"]
