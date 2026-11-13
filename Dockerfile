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

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ccache \
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

# Empty lets Ninja pick the job count. Each nvcc job can take several GiB, so a small builder sets
# it explicitly.
ARG BUILD_JOBS=
# The build directory keeps Ninja incremental on one machine. ccache hashes sources instead of
# mtimes, so a fresh checkout in CI, which restores only the ccache directory, recompiles only
# changed files.
ENV CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=4G
RUN --mount=type=cache,id=ninfer-sm89-cuda13.2-release,target=/build,sharing=locked \
    --mount=type=cache,id=ninfer-ccache,target=/root/.cache/ccache,sharing=locked \
    cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
        -DNINFER_BUILD_QWEN3_8_27B=ON \
        -DNINFER_BUILD_QWEN3_6_35B_A3B=OFF \
    && cmake --build /build ${BUILD_JOBS:+--parallel "$BUILD_JOBS"} --target ninfer ninfer-serve \
    && install -D /build/apps/ninfer /opt/ninfer/bin/ninfer \
    && install -D /build/apps/ninfer-serve /opt/ninfer/bin/ninfer-serve \
    && ccache --show-stats

# The binaries link only cudart, which the base image has, and NVML and the driver, which the NVIDIA
# Container Toolkit mounts.
FROM nvidia/cuda:13.2.0-base-ubuntu24.04

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

COPY --from=build /opt/ninfer/bin/ninfer /usr/local/bin/ninfer
COPY --from=build /opt/ninfer/bin/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=web /web/dist /opt/ninfer/web
# The model and adapters stay out of the image. The entrypoint downloads the pinned files into the
# /var/lib/ninfer volume on first start, so an image update downloads only what changed.
COPY docker/assets.txt /opt/ninfer/assets.txt
COPY --chmod=0755 docker/fetch-assets.sh /usr/local/bin/ninfer-fetch
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/ninfer-entrypoint

ENV NINFER_HOME=/var/lib/ninfer
WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM
VOLUME ["/var/lib/ninfer", "/var/cache/ninfer"]
# The start period covers the first model download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=2h --start-interval=10s \
    CMD curl --fail --silent --show-error http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["ninfer-entrypoint"]
