#!/usr/bin/env bash
# Build llama.cpp's own tools (llama-cli, llama-server, llama-bench) from the
# pinned submodule, in a SEPARATE build tree from microvllm.
#
# Two reasons these are not built as part of the main project:
#   1. llama-server is the external reference baseline for the Phase 6 benchmarks.
#      It must be the stock upstream binary, not something our CMake reconfigured.
#   2. Building tools inside our tree would add minutes to every incremental build.
#
#   bash scripts/build_llama_tools.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
BUILD="${LLAMA_TOOLS_BUILD_DIR:-$HOME/build/llama.cpp-tools}"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "third_party/llama.cpp is empty. Run: git submodule update --init --recursive" >&2
  exit 1
fi

echo "==> Source: $SRC ($(git -C "$SRC" describe --tags 2>/dev/null || git -C "$SRC" rev-parse --short HEAD))"
echo "==> Build:  $BUILD"

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_CCACHE=ON

cmake --build "$BUILD" -j "$(nproc)"

echo
echo "==> Built binaries:"
find "$BUILD/bin" -maxdepth 1 -type f -executable -name 'llama-*' 2>/dev/null | sort | sed 's/^/  /'
echo
echo "Smoke test:"
echo "  $BUILD/bin/llama-cli -m $ROOT/models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p 'Hello' -n 32 -no-cnv"
