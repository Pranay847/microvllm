#!/usr/bin/env bash
# One-time WSL2 toolchain setup for microvllm.
#
# Run this from inside WSL:   bash scripts/setup_wsl.sh
# It uses sudo and will prompt for your password.
set -euo pipefail

PACKAGES=(
  cmake                  # 3.28.3 on Ubuntu 24.04; microvllm needs >= 3.25 for add_subdirectory(SYSTEM)
  ninja-build
  pkg-config
  ccache                 # llama.cpp is a large rebuild; this matters
  libssl-dev             # only needed for a standalone llama-server build (LLAMA_OPENSSL)
)
# Note: libcurl is NOT required. LLAMA_CURL was deprecated as of llama.cpp b10103 --
# llama.cpp vendors cpp-httplib for model downloads instead.

echo "==> Installing: ${PACKAGES[*]}"
sudo apt-get update
sudo apt-get install -y "${PACKAGES[@]}"

echo
echo "==> Verifying toolchain"
fail=0
check() {
  local name="$1"; shift
  if command -v "$name" >/dev/null 2>&1; then
    printf '  %-10s %s\n' "$name" "$("$@" 2>&1 | head -1)"
  else
    printf '  %-10s MISSING\n' "$name"
    fail=1
  fi
}
check gcc     gcc --version
check g++     g++ --version
check clang++ clang++ --version
check cmake   cmake --version
check ninja   ninja --version
check ccache  ccache --version

# ThreadSanitizer is the reason this project builds on Linux rather than Windows.
# Verify the runtime is actually present before relying on it.
echo
echo "==> Verifying sanitizer runtimes"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/t.cpp" <<'EOF'
#include <thread>
int g = 0;
int main() {
    std::thread a([]{ g++; });
    std::thread b([]{ g++; });
    a.join(); b.join();
    return 0;
}
EOF
for san in thread address undefined; do
  if g++ -std=c++20 -fsanitize="$san" -g -O1 "$tmp/t.cpp" -o "$tmp/t.$san" 2>/dev/null; then
    printf '  %-10s OK\n' "$san"
  else
    printf '  %-10s FAILED TO LINK\n' "$san"
    fail=1
  fi
done

echo
if [ "$fail" -ne 0 ]; then
  echo "Setup incomplete -- see MISSING/FAILED entries above."
  exit 1
fi
echo "Toolchain ready."
echo "Next: bash scripts/fetch_model.sh"
