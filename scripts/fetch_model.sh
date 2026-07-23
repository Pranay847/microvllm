#!/usr/bin/env bash
# Download GGUF models into models/ (gitignored).
#
#   bash scripts/fetch_model.sh            # primary model only (Qwen2.5-0.5B, ~400 MB)
#   bash scripts/fetch_model.sh --all      # also TinyLlama-1.1B (~670 MB)
#
# Primary model is deliberately small: faster iteration, and more decode headroom on a
# 15 W CPU makes the batching benchmarks less dominated by raw compute.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/models"
mkdir -p "$MODEL_DIR"

HF="${HF_ENDPOINT:-https://huggingface.co}"

# name | repo | filename
PRIMARY="Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q4_k_m.gguf"
SECONDARY="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF|tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

fetch() {
  local repo="${1%%|*}"
  local file="${1##*|}"
  local dest="$MODEL_DIR/$file"

  if [ -s "$dest" ]; then
    echo "==> $file already present ($(du -h "$dest" | cut -f1)), skipping"
    return 0
  fi

  local url="$HF/$repo/resolve/main/$file"
  echo "==> Downloading $file"
  echo "    from $url"
  # --continue-at - so an interrupted 400 MB download resumes instead of restarting.
  curl -fL --continue-at - --progress-bar -o "$dest" "$url"
  echo "    done: $(du -h "$dest" | cut -f1)"
}

fetch "$PRIMARY"
if [ "${1:-}" = "--all" ]; then
  fetch "$SECONDARY"
fi

echo
echo "Models in $MODEL_DIR:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || echo "  (none)"
