#!/usr/bin/env bash
# Donor retention, measured against the real model on strictly sequential traffic.
#
# Each request is sent only after the previous one has fully returned, so no two requests
# ever overlap in time. Without donor retention there is never a live sequence to copy
# from and the hit rate is exactly zero -- which is the case this feature exists for.
#
#   bash scripts/bench_prefix_sequential.sh [requests]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${MICROVLLM_BIN:-$HOME/build/microvllm/wsl-release/src/microvllm}"
MODEL="${MICROVLLM_MODEL:-$ROOT/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
N="${1:-6}"

# A long shared system prompt, several whole blocks -- the shape of chat traffic.
SYSTEM=""
for _ in $(seq 1 12); do
  SYSTEM+="You are a careful assistant. Follow instructions precisely and answer concisely. "
done

metric() { grep -E "^microvllm_$1 " "$2" | awk '{print $2}'; }

run_arm() {
  local donors="$1"
  local port="$2"
  local log="/tmp/mv_donors_${donors}.log"

  "$BIN" --model "$MODEL" --threads 8 --port "$port" --batch-size 2 \
         --prefix-donors "$donors" --quiet >"$log" 2>&1 &
  local srv=$!

  local i
  for i in $(seq 1 150); do
    curl -s "localhost:$port/health" >/dev/null 2>&1 && break
    sleep 0.3
  done
  if ! curl -s "localhost:$port/health" >/dev/null 2>&1; then
    echo "  donors=$donors: server never became healthy (see $log)"
    kill -9 "$srv" 2>/dev/null
    return 1
  fi

  local start elapsed
  start=$(date +%s.%N)
  for i in $(seq 1 "$N"); do
    # Strictly sequential: this curl returns before the next is issued.
    curl -s -m 300 -X POST "localhost:$port/generate" \
      --data-binary "$(printf '{"prompt":%s,"max_tokens":8,"temperature":0}' \
                       "$(printf '%s' "$SYSTEM Question $i: name a colour." | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
      >/dev/null
  done
  elapsed=$(python3 -c "import time;print(f'{time.time()-$start:.1f}')" 2>/dev/null || echo "?")

  curl -s "localhost:$port/metrics" >/tmp/mv_metrics_$donors.txt
  local m=/tmp/mv_metrics_$donors.txt
  printf '  donors=%-2s hits=%-4s tokens_saved=%-6s retained=%-4s held=%-3s wall=%ss\n' \
    "$donors" \
    "$(metric prefix_cache_hits_total "$m")" \
    "$(metric prefix_tokens_saved_total "$m")" \
    "$(metric prefix_donors_retained_total "$m")" \
    "$(metric prefix_donors_held "$m")" \
    "$elapsed"

  kill -INT "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
}

echo "sequential prefix sharing: $N requests behind a shared system prompt"
echo "(each sent only after the previous returned, so none overlap in time)"
run_arm 0 8940
run_arm 4 8941
