#!/usr/bin/env bash
# With no arguments, or only flags, this starts the default server profile and appends the flags.
# ninfer-serve keeps the last value of a repeated flag, so an appended flag overrides a default.
# Any other command runs as given, after the asset download when it is ninfer or ninfer-serve
# without --help.
set -euo pipefail

home="${NINFER_HOME:-/var/lib/ninfer}"

fetch() {
  if [[ "${NINFER_FETCH:-1}" != 0 ]]; then
    ninfer-fetch
  fi
}

case "${1:-}" in
  "" | -*)
    fetch
    exec ninfer-serve "$home/models/qwen3_8_27b.ninfer" \
      --model-id qwen3.8-27b --host 0.0.0.0 --port 8080 \
      --max-context 262144 --kv-capacity 262144 \
      --max-concurrency 4 --max-pending-requests 16 \
      --prefill-chunk 1024 --kv-dtype rk4v4 \
      --spec mtp --draft-tokens 3 --lm-head-draft \
      --prefix-checkpoint-policy rolling-tool \
      --continuation-cache l1-l2-l3 \
      --continuation-cache-dir /var/cache/ninfer \
      --continuation-cache-namespace local \
      --continuation-cache-l1-mib 768 \
      --continuation-cache-l2-mib 16384 \
      --continuation-cache-l3-mib 49152 \
      --continuation-cache-filesystem-reserve-mib 4096 \
      --lora-dir "$home/lora" --lora-slots 2 \
      --web-dir /opt/ninfer/web \
      "$@"
    ;;
  ninfer | ninfer-serve)
    if [[ "${2:-}" != --help && "${2:-}" != -h ]]; then
      fetch
    fi
    exec "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
