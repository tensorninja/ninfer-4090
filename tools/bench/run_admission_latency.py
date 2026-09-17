#!/usr/bin/env python3
"""Measure admission latency behind an in-flight long prefill.

Fires one long cold prompt, then after `--delay` seconds one short prompt, and reports both
requests' TTFT decomposition (prepare / queue / prefill) from the server's structured request
log. With a single prefill owner the short request queues for the whole remaining long prefill;
with a prefilling set it is admitted at the next prefill unit and finishes its own chunks first.

Server requirements are the same as run_decode_scaling.py (request log, cache off, no prefix reuse).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_decode_scaling import (  # noqa: E402
    Client, calibrate_chars_per_token, load_corpus, make_prompt, read_log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--long-tokens", type=int, default=131072)
    parser.add_argument("--short-tokens", type=int, default=8192)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    corpus = load_corpus()
    client = Client(args.url, 3600.0)
    chars_per_token = calibrate_chars_per_token(client, args.model, corpus)
    runs = []
    for repeat in range(args.repeats):
        long_prompt = make_prompt(corpus, int(args.long_tokens * chars_per_token), 991 * (repeat + 1))
        short_prompt = make_prompt(corpus, int(args.short_tokens * chars_per_token),
                                   577 * (repeat + 7))
        results: dict[str, dict] = {}

        def fire(name: str, prompt: str, seed: int) -> None:
            results[name] = client.chat(args.model, prompt, args.max_tokens, seed, False)

        start_ms = int(time.time() * 1000)
        long_thread = threading.Thread(target=fire, args=("long", long_prompt, 11 + repeat))
        long_thread.start()
        time.sleep(args.delay)
        short_started = time.time()
        fire("short", short_prompt, 23 + repeat)
        short_client_seconds = time.time() - short_started
        long_thread.join()
        end_ms = int(time.time() * 1000)
        time.sleep(2.0)
        done, _ = read_log(args.request_log, start_ms, end_ms)
        rows = []
        for record in sorted(done, key=lambda r: r["timestamp_unix_ms"]):
            timings = record["timings_seconds"]
            result = record["result"]
            rows.append({
                "prompt_tokens": result["prompt_tokens"],
                "computed_prefill_tokens": result["computed_prefill_tokens"],
                "ttft_s": round(timings["ttft"], 3),
                "prepare_s": round(timings["prepare"], 3),
                "queue_s": round(timings["queue"], 3),
                "prefill_s": round(timings["prefill"], 3),
                "decode_s": round(timings["decode"], 3),
                "completion_tokens": result["completion_tokens"],
            })
        run = {"repeat": repeat, "short_client_seconds": short_client_seconds, "requests": rows}
        runs.append(run)
        print(json.dumps(run), flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "artifact_type": "ninfer_admission_latency_report",
            "schema_version": 1,
            "label": args.label,
            "generated_unix": time.time(),
            "model": args.model,
            "long_tokens": args.long_tokens,
            "short_tokens": args.short_tokens,
            "delay_seconds": args.delay,
            "max_tokens": args.max_tokens,
            "runs": runs,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
