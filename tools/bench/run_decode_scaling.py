#!/usr/bin/env python3
"""Measure decode-round scaling against a live ninfer-serve.

For every (concurrency, prompt_tokens) point the driver fires B concurrent chat completions
with distinct natural-text prompts of about the requested length, then reads the structured
request log written by the server: per-request prefill/decode rates and speculative acceptance
from `request_done`, and the steady-state aggregate decode rate from the `throughput` records
whose scheduler snapshot shows exactly B decode-ready lanes and no prefill in flight.

The server must run with `--request-log-jsonl PATH --log-stats-interval-ms 1000`; for a clean
decode number also `--continuation-cache off --no-prefix-reuse`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def load_corpus() -> str:
    parts: list[str] = []
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n\n".join(parts)
    if len(text) < 200_000:
        raise SystemExit("docs corpus too small for prompt construction")
    return text


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[tuple[float, int, float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                sm, watts, temp = [x.strip() for x in out.split(",")]
                self.samples.append((time.time(), int(float(sm)), float(watts), int(float(temp))))
            except Exception:
                pass
            self._stop.wait(1.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def window(self, start: float, end: float) -> dict:
        rows = [s for s in self.samples if start <= s[0] <= end]
        if not rows:
            return {}
        return {
            "sm_mhz_min": min(r[1] for r in rows),
            "sm_mhz_p50": percentile([r[1] for r in rows], 0.5),
            "power_w_p50": percentile([r[2] for r in rows], 0.5),
            "power_w_max": max(r[2] for r in rows),
            "temp_c_max": max(r[3] for r in rows),
        }


class Client:
    def __init__(self, url: str, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, prompt: str, max_tokens: int, seed: int,
             thinking: bool) -> dict:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": seed,
            "stream": False,
            "enable_thinking": thinking,
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/v1/chat/completions", data=data,
            headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        payload["_client_seconds"] = time.time() - started
        return payload


def make_prompt(corpus: str, chars: int, offset: int) -> str:
    if chars <= 0:
        body = ""
    else:
        start = offset % max(1, len(corpus) - chars)
        body = corpus[start:start + chars]
    return (
        "The following is an excerpt from an engineering documentation set.\n\n"
        f"<document>\n{body}\n</document>\n\n"
        "Write a detailed, well-structured technical essay of at least 1500 words that explains "
        "the design decisions described above, their trade-offs, and how you would extend the "
        "system. Do not stop early."
    )


def calibrate_chars_per_token(client: Client, model: str, corpus: str) -> float:
    chars = 40_000
    prompt = make_prompt(corpus, chars, 12_345)
    payload = client.chat(model, prompt, 1, 1, False)
    tokens = payload["usage"]["prompt_tokens"]
    return len(prompt) / tokens


def read_log(path: Path, start_ms: int, end_ms: int) -> tuple[list[dict], list[dict]]:
    done: list[dict] = []
    intervals: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = record.get("timestamp_unix_ms", 0)
            if ts < start_ms or ts > end_ms + 3000:
                continue
            if record.get("event") == "request_done":
                done.append(record)
            elif record.get("event") == "throughput":
                intervals.append(record)
    return done, intervals


def summarize_point(concurrency: int, done: list[dict], intervals: list[dict]) -> dict:
    per_request = []
    for record in done:
        timings = record["timings_seconds"]
        result = record["result"]
        spec = record.get("speculative", {})
        rounds = spec.get("rounds", 0)
        per_request.append({
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "finish_reason": result["finish_reason"],
            "prefill_tok_s": (result["computed_prefill_tokens"] / timings["prefill"]
                              if timings["prefill"] > 0 else None),
            "decode_tok_s": (result["completion_tokens"] / timings["decode"]
                             if timings["decode"] > 0 else None),
            "rounds_per_s": rounds / timings["decode"] if timings["decode"] > 0 and rounds else None,
            "tokens_per_round": result["completion_tokens"] / rounds if rounds else None,
            "acceptance": (spec.get("accepted_tokens", 0) / spec["drafted_tokens"]
                           if spec.get("drafted_tokens") else None),
            "ttft_s": timings["ttft"],
            "queue_s": timings["queue"],
            "prepare_s": timings["prepare"],
            "adapter": record["request"].get("adapter", ""),
        })
    steady = []
    for record in intervals:
        sched = record["scheduler"]
        tokens = record["tokens"]
        batch = record.get("decode_batch", {})
        if (sched["prefilling"] == 0 and sched["running"] == concurrency
                and sched["decode_ready"] == concurrency and tokens["computed_prefill"] == 0
                and tokens["committed_decode"] > 0 and record["interval_seconds"] > 0.5):
            steady.append({
                "agg_tok_s": tokens["committed_decode"] / record["interval_seconds"],
                "rounds_per_s": batch.get("rounds", 0) / record["interval_seconds"],
                "avg_batch": batch.get("average_size"),
            })
    agg = [s["agg_tok_s"] for s in steady]
    rps = [s["rounds_per_s"] for s in steady if s["rounds_per_s"]]
    return {
        "requests": per_request,
        "steady_intervals": len(steady),
        "aggregate_tok_s_p50": percentile(agg, 0.5),
        "aggregate_tok_s_p90": percentile(agg, 0.9),
        "aggregate_tok_s_max": max(agg) if agg else None,
        "rounds_per_s_p50": percentile(rps, 0.5),
        "round_ms_p50": (1000.0 / percentile(rps, 0.5)) if rps else None,
        "per_request_decode_tok_s_p50": percentile(
            [r["decode_tok_s"] for r in per_request if r["decode_tok_s"]], 0.5),
        "prefill_tok_s_p50": percentile(
            [r["prefill_tok_s"] for r in per_request if r["prefill_tok_s"]], 0.5),
        "acceptance_p50": percentile(
            [r["acceptance"] for r in per_request if r["acceptance"] is not None], 0.5),
        "tokens_per_round_p50": percentile(
            [r["tokens_per_round"] for r in per_request if r["tokens_per_round"]], 0.5),
    }


def run_point(client: Client, model: str, corpus: str, chars_per_token: float,
              concurrency: int, prompt_tokens: int, max_tokens: int, thinking: bool,
              log_path: Path, sampler: GpuSampler, point_index: int) -> dict:
    chars = int(prompt_tokens * chars_per_token)
    prompts = [make_prompt(corpus, chars, 7919 * (point_index * 8 + i) + 101 * prompt_tokens)
               for i in range(concurrency)]
    start = time.time()
    start_ms = int(start * 1000)
    payloads: list[dict] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(client.chat, model, prompts[i], max_tokens,
                               1000 + point_index * 16 + i, thinking)
                   for i in range(concurrency)]
        for future in futures:
            try:
                payloads.append(future.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc)[:300])
    end = time.time()
    end_ms = int(end * 1000)
    time.sleep(2.5)
    done, intervals = read_log(log_path, start_ms, end_ms)
    summary = summarize_point(concurrency, done, intervals)
    summary.update({
        "concurrency": concurrency,
        "prompt_tokens_requested": prompt_tokens,
        "model": model,
        "wall_seconds": end - start,
        "errors": errors,
        "usage_prompt_tokens": [p.get("usage", {}).get("prompt_tokens") for p in payloads],
        "usage_completion_tokens": [p.get("usage", {}).get("completion_tokens") for p in payloads],
        "gpu": sampler.window(start, end),
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,2,3")
    parser.add_argument("--prompt-tokens", default="8192,65536,131072")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    concurrencies = [int(x) for x in args.concurrency.split(",") if x]
    lengths = [int(x) for x in args.prompt_tokens.split(",") if x]
    corpus = load_corpus()
    client = Client(args.url, args.timeout)
    chars_per_token = calibrate_chars_per_token(client, args.model, corpus)
    print(f"calibrated chars/token = {chars_per_token:.3f}", flush=True)

    sampler = GpuSampler()
    sampler.start()
    points = []
    index = 0
    try:
        for prompt_tokens in lengths:
            for concurrency in concurrencies:
                index += 1
                point = run_point(client, args.model, corpus, chars_per_token, concurrency,
                                  prompt_tokens, args.max_tokens, args.thinking,
                                  args.request_log, sampler, index)
                points.append(point)
                agg = point["aggregate_tok_s_p50"]
                print(
                    f"B={concurrency} L={prompt_tokens} usage={point['usage_prompt_tokens']} "
                    f"agg_p50={agg and round(agg, 1)} steady_n={point['steady_intervals']} "
                    f"round_ms={point['round_ms_p50'] and round(point['round_ms_p50'], 1)} "
                    f"per_req_decode={point['per_request_decode_tok_s_p50'] and round(point['per_request_decode_tok_s_p50'], 1)} "
                    f"prefill={point['prefill_tok_s_p50'] and round(point['prefill_tok_s_p50'], 0)} "
                    f"accept={point['acceptance_p50'] and round(point['acceptance_p50'], 3)} "
                    f"tok/round={point['tokens_per_round_p50'] and round(point['tokens_per_round_p50'], 2)} "
                    f"gpu={point['gpu']} errors={point['errors']}",
                    flush=True)
    finally:
        sampler.stop()

    report = {
        "artifact_type": "ninfer_decode_scaling_report",
        "schema_version": 1,
        "label": args.label,
        "generated_unix": time.time(),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "chars_per_token": chars_per_token,
        "points": points,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
