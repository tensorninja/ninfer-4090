#!/usr/bin/env python3
"""Closed-loop multi-session agent swarm against a live ninfer-serve.

Every session is a growing multi-turn conversation routed with `prompt_cache_key`, so turns
after the first exercise the continuation path (L1 turn checkpoint or L2/L3 restore) and the
lane-swap behaviour when sessions outnumber lanes. Each session issues its next turn as soon as
the previous one returns. The report reads the structured request log: TTFT and its
decomposition by reuse path and cache tier, the served throughput, and the share of queueing time
during which a lane was free (head-of-line blocking rather than capacity).

Run the server with `--request-log-jsonl PATH --log-stats-interval-ms 1000` and the continuation
cache configured as it is served.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_decode_scaling import (  # noqa: E402
    GpuSampler, load_corpus, percentile, read_log)


def chat(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def excerpt(corpus: str, chars: int, offset: int) -> str:
    start = offset % max(1, len(corpus) - chars)
    return corpus[start:start + chars]


class Session(threading.Thread):
    def __init__(self, index: int, args: argparse.Namespace, corpus: str, cpt: float) -> None:
        super().__init__(daemon=True)
        self.index = index
        self.args = args
        self.corpus = corpus
        self.cpt = cpt
        self.turn_records: list[dict] = []
        self.failures: list[str] = []

    def run(self) -> None:
        key = f"swarm-{self.args.label}-{self.index}-{int(time.time())}"
        messages = [{
            "role": "user",
            "content": ("You are reviewing this engineering documentation set.\n\n<document>\n"
                        + excerpt(self.corpus, int(self.args.initial_tokens * self.cpt),
                                  100_003 * (self.index + 1))
                        + "\n</document>\n\nSummarise the main design decisions in about 150 words."),
        }]
        for turn in range(self.args.turns):
            if turn > 0:
                messages.append({
                    "role": "user",
                    "content": ("Here is a further excerpt.\n\n<document>\n"
                                + excerpt(self.corpus, int(self.args.turn_tokens * self.cpt),
                                          7_919 * (self.index + 1) * (turn + 3))
                                + "\n</document>\n\nRelate it to what you said before, in about "
                                  "150 words."),
                })
            body = {
                "model": self.args.model,
                "messages": messages,
                "max_tokens": self.args.max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 1000 * self.index + turn,
                "stream": False,
                "enable_thinking": False,
                "prompt_cache_key": key,
            }
            started = time.time()
            try:
                payload = chat(self.args.url, body, self.args.timeout)
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"turn {turn}: {str(exc)[:200]}")
                return
            elapsed = time.time() - started
            content = payload["choices"][0]["message"].get("content") or ""
            messages.append({"role": "assistant", "content": content})
            self.turn_records.append({
                "turn": turn,
                "client_seconds": elapsed,
                "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
            })


def summarize(done: list[dict], intervals: list[dict], lanes: int) -> dict:
    def path_of(record: dict) -> str:
        return f"{record['result']['prefix_reuse_path']}|{record['continuation_cache']['source']}"

    by_path: dict[str, list[dict]] = collections.defaultdict(list)
    for record in done:
        by_path[path_of(record)].append(record)
    paths = {}
    for path, records in sorted(by_path.items()):
        t = [r["timings_seconds"] for r in records]
        paths[path] = {
            "n": len(records),
            "ttft_p50": percentile([x["ttft"] for x in t], 0.5),
            "ttft_p90": percentile([x["ttft"] for x in t], 0.9),
            "ttft_mean": statistics.fmean(x["ttft"] for x in t),
            "prepare_mean": statistics.fmean(x["prepare"] for x in t),
            "queue_mean": statistics.fmean(x["queue"] for x in t),
            "restore_mean": statistics.fmean(x["restore"] for x in t),
            "prefill_mean": statistics.fmean(x["prefill"] for x in t),
            "publish_mean": statistics.fmean(x["publish"] for x in t),
            "computed_prefill_tokens_mean": statistics.fmean(
                r["result"]["computed_prefill_tokens"] for r in records),
        }
    all_t = [r["timings_seconds"] for r in done]
    waiting_seconds = 0.0
    waiting_free_lane = 0.0
    waiting_free_lane_prefilling = 0.0
    gpu_seconds = 0.0
    for record in intervals:
        sched = record["scheduler"]
        seconds = record["interval_seconds"]
        gpu_seconds += seconds
        if sched["waiting"] > 0:
            waiting_seconds += seconds
            if sched["running"] < lanes:
                waiting_free_lane += seconds
                if sched["prefilling"] > 0:
                    waiting_free_lane_prefilling += seconds
    span_ms = (max(r["timestamp_unix_ms"] for r in done) - min(r["timestamp_unix_ms"] for r in done)
               if done else 0)
    return {
        "requests": len(done),
        "ttft_p50": percentile([x["ttft"] for x in all_t], 0.5),
        "ttft_p90": percentile([x["ttft"] for x in all_t], 0.9),
        "ttft_mean": statistics.fmean(x["ttft"] for x in all_t) if all_t else None,
        "decomposition_mean": {
            k: statistics.fmean(x[k] for x in all_t) if all_t else None
            for k in ("prepare", "queue", "restore", "prefill", "publish")
        },
        "requests_per_minute": (len(done) / (span_ms / 60000.0)) if span_ms > 0 else None,
        "computed_prefill_tokens_total": sum(r["result"]["computed_prefill_tokens"] for r in done),
        "completion_tokens_total": sum(r["result"]["completion_tokens"] for r in done),
        "decode_tok_s_p50": percentile(
            [r["result"]["completion_tokens"] / r["timings_seconds"]["decode"]
             for r in done if r["timings_seconds"]["decode"] > 0], 0.5),
        "cache_source": dict(collections.Counter(
            r["continuation_cache"]["source"] for r in done)),
        "reuse_path": dict(collections.Counter(r["result"]["prefix_reuse_path"] for r in done)),
        "miss_reason": dict(collections.Counter(
            r["continuation_cache"]["final_miss_reason"] for r in done)),
        "waiting_seconds": waiting_seconds,
        "waiting_with_free_lane_seconds": waiting_free_lane,
        "waiting_with_free_lane_share": (waiting_free_lane / waiting_seconds
                                         if waiting_seconds else None),
        "waiting_with_free_lane_and_prefill_seconds": waiting_free_lane_prefilling,
        "by_path": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=3)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--initial-tokens", type=int, default=24000)
    parser.add_argument("--turn-tokens", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--label", default="swarm")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    corpus = load_corpus()
    probe = chat(args.url, {"model": args.model, "max_tokens": 1, "enable_thinking": False,
                            "messages": [{"role": "user",
                                          "content": excerpt(corpus, 40_000, 5)}]}, args.timeout)
    cpt = 40_000 / probe["usage"]["prompt_tokens"]
    print(f"chars/token {cpt:.3f}", flush=True)

    sampler = GpuSampler()
    sampler.start()
    start = time.time()
    sessions = [Session(i, args, corpus, cpt) for i in range(args.sessions)]
    for session in sessions:
        session.start()
    for session in sessions:
        session.join()
    end = time.time()
    sampler.stop()
    time.sleep(2.5)
    done, intervals = read_log(args.request_log, int(start * 1000), int(end * 1000))
    summary = summarize(done, intervals, args.lanes)
    summary["wall_seconds"] = end - start
    summary["client_failures"] = [f for s in sessions for f in s.failures]
    summary["gpu"] = sampler.window(start, end)
    report = {
        "artifact_type": "ninfer_session_swarm_report",
        "schema_version": 1,
        "label": args.label,
        "generated_unix": time.time(),
        "configuration": {k: v for k, v in vars(args).items() if k not in ("out", "request_log")},
        "summary": summary,
        "sessions": [{"index": s.index, "turns": s.turn_records} for s in sessions],
    }
    print(json.dumps(summary, indent=1, default=str), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
