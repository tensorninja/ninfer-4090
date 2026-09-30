#!/usr/bin/env python3
"""Measure chat and System One decisions served together by one ninfer-serve.

One process holds one weight copy, one scheduler, one set of lanes and one adapter bank. This
driver runs a closed loop of System One decisions (a decision adapter over a cached state) and two
chat streams (the base weights and a chat adapter), first each alone and then together, and one
cold decision idle and then beside the chat streams.

Every figure comes from the server's --request-log-jsonl records, matched to the request that
produced them by id: a chat request by the x-request-id its response carries, a decision by the
x-typesafe-request-id the driver sends. HTTP round-trip time is never used. Per phase it reports
decision latency (the response's latency_ms: restore plus execution) and end-to-end time (with the
queue wait), decisions per second, the LoRA bank's swaps and slot waits from /telemetry, and the SM
clock and board power range, because sustained load on a 24 GB RTX 4090 can throttle and read as a
regression. Per chat stream it reports two rates. The output rate is completion tokens over the
wall time after the first token: what a streaming client sees, and what decisions compete for,
because the execution thread runs one unit at a time and a decision's prefill units take turns
with the decode rounds. The round rate is completion tokens over the seconds of the decode rounds
the request joined (the dashboard's decode rate): how fast the decode rounds themselves ran.

Chat streams stop first and finish their requests in flight while the decisions still run, so
every chat request of the combined phase ran beside decisions from start to end, and a decision
counts toward that phase only if it completed while chat was in flight.

The run fails unless each chat stream's greedy text is identical in every measured request: a
decision is prefill-only and never joins a decode batch, so decisions must not change chat output.
Each stream's first request, in warm-up, prefills its prompt cold; every measured request then
takes a reuse path, which the report shows beside the comparison.

Start the server with prefix reuse and the continuation cache at their defaults, a chat adapter
that writes long answers and a decision adapter in DIR, and the request log this driver reads, for
example:

  ninfer-serve models/qwen3_8_27b.ninfer --kv-dtype bf16 --spec mtp --draft-tokens 3 \\
      --lm-head-draft --max-concurrency 4 --max-context 32768 --lora-dir DIR --lora-slots 2 \\
      --request-log-jsonl /tmp/dual.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from decision_probe import GpuSampler, corpus, make_questions, make_state, percentile

BASE_PROMPT = ("Write a detailed technical explanation of how a modern out-of-order CPU core "
               "executes instructions. Cover fetch, decode, register renaming, scheduling, "
               "execution, and retirement, and give a worked example for every stage.")
ADAPTER_PROMPT = ("Work through ten algebra and number theory problems of increasing difficulty. "
                  "State each problem, then solve it step by step, explaining every step in full.")


class RecordLog:
    """Completed chat and decision records from the server's JSONL log, keyed by client id."""

    EVENTS = ("request_done", "request_error", "decision_done", "decision_error")

    def __init__(self, path):
        self.path = path
        self.offset = os.path.getsize(path) if os.path.exists(path) else 0
        self.records = {}
        self.lock = threading.Lock()

    def _drain(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line.endswith("\n"):
                    break  # a torn trailing line is still being written
                self.offset = f.tell()
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") in self.EVENTS:
                    self.records[record["request"]["x_request_id"]] = record

    def wait(self, x_request_id, timeout_s=30.0):
        deadline = time.time() + timeout_s
        while True:
            with self.lock:
                self._drain()
                record = self.records.pop(x_request_id, None)
            if record is not None:
                return record
            if time.time() > deadline:
                raise RuntimeError(f"no record for request {x_request_id} in {self.path}")
            time.sleep(0.05)


def call(url, api_key, payload=None, timeout=600, headers=None):
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.headers, json.loads(error.read() or b"{}")


class Probe:
    def __init__(self, args):
        self.args = args
        self.root = args.url.rstrip("/")
        self.log = RecordLog(args.log)
        self.words = corpus()

    def telemetry(self):
        status, _, body = call(self.root + "/telemetry", self.args.api_key, timeout=30)
        if status != 200:
            raise RuntimeError(f"/telemetry returned HTTP {status}")
        return body

    def decide(self, state, questions):
        request_id = uuid.uuid4().hex
        payload = {"model": self.args.decision_model, "state": state, "questions": questions}
        status, _, body = call(self.root + "/typesafe/v1/systemone", self.args.api_key, payload,
                               self.args.timeout, {"x-typesafe-request-id": request_id})
        record = self.log.wait(request_id)
        if status != 200 or record["event"] != "decision_done":
            raise RuntimeError(f"decision failed with HTTP {status}: {json.dumps(body)[:400]}")
        return record

    def chat(self, stream):
        payload = {"model": stream["model"],
                   "messages": [{"role": "user", "content": stream["prompt"]}],
                   "max_tokens": self.args.chat_tokens, "temperature": 0,
                   "enable_thinking": False}
        status, headers, body = call(self.root + "/v1/chat/completions", self.args.api_key,
                                     payload, self.args.timeout)
        if status != 200:
            raise RuntimeError(f"chat failed with HTTP {status}: {json.dumps(body)[:400]}")
        record = self.log.wait(headers.get("x-request-id"))
        if record["event"] != "request_done":
            raise RuntimeError(f"chat request failed: {json.dumps(record)[:400]}")
        return {"record": record, "text": body["choices"][0]["message"].get("content") or ""}

    def calibrate(self, target, seed):
        """Word count whose state encodes to about `target` tokens, and that state."""
        count = max(1, int(target * self.args.words_per_token))
        for _ in range(4):
            state = make_state(self.words, count, seed)
            tokens = self.decide(state, make_questions(1))["request"]["state_tokens"]
            if abs(tokens - target) <= max(8, target // 100):
                break
            count = max(1, round(count * target / tokens))
        return make_state(self.words, count, seed), count

    def run(self, name, seconds, streams, decisions, extra=None):
        """Runs the chat streams and/or the decision loop for `seconds`, or around `extra`.

        The chat streams stop first and their requests in flight finish beside the decision loop;
        a decision that completes after the last of them is not counted.
        """
        before = self.telemetry()
        chat_stop, chats_done, decision_stop = threading.Event(), threading.Event(), threading.Event()
        chats = {stream["name"]: [] for stream in streams}
        counted = []
        errors = []

        def chat_worker(stream):
            try:
                while not chat_stop.is_set():
                    chats[stream["name"]].append(self.chat(stream))
            except Exception as error:
                errors.append(error)

        def decision_worker():
            try:
                while not decision_stop.is_set():
                    record = self.decide(self.state, self.questions)
                    if streams and chats_done.is_set():
                        break
                    counted.append(record)
            except Exception as error:
                errors.append(error)

        chat_workers = [threading.Thread(target=chat_worker, args=(stream,)) for stream in streams]
        decision_workers = [threading.Thread(target=decision_worker)] if decisions else []
        with GpuSampler() as sampler:
            started = time.time()
            for worker in chat_workers + decision_workers:
                worker.start()
            try:
                if extra is not None:
                    time.sleep(self.args.settle)
                    outcome = extra()
                else:
                    time.sleep(seconds / 2)
                    midpoint = self.telemetry()
                    time.sleep(seconds / 2)
            finally:
                chat_stop.set()
                for worker in chat_workers:
                    worker.join()
                chats_done.set()
                window = time.time() - started
                decision_stop.set()
                for worker in decision_workers:
                    worker.join()
        if errors:
            raise errors[0]
        after = self.telemetry()
        phase = {"name": name, "chats": chats, "decisions": counted,
                 "decision_seconds": window, "gpu": sampler.summary(),
                 "stages": after["adapters"]["stages"] - before["adapters"]["stages"],
                 "slot_waits": after["adapters"]["slot_waits"] - before["adapters"]["slot_waits"]}
        if extra is not None:
            phase["cold"] = outcome
        else:
            phase["midpoint"] = midpoint
        print(f"{name}: {len(counted)} decisions, "
              + ", ".join(f"{key} {len(value)} chats" for key, value in chats.items()),
              file=sys.stderr, flush=True)
        return phase

    def cold(self, seeds):
        """Decisions on states no lane or tier holds, so every state token is prefilled."""
        records = []
        for seed in seeds:
            record = self.decide(make_state(self.words, self.state_words, seed), self.questions)
            if record["result"]["reused_state_tokens"] != 0:
                raise RuntimeError("a cold decision reused state tokens")
            records.append(record)
        return records


def latency_ms(record):
    timings = record["timings_seconds"]
    return 1000.0 * (timings["restore"] + timings["execution"])


def end_to_end_ms(record):
    return 1000.0 * record["timings_seconds"]["total"]


def output_rate(chat):
    """Completion tokens over the wall time after the first token."""
    timings = chat["record"]["timings_seconds"]
    after_first = timings["total"] - timings["ttft"]
    return chat["record"]["result"]["completion_tokens"] / after_first if after_first > 0 else None


def round_rate(chat):
    """Completion tokens over the seconds of the decode rounds the request joined."""
    decode = chat["record"]["timings_seconds"]["decode"]
    return chat["record"]["result"]["completion_tokens"] / decode if decode > 0 else None


def reuse_path(chat):
    record = chat["record"]
    return f"{record['result']['prefix_reuse_path']}/{record['continuation_cache']['source']}"


def summarize(phase, streams):
    decisions = phase["decisions"]
    latencies = [latency_ms(record) for record in decisions]
    summary = {"phase": phase["name"], "decisions": len(decisions),
               "stages": phase["stages"], "slot_waits": phase["slot_waits"], **phase["gpu"]}
    if decisions:
        totals = [end_to_end_ms(record) for record in decisions]
        summary.update({
            "latency_p50_ms": statistics.median(latencies),
            "latency_p90_ms": percentile(latencies, 0.9),
            "end_to_end_p50_ms": statistics.median(totals),
            "end_to_end_p90_ms": percentile(totals, 0.9),
            "decisions_per_second": len(decisions) / phase["decision_seconds"],
            "state_sources": sorted({record["result"]["state_source"] for record in decisions}),
            "branch_passes": statistics.median(r["result"]["branch_passes"] for r in decisions),
        })
    for stream in streams:
        chats = phase["chats"].get(stream["name"], [])
        outputs = [rate for rate in (output_rate(chat) for chat in chats) if rate is not None]
        rounds = [rate for rate in (round_rate(chat) for chat in chats) if rate is not None]
        if outputs and rounds:
            speculative = [chat["record"]["speculative"] for chat in chats]
            drafted = sum(entry["drafted_tokens"] for entry in speculative)
            summary[stream["name"]] = {
                "requests": len(chats),
                "output_tok_s": statistics.median(outputs),
                "round_tok_s": statistics.median(rounds),
                "completion_tokens": statistics.median(
                    chat["record"]["result"]["completion_tokens"] for chat in chats),
                "mtp_accept": (sum(entry["accepted_tokens"] for entry in speculative) / drafted
                               if drafted else None),
            }
    return summary


def summarize_cold(label, records):
    timings = [record["timings_seconds"] for record in records]
    state_tokens = [record["request"]["state_tokens"] for record in records]
    return {"condition": label, "n": len(records),
            "state_tokens": statistics.median(state_tokens),
            "latency_ms": statistics.median(latency_ms(record) for record in records),
            "state_ms": statistics.median(1000.0 * entry["state"] for entry in timings),
            "branch_ms": statistics.median(1000.0 * entry["branch"] for entry in timings),
            "state_tok_s": statistics.median(record["rates"]["state_tok_s"] for record in records)}


def identity(phases, streams):
    """Each stream's greedy text against its first measured response, with the reuse paths."""
    report = {}
    for stream in streams:
        chats = [chat for phase in phases for chat in phase["chats"].get(stream["name"], [])]
        if not chats:
            continue
        reference = chats[0]["text"]
        report[stream["name"]] = {
            "requests": len(chats),
            "identical": sum(chat["text"] == reference for chat in chats),
            "paths": sorted({reuse_path(chat) for chat in chats}),
        }
    return report


def print_report(summaries, cold, texts, streams):
    print("| phase | decisions | latency ms p50 / p90 | end-to-end ms p50 / p90 | decisions/s | "
          "swaps | slot waits | SM MHz min / median | max W |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        decisions = (f"{s['latency_p50_ms']:.1f} / {s['latency_p90_ms']:.1f} | "
                     f"{s['end_to_end_p50_ms']:.1f} / {s['end_to_end_p90_ms']:.1f} | "
                     f"{s['decisions_per_second']:.2f}") if s["decisions"] else "- | - | -"
        clock = (f"{s['sm_mhz_min']:.0f} / {s['sm_mhz_median']:.0f}"
                 if s["sm_mhz_min"] is not None else "-")
        power = f"{s['power_w_max']:.0f}" if s["power_w_max"] is not None else "-"
        print(f"| {s['phase']} | {s['decisions']} | {decisions} | {s['stages']} | "
              f"{s['slot_waits']} | {clock} | {power} |")
    print()
    print("| phase | stream | requests | output tok/s | decode-round tok/s | MTP accept |")
    print("|---|---|---:|---:|---:|---:|")
    for s in summaries:
        for stream in streams:
            entry = s.get(stream["name"])
            if entry is None:
                continue
            accept = f"{100 * entry['mtp_accept']:.1f}%" if entry["mtp_accept"] is not None else "-"
            print(f"| {s['phase']} | {stream['name']} | {entry['requests']} | "
                  f"{entry['output_tok_s']:.1f} | {entry['round_tok_s']:.1f} | {accept} |")
    print()
    print("| cold decision | n | state tokens | latency ms | state ms | branch ms | state tok/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for c in cold:
        print(f"| {c['condition']} | {c['n']} | {c['state_tokens']:.0f} | {c['latency_ms']:.1f} | "
              f"{c['state_ms']:.1f} | {c['branch_ms']:.1f} | {c['state_tok_s']:.0f} |")
    print()
    for name, entry in texts.items():
        print(f"chat text {name}: {entry['identical']}/{entry['requests']} identical "
              f"(reuse paths {', '.join(entry['paths'])})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--log", required=True, help="the server's --request-log-jsonl path")
    parser.add_argument("--decision-model", default="jev-latest")
    parser.add_argument("--chat-adapter", required=True,
                        help="pool name of the generative adapter the second stream selects")
    parser.add_argument("--state-tokens", type=int, default=8192)
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--chat-tokens", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=90.0, help="duration of each phase")
    parser.add_argument("--cold", type=int, default=3, help="cold decisions per condition")
    parser.add_argument("--settle", type=float, default=5.0,
                        help="seconds the chat streams decode before the cold decisions start")
    parser.add_argument("--words-per-token", type=float, default=0.75)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="", help="also write every summary and record here")
    args = parser.parse_args()

    probe = Probe(args)
    start = probe.telemetry()
    pool = {entry["name"]: entry for entry in start["adapters"]["pool"]}
    if pool.get(args.chat_adapter, {}).get("kind") != "generative":
        raise SystemExit(f"{args.chat_adapter} is not a generative adapter in the server's pool")
    if not any(entry["kind"] == "decision" for entry in pool.values()):
        raise SystemExit("the server's pool holds no decision adapter")
    streams = [{"name": "base", "model": start["model_id"], "prompt": BASE_PROMPT},
               {"name": args.chat_adapter, "model": pool[args.chat_adapter]["model_id"],
                "prompt": ADAPTER_PROMPT}]

    # Warm-up: calibrate and retain the decision state, stage both adapters, and send each
    # chat stream's cold first request, so every measured request takes the steady-state path.
    probe.state, probe.state_words = probe.calibrate(args.state_tokens, args.seed)
    probe.questions = make_questions(args.questions)
    for _ in range(3):
        probe.decide(probe.state, probe.questions)
    warm = probe.run("warm-up", 0.0, streams, decisions=False,
                     extra=lambda: probe.cold([args.seed + 1000]))

    phases = [probe.run("decisions alone", args.seconds, [], decisions=True),
              probe.run("chat alone", args.seconds, streams, decisions=False),
              probe.run("together", args.seconds, streams, decisions=True)]
    idle = probe.cold(range(args.seed + 1001, args.seed + 1001 + args.cold))
    loaded = probe.run("cold beside chat", 0.0, streams, decisions=False,
                       extra=lambda: probe.cold(range(args.seed + 2001,
                                                      args.seed + 2001 + args.cold)))

    summaries = [summarize(phase, streams) for phase in phases]
    cold = [summarize_cold("idle", idle), summarize_cold("beside chat", loaded["cold"])]
    texts = identity(phases[1:] + [loaded], streams)
    print_report(summaries, cold, texts, streams)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": vars(args), "summaries": summaries, "cold": cold,
                       "texts": texts, "warm_up_chats": {name: [chat["text"] for chat in chats]
                                                         for name, chats in warm["chats"].items()},
                       "phases": phases + [loaded], "idle_cold": idle}, f, indent=1)
    return 0 if all(entry["identical"] == entry["requests"] for entry in texts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
