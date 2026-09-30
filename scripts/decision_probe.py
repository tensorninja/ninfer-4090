#!/usr/bin/env python3
"""Drive ninfer-serve System One decisions and report decision latency.

A point is a state size and a question count. Latencies come from the structured request log's
decision_done record, never from the HTTP round trip, which also holds request parsing, rendering
and queueing. A decision's latency is its `latency_ms`: the restore of a cached state image, if
any, plus execution from admission to result, which splits into state and branch time.

Modes must match how the server was started:
  cached  prefix reuse on. The first request of a point computes and retains the state; the
          measured repeats must restore all of it (reused_state_tokens == state_tokens).
  cold    prefix reuse off and the continuation cache off. Every measured request must compute
          its whole state (reused_state_tokens == 0).
A request that violates its mode is reported as invalid and left out of the statistics.

SM clock and board power are sampled with nvidia-smi during every measured request, because
sustained prefill on a 24 GB RTX 4090 can throttle and read as a regression.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

FALLBACK_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
                  "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
                  "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
                  "yankee", "zulu"]


def corpus():
    # Deterministic pseudo-text with a realistic token/word ratio and no long repeated span.
    src = "/usr/share/dict/words"
    if os.path.exists(src):
        with open(src) as f:
            words = [w.strip() for w in f if w.strip().isalpha()]
        if words:
            return words
    return FALLBACK_WORDS


def make_state(words, count, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(words) for _ in range(count))


def make_questions(count):
    # Cycles the three System One question types so option counts and branch lengths are mixed.
    questions = {}
    for index in range(count):
        kind = index % 3
        if kind == 0:
            questions[f"q{index}"] = {"type": "noul",
                                      "instructions": f"Does the text mention topic number {index}?"}
        elif kind == 1:
            questions[f"q{index}"] = {
                "type": "choice",
                "instructions": "Which category best fits the text?",
                "criteria": {"news": None, "fiction": None,
                             "technical": "Documentation, specifications or code",
                             "other": None}}
        else:
            questions[f"q{index}"] = {"type": "score",
                                      "instructions": "How formal is the text?",
                                      "criteria": ["informal", "neutral", "formal"]}
    return questions


def post(url, api_key, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


class RequestLog:
    """Reads decision_done records appended to the server's JSONL request log."""

    def __init__(self, path):
        self.path = path
        self.offset = os.path.getsize(path) if os.path.exists(path) else 0

    def next_decision(self, timeout_s=10.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if os.path.exists(self.path):
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
                        if record.get("event") in ("decision_done", "decision_error"):
                            return record
            time.sleep(0.05)
        raise RuntimeError(f"no decision record appeared in {self.path}")


class GpuSampler:
    """nvidia-smi SM clock and board power at 100 ms while one request runs."""

    def __enter__(self):
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def __exit__(self, *exc):
        self.process.terminate()
        output, _ = self.process.communicate(timeout=5)
        self.samples = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            try:
                self.samples.append(tuple(float(field) for field in fields[:3]))
            except ValueError:
                continue
        return False

    def summary(self):
        if not self.samples:
            return {"sm_mhz_min": None, "sm_mhz_median": None, "power_w_max": None,
                    "temperature_c_max": None}
        clocks = [sample[0] for sample in self.samples]
        return {"sm_mhz_min": min(clocks), "sm_mhz_median": statistics.median(clocks),
                "power_w_max": max(sample[1] for sample in self.samples),
                "temperature_c_max": max(sample[2] for sample in self.samples)}


def decide(args, log, state, questions, sample):
    payload = {"model": args.model, "state": state, "questions": questions}
    if sample:
        with GpuSampler() as sampler:
            status, body = post(args.endpoint, args.api_key, payload, args.timeout)
        gpu = sampler.summary()
    else:
        status, body = post(args.endpoint, args.api_key, payload, args.timeout)
        gpu = None
    record = log.next_decision()
    if status != 200 or record.get("event") != "decision_done":
        raise RuntimeError(f"decision failed with HTTP {status}: {json.dumps(body)[:400]}")
    return record, gpu


def calibrate(args, log, words, target, seed):
    """Word count whose state encodes to about `target` tokens, and the state text."""
    count = max(1, int(target * args.words_per_token))
    for _ in range(4):
        state = make_state(words, count, seed)
        record, _ = decide(args, log, state, make_questions(1), sample=False)
        tokens = record["request"]["state_tokens"]
        if record["request"]["state_truncated"]:
            count = int(count * 0.95)
            continue
        if abs(tokens - target) <= max(8, target // 100):
            return state
        count = max(1, round(count * target / tokens))
    return make_state(words, count, seed)


def measure_point(args, log, state, question_count):
    questions = make_questions(question_count)
    if args.mode == "cached":
        # Computes and retains the state (and warms this question mix's pass widths).
        decide(args, log, state, questions, sample=False)
    else:
        decide(args, log, state, questions, sample=False)  # warms this point's routes
    rows = []
    for _ in range(args.repeats):
        record, gpu = decide(args, log, state, questions, sample=True)
        request, result = record["request"], record["result"]
        timings = record["timings_seconds"]
        reused = result["reused_state_tokens"]
        expected = request["state_tokens"] if args.mode == "cached" else 0
        rows.append({
            "state_tokens": request["state_tokens"],
            "questions": request["questions"],
            "options": request["options"],
            "branch_tokens": request["branch_tokens"],
            "mode": args.mode,
            "valid": reused == expected,
            "reused_state_tokens": reused,
            "state_source": result["state_source"],
            "branch_passes": result["branch_passes"],
            "long_branch_chunks": result["long_branch_chunks"],
            "latency_ms": 1000.0 * (timings["restore"] + timings["execution"]),
            "execution_ms": 1000.0 * timings["execution"],
            "restore_ms": 1000.0 * timings["restore"],
            "state_ms": 1000.0 * timings["state"],
            "branch_ms": 1000.0 * timings["branch"],
            "queue_ms": 1000.0 * timings["queue"],
            "prepare_ms": 1000.0 * timings["prepare"],
            **gpu,
        })
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(rows):
    valid = [row for row in rows if row["valid"]]
    if not valid:
        return None
    first = valid[0]
    return {
        "state_tokens": first["state_tokens"], "questions": first["questions"],
        "options": first["options"], "branch_tokens": first["branch_tokens"],
        "mode": first["mode"], "n": len(valid), "invalid": len(rows) - len(valid),
        "branch_passes": first["branch_passes"],
        "latency_ms": statistics.median(row["latency_ms"] for row in valid),
        "latency_p90_ms": percentile([row["latency_ms"] for row in valid], 0.9),
        "restore_ms": statistics.median(row["restore_ms"] for row in valid),
        "state_ms": statistics.median(row["state_ms"] for row in valid),
        "branch_ms": statistics.median(row["branch_ms"] for row in valid),
        "prepare_ms": statistics.median(row["prepare_ms"] for row in valid),
        "sm_mhz_min": min((row["sm_mhz_min"] for row in valid if row["sm_mhz_min"] is not None),
                          default=None),
        "power_w_max": max((row["power_w_max"] for row in valid if row["power_w_max"] is not None),
                           default=None),
    }


def print_table(summaries):
    print("| state tokens | Q | options | mode | n | latency ms (median / p90) | restore ms | "
          "state ms | branch ms | passes | prepare ms | min SM MHz | max W |")
    print("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        clock = f"{s['sm_mhz_min']:.0f}" if s["sm_mhz_min"] is not None else "-"
        power = f"{s['power_w_max']:.0f}" if s["power_w_max"] is not None else "-"
        print(f"| {s['state_tokens']} | {s['questions']} | {s['options']} | {s['mode']} | "
              f"{s['n']} | {s['latency_ms']:.1f} / {s['latency_p90_ms']:.1f} | "
              f"{s['restore_ms']:.1f} | {s['state_ms']:.1f} | {s['branch_ms']:.1f} | "
              f"{s['branch_passes']} | {s['prepare_ms']:.1f} | {clock} | {power} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="server root; requests go to <url>/typesafe/v1/systemone")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--log", default="/tmp/ninfer-reqlog.jsonl",
                        help="the server's --request-log JSONL path")
    parser.add_argument("--mode", choices=("cached", "cold"), required=True,
                        help="must match the server's prefix-reuse and continuation settings")
    parser.add_argument("--states", default="300,8192,65000",
                        help="target state sizes in tokens (kev truncates states above 65,535)")
    parser.add_argument("--questions", default="1,6,16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--words-per-token", type=float, default=0.75)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="", help="also write every row and summary here")
    args = parser.parse_args()
    args.endpoint = args.url.rstrip("/") + "/typesafe/v1/systemone"

    log = RequestLog(args.log)
    words = corpus()
    rows, summaries = [], []
    for state_index, target in enumerate(int(value) for value in args.states.split(",")):
        # Each state size gets its own text, so a cached point never restores another's state.
        state = calibrate(args, log, words, target, args.seed + state_index)
        for question_count in (int(value) for value in args.questions.split(",")):
            point = measure_point(args, log, state, question_count)
            rows.extend(point)
            summary = summarize(point)
            if summary is None:
                print(f"state {target} Q={question_count}: every request violated mode "
                      f"{args.mode} (reused tokens {[row['reused_state_tokens'] for row in point]})",
                      file=sys.stderr)
                continue
            summaries.append(summary)
            print(json.dumps(summary), file=sys.stderr, flush=True)
    print_table(summaries)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"mode": args.mode, "rows": rows, "summaries": summaries}, f, indent=1)
    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())
