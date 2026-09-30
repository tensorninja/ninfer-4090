"""Run the TypeSafe Python SDK against a real ninfer-serve System One endpoint.

The server is started once with a LoRA pool holding a decision adapter (and, optionally, a
generative one). The wire contract the SDK does not exercise is checked over raw HTTP, and a greedy
chat reply (through the generative adapter when one is given) must stay bit-identical while
decisions run beside it. Then this file re-runs itself under each `--sdk-python` interpreter (a
virtual environment with one `typesafe-sdk` version installed) and drives the SDK's own client against
`base_url=http://host:port/typesafe`: models.list, system_one with all three question types by
object and by dictionary, the default-model alias, a 401, a 404 and a server-side 422.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_KEY = "systemone-smoke-key"
DEFAULT_MODEL = "jev-latest"
STATE = (
    "Subject: charged twice\n"
    "I was charged twice for my subscription this month and I need one of the charges refunded "
    "before my rent is due on Friday. This is the second time this has happened."
)


class SmokeFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http(
    base_url: str,
    method: str,
    path: str,
    payload: Any | None = None,
    headers: dict[str, str] | None = None,
    raw_body: bytes | None = None,
) -> tuple[int, dict[str, str], Any]:
    """One request -> (status, lower-cased headers, decoded JSON body or text)."""
    body = raw_body
    sent = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if body is not None:
        sent["Content-Type"] = "application/json"
    request = urllib.request.Request(base_url + path, data=body, headers=sent, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status, received, text = response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        status, received, text = error.code, error.headers, error.read()
    lowered = {key.lower(): value for key, value in received.items()}
    try:
        decoded: Any = json.loads(text)
    except ValueError:
        decoded = text.decode("utf-8", errors="replace")
    return status, lowered, decoded


def authorized(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}", **(extra or {})}


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeFailure(f"ninfer-serve exited during startup with code {process.returncode}")
        try:
            status, _, body = http(base_url, "GET", "/health")
            if status == 200 and body == {"status": "ok"}:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise SmokeFailure(f"ninfer-serve was not healthy after {timeout:g}s")


def check_wire(base_url: str, adapter: str, generative: str | None, model_id: str) -> dict[str, Any]:
    """The contract below the SDK: request ids, error shapes, routing and model separation."""
    request = {
        "state": STATE,
        "model": adapter,
        "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    }

    status, headers, body = http(base_url, "GET", "/typesafe/v1/models")
    require(status == 401, f"an unauthenticated models request returned {status}")
    require(isinstance(body, dict) and isinstance(body.get("detail"), str), f"401 body {body!r}")
    require(headers.get("www-authenticate") == "Bearer", "401 lacks www-authenticate: Bearer")
    require(re.fullmatch(r"[0-9a-f]{32}", headers.get("x-typesafe-request-id", "")) is not None,
            "a generated x-typesafe-request-id is not uuid4 hex")

    status, headers, body = http(base_url, "POST", "/typesafe/v1/systemone", request,
                                 authorized({"x-typesafe-request-id": "smoke-echo-1"}))
    require(status == 200, f"a decision returned {status}: {body!r}")
    require(headers.get("x-typesafe-request-id") == "smoke-echo-1", "request id was not echoed")
    require(list(body) == ["model", "answers", "usage", "latency_ms"], f"body keys {list(body)}")
    require(body["model"] == adapter, f"response model {body['model']!r} is not {adapter!r}")

    status, headers, body = http(base_url, "POST", "/typesafe/v1/systemone",
                                 {"model": adapter, "questions": request["questions"]}, authorized())
    require(status == 422 and isinstance(body.get("detail"), list), f"missing state: {status} {body!r}")
    require(body["detail"][0]["loc"] == ["body", "state"], f"422 loc {body['detail'][0]!r}")
    require("x-typesafe-request-id" in headers, "a 422 lacks x-typesafe-request-id")

    status, _, body = http(base_url, "POST", "/typesafe/v1/systemone", raw_body=b"{",
                           headers=authorized())
    require(status == 422 and body["detail"][0]["type"] == "json_invalid",
            f"undecodable JSON: {status} {body!r}")

    status, headers, body = http(base_url, "GET", "/typesafe/v1/unknown", headers=authorized())
    require(status == 404 and body == {"detail": "Not Found"}, f"unknown route: {status} {body!r}")
    require("x-typesafe-request-id" in headers, "a route 404 lacks x-typesafe-request-id")

    status, _, body = http(base_url, "GET", "/v1/models", headers=authorized())
    require(status == 200, f"OpenAI /v1/models returned {status}")
    openai_ids = [entry["id"] for entry in body["data"]]
    require(all(adapter not in identifier for identifier in openai_ids),
            f"a decision adapter is listed as an OpenAI model: {openai_ids}")

    chat = {"model": f"{model_id}-{adapter}", "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}
    status, _, body = http(base_url, "POST", "/v1/chat/completions", chat, authorized())
    require(status == 404, f"chat accepted the decision adapter: {status} {body!r}")

    if generative is not None:
        status, _, body = http(base_url, "POST", "/typesafe/v1/systemone",
                               {**request, "model": generative}, authorized())
        require(status == 404 and isinstance(body.get("detail"), str),
                f"System One accepted the generative adapter: {status} {body!r}")
    return {"wire": "ok", "openai_models": openai_ids}


def check_chat_under_decisions(base_url: str, chat_model: str, adapter: str) -> dict[str, Any]:
    """A greedy chat reply is bit-identical with and without decisions running beside it. The
    server runs with prefix reuse off, so both replies come from a cold prefill."""
    chat = {"model": chat_model, "max_tokens": 96, "temperature": 0,
            "messages": [{"role": "user", "content": "List five uses of a paper clip."}]}
    status, _, alone = http(base_url, "POST", "/v1/chat/completions", chat, authorized())
    require(status == 200, f"the lone chat returned {status}: {alone!r}")

    stop = threading.Event()
    decisions: list[tuple[float, float, int]] = []

    def load() -> None:
        index = 0
        while not stop.is_set():
            body = {"state": f"{STATE}\nTicket {index}: the customer wrote again.",
                    "model": adapter,
                    "questions": {"billing": {"type": "noul",
                                              "instructions": "Is this about billing?"},
                                  "urgency": {"type": "score",
                                              "instructions": "How urgent is this?",
                                              "criteria": ["Can wait", "This week", "Today"]}}}
            started = time.monotonic()
            code, _, _ = http(base_url, "POST", "/typesafe/v1/systemone", body, authorized())
            decisions.append((started, time.monotonic(), code))
            index += 1

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    while not decisions:
        time.sleep(0.05)
    chat_started = time.monotonic()
    status, _, loaded = http(base_url, "POST", "/v1/chat/completions", chat, authorized())
    chat_ended = time.monotonic()
    stop.set()
    thread.join(timeout=600)
    require(status == 200, f"the chat under decision load returned {status}: {loaded!r}")
    require(all(code == 200 for _, _, code in decisions),
            f"a decision under chat load failed: {[code for _, _, code in decisions]}")
    overlapping = sum(started < chat_ended and ended > chat_started
                      for started, ended, _ in decisions)
    require(overlapping > 0, "no decision ran while the chat was in flight")
    require(alone["choices"][0] == loaded["choices"][0],
            f"the greedy chat changed under decision load:\n{alone['choices'][0]!r}\n"
            f"{loaded['choices'][0]!r}")
    return {"chat_model": chat_model, "completion_tokens": loaded["usage"]["completion_tokens"],
            "overlapping_decisions": overlapping, "bit_identical": True}


def run_sdk_client(base_url: str, adapter: str) -> dict[str, Any]:
    """Client mode: runs under an interpreter with one typesafe-sdk version installed."""
    import typesafe_sdk as ts

    base = base_url + "/typesafe"
    no_retry = ts.RetryPolicy(max_retries=0)
    client = ts.TypeSafeClient(api_key=API_KEY, base_url=base, retry=no_retry, timeout=600.0)

    models = client.models.list()
    names = [model.name for model in models.models]
    require(adapter in names and DEFAULT_MODEL in names, f"models.list names {names}")
    for model in models.models:
        require(bool(model.description), f"model card {model.name} has no description")
        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", model.release_date) is not None,
                f"model card {model.name} release date {model.release_date!r}")

    objects = {
        "billing": ts.Noul(instructions="Is this message about billing?"),
        "tone": ts.Choice(
            instructions="What is the tone of this message?",
            criteria={"calm": None, "angry": "An upset or hostile message", "excited": None},
        ),
        "urgency": ts.Score(
            instructions="How urgent is this message?",
            criteria=["Can wait", "Needs attention this week", "Needs attention today"],
        ),
    }
    result = client.system_one(state=STATE, questions=objects)
    require(result.model == adapter, f"default-model response names {result.model!r}")
    require(bool(result.request_id), "the response has no request id")
    noul = result.nouls["billing"].noul
    require(0.0 <= noul <= 1.0, f"noul {noul}")
    tone = result.choices["tone"]
    require(set(tone.probabilities) == {"calm", "angry", "excited"}, f"choice keys {tone.probabilities}")
    require(tone.choice == max(tone.probabilities, key=tone.probabilities.__getitem__),
            "the choice is not its most likely option")
    require(abs(sum(tone.probabilities.values()) - 1.0) < 1e-3, "choice probabilities do not sum to 1")
    urgency = result.scores["urgency"]
    require(set(urgency.probabilities) == {0, 1, 2}, f"score keys {urgency.probabilities}")
    require(urgency.legend[2] == "Needs attention today", f"score legend {urgency.legend}")
    require(0.0 <= urgency.score <= 2.0 and 0.0 <= urgency.confidence <= 1.0, "score out of range")
    require(result.usage.input_tokens > 0 and result.usage.output_tokens > 0, f"usage {result.usage}")

    dictionaries = {
        "billing": {"type": "noul", "instructions": "Is this message about billing?"},
        "tone": {"type": "choice", "instructions": "What is the tone of this message?",
                 "criteria": {"calm": None, "angry": "An upset or hostile message", "excited": None}},
        "urgency": {"type": "score", "instructions": "How urgent is this message?",
                    "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]},
    }
    explicit = client.system_one(state=STATE, questions=dictionaries, model=adapter)
    require(explicit.model == adapter, f"explicit-model response names {explicit.model!r}")
    drift = max(
        abs(explicit.nouls["billing"].noul - noul),
        *(abs(explicit.choices["tone"].probabilities[key] - value)
          for key, value in tone.probabilities.items()),
        *(abs(explicit.scores["urgency"].probabilities[key] - value)
          for key, value in urgency.probabilities.items()),
    )
    require(drift <= 0.03, f"the repeated state moved a probability by {drift}")

    try:
        client.system_one(state=STATE, questions=objects, model="no-such-model")
        raise SmokeFailure("an unknown model was answered")
    except ts.TypeSafeNotFoundError as error:
        require(error.request_id is not None, "the 404 has no request id")
        not_found = str(error)

    try:
        client.system_one(state=STATE, questions={"tone": {"type": "choice", "criteria": {}}})
        raise SmokeFailure("a choice without options was answered")
    except ts.TypeSafeUnprocessableEntityError as error:
        unprocessable = str(error)

    wrong = ts.TypeSafeClient(api_key="wrong-key", base_url=base, retry=no_retry)
    try:
        wrong.models.list()
        raise SmokeFailure("a wrong API key listed models")
    except ts.TypeSafeAuthenticationError as error:
        require(error.request_id is not None, "the 401 has no request id")
        unauthorized = str(error)

    return {
        "sdk": ts.__version__,
        "models": names,
        "answers": {"billing": noul, "tone": tone.choice, "urgency": urgency.score},
        "usage": {"input_tokens": result.usage.input_tokens,
                  "output_tokens": result.usage.output_tokens},
        "repeat_max_abs_delta": drift,
        "errors": {"404": not_found, "422": unprocessable, "401": unauthorized},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, default=Path("models/qwen3_8_27b.ninfer"))
    parser.add_argument("--server-bin", type=Path, default=Path("build/apps/ninfer-serve"))
    parser.add_argument("--lora-dir", type=Path)
    parser.add_argument("--adapter", required=True, help="the decision adapter's pool name")
    parser.add_argument("--generative-adapter", help="a generative adapter of the same pool")
    parser.add_argument("--sdk-python", type=Path, action="append", default=[],
                        help="an interpreter with typesafe-sdk installed; repeatable")
    parser.add_argument("--server-arg", action="append", default=[],
                        help="an extra ninfer-serve argument; repeatable")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    if args.client_mode:
        print(json.dumps(run_sdk_client(args.base_url, args.adapter)))
        return

    if args.lora_dir is None or not args.sdk_python:
        parser.error("--lora-dir and at least one --sdk-python are required")
    artifact = args.artifact.resolve()
    server_bin = args.server_bin.resolve()
    require(artifact.is_file(), f"artifact does not exist: {artifact}")
    require(server_bin.is_file(), f"server binary does not exist: {server_bin}")

    port = args.port or free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(server_bin), str(artifact),
        "--host", "127.0.0.1", "--port", str(port),
        "--api-key", API_KEY,
        "--lora-dir", str(args.lora_dir.resolve()),
        "--max-context", "4096", "--kv-capacity", "8192",
        "--max-concurrency", "2", "--log-stats-interval-ms", "0",
        "--no-prefix-reuse",
        *args.server_arg,
    ]
    with tempfile.TemporaryDirectory(prefix="ninfer-systemone-sdk-") as temporary:
        server_log = Path(temporary) / "server.log"
        with server_log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
            try:
                wait_for_server(base_url, process, args.startup_timeout)
                status, _, body = http(base_url, "GET", "/v1/models", headers=authorized())
                require(status == 200, f"/v1/models returned {status}")
                model_id = body["data"][0]["id"]
                chat_model = (f"{model_id}-{args.generative_adapter}"
                              if args.generative_adapter else model_id)
                report: dict[str, Any] = {
                    "wire": check_wire(base_url, args.adapter, args.generative_adapter, model_id),
                    "chat_under_decisions": check_chat_under_decisions(base_url, chat_model,
                                                                       args.adapter),
                    "sdk": [],
                }
                for interpreter in args.sdk_python:
                    completed = subprocess.run(
                        [str(interpreter), str(Path(__file__).resolve()), "--client-mode",
                         "--base-url", base_url, "--adapter", args.adapter],
                        capture_output=True, text=True, check=False,
                    )
                    if completed.returncode != 0:
                        raise SmokeFailure(f"SDK client {interpreter} failed:\n{completed.stderr}")
                    report["sdk"].append(json.loads(completed.stdout.strip().splitlines()[-1]))
                print(json.dumps(report, ensure_ascii=False, indent=2))
            except Exception as error:
                output.flush()
                tail = server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                raise SystemExit(f"{error}\n\nlast server log lines:\n" + "\n".join(tail)) from error
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())
