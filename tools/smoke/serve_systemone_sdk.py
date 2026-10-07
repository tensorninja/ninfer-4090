"""Run TypeSafe and optional OpenAI Python SDKs against one real ninfer-serve process.

The server is started once with a LoRA pool holding a decision adapter (and, optionally, a
generative one). The wire contract the SDK does not exercise is checked over raw HTTP, and a greedy
chat reply (through the generative adapter when one is given) must stay bit-identical while
decisions run beside it. Then this file re-runs itself under each `--sdk-python` interpreter (a
virtual environment with one `typesafe-sdk` version installed) and drives the SDK's own client against
`base_url=http://host:port/typesafe`: models.list, system_one with all three question types by
object and by dictionary, the default-model alias, a 401, a 404 and a server-side 422.
An optional `--openai-sdk-python` environment with openai>=3.26.0 exercises decisions.create,
discovery, typed answers and errors. Raw HTTP checks cover both surfaces, equivalent text inputs,
probability parity and strict overflow rejection. Use Python 3.11 environments; no SDK is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
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
    require(adapter in openai_ids, f"the decision adapter is absent from OpenAI models: {openai_ids}")
    require(DEFAULT_MODEL not in openai_ids and "gpt-6-luna" not in openai_ids,
            f"an unsupported alias is listed as an OpenAI model: {openai_ids}")

    for name in (adapter, f"{model_id}-{adapter}"):
        chat = {"model": name, "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]}
        status, _, body = http(base_url, "POST", "/v1/chat/completions", chat, authorized())
        require(status == 404, f"chat accepted the decision adapter: {status} {body!r}")

    status, _, body = http(base_url, "POST", "/v1/responses",
                           {"model": adapter, "input": "hi", "max_output_tokens": 16}, authorized())
    require(status == 404 and "error" in body, f"Responses accepted a decision model: {status} {body!r}")
    message = {"model": adapter, "max_tokens": 1,
               "messages": [{"role": "user", "content": "hi"}]}
    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        status, _, body = http(base_url, "POST", path, message, authorized())
        require(status == 404 and body.get("type") == "error",
                f"{path} accepted a decision-only model: {status} {body!r}")
    status, _, body = http(base_url, "POST", "/v1/messages",
                           {**message, "model": "claude-smoke-alias"}, authorized())
    require(status == 200 and body["model"] == "claude-smoke-alias",
            f"Anthropic model-name fallback changed: {status} {body!r}")

    if generative is not None:
        status, _, body = http(base_url, "POST", "/typesafe/v1/systemone",
                               {**request, "model": generative}, authorized())
        require(status == 404 and isinstance(body.get("detail"), str),
                f"System One accepted the generative adapter: {status} {body!r}")
    return {"wire": "ok", "openai_models": openai_ids}


def openai_questions() -> list[dict[str, Any]]:
    return [
        {"type": "predicate", "name": "billing", "instructions": "Is this message about billing?"},
        {"type": "choice", "name": "tone", "instructions": "What is the tone of this message?",
         "choices": [{"value": "calm"}, {"value": "angry", "description": "An upset or hostile message"},
                     {"value": "excited"}]},
        {"type": "score", "name": "urgency", "instructions": "How urgent is this message?",
         "levels": [{"label": "Can wait"}, {"label": "Needs attention this week"},
                    {"label": "Needs attention today", "description": "Requires action now"}]},
    ]


def check_openai_answer(body: dict[str, Any], adapter: str,
                        questions: list[dict[str, Any]]) -> None:
    require(set(body) == {"model", "answers", "usage"}, f"OpenAI response keys: {list(body)}")
    require(body["model"] == adapter, f"OpenAI response model: {body['model']!r}")
    answers = body["answers"]
    require(isinstance(answers, list) and len(answers) == len(questions), "answer count/order lost")
    for question, answer in zip(questions, answers):
        kind = question["type"]
        require(answer["type"] == kind and answer["name"] == question.get("name"),
                f"answer type/name changed: {answer!r}")
        if kind == "predicate":
            require(set(answer) == {"type", "name", "probability"}, f"predicate keys: {answer!r}")
            require(0 <= answer["probability"] <= 1, f"predicate probability: {answer!r}")
            continue
        require(set(answer) == {"type", "name", kind, "probabilities", "confidence"},
                f"{kind} keys: {answer!r}")
        options = question["choices" if kind == "choice" else "levels"]
        distribution = answer["probabilities"]
        require(len(distribution) == len(options), f"option count changed: {answer!r}")
        probabilities = [entry["probability"] for entry in distribution]
        require(all(0 <= p <= 1 for p in probabilities)
                and abs(math.fsum(probabilities) - 1) < 1e-5, f"invalid distribution: {answer!r}")
        p = [value / math.fsum(probabilities) for value in probabilities]
        mode = max(range(len(p)), key=p.__getitem__)
        if kind == "choice":
            for option, entry in zip(options, distribution):
                require(set(entry) == {"value", "probability"}
                        and type(entry["value"]) is type(option["value"])
                        and entry["value"] == option["value"], f"typed choice changed: {entry!r}")
            require(type(answer["choice"]) is type(options[mode]["value"])
                    and answer["choice"] == options[mode]["value"], f"wrong choice: {answer!r}")
            confidence = (max(p) - 1 / len(p)) / (1 - 1 / len(p))
        else:
            for index, (option, entry) in enumerate(zip(options, distribution)):
                require(set(entry) == {"value", "label", "probability"}
                        and type(entry["value"]) is int and entry["value"] == index
                        and entry["label"] == option["label"], f"score level changed: {entry!r}")
            score = math.fsum(i * value for i, value in enumerate(probabilities))
            require(abs(answer["score"] - score) < 1e-7, f"score is not weighted index: {answer!r}")
            spread = math.fsum(abs(i - (len(p) - 1) / 2) for i in range(len(p))) / len(p)
            confidence = max(0, 1 - math.fsum(value * abs(i - mode)
                                             for i, value in enumerate(p)) / spread)
        require(abs(answer["confidence"] - confidence) < 1e-7, f"confidence formula: {answer!r}")
    usage = body["usage"]
    require(set(usage) == {"input_tokens", "input_tokens_details", "output_tokens",
                           "output_tokens_details", "total_tokens"}, f"usage keys: {usage!r}")
    require(type(usage["input_tokens"]) is int and usage["input_tokens"] > 0
            and usage["total_tokens"] == usage["input_tokens"] and usage["output_tokens"] == 0,
            f"prefill-only usage: {usage!r}")
    details = usage["input_tokens_details"]
    require(set(details) == {"cached_tokens", "cache_write_tokens"}
            and 0 <= details["cached_tokens"] <= usage["input_tokens"]
            and details["cache_write_tokens"] == 0
            and usage["output_tokens_details"] == {"reasoning_tokens": 0}, f"usage details: {usage!r}")


def check_openai_wire(base_url: str, adapter: str, model_id: str) -> dict[str, Any]:
    path = "/v1/decisions"
    questions = openai_questions()
    payload = {"model": adapter, "input": STATE, "questions": questions}
    status, _, cards = http(base_url, "GET", "/typesafe/v1/models", headers=authorized())
    require(status == 200, f"TypeSafe discovery: {status} {cards!r}")
    typesafe_card = next(card for card in cards["models"] if card["name"] == adapter)
    status, _, card = http(base_url, "GET", f"/v1/models/{urllib.parse.quote(adapter, safe='')}",
                           headers=authorized())
    require(status == 200 and card["object"] == "model" and card["id"] == adapter
            and card["supported_endpoints"] == [path] and card["modalities"]["vision"] is False
            and card["context_window"] == min(73728, typesafe_card["max_context"])
            and card["max_state_tokens"] == min(65536, typesafe_card["max_context"]),
            f"OpenAI decision model card: {status} {card!r}")
    status, headers, baseline = http(base_url, "POST", path, payload, authorized())
    require(status == 200 and bool(headers.get("x-request-id")),
            f"OpenAI decision: {status} {baseline!r}")
    check_openai_answer(baseline, adapter, questions)

    ts_questions = {
        "billing": {"type": "noul", "instructions": questions[0]["instructions"]},
        "tone": {"type": "choice", "instructions": questions[1]["instructions"],
                 "criteria": {item["value"]: item.get("description") for item in questions[1]["choices"]}},
        "urgency": {"type": "score", "instructions": questions[2]["instructions"],
                    "criteria": [item["label"] + (": " + item["description"]
                                                  if item.get("description") else "")
                                 for item in questions[2]["levels"]]},
    }
    status, _, typesafe = http(base_url, "POST", "/typesafe/v1/systemone",
                               {"model": adapter, "state": STATE, "questions": ts_questions}, authorized())
    require(status == 200, f"parity TypeSafe request: {status} {typesafe!r}")
    deltas = [abs(baseline["answers"][0]["probability"] - typesafe["answers"]["billing"]["noul"])]
    for answer in baseline["answers"][1:]:
        reference = typesafe["answers"][answer["name"]]
        deltas.extend(abs(entry["probability"] - reference["probabilities"][str(entry["value"])])
                      for entry in answer["probabilities"])
    tolerance = 5e-5 + 1e-8 + (0.03 if typesafe_card["prefix_reuse"] else 0)
    require(max(deltas) <= tolerance, f"cross-surface probability drift: {max(deltas)} > {tolerance}")
    require(typesafe["usage"]["input_tokens"] == baseline["usage"]["input_tokens"],
            "cross-surface rendered input token counts differ")

    suffix = "  Keep this whitespace.\t"
    messages = [{"role": "user", "content": [{"type": "input_text", "text": STATE[:8]},
                                              {"type": "input_text", "text": STATE[8:]}]},
                {"role": "user", "content": suffix}]
    rendered = STATE + "\n\n" + suffix
    responses = []
    for value, safety in ((rendered, None), (messages, "s" * 128)):
        status, _, body = http(base_url, "POST", path,
                               {**payload, "input": value, "safety_identifier": safety}, authorized())
        require(status == 200, f"text/metadata request: {status} {body!r}")
        check_openai_answer(body, adapter, questions)
        responses.append(body)
    require(responses[0]["usage"]["input_tokens"] == responses[1]["usage"]["input_tokens"],
            "message concatenation or metadata changed input tokens")
    for first, second in zip(responses[0]["answers"], responses[1]["answers"]):
        left = ([first["probability"]] if first["type"] == "predicate"
                else [p["probability"] for p in first["probabilities"]])
        right = ([second["probability"]] if second["type"] == "predicate"
                 else [p["probability"] for p in second["probabilities"]])
        require(max(abs(a - b) for a, b in zip(left, right)) <= tolerance,
                "equivalent text/messages or safety metadata changed probabilities")

    typed = [{"type": "choice", "name": "duplicate", "instructions": "Choose the best label.",
              "choices": [{"value": value} for value in (True, "true", False, "false", "true")]},
             {**questions[0], "name": "duplicate"},
             {key: value for key, value in questions[2].items() if key != "name"}]
    status, _, body = http(base_url, "POST", path, {**payload, "questions": typed}, authorized())
    require(status == 200, f"typed/duplicate request: {status} {body!r}")
    check_openai_answer(body, adapter, typed)
    if not typesafe_card["prefix_reuse"]:
        require(all(result["usage"]["input_tokens_details"]["cached_tokens"] == 0
                    for result in (baseline, *responses, body)), "cold decisions reported cached tokens")

    bad = [("missing model", {key: value for key, value in payload.items() if key != "model"}),
           ("unknown field", {**payload, "temperature": 0}),
           ("numeric choice", {**payload, "questions": [
               {**questions[1], "choices": [{"value": 1}, {"value": 2}]}]}),
           ("null name", {**payload, "questions": [{**questions[0], "name": None}]}),
           ("empty questions", {**payload, "questions": []}),
           ("too many questions", {**payload, "questions": [questions[0]] * 201}),
           ("safety identifier", {**payload, "safety_identifier": "s" * 129}),
           ("non-user message", {**payload, "input": [{"role": "system", "content": STATE}]}),
           ("state overflow", {**payload, "input": " x" * 70000}),
           ("branch overflow", {**payload, "questions": [
               {**questions[0], "instructions": " x" * 74000}]})]
    for name, invalid in bad:
        status, _, error = http(base_url, "POST", path, invalid, authorized())
        require(status == 400 and set(error) == {"error"}
                and set(error["error"]) == {"message", "type", "param", "code"},
                f"{name}: {status} {error!r}")
    image = [{"role": "user", "content": [
        {"type": "input_text", "text": STATE},
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]
    status, _, error = http(base_url, "POST", path, {**payload, "input": image}, authorized())
    require(status == 400 and error["error"]["code"] == "unsupported_modality",
            f"an image was not explicitly rejected: {status} {error!r}")
    for model in ("no-such-model", "gpt-6-luna", DEFAULT_MODEL, model_id):
        status, _, error = http(base_url, "POST", path, {**payload, "model": model}, authorized())
        require(status == 404 and "error" in error, f"invalid decision model {model}: {status} {error!r}")
    status, _, error = http(base_url, "POST", path, {**payload, "safety_identifier": API_KEY})
    require(status == 401 and "error" in error, f"unauthenticated decision: {status} {error!r}")
    return {"wire": "ok", "model": card, "parity_max_abs_delta": max(deltas),
            "parity_tolerance": tolerance, "usage": baseline["usage"]}


def check_chat_under_decisions(base_url: str, chat_model: str, adapter: str) -> dict[str, Any]:
    """A greedy chat reply is bit-identical with and without decisions running beside it. The
    server runs with prefix reuse off, so both replies come from a cold prefill."""
    chat = {"model": chat_model, "max_tokens": 96, "temperature": 0,
            "messages": [{"role": "user", "content": "List five uses of a paper clip."}]}
    status, _, alone = http(base_url, "POST", "/v1/chat/completions", chat, authorized())
    require(status == 200, f"the lone chat returned {status}: {alone!r}")

    stop = threading.Event()
    decisions: list[tuple[str, float, float, int]] = []
    failures: list[str] = []

    def load(protocol: str, ready: threading.Event) -> None:
        index = 0
        try:
            while not stop.is_set():
                state = f"{STATE}\n{protocol} ticket {index}: the customer wrote again."
                if protocol == "systemone":
                    path = "/typesafe/v1/systemone"
                    body = {"state": state, "model": adapter,
                            "questions": {"billing": {"type": "noul",
                                                       "instructions": "Is this about billing?"},
                                          "urgency": {"type": "score",
                                                       "instructions": "How urgent is this?",
                                                       "criteria": ["Can wait", "This week", "Today"]}}}
                else:
                    path = "/v1/decisions"
                    body = {"input": state, "model": adapter, "questions": openai_questions()}
                started = time.monotonic()
                code, _, _ = http(base_url, "POST", path, body, authorized())
                decisions.append((protocol, started, time.monotonic(), code))
                ready.set()
                index += 1
        except Exception as error:
            failures.append(f"{protocol}: {error}")
            ready.set()

    workers = [(protocol, threading.Event()) for protocol in ("systemone", "openai_decisions")]
    threads = [threading.Thread(target=load, args=(protocol, ready), daemon=True)
               for protocol, ready in workers]
    try:
        for thread in threads:
            thread.start()
        for protocol, ready in workers:
            require(ready.wait(timeout=600), f"{protocol} load did not complete a request")
        require(not failures, f"decision load failed: {failures}")
        chat_started = time.monotonic()
        status, _, loaded = http(base_url, "POST", "/v1/chat/completions", chat, authorized())
        chat_ended = time.monotonic()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=610)
    require(not failures, f"decision load failed: {failures}")
    require(status == 200, f"the chat under decision load returned {status}: {loaded!r}")
    require(all(code == 200 for _, _, _, code in decisions),
            f"a decision under chat load failed: {[code for _, _, _, code in decisions]}")
    overlapping = {protocol: sum(kind == protocol and started < chat_ended and ended > chat_started
                                 for kind, started, ended, _ in decisions)
                   for protocol, _ in workers}
    require(all(overlapping.values()), f"both decision surfaces must overlap chat: {overlapping}")
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


def run_openai_sdk_client(base_url: str, adapter: str) -> dict[str, Any]:
    import openai

    version = re.match(r"(\d+)\.(\d+)\.(\d+)", openai.__version__)
    require(version is not None and tuple(int(part) for part in version.groups()) >= (3, 26, 0),
            f"openai>=3.26.0 is required, found {openai.__version__}")
    client = openai.OpenAI(api_key=API_KEY, base_url=base_url + "/v1", max_retries=0, timeout=600.0)
    try:
        names = [model.id for model in client.models.list()]
        require(adapter in names and DEFAULT_MODEL not in names and "gpt-6-luna" not in names,
                f"OpenAI SDK discovery: {names}")
        require(client.models.retrieve(adapter).id == adapter, "OpenAI SDK model lookup failed")
        questions = openai_questions()
        questions.append({"type": "choice", "instructions": "Choose the best label.",
                          "choices": [{"value": True}, {"value": "true"}]})
        result = client.decisions.create(model=adapter, input=STATE, questions=questions)
        check_openai_answer(result.model_dump(mode="json"), adapter, questions)
        require(bool(result._request_id), "OpenAI SDK response has no request id")
        require(type(result.answers[-1].probabilities[0].value) is bool
                and type(result.answers[-1].probabilities[1].value) is str,
                "OpenAI SDK lost boolean versus string choice values")
        try:
            client.decisions.create(model="no-such-model", input=STATE, questions=questions)
            raise SmokeFailure("OpenAI SDK answered an unknown model")
        except openai.NotFoundError as error:
            require(bool(error.request_id), "OpenAI SDK 404 has no request id")
            not_found = str(error)
        try:
            client.decisions.create(model=adapter, input=STATE, questions=[])
            raise SmokeFailure("OpenAI SDK answered an empty question array")
        except openai.BadRequestError as error:
            require(bool(error.request_id), "OpenAI SDK 400 has no request id")
            bad_request = str(error)
        with openai.OpenAI(api_key="wrong-key", base_url=base_url + "/v1", max_retries=0) as wrong:
            try:
                wrong.decisions.create(model=adapter, input=STATE, questions=questions)
                raise SmokeFailure("OpenAI SDK authenticated a wrong key")
            except openai.AuthenticationError as error:
                unauthorized = str(error)
        return {"sdk": openai.__version__, "models": names, "answers": result.model_dump()["answers"],
                "usage": result.usage.model_dump(),
                "errors": {"404": not_found, "400": bad_request, "401": unauthorized}}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--openai-client-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, default=Path("models/qwen3_8_27b.ninfer"))
    parser.add_argument("--server-bin", type=Path, default=Path("build/apps/ninfer-serve"))
    parser.add_argument("--lora-dir", type=Path)
    parser.add_argument("--adapter", required=True, help="the decision adapter's pool name")
    parser.add_argument("--generative-adapter", help="a generative adapter of the same pool")
    parser.add_argument("--sdk-python", type=Path, action="append", default=[],
                        help="a Python 3.11 interpreter with typesafe-sdk installed; repeatable")
    parser.add_argument("--openai-sdk-python", type=Path,
                        help="optional Python 3.11 interpreter with openai>=3.26.0 installed")
    parser.add_argument("--server-arg", action="append", default=[],
                        help="an extra ninfer-serve argument; repeatable")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    require(sys.version_info[:2] == (3, 11), "select a Python 3.11 interpreter for this smoke")

    if args.client_mode:
        print(json.dumps(run_sdk_client(args.base_url, args.adapter)))
        return
    if args.openai_client_mode:
        print(json.dumps(run_openai_sdk_client(args.base_url, args.adapter)))
        return

    if args.lora_dir is None or not args.sdk_python:
        parser.error("--lora-dir and at least one --sdk-python are required")
    artifact = args.artifact.resolve()
    server_bin = args.server_bin.resolve()
    require(artifact.is_file(), f"artifact does not exist: {artifact}")
    require(server_bin.is_file(), f"server binary does not exist: {server_bin}")
    interpreters = [(interpreter, "typesafe_sdk", "--client-mode") for interpreter in args.sdk_python]
    if args.openai_sdk_python:
        interpreters.append((args.openai_sdk_python, "openai", "--openai-client-mode"))
    for interpreter, module, _ in interpreters:
        probe = subprocess.run(
            [str(interpreter), "-c", f"import sys, {module}; assert sys.version_info[:2] == (3, 11)"],
            capture_output=True, text=True, check=False,
        )
        require(probe.returncode == 0, f"{interpreter} requires Python 3.11 and {module}: {probe.stderr}")

    port = args.port or free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(server_bin), str(artifact),
        "--host", "127.0.0.1", "--port", str(port),
        "--api-key", API_KEY,
        "--lora-dir", str(args.lora_dir.resolve()),
        "--systemone-default", args.adapter,
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
                    "openai_wire": check_openai_wire(base_url, args.adapter, model_id),
                    "chat_under_decisions": check_chat_under_decisions(base_url, chat_model,
                                                                       args.adapter),
                    "sdk": [],
                    "openai_sdk": None,
                }
                for interpreter, module, mode in interpreters:
                    completed = subprocess.run(
                        [str(interpreter), str(Path(__file__).resolve()), mode,
                         "--base-url", base_url, "--adapter", args.adapter],
                        capture_output=True, text=True, check=False,
                    )
                    if completed.returncode != 0:
                        raise SmokeFailure(f"SDK client {interpreter} failed:\n{completed.stderr}")
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                    if module == "openai":
                        report["openai_sdk"] = result
                    else:
                        report["sdk"].append(result)
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
