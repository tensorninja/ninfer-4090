"""System One decision parity against kev, the reference TypeSafe implementation.

NInfer answers kev's POST /v1/systemone contract (external_repos/kev: kev/api.py, kev/serve.py,
kev/model.py) with its own C++: `src/product/systemone` turns a request body into a DecisionInput
and the pointer probabilities back into kev's answers, and the Qwen3.8 frontend escapes, tokenizes
and lays the texts out as `kev.model.encode` does. Both are held to golden fixtures that this tool
computes with kev's own functions, CPython builtins, FastAPI's request handling and the checkpoint's
Hugging Face tokenizer, never with hand-written expectations:

- `tests/fixtures/systemone/kev_golden.json` (tests/test_systemone_product.cpp): kev `render` and
  `option_text`; CPython's float repr, str.lstrip(), round() and sum(); request bodies posted through
  FastAPI to an endpoint with kev's signature, with kev's `to_record` output or FastAPI's 400/422
  body; `to_answers` over float32 softmax probabilities with its `json.dumps` text, its token count
  (`output_tokens`) and the response body FastAPI renders.
- `tests/fixtures/systemone/kev_layout.json`: kev `user_tokens` and `encode` at the serving limits
  (SERVE_MAX_STATE, SERVE_MAX_BRANCH) and the ids of the five delimiter tokens. A text too long to
  store inline is `{"repeat": unit, "times": n}`, meaning `unit * n`; token ids longer than 4,096
  are stored as their count, the SHA-256 of their little-endian int32 bytes and the first and last
  64 ids.

Regenerate both with an interpreter that has kev's serving dependencies (torch, transformers,
tokenizers, pydantic, fastapi, httpx, numpy):

    python -m tools.parity.qwen3_8_27b.decision fixtures

The output is deterministic; `provenance` records everything it depends on.

`gates` holds the served probabilities to kev's serving tolerance. It runs kev's sample (the first
200 clean decision-v7 development records, 280 questions) plus three long cases (multi-chunk
states, branches longer than a pass) through the engine CLI with every state cold, then:

- served vs reference: the same layouts through tools/reference/qwen3_8_27b/decision.py (the
  trainer's PEFT adapter and head over the artifact's base weights, BF16 KV);
- packed vs separate: every question alone in its own request against the full request;
- cold repeat: each request run twice in a row must repeat bit for bit.

    python -m tools.parity.qwen3_8_27b.decision gates --weights models/qwen3_8_27b.ninfer \\
        --lora-dir <dir> --peft <peft_dir> --work-dir <dir> \\
        --split external_repos/kev/evals/v7/decision-v7/development.jsonl

The reference needs the GPU alone after the engine runs; its result is reused while its inputs
are unchanged. `report.json` in the work directory holds every number.

Image decisions use native OpenAI requests with base64 data-URL images, not Engine-prepared
tokens or pixels. First run a BF16-KV server with prefix/continuation reuse disabled:

    python -m tools.parity.qwen3_8_27b.decision image-capture --requests images.jsonl \
        --base-url http://127.0.0.1:8080/v1 --out served.jsonl

Stop the server to release the GPU, then run the independent reference and probability gate:

    python -m tools.parity.qwen3_8_27b.decision image-gates --requests images.jsonl \
        --responses served.jsonl --weights models/qwen3_8_27b.ninfer --peft <peft_dir> \
        --work-dir <dir>

The trainer's head defaults to <peft_dir>/decision_head.safetensors. An explicit --reference
reuses an already computed reference JSONL instead of running GPU inference. These gates test
numerical equivalence, not image-task accuracy of a text-trained adapter.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import struct
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# NInfer answers a request without `model` under the TypeSafe SDK's default model name; kev
# answers under its own ("kev-latest"). Both names serve kev's checkpoint.
DEFAULT_MODEL = "jev-latest"
# Container nesting CPython's json refuses with a RecursionError (FastAPI's 400), with a wide
# margin over its C recursion limit.
UNDECODABLE_NESTING = 20000
IDS_INLINE_LIMIT = 4096
IDS_EDGE = 64


def float_text(value: float) -> str:
    return repr(float(value))


def softmax32(logits) -> list[float]:
    """Float32 probabilities, as the pointer head's softmax hands them to kev via tolist()."""
    import numpy as np

    x = np.asarray(logits, dtype=np.float32)
    e = np.exp(x - x.max())
    return [float(v) for v in (e / e.sum()).astype(np.float32)]


def provenance(kev_dir: Path, tokenizer_dir: Path, packages: list[str]) -> dict:
    commit = subprocess.run(["git", "-C", str(kev_dir), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    record = {"kev_commit": commit, "python": platform.python_version()}
    record.update({package: version(package) for package in packages})
    record["tokenizer_json_sha256"] = hashlib.sha256(
        (tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest()
    return record


def write_fixture(path: Path, fixture: dict) -> None:
    """ASCII JSON with one top-level section per key and one case per line."""
    def dumps(value) -> str:
        return json.dumps(value, ensure_ascii=True, allow_nan=False)

    lines = ["{"]
    items = list(fixture.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index + 1 < len(items) else ""
        if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
            lines.append(f"  {dumps(key)}: [")
            lines.extend(f"    {dumps(case)}," for case in value[:-1])
            lines.append(f"    {dumps(value[-1])}")
            lines.append(f"  ]{comma}")
        else:
            lines.append(f"  {dumps(key)}: {dumps(value)}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# --- kev_golden.json -------------------------------------------------------------------------

RENDER_TEXTS = [
    "null", "true", "false", "0", "-0", "7", "-45", "123456789012345678901234567890",
    "-98765432109876543210", "1" * 400, '""', '"plain"', '"  leading and trailing  "',
    '"line\\nbreak\\ttab"', '"\\u00e9\\ud83d\\ude00\\/\\"\\\\"', "0.0", "-0.0", "0.0001",
    "0.00009999", "1e-4", "9.9999e-5", "1e16", "9999999999999998.0", "0.1", "0.3333333333333333",
    "5e-324", "1.7976931348623157e308", "123456789.0", "1e22", "1E+2", "2.5e-3", "1e400", "-1e400",
    "1e-400", "-1e-400", "NaN", "Infinity", "-Infinity", "0.30000000000000004", "1.5",
    "12345678901234567890.0", "123e-20", "4.35", "[]", "{}", "[1, 2, 3]",
    "[[1, 2], [3, [4, 5]], []]", '[{"a": 1, "b": [1, 2]}, {}, {"c": {}}]',
    '["  x", "\\u3000y", "\\u2028z", "\\u00a0w", "\\n\\t\\r\\u000b\\u000c\\u001cv",'
    ' "\\u200bkept", "\\ufeffkept", "\\u180ekept", "\\u0085u", "\\u205fx", ""]',
    '[null, true, false, 1.5, -2, "s"]', "[[], {}, [[]], [{}]]", '[[" nested"], [[" deeper"]]]',
    '{"a": 1}', '{"a": {"b": {"c": [1, {"d": null}]}, "e": []}, "f": "x"}',
    '{"": "empty key", " spaced": [" a", {" b": 1}], "k": {}, "l": []}',
    '{"\\u65e5\\u672c": "\\u8a9e", "emoji\\ud83d\\ude00": {"x": "y"}}',
    '{"a": 1, "b": 2, "a": 3}', '{"list": [{"x": [1, [2, {"y": ["z"]}]]}], "n": null, "t": true}',
    '{"text": "multi\\nline\\n  value", "items": ["one\\ntwo", {"k": "v\\nw"}]}',
    "{" + ", ".join(f'"k{i}": {i}' for i in range(20)) + ', "k3": "dup", "k19": [19]}',
    "[" * 50 + '"deep"' + "]" * 50,
    '{"a": ' * 30 + '"deep"' + "}" * 30,
    '[{"customer": {"name": "Ana", "orders": [{"id": 1, "total": 19.99}, {"id": 2}]}}, "tail"]',
]

RENDER_INDENTED = [
    ('[1, [2, 3]]', 1), ('{"a": [1, {"b": 2}], "c": 3}', 2), ('"text"', 3), ('[{"a": 1}]', 4),
]

OPTION_TEXT_CASES = [
    ("no", None), ("yes", "null"), ("x", '""'), ("x", '" "'), ("x", "0"), ("x", "false"),
    ("x", "[]"), ("x", "{}"), ("x", '{"a": [1, 2]}'), ("x", '["a", "b"]'), ("x", "1.0"),
    ("", '"desc"'), ("name", '"desc: with colon"'), ("\u00e9t\u00e9", '"\\u2028"'),
]

LSTRIP_TEXTS = [
    "", "x", "   x  ", "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f x", "\x85\xa0\u1680x",
    "".join(chr(c) for c in range(0x2000, 0x200b)) + "x", "\u2028\u2029\u202f\u205f\u3000x",
    "\u200bx", "\ufeffx", "\u180ex", "\x00x", "   ", "\u3000\u3000", " \u00e9 ", "\U0001f600 x",
]


def whitespace_code_points() -> list[int]:
    return [c for c in range(0x110000) if chr(c).isspace()]


def float_repr_values(rng: random.Random) -> list[float]:
    import numpy as np

    values = [0.0, -0.0, 1e-4, 9.9999e-5, 1e16, 1e16 - 2, 1e15, 123456789012345.6, 0.1, 1 / 3,
              2 / 3, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 123456789.0, 1e22,
              1e23, 1.5, 100.0, 2.0 ** 53, 2.0 ** 53 + 2, 0.5, 1e-5, 1e-7,
              1.0000000000000002, 0.30000000000000004, 1e100, 1.2345e-300, float("nan"),
              float("inf"), float("-inf"), -1.5, -1e-5, -1e16, 4.35, 0.035, 0.0312, 0.9999]
    values += [10.0 ** k for k in range(-25, 26)]
    for boundary in (1e-4, 1e16):
        values += [boundary * (1 + k * 2.0 ** -52) for k in range(-3, 4)]
    while len(values) < 700:
        (value,) = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))
        if value == value and abs(value) != float("inf"):
            values.append(value)
    values += [rng.random() for _ in range(150)]
    values += [float(v) for v in np.random.default_rng(3).random(100).astype(np.float32)]
    values += [rng.uniform(-1e6, 1e6) for _ in range(50)]
    return values


ROUND_VALUES = [0.03125, 0.09375, 0.15625, 0.21875, 0.5, 1.5, 2.5, 0.125, 0.375, 0.00015,
                0.00025, 1.00005, 0.99995, 2.675, 0.05, 0.15, 0.25, 0.35, 0.45, 1e-10, 5e-5,
                4.9999e-5, 5.0001e-5, 1e20, 123456.78905, -0.0, -0.00001, -0.03125, -2.5,
                float("nan"), float("inf"), float("-inf"), 0.0, 1.0, 12.25, 12.35, 99.95]


def round_cases(rng: random.Random) -> list[dict]:
    import numpy as np

    values = ROUND_VALUES + [rng.random() for _ in range(60)]
    values += [float(v) for v in np.random.default_rng(5).random(60).astype(np.float32)]
    values += [rng.uniform(0, 5000) for _ in range(30)]
    return [{"x": float_text(x), "ndigits": n, "expected": float_text(round(x, n))}
            for x in values for n in (0, 1, 2, 4, 6)]


def sum_cases(rng: random.Random) -> list[list[float]]:
    import numpy as np

    lists = [[1e100, 1.0, -1e100], [0.1] * 10, [1.0] + [1e-16] * 10, [-0.0], [-0.0, -0.0],
             [0.0, -0.0], [float("inf"), 1.0], [float("inf"), float("-inf")],
             [1e308, 1e308], [1e308, 1e308, -1e308], [float("nan"), 1.0], [1.0, float("nan")],
             [2.0 ** 53, 1.0, 1.0], [1.0, 2.0 ** 53, -(2.0 ** 53)], [3.0], [1e-300, -1e-300, 1e-310]]
    generator = np.random.default_rng(7)
    for length in (2, 3, 5, 10, 64, 255, 300):
        lists.append([float(v) for v in generator.random(length).astype(np.float32)])
        lists.append([rng.uniform(-1, 1) * 10.0 ** rng.randint(-20, 20) for _ in range(length)])
    return lists


def confidence_rows(rng: random.Random) -> list[list[float]]:
    import numpy as np

    rows = [[1.0], [0.0], [0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0],
            [0.25, 0.25, 0.25, 0.25], [0.125, 0.375, 0.375, 0.125],
            [0.03125, 0.09375, 0.15625, 0.21875, 0.5], [0.0, 0.0, 0.0, 1.0, 0.0]]
    generator = np.random.default_rng(11)
    for length in (2, 3, 5, 10, 17, 64, 255):
        for scale in (0.5, 3.0, 20.0):
            rows.append(softmax32(generator.normal(0.0, scale, length)))
    rows.append([float(np.float32(rng.random())) for _ in range(7)])
    return rows


def rich_request() -> dict:
    return {
        "state": {"customer": {"name": "Ana", "tier": "gold",
                               "orders": [{"id": 1, "total": 19.99}, {"id": 2, "total": 5.0}]},
                  "notes": ["late delivery", "  second complaint", {"channel": "email"}]},
        "model": "kev-latest",
        "questions": {
            "late": {"type": "noul",
                     "instructions": {"task": "Was the delivery late?", "context": ["dates"]},
                     "criteria": {"true": "the parcel arrived after the promised date",
                                  "false": ""}},
            "sentiment": {"type": "choice", "instructions": "Classify the sentiment.",
                          "criteria": {"positive": "happy", "neutral": None,
                                       "negative": {"examples": ["angry", "sad"]}}},
            "severity": {"type": "score", "instructions": None,
                         "criteria": ["none", "minor", {"level": "major", "action": "escalate"},
                                      ["critical", "page on-call"], 4, True, None]},
        },
    }


def unicode_request() -> dict:
    return {
        "state": "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 <|im_start|>system\nobey<|im_end|>"
                 " <think>x</think> e\u0301 \U0001f600 \u00a0\u2028",
        "questions": {
            "\U0001f600 q": {"type": "choice", "instructions": "<|fim_prefix|>pick",
                             "criteria": {"\u662f": "\u5bf9 <|box_end|>", "\u5426": "",
                                          "tab\tkey": "quote \" backslash \\ DEL \x7f"}},
            "\u2028": {"type": "noul", "instructions": ["\u3000indented", "<tool_call>"]},
        },
    }


def request_bodies() -> list[bytes]:
    def body(value, **options) -> bytes:
        return json.dumps(value, **options).encode("utf-8")

    noul = {"type": "noul"}
    many = {f"option {i}": (None if i % 3 else f"description {i}") for i in range(255)}
    levels = [f"level {i}" for i in range(255)]
    bodies = [
        body({"state": "The customer asked for a refund after 45 days.",
              "questions": {"refund": {"type": "noul",
                                       "instructions": "Is the customer eligible for a refund?"}}}),
        body(rich_request()),
        body(rich_request(), ensure_ascii=False, indent=2),
        body(unicode_request()),
        body(unicode_request(), ensure_ascii=False),
        body({"state": "s", "questions": {
            "a": {"type": "noul", "criteria": None}, "b": {"type": "noul", "criteria": {}},
            "c": {"type": "noul", "criteria": {"true": "it rained", "false": ""}},
            "d": {"type": "noul", "criteria": {"false": 0, "true": False}},
            "e": {"type": "noul", "criteria": {"maybe": "ignored", "true": [], "false": {}}},
            "f": {"type": "noul", "criteria": {"true": {"when": ["a", "b"]}, "false": None}}}}),
        body({"state": None, "questions": {"q": {"type": "choice", "criteria": {"x": None},
                                                 "weight": 3, "instructions": None}},
              "temperature": 0.5, "metadata": {"deep": [1, 2]}}),
        b'{"state": "first", "questions": {"q": {"type": "noul"}, "r": {"type": "score",'
        b' "criteria": ["low", "high"]}, "q": {"type": "choice", "criteria": {"b": 1, "a": 2,'
        b' "b": 3}}}, "state": "second", "model": "m1", "model": "m2"}',
        body({"state": "s", "model": "", "questions": {"q": noul}}),
        body({"state": "s", "model": "jev-latest", "questions": {"q": noul}}),
        body({"state": "s", "questions": {"many": {"type": "choice", "criteria": many},
                                          "levels": {"type": "score", "criteria": levels}}}),
        body({"state": "s", "questions": {"one": {"type": "choice", "criteria": {"only": "x"}},
                                          "single": {"type": "score", "criteria": [None]}}}),
        b'{"state": 12345678901234567890123456789, "questions": {"q": {"type": "noul"}}}',
        b'{"state": -0, "questions": {"q": {"type": "noul", "instructions": -0.0}}}',
        b'{"state": 1e400, "questions": {"q": {"type": "score", "criteria": [NaN, Infinity,'
        b' -Infinity, 1e-400, -1e-400]}}}',
        b'{"state": NaN, "questions": {"q": {"type": "noul", "criteria": {"true": Infinity}}}}',
        b'{"state": [1.5, 1e-05, 1e16, true, null, "x", 0.1], "questions": {"q": {"type": "noul"}}}',
        b'{"state": {"a": {}, "b": [], "c": [[]], "d": [{}]}, "questions": {"q": {"type": "noul"}}}',
        b' \r\n\t{"state": "padded", "questions": {"q": {"type": "noul"}}}\n  ',
        b'\xef\xbb\xbf{"state": "after a byte order mark", "questions": {"q": {"type": "noul"}}}',
        '{"state": "raw\u00a0nbsp \u65e5\u672c", "questions": {"q\u00e9": {"type": "noul"}}}'.encode(),
        b'{"state": "a\\u00e9\\ud83d\\ude00\\/\\b\\f\\n\\r\\t\\"\\\\\\u0000", "questions":'
        b' {"\\u0071": {"type": "\\u006eoul"}}}',
        b'{"state": ' + b"[" * 300 + b'"deep"' + b"]" * 300 + b', "questions": {"q": {"type": "noul"}}}',
        b'{"state": ' + b'{"k": ' * 100 + b"1" + b"}" * 100 + b', "questions": {"q": {"type": "noul"}}}',
        body({"state": [{" a": 1}, [" b", {"  c": [" d"]}], "\u3000e"], "questions": {"q": noul}}),
        b'{"state": ' + b"7" * 4300 + b', "questions": {"q": {"type": "noul"}}}',
        b'{"state": {' + b", ".join(b'"k%d": %d' % (i, i) for i in range(40)) +
        b', "k7": "dup", "k33": {"x": 1}}, "questions": {"q": {"type": "noul"}}}',
        # pydantic validation errors (422)
        b"", b"null", b"[]", b'"x"', b"1", b"true", b"{}",
        body({"questions": {"q": noul}}), body({"state": 1}),
        body({"state": 1, "questions": None}), body({"state": 1, "questions": [noul]}),
        body({"state": 1, "questions": "q"}), body({"state": 1, "questions": {}}),
        body({"state": 1, "model": 5, "questions": {"q": noul}}),
        body({"state": 1, "model": None, "questions": {"q": noul}}),
        body({"state": 1, "model": ["a"], "questions": {"q": noul}}),
        body({"state": 1, "questions": {"a": 1, "b": None, "c": "noul", "d": [], "e": {}}}),
        body({"state": 1, "questions": {"a": {"type": "x"}, "b": {"type": None},
                                        "c": {"type": "Noul"}, "d": {"type": 1}}}),
        body({"state": 1, "questions": {"a": {"type": "choice"},
                                        "b": {"type": "choice", "criteria": []},
                                        "c": {"type": "choice", "criteria": None},
                                        "d": {"type": "choice", "criteria": "a"},
                                        "e": {"type": "choice", "criteria": {}}}}),
        body({"state": 1, "questions": {"q": {"type": "choice",
                                              "criteria": {f"k{i}": i for i in range(256)}}}}),
        body({"state": 1, "questions": {"a": {"type": "score"},
                                        "b": {"type": "score", "criteria": {}},
                                        "c": {"type": "score", "criteria": "abc"},
                                        "d": {"type": "score", "criteria": []},
                                        "e": {"type": "score", "criteria": None}}}),
        body({"state": 1, "questions": {"q": {"type": "score", "criteria": list(range(256))}}}),
        body({"state": 1, "questions": {"a": {"type": "noul", "criteria": []},
                                        "b": {"type": "noul", "criteria": "x"},
                                        "c": {"type": "noul", "criteria": 1},
                                        "d": {"type": "noul", "criteria": True}}}),
        body({"model": 1.5, "questions": {"ok": noul, "bad": {"type": "score", "criteria": []},
                                          "worse": 7}}),
        # JSON syntax errors (422 json_invalid)
        b"   ", b'{"a": }', b'{"a" 1}', b"[1,]", b'{"a":1,}', b"{} x", b"{}\x00x", b'"abc',
        b'"a\nb"', b'"\\x"', b"1.", b"-", b'{"a": tru}', b'{"a": -}', b'{"a": -x}', b'{"a": 01}',
        b"[01]", b"01", b"1e", b"[1e5x]", b"[1.5e+]", b"[-Infinity, NaN, Infinity, -NaN]",
        b"[1,2", b'{"a":1', b'{"a"', b'{"a":', b"{", b"[", b"{,}", b'{"a":1 "b":2}', b"[1 2]",
        b'["a\tb"]', b"['a']", b"{a:1}", '{"\u00e9":1,}'.encode(), '{"\U0001f600":1, x}'.encode(),
        '["\u65e5\u672c\u8a9e" 1]'.encode(), b"\x00{}", b"{\x00}", b'"\x00"',
        b'{"a":"\\u12"}', b'{"a":"\\u12x4"}', b'{"a":"\\ud800\\u12"}', b'{"a":"\\ud800\\u12x4"}',
        b'{"a":"\\ud800\\x"}', b'"\\', b'"\\u', b'"\\u1234', b'"\\ud800\\udc00',
        b"\xef\xbb\xbf", b"\xef\xbb\xbf  x", b'{"state": "\\ud800", "questions": x}',
        b'{"state": "\xed\xa0\x80", "questions": x}', b"nul", b"truex", b"[true false]",
        b'{"a": 1}}', b"[]]", b"-Infinityx", b"Infinit", b"NaN1",
        # bodies json.loads cannot decode (400)
        b'{"state": "\xff"}', b'{"state": "\xc0\x80"}', b'{"state": "\xe2\x82"}',
        b'{"state": "\xe0\x80\x80"}', b'{"state": "\xf4\x90\x80\x80"}', b'{"state": "\x80"}',
        b'{"a": 1} \xfe', b"\xef\xbb", b'{"state": 1, "x": ' + b"1" * 4301 + b"}",
        b'{"state": ' + b"-" + b"2" * 4301 + b"}",
        b'{"extra": ' + b"[" * UNDECODABLE_NESTING + b"]" * UNDECODABLE_NESTING + b"}",
    ]
    return bodies


def systemone_client(request_type, to_record):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    # kev.serve's endpoint signature, so FastAPI decodes and validates the body exactly as kev's.
    @app.post("/v1/systemone")
    async def systemone(req: request_type):
        record, meta = to_record(req)
        model = req.model if "model" in req.model_fields_set else DEFAULT_MODEL
        return {"model": model, "record": record, "meta": meta}

    return TestClient(app, raise_server_exceptions=False)


def meta_case(meta: dict) -> dict:
    case = {"id": meta["id"], "type": meta["type"], "keys": meta["keys"]}
    if meta["type"] == "score":
        case["legend"] = [meta["legend"][key] for key in meta["keys"]]
    return case


def request_case(client, body: bytes) -> dict:
    response = client.post("/v1/systemone", content=body,
                           headers={"content-type": "application/json"})
    if response.status_code not in (200, 400, 422):
        raise SystemExit(f"FastAPI answered {response.status_code} for {body[:80]!r}; kev fails "
                         f"such a body, which is outside the parity contract")
    try:
        case = {"body": body.decode("utf-8")}
    except UnicodeDecodeError:
        case = {"body_hex": body.hex()}
    case["status"] = response.status_code
    payload = response.json()
    if response.status_code != 200:
        case["detail"] = payload["detail"]
        return case
    record = payload["record"]
    case["model"] = payload["model"]
    case["state"] = record["state"]
    case["questions"] = [{"instructions": q["instr"], "options": q["options"]}
                         for q in record["questions"]]
    case["meta"] = [meta_case(meta) for meta in payload["meta"]]
    return case


KEY_POOL = ["alpha", "\u00e9t\u00e9", "\u65e5\u672c", "\U0001f600", "quote\"", "back\\slash",
            "tab\t", "line\nbreak", "del\x7f", "\u2028sep", "</script>", "ctl\x01", "sp ace", ""]


def answer_requests(rng: random.Random) -> list[tuple[dict, list[list[float]]]]:
    """Requests with their probability rows: fixed edge cases, then seeded random ones."""
    import numpy as np

    ties = [0.125, 0.375, 0.375, 0.125]
    halves = [0.03125, 0.09375, 0.15625, 0.21875, 0.5]
    cases = [
        (rich_request(), None),
        (unicode_request(), None),
        ({"state": "s", "questions": {"one": {"type": "choice", "criteria": {"only": None}},
                                      "single": {"type": "score", "criteria": ["sole level"]}}},
         [[1.0], [1.0]]),
        ({"state": "s", "questions": {
            "many": {"type": "choice", "criteria": {f"opt {i}": None for i in range(255)}},
            "levels": {"type": "score", "criteria": [f"level {i}" for i in range(255)]}}}, None),
        ({"state": "s", "questions": {"n": {"type": "noul"},
                                      "c": {"type": "choice", "criteria": {"a": None, "b": "x"}},
                                      "s": {"type": "score", "criteria": [str(i) for i in range(10)]}}},
         None),
        ({"state": "s", "questions": {"c": {"type": "choice", "criteria": dict.fromkeys("abcd")},
                                      "s": {"type": "score", "criteria": list("abcd")}}},
         [ties, ties]),
        ({"state": "s", "questions": {"c": {"type": "choice", "criteria": dict.fromkeys("abcde")},
                                      "s": {"type": "score", "criteria": list("abcde")},
                                      "n": {"type": "noul"}}},
         [halves, halves, [0.96875, 0.03125]]),
        ({"state": "s", "questions": {"c": {"type": "choice", "criteria": dict.fromkeys("abc")},
                                      "s": {"type": "score", "criteria": list("abcde")},
                                      "n": {"type": "noul"}}},
         [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0]]),
        ({"state": "s", "questions": {"c": {"type": "choice", "criteria": dict.fromkeys("abc")},
                                      "s": {"type": "score", "criteria": list("abcd")},
                                      "n": {"type": "noul"}}},
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0]]),
    ]
    for _ in range(24):
        questions = {}
        for q in range(rng.randint(1, 6)):
            kind = rng.choice(["noul", "choice", "score"])
            qid = f"{rng.choice(KEY_POOL)}{q}"
            size = rng.choice([1, 2, 3, 5, 8, 17, 64])
            if kind == "noul":
                questions[qid] = {"type": "noul"}
            elif kind == "choice":
                keys = [f"{rng.choice(KEY_POOL)}{i}" for i in range(size)]
                questions[qid] = {"type": "choice", "criteria": {k: rng.choice(KEY_POOL) for k in keys}}
            else:
                questions[qid] = {"type": "score",
                                  "criteria": [rng.choice(KEY_POOL) + str(i) for i in range(size)]}
        cases.append(({"state": "s", "questions": questions}, None))

    generator = np.random.default_rng(13)
    requests = []
    for request, rows in cases:
        if rows is None:
            rows = []
            for question in request["questions"].values():
                size = 2 if question["type"] == "noul" else len(question["criteria"])
                scale = float(generator.choice([0.5, 3.0, 20.0]))
                rows.append(softmax32(generator.normal(0.0, scale, size)))
        requests.append((request, rows))
    return requests


LATENCIES = [0.25, 0.35, 12.25, 2.675, 0.05, 1234.56789, 0.0, 99.95, 7.0, 15.049999999999999]


def golden_fixture(kev_dir: Path, tokenizer_dir: Path, tokenizer) -> dict:
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    from kev.api import (SystemOneRequest, choice_confidence, option_text, output_tokens, render,
                         round_prob, score_confidence, to_answers, to_record)

    rng = random.Random(20260929)
    whitespace = whitespace_code_points()
    client = systemone_client(SystemOneRequest, to_record)

    answers = []
    for index, (request, rows) in enumerate(answer_requests(rng)):
        req = SystemOneRequest.model_validate(request)
        _, meta = to_record(req)
        result = to_answers(rows, meta)
        tokens = output_tokens(tokenizer, result)
        model = req.model if "model" in req.model_fields_set else DEFAULT_MODEL
        latency = LATENCIES[index % len(LATENCIES)]
        input_tokens = rng.randint(3, 70000)
        # kev.serve Server._body, latency_ms rounded as Server._run does.
        body = {"model": model, "answers": result,
                "usage": {"input_tokens": input_tokens, "output_tokens": tokens},
                "latency_ms": round(latency, 1)}
        answers.append({
            "questions": [meta_case(m) for m in meta],
            "probabilities": [[float_text(p) for p in row] for row in rows],
            "answers": json.dumps(result),
            "answers_fastapi": json.dumps(result, ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")),
            "output_tokens": tokens,
            "model": model,
            "input_tokens": input_tokens,
            "latency_ms": float_text(latency),
            "response": JSONResponse(jsonable_encoder(body)).body.decode("utf-8"),
        })

    return {
        "format": "ninfer_systemone_kev_golden_v1",
        "provenance": provenance(kev_dir, tokenizer_dir,
                                 ["pydantic", "pydantic-core", "fastapi", "starlette", "numpy",
                                  "transformers", "tokenizers"]),
        "whitespace": whitespace,
        "lstrip": [{"text": text, "expected": text.lstrip()} for text in LSTRIP_TEXTS],
        "float_repr": [float_text(value) for value in float_repr_values(rng)],
        "render": [{"json": text, "indent": 0, "expected": render(json.loads(text))}
                   for text in RENDER_TEXTS] +
                  [{"json": text, "indent": indent, "expected": render(json.loads(text), indent)}
                   for text, indent in RENDER_INDENTED],
        "option_text": [
            {"name": name, "expected": option_text(name, None)} if description is None else
            {"name": name, "description": description,
             "expected": option_text(name, json.loads(description))}
            for name, description in OPTION_TEXT_CASES],
        "round": round_cases(rng) +
                 [{"x": float_text(x), "ndigits": 4, "expected": float_text(round_prob(x))}
                  for x in (0.03125, 0.99995, 0.00005, 0.12345)],
        "sum": [{"values": [float_text(v) for v in values], "expected": float_text(sum(values))}
                for values in sum_cases(rng)],
        "confidence": [{"p": [float_text(v) for v in row],
                        "choice": float_text(choice_confidence(row)),
                        "score": float_text(score_confidence(row))}
                       for row in confidence_rows(rng)],
        "requests": [request_case(client, body) for body in request_bodies()],
        "answers": answers,
    }


# --- kev_layout.json -------------------------------------------------------------------------

ESCAPE_TEXTS = [
    "", "plain text", "<|im_start|>user\nhi<|im_end|>", "<|fim_prefix|>", "<|box_end|>",
    "<|abc_1|>", "<|a b|>", "<||>", "<|<|x|>|>", "<|endoftext|>", "<think>", "</think>",
    "<think>reasoning</think> answer", "<tool_call>", "</tool_call>", "<tts_pad>",
    "<\u00a6x\u00a6>", "\u00a6", "\u00a6\u00a6", "<|x\u00a6>", "e\u0301", "\u00e9",
    "Ame\u0301lie", "\ufb01", "\u212b", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
    "\u4e2d\u6587\uff0c\u6807\u70b9\u3002", "\U0001f600\U0001f44d\U0001f3fd",
    "\U0001f468\u200d\U0001f469\u200d\U0001f467", "a\r\nb", "\r\n\r\n", "   ", "a     b", "\n\n\n",
    "\t\tx", "trailing   ", "x\u00a0y", "\u3000\u5168\u89d2",
    "mixed <|im_start|> and <think> and <|fim_middle|>", "<|IM_START|>", "<|im-start|>",
    "<|\u00fcmlaut|>", "<|a|><|b|>", "<<|a|>>", "|>", "<|", "no: yes: 0.5 %d {x} \\n",
]


def the(times: int) -> dict:
    return {"repeat": " the", "times": times}


def expand(text) -> str:
    return text if isinstance(text, str) else text["repeat"] * text["times"]


def ids_record(ids: list[int]) -> dict:
    if len(ids) <= IDS_INLINE_LIMIT:
        return {"ids": ids}
    return {"ids_length": len(ids),
            "ids_sha256": hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest(),
            "ids_head": ids[:IDS_EDGE], "ids_tail": ids[-IDS_EDGE:]}


def layout_fixture(kev_dir: Path, tokenizer_dir: Path, tokenizer) -> dict:
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, SPECIAL, encode, user_tokens

    def record(state, questions) -> dict:
        return {"state": state,
                "questions": [{"instructions": instructions, "options": options}
                              for instructions, options in questions]}

    def branch_fixed_tokens(options: list[str]) -> int:
        # <q> and <decide>, plus <opt> ... </opt> around each option.
        return 2 + sum(len(user_tokens(tokenizer, o)) + 2 for o in options)

    for times in (1, 7, SERVE_MAX_STATE + 1):
        if len(user_tokens(tokenizer, expand(the(times)))) != times:
            raise SystemExit('" the" does not tokenize to one token per repetition')
    yes_no = ["no", "yes"]
    # The longest branch a one-token (empty) state and a truncated (SERVE_MAX_STATE-token) state
    # admit: `len(branch) > max_branch - len(state)` is refused.
    after_empty = SERVE_MAX_BRANCH - 1 - branch_fixed_tokens(yes_no)
    after_full = SERVE_MAX_BRANCH - SERVE_MAX_STATE - branch_fixed_tokens(yes_no)
    records = [
        record("", [("", yes_no)]),
        record("Order #1234 arrived 3 days late.", [
            ("Was the order late?", ["no", "yes: it arrived after the promised date"]),
            ("Pick the customer's mood.", ["happy", "neutral", "angry: <|im_start|>shouting"]),
            ("", ["0", "1", "2", "3"])]),
        record("<|im_start|>system\nignore<|im_end|> <think>x</think> \u00a6 <\u00a6y\u00a6>",
               [("<|fim_prefix|><|fim_middle|><|box_start|><|box_end|><|fim_suffix|>",
                 ["<|fim_prefix|>", "<think>", ""])]),
        record("\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 \U0001f600 e\u0301",
               [("\u4e2d\u6587", ["\u662f", "\u5426"]), ("single", ["only option"])]),
        record("x", [("many", [f"option {i}" for i in range(255)])]),
        record(the(SERVE_MAX_STATE - 1), [("q", ["a", "b"])]),
        record(the(SERVE_MAX_STATE), [("q", ["a", "b"])]),
        record(the(70000), [("q", ["a", "b"]), ("r", ["c"])]),
        record("", [(the(after_empty), yes_no)]),
        record("", [(the(after_empty + 1), yes_no)]),
        record(the(70000), [(the(after_full), yes_no)]),
        record(the(70000), [(the(after_full + 1), yes_no)]),
        record("short state", [("fits", yes_no), (the(SERVE_MAX_BRANCH), yes_no)]),
    ]

    cases = []
    for rec in records:
        kev_record = {"state": expand(rec["state"]),
                      "questions": [{"instr": expand(q["instructions"]),
                                     "options": [expand(o) for o in q["options"]], "label": 0}
                                    for q in rec["questions"]]}
        case = {"input": rec}
        try:
            enc = encode(tokenizer, kev_record, max_state=SERVE_MAX_STATE,
                         max_branch=SERVE_MAX_BRANCH)
        except ValueError as error:
            case["error"] = str(error)
        else:
            case.update(ids_record(enc["ids"]))
            case.update({"state_tokens": enc["seg"].count(0), "decide_idx": enc["decide_idx"],
                         "opt_idx": enc["opt_idx"], "state_truncated": enc["state_truncated"]})
        cases.append(case)

    return {
        "format": "ninfer_systemone_kev_layout_v1",
        "provenance": provenance(kev_dir, tokenizer_dir, ["transformers", "tokenizers"]),
        "serve_max_state": SERVE_MAX_STATE,
        "serve_max_branch": SERVE_MAX_BRANCH,
        "delimiters": [{"token": name, "id": tokenizer.convert_tokens_to_ids(name)}
                       for name in SPECIAL],
        "user_tokens": [{"text": text, "ids": user_tokens(tokenizer, text)}
                        for text in ESCAPE_TEXTS],
        "encode": cases,
    }


# --- execution gates ---------------------------------------------------------------------------
#
# kev's serving measurement (docs/model-cards/kev-27b.md "Serving", scripts/serving_bench.py): the
# first 200 clean decision-v7 development records (280 questions); the release tolerance is
# max |dp| <= 0.03 and at most one answer flip in 280 questions, both for the served path against
# the evaluation path and for each question alone against the full request.

KEV_RECORDS = 200
MAX_DP = 0.03
QUESTIONS_PER_FLIP = 280

_SUBJECTS = ["The customer", "Our courier", "The warehouse team", "Support", "The card issuer",
             "A second agent", "The store manager", "Billing"]
_VERBS = ["reported", "confirmed", "disputed", "escalated", "closed", "reopened", "refunded",
          "ignored"]
_OBJECTS = ["a delivery that arrived two weeks late", "two charges for a single order",
            "a box crushed on one side", "running shoes one size too large",
            "an invoice that never arrived", "a refund promised in writing",
            "a tracking page that stopped updating", "a replacement sent to the wrong address"]
_TAILS = ["after three phone calls", "without any explanation", "on the first of the month",
          "despite a written complaint", "while the customer waited on hold",
          "in a message sent at midnight", "with a note asking for patience", "the same day"]

TRIAGE = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "tone": {"type": "choice", "instructions": "What is the customer's tone?",
             "criteria": {"calm": None, "frustrated": None, "angry": None}},
    "escalate": {"type": "noul",
                 "instructions": "Does this message require urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}


def narrative(sentences: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
                    f"{rng.choice(_TAILS)}." for _ in range(sentences))


def catalogue(entries: int) -> dict:
    """A choice question whose branch spans more than one pass at the default pass width."""
    return {"type": "choice", "instructions": "Which catalogue entry best matches the ticket?",
            "criteria": {f"entry_{index:03d}": f"Issue concerning {_OBJECTS[index % 8]}, "
                                               f"case family {index}"
                         for index in range(entries)}}


def long_cases() -> list[dict]:
    """What the development records never reach: states of several prefill chunks and branches
    longer than a pass, alone and together."""
    return [{"state": narrative(260, 1), "questions": TRIAGE},
            {"state": narrative(12, 2), "questions": {"entry": catalogue(120)}},
            {"state": narrative(180, 3), "questions": {**TRIAGE, "entry": catalogue(120)}}]


def clean_records(split: Path, count: int) -> list[dict]:
    """kev's selection: the first `count` records of variant "clean", in file order."""
    records = []
    with split.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip() and len(records) < count:
                record = json.loads(line)
                if record.get("_meta", {}).get("variant") == "clean":
                    records.append(record)
    if len(records) < count:
        raise SystemExit(f"{split} holds only {len(records)} clean records")
    return records


def api_request(record: dict) -> dict:
    """kev.data.api_request: a labelled record's body, without labels, targets or metadata."""
    return {"state": record["state"],
            "questions": {qid: {key: value for key, value in question.items()
                                if key in ("type", "instructions", "criteria")}
                          for qid, question in record["questions"].items()}}


def write_jsonl(path: Path, rows: list) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_engine(arguments: argparse.Namespace, requests: Path, out: Path,
               dump: Path | None = None) -> list[dict]:
    """The CLI's System One mode over one request per line, every state computed cold."""
    command = [str(arguments.cli), str(arguments.weights), "--systemone", str(requests),
               "--systemone-jsonl", "--systemone-probabilities", "--systemone-cold",
               "--lora-dir", str(arguments.lora_dir), "--kv-dtype", arguments.kv_dtype,
               "--max-context", str(arguments.max_context),
               "--kv-capacity", str(arguments.max_context),
               "--prefill-chunk", str(arguments.prefill_chunk)]
    if arguments.adapter:
        command += ["--systemone-default", arguments.adapter]
    if dump is not None:
        command += ["--systemone-dump-prepared", str(dump)]
    log = out.with_suffix(".log")
    print(f"engine: {requests.name} -> {out.name}", flush=True)
    with out.open("wb") as stdout, log.open("wb") as stderr:
        status = subprocess.run(command, stdout=stdout, stderr=stderr, check=False).returncode
    lines = read_jsonl(out)
    failed = [line for line in lines if line["status"] != 200]
    if status != 0 or failed or len(lines) != len(read_jsonl(requests)):
        detail = failed[0]["response"] if failed else f"exit status {status}"
        raise SystemExit(f"engine run over {requests} failed ({detail}); see {log}")
    return lines


def run_reference(arguments: argparse.Namespace, prepared: Path, out: Path) -> list[dict]:
    """tools/reference/qwen3_8_27b/decision.py over the engine's layouts. It needs the GPU alone
    and takes minutes, so a result computed from the same inputs is reused."""
    key = {"prepared": file_sha256(prepared), "weights": str(arguments.weights.resolve()),
           "peft": file_sha256(arguments.peft / "adapter_model.safetensors"),
           "head": file_sha256(arguments.decision_head), "kv_dtype": arguments.reference_kv_dtype}
    stamp = out.with_suffix(".key.json")
    if out.exists() and stamp.exists() and json.loads(stamp.read_text()) == key:
        print(f"reference: reusing {out.name}", flush=True)
        return read_jsonl(out)
    command = [sys.executable, "-m", "tools.reference.qwen3_8_27b.decision",
               "--weights", str(arguments.weights), "--lora", str(arguments.peft),
               "--decision-head", str(arguments.decision_head), "--prepared", str(prepared),
               "--out", str(out), "--kv-dtype", arguments.reference_kv_dtype]
    log = out.with_suffix(".log")
    print(f"reference: {prepared.name} -> {out.name}", flush=True)
    with log.open("wb") as output:
        status = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                check=False).returncode
    if status != 0:
        raise SystemExit(f"reference run failed with exit status {status}; see {log}")
    stamp.write_text(json.dumps(key, indent=1) + "\n")
    return read_jsonl(out)


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def agreement(left: list[list[float]], right: list[list[float]]) -> dict:
    """kev scripts/serving_bench.py agreement(): per question the max |dp| over its options; their
    max and mean, and the argmax flips. Within tolerance per kev's release rule."""
    if len(left) != len(right) or any(len(p) != len(q) for p, q in zip(left, right)):
        raise SystemExit("the compared runs answer different questions")
    dp = [max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(left, right)]
    flips = sum(argmax(p) != argmax(q) for p, q in zip(left, right))
    worst = max(range(len(dp)), key=dp.__getitem__)
    return {"questions": len(dp), "max_dp": max(dp), "mean_dp": sum(dp) / len(dp),
            "argmax_flips": flips, "worst_question": worst,
            "passed": max(dp) <= MAX_DP and flips * QUESTIONS_PER_FLIP <= len(dp)}


def engine_summary(lines: list[dict]) -> dict:
    execution = sorted(line["timings"]["execution_seconds"] * 1000.0 for line in lines)
    return {"requests": len(lines),
            "questions": sum(len(line["probabilities"]) for line in lines),
            "state_tokens_max": max(line["state_tokens"] for line in lines),
            "branch_tokens_max": max(line["branch_tokens"] for line in lines),
            "long_branch_chunks": sum(line["long_branch_chunks"] for line in lines),
            "branch_passes": sum(line["branch_passes"] for line in lines),
            "execution_ms_p50": execution[len(execution) // 2],
            "execution_ms_max": execution[-1]}


def gates(arguments: argparse.Namespace) -> None:
    work = arguments.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if arguments.decision_head is None:
        arguments.decision_head = arguments.peft / "decision_head.safetensors"
    records = clean_records(arguments.split, arguments.records)
    bodies = [api_request(record) for record in records] + long_cases()
    kev_questions = sum(len(record["questions"]) for record in records)

    # Every request twice in a row: the second is the repeated cold request.
    pairs = work / "packed-pairs.jsonl"
    write_jsonl(pairs, [body for body in bodies for _ in range(2)])
    separate = work / "separate.jsonl"
    write_jsonl(separate, [{"state": body["state"], "questions": {qid: question}}
                           for body in bodies for qid, question in body["questions"].items()])

    dumped = work / "prepared-pairs.jsonl"
    served_pairs = run_engine(arguments, pairs, work / "served-pairs.jsonl", dump=dumped)
    served, repeated = served_pairs[0::2], served_pairs[1::2]
    alone = run_engine(arguments, separate, work / "separate-served.jsonl")
    prepared = work / "prepared.jsonl"
    write_jsonl(prepared, read_jsonl(dumped)[0::2])
    reference = run_reference(arguments, prepared, work / "reference.jsonl")

    def flat(lines: list[dict], field: str) -> list[list[float]]:
        return [row for line in lines for row in line[field]]

    engine = flat(served, "probabilities")
    oracle = flat(reference, "questions")
    report = {
        "configuration": {
            "cli": str(arguments.cli), "weights": str(arguments.weights),
            "lora_dir": str(arguments.lora_dir), "adapter": arguments.adapter,
            "peft": str(arguments.peft), "decision_head": str(arguments.decision_head),
            "split": str(arguments.split), "records": len(records),
            "kev_questions": kev_questions, "long_cases": len(bodies) - len(records),
            "kv_dtype": arguments.kv_dtype, "prefill_chunk": arguments.prefill_chunk,
            "max_context": arguments.max_context,
            "reference_kv_dtype": arguments.reference_kv_dtype,
            "temperature": reference[0]["temperature"]},
        "served_vs_reference": {
            "kev_sample": agreement(engine[:kev_questions], oracle[:kev_questions]),
            "all": agreement(engine, oracle)},
        "packed_vs_separate": agreement(engine, flat(alone, "probabilities")),
        "cold_repeat": {"questions": len(engine),
                        "passed": engine == flat(repeated, "probabilities")},
        "engine": engine_summary(served),
    }
    (work / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"{'gate':<34}{'questions':>10}{'max |dp|':>10}{'mean |dp|':>11}{'flips':>7}  result")
    rows = [("served vs reference (kev sample)", report["served_vs_reference"]["kev_sample"]),
            ("served vs reference (all)", report["served_vs_reference"]["all"]),
            ("packed vs separate", report["packed_vs_separate"])]
    for name, result in rows:
        print(f"{name:<34}{result['questions']:>10}{result['max_dp']:>10.4f}"
              f"{result['mean_dp']:>11.5f}{result['argmax_flips']:>7}  "
              f"{'pass' if result['passed'] else 'FAIL'}")
    print(f"{'cold repeat bit-identical':<34}{len(engine):>10}{'':>28}  "
          f"{'pass' if report['cold_repeat']['passed'] else 'FAIL'}")
    print(f"report: {work / 'report.json'}")
    if not all(result["passed"] for _, result in rows) or not report["cold_repeat"]["passed"]:
        raise SystemExit(1)


def openai_probabilities(request: dict, response: dict) -> list[list[float]]:
    """Read native typed answers in request order, rejecting a mismatched capture."""
    if response.get("model") != request["model"]:
        raise ValueError("response model does not match the request's adapter")
    answers = response["answers"]
    if len(answers) != len(request["questions"]):
        raise ValueError("response question count mismatch")
    rows = []
    for question, answer in zip(request["questions"], answers, strict=True):
        kind = question["type"]
        if answer["type"] != kind or answer["name"] != question.get("name"):
            raise ValueError("response question order/type/name mismatch")
        if kind == "predicate":
            probability = float(answer["probability"])
            row = [1.0 - probability, probability]
        else:
            distribution = answer["probabilities"]
            expected = ([entry["value"] for entry in question["choices"]]
                        if kind == "choice" else list(range(len(question["levels"]))))
            actual = [entry["value"] for entry in distribution]
            if (actual != expected or any(type(a) is not type(b)
                                          for a, b in zip(actual, expected))):
                raise ValueError("response option order/value/type mismatch")
            if kind == "score" and [entry["label"] for entry in distribution] != [
                entry["label"] for entry in question["levels"]
            ]:
                raise ValueError("response level labels mismatch")
            row = [float(entry["probability"]) for entry in distribution]
        if (not row or any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in row)
                or abs(sum(row) - 1.0) > 1e-5):
            raise ValueError("response probabilities are not a finite normalized distribution")
        rows.append(row)
    usage = response["usage"]
    if (usage["output_tokens"] != 0 or usage["output_tokens_details"]["reasoning_tokens"] != 0
            or usage["total_tokens"] != usage["input_tokens"]):
        raise ValueError("decision capture must be prefill-only")
    if usage["input_tokens_details"]["cached_tokens"] != 0:
        raise ValueError("image qualification needs cold requests; disable prefix/continuation reuse")
    return rows


def image_capture(arguments: argparse.Namespace) -> None:
    """Capture native replies without importing the numerical stack or occupying another GPU."""
    import os
    from urllib.request import Request, urlopen

    requests = read_jsonl(arguments.requests)
    if not requests:
        raise SystemExit("requests JSONL is empty")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    responses = []
    for body in requests:
        request = Request(arguments.base_url.rstrip("/") + "/decisions",
                          data=json.dumps(body).encode("utf-8"), headers=headers)
        with urlopen(request, timeout=arguments.timeout) as reply:
            response = json.load(reply)
        openai_probabilities(body, response)
        responses.append(response)
    write_jsonl(arguments.out, responses)
    print(f"captured {len(responses)} cold requests to {arguments.out}; "
          "stop the server before image-gates")


def image_gates(arguments: argparse.Namespace) -> None:
    """BF16 first: independently prepare the original native requests, then compare responses."""
    work = arguments.work_dir
    work.mkdir(parents=True, exist_ok=True)
    requests = read_jsonl(arguments.requests)
    responses = read_jsonl(arguments.responses)
    if not requests or len(requests) != len(responses):
        raise SystemExit("requests and responses must contain the same nonzero number of lines")
    engine = [row for request, response in zip(requests, responses, strict=True)
              for row in openai_probabilities(request, response)]
    reference = arguments.reference
    head = arguments.decision_head or arguments.peft / "decision_head.safetensors"
    if reference is None:
        reference = work / "image-reference.jsonl"
        command = [sys.executable, "-m", "tools.reference.qwen3_8_27b.decision",
                   "--weights", str(arguments.weights.resolve()),
                   "--lora", str(arguments.peft.resolve()),
                   "--decision-head", str(head.resolve()),
                   "--openai-requests", str(arguments.requests.resolve()),
                   "--out", str(reference.resolve()), "--kv-dtype", "bf16",
                   "--prefill-chunk", str(arguments.reference_prefill_chunk)]
        log = work / "image-reference.log"
        with log.open("wb") as output:
            status = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                    check=False).returncode
        if status:
            raise SystemExit(f"image reference failed ({status}); see {log}")
    oracle = read_jsonl(reference)
    if len(oracle) != len(requests):
        raise SystemExit("reference request count mismatch")
    for request, response, result in zip(requests, responses, oracle, strict=True):
        if (result.get("format") != "ninfer-decision-probabilities"
                or result.get("format_version") != 1
                or len(result["questions"]) != len(request["questions"])):
            raise SystemExit("reference format/question count mismatch")
        if result.get("kv_dtype") != "bf16":
            raise SystemExit("image qualification requires a BF16-KV reference")
        if any(not row or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row)
               or abs(sum(row) - 1) > 1e-5 for row in result["questions"]):
            raise SystemExit("reference probabilities are not finite normalized distributions")
        if response["usage"]["input_tokens"] != result["input_tokens"]:
            raise SystemExit("Engine token usage differs from independent native preparation")
    if not any(result["image_grid_thw"] for result in oracle):
        raise SystemExit("image qualification needs at least one image-bearing state")
    comparison = agreement(engine, [row for result in oracle for row in result["questions"]])
    report = {"configuration": {
        "weights": str(arguments.weights), "peft": str(arguments.peft), "head": str(head),
        "requests": str(arguments.requests), "responses": str(arguments.responses),
        "reference": str(reference), "engine_kv_dtype_declared": "bf16",
        "reference_kv_dtype": "bf16",
        "reference_prefill_chunks": sorted({result["prefill_chunk"] for result in oracle}),
        "requests_count": len(requests)}, "served_vs_reference": comparison,
        "input_tokens_exact": True,
        "image_grids": [result["image_grid_thw"] for result in oracle],
        "rope_deltas": [result["rope_delta"] for result in oracle]}
    (work / "image-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=1))
    if not comparison["passed"]:
        raise SystemExit(1)


def fixtures(arguments: argparse.Namespace) -> None:
    kev_dir = arguments.kev.resolve()
    sys.path.insert(0, str(kev_dir))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(arguments.tokenizer))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    golden = arguments.output_dir / "kev_golden.json"
    write_fixture(golden, golden_fixture(kev_dir, arguments.tokenizer, tokenizer))
    print(f"wrote {golden}")
    layout = arguments.output_dir / "kev_layout.json"
    write_fixture(layout, layout_fixture(kev_dir, arguments.tokenizer, tokenizer))
    print(f"wrote {layout}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("fixtures", help="regenerate the kev golden fixtures")
    generate.add_argument("--kev", type=Path, default=ROOT / "external_repos" / "kev",
                          help="kev checkout")
    generate.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "Qwen3.8-27B",
                          help="Hugging Face tokenizer directory of the served checkpoint")
    generate.add_argument("--output-dir", type=Path,
                          default=ROOT / "tests" / "fixtures" / "systemone")
    generate.set_defaults(run=fixtures)

    gate = commands.add_parser(
        "gates", help="served vs reference, packed vs separate and cold repeat through the CLI")
    gate.add_argument("--cli", type=Path, default=ROOT / "build-sm89" / "apps" / "ninfer")
    gate.add_argument("--weights", type=Path, required=True, help="Qwen3.8-27B .ninfer artifact")
    gate.add_argument("--lora-dir", type=Path, required=True,
                      help="adapter directory holding the decision .lora.ninfer")
    gate.add_argument("--adapter", help="decision adapter the default model binds (--systemone-"
                                        "default); needed when the directory holds several")
    gate.add_argument("--peft", type=Path, required=True,
                      help="the same adapter's PEFT directory, for the reference")
    gate.add_argument("--decision-head", type=Path,
                      help="the trainer's head (default: <peft>/decision_head.safetensors)")
    gate.add_argument("--split", type=Path, required=True,
                      help="labelled kev split, e.g. external_repos/kev/evals/v7/decision-v7/"
                           "development.jsonl")
    gate.add_argument("--records", type=int, default=KEV_RECORDS)
    gate.add_argument("--work-dir", type=Path, required=True)
    gate.add_argument("--kv-dtype", default="rk4v4", help="engine KV storage")
    gate.add_argument("--prefill-chunk", type=int, default=1024, help="engine pass width")
    gate.add_argument("--max-context", type=int, default=8192)
    gate.add_argument("--reference-kv-dtype", choices=("bf16", "int8"), default="bf16")
    gate.set_defaults(run=gates)

    capture = commands.add_parser("image-capture", help="capture cold native OpenAI image decisions")
    capture.add_argument("--requests", type=Path, required=True, help="native requests JSONL; "
                         "use data-URL images, actual adapter pool names, a BF16-KV cold server")
    capture.add_argument("--base-url", required=True, help="server API root ending in /v1")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--timeout", type=float, default=600)
    capture.set_defaults(run=image_capture)

    image = commands.add_parser("image-gates", help="independent BF16 image-decision parity; "
                                "stop the server first to release the GPU")
    image.add_argument("--requests", type=Path, required=True)
    image.add_argument("--responses", type=Path, required=True, help="image-capture JSONL")
    image.add_argument("--weights", type=Path, required=True)
    image.add_argument("--peft", type=Path, required=True)
    image.add_argument("--decision-head", type=Path)
    image.add_argument("--reference", type=Path, help="already computed native reference JSONL")
    image.add_argument("--reference-prefill-chunk", type=int, default=1024)
    image.add_argument("--work-dir", type=Path, required=True)
    image.set_defaults(run=image_gates)

    arguments = parser.parse_args()
    arguments.run(arguments)


if __name__ == "__main__":
    main()
