"""Parse LLM replies into typed results.

Separate from graph.py so the parsing logic is unit-testable without importing the LLM client,
and so graph.py stays focused on node wiring.
"""
import json
import re
from typing import NamedTuple


class VerifyResult(NamedTuple):
    ok: bool
    issue: str


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_verify_result(text: str) -> VerifyResult:
    """Defensively parse the verifier's reply into (ok, issue).

    The model may wrap the JSON in prose, markdown fences, or <think> blocks. Strip think-blocks,
    then try every fenced block and every bare {...}, keeping the last one that is a JSON object
    containing "ok". On total failure default to ok=False so an unsure verifier triggers a revise
    (bounded by MAX_ITERATIONS). This is the LLM-output boundary, so loose parsing / Any is fine.
    """
    cleaned = _THINK_RE.sub("", text)
    candidates: list[str] = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    candidates += _JSON_OBJ_RE.findall(cleaned)
    result = VerifyResult(ok=False, issue="could not parse verifier reply")
    for chunk in candidates:
        # EAFP: try to read "ok"/"issue"; a non-object (list/str/number) or a
        # missing "ok" key just means this chunk isn't our verdict -> skip it.
        try:
            obj = json.loads(chunk)
            result = VerifyResult(ok=bool(obj["ok"]), issue=str(obj.get("issue", "")))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return result