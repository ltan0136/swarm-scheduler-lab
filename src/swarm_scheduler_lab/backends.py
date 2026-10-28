"""Candidate generators. Model output is data, never executable code."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .policy import Policy, random_policy


WEIGHT_NAMES = (
    "processing_time", "remaining_work", "remaining_operations",
    "earliest_start", "machine_load",
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # Authorization must never follow a provider's redirect to another host.
        return None


def urlopen(request: Request, *, timeout: float):
    """Preserve proxy/TLS defaults while rejecting every HTTP redirect."""
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


@dataclass(frozen=True)
class Generation:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    token_source: str


def make_prompt(history: list[dict], direction: str) -> str:
    """Only callers' training evaluations enter model context.

    Keep a compact best-plus-recent view so the input reservation is bounded.
    A receiver may include explicitly migrated training feedback in this list.
    """
    best = min(history, key=lambda row: row["train_mean"])
    selected = [best]
    for row in history[-2:]:
        if row is not best:
            selected.append(row)
    feedback = [
        {"weights": [row["policy"]["weights"][k] for k in WEIGHT_NAMES],
         "tie": row["policy"]["tie_break"],
         "training_mean_makespan": row["train_mean"]}
        for row in selected
    ]
    return (
        "Search a restricted dispatching rule for job-shop scheduling. "
        "Minimize training mean makespan. Propose ONE JSON object only: "
        '{"weights":{"processing_time":0,"remaining_work":0,'
        '"remaining_operations":0,"earliest_start":0,"machine_load":0},'
        '"tie_break":"job_id"}. '
        "Each finite weight must be in [-10,10]. Lower weighted sum has "
        "priority. Each feature is divided by its maximum over currently ready "
        "operations (zero if that maximum is zero). "
        "Ties allowed: job_id, shortest_processing_time, most_work_remaining. "
        f"Search direction: {direction}. "
        f"Feature order: {','.join(WEIGHT_NAMES)}. "
        "Only training feedback follows; no evaluation-set feedback is available: "
        + json.dumps(feedback, separators=(",", ":"), allow_nan=False)
    )


class MockBackend:
    """Offline seeded search surrogate; its output is not LLM evidence."""

    token_source = "estimated_mock"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def generate(self, prompt: str, history: list[dict], max_output_tokens: int) -> Generation:
        # The seed, own/migrated history, and ordered calls determine every result.
        parent = min(history, key=lambda row: row["train_mean"])["policy"]
        if self.rng.random() < 0.25:
            candidate = random_policy(self.rng).to_dict()
        else:
            weights = {
                key: round(max(-10.0, min(10.0, value + self.rng.gauss(0, 1.2))), 6)
                for key, value in parent["weights"].items()
            }
            candidate = {"weights": weights, "tie_break": parent["tie_break"]}
        content = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        # Truncation is a failed proposal, exactly as a real output limit can be.
        estimated_output = math.ceil(len(content.encode("utf-8")) / 4)
        if estimated_output > max_output_tokens:
            content = content[: max_output_tokens * 4]
        return Generation(
            content, math.ceil(len(prompt.encode("utf-8")) / 4),
            min(estimated_output, max_output_tokens), self.token_source,
        )


class OpenAICompatibleBackend:
    """A single HTTP attempt per generate(), without SDK hidden retries."""

    token_source = "provider_reported"

    def __init__(
        self, *, model: str | None = None, base_url: str | None = None,
        api_key_env: str = "SWARM_API_KEY", request_timeout: float = 60,
    ):
        self.model = model or os.environ.get("SWARM_MODEL")
        self.base_url = (base_url or os.environ.get("SWARM_BASE_URL") or
                         "https://api.openai.com/v1").rstrip("/")
        if not self.model:
            raise ValueError("Set --model or SWARM_MODEL for the real backend")
        if not api_key_env or not os.environ.get(api_key_env):
            raise ValueError(f"Required credential variable is unset: {api_key_env}")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
        ):
            raise ValueError("Model base URL must use HTTPS (or local HTTP for development)")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Model base URL needs a hostname and no credentials, query or fragment")
        if request_timeout <= 0 or not math.isfinite(request_timeout):
            raise ValueError("request_timeout must be positive and finite")
        self._api_key = os.environ[api_key_env]
        self.request_timeout = request_timeout

    def generate(self, prompt: str, history: list[dict], max_output_tokens: int) -> Generation:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                # Bound untrusted response size; never include response body in errors.
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("Model response exceeds the 2 MB limit")
            data = json.loads(raw)
        except HTTPError as error:
            raise RuntimeError(f"Model HTTP request failed with status {error.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Model request failed (network or timeout)") from None
        except (ValueError, UnicodeDecodeError):
            raise RuntimeError("Model returned an invalid JSON response") from None
        usage = data.get("usage") or {}

        def count(key: str) -> int | None:
            value = usage.get(key)
            return value if type(value) is int and value >= 0 else None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # Return a deliberately invalid candidate to preserve any real usage
            # returned with a malformed envelope. Search rejects it and charges
            # the failed proposal, without inventing a successful response.
            content = ""
        if not isinstance(content, str):
            content = ""
        return Generation(content, count("prompt_tokens"), count("completion_tokens"), self.token_source)


def parse_candidate(content: str) -> Policy:
    """Reject markdown, nonstandard JSON numbers, and schema violations."""
    def reject_constant(value: str):
        raise ValueError(f"Non-finite JSON numeric constant: {value}")

    data = json.loads(content, parse_constant=reject_constant)
    return Policy.from_dict(data)
