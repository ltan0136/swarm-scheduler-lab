"""The entire model-controlled surface: five bounded weights and a tie rule.

``candidate_filter`` records which ready-operation set the weights rank. It is
fixed per experiment (a search-space condition), not chosen by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

from jsonschema import Draft202012Validator


FEATURES = (
    "processing_time",
    "remaining_work",
    "remaining_operations",
    "earliest_start",
    "machine_load",
)
TIE_BREAKS = ("job_id", "shortest_processing_time", "most_work_remaining")
# "none": every job's next operation is ready, even one that cannot start yet.
# "pruned": JobShopLib's default dominated + non-immediate operation filters.
CANDIDATE_FILTERS = ("none", "pruned")
POLICY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["weights"],
    "properties": {
        "weights": {
            "type": "object",
            "additionalProperties": False,
            "required": list(FEATURES),
            "properties": {
                feature: {"type": "number", "minimum": -10, "maximum": 10}
                for feature in FEATURES
            },
        },
        "tie_break": {"type": "string", "enum": list(TIE_BREAKS), "default": "job_id"},
        "candidate_filter": {"type": "string", "enum": list(CANDIDATE_FILTERS), "default": "none"},
    },
}
_VALIDATOR = Draft202012Validator(POLICY_SCHEMA)


def _spt_weights() -> dict[str, float]:
    return {feature: 1.0 if feature == "processing_time" else 0.0 for feature in FEATURES}


@dataclass(frozen=True)
class Policy:
    """Lower weighted, max-normalized score has higher dispatch priority."""

    weights: dict[str, float] = field(default_factory=_spt_weights)
    tie_break: str = "job_id"
    candidate_filter: str = "none"

    def __post_init__(self) -> None:
        data = {"weights": self.weights, "tie_break": self.tie_break, "candidate_filter": self.candidate_filter}
        errors = sorted(_VALIDATOR.iter_errors(data), key=lambda error: str(error.path))
        if errors:
            raise ValueError("invalid policy: " + errors[0].message)
        # JSON Schema numeric bounds alone do not reject NaN in all libraries.
        if any(not math.isfinite(value) for value in self.weights.values()):
            raise ValueError("policy weights must be finite")
        object.__setattr__(self, "weights", {name: float(self.weights[name]) for name in FEATURES})

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "tie_break": self.tie_break,
                "candidate_filter": self.candidate_filter}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        errors = sorted(_VALIDATOR.iter_errors(data), key=lambda error: str(error.path))
        if errors:
            raise ValueError("invalid policy: " + errors[0].message)
        return cls(data["weights"], data.get("tie_break", "job_id"), data.get("candidate_filter", "none"))


def random_policy(rng: random.Random, candidate_filter: str = "none") -> Policy:
    # The filter is not drawn, so the random stream is identical for every filter.
    return Policy(
        {name: round(rng.uniform(-10, 10), 6) for name in FEATURES},
        rng.choice(TIE_BREAKS),
        candidate_filter,
    )


def initial_policies(seed: int, candidate_filter: str = "none") -> list[Policy]:
    """The identical three starting candidates for every search organization."""
    return [
        Policy(candidate_filter=candidate_filter),
        Policy({name: -1.0 if name == "remaining_work" else 0.0 for name in FEATURES},
               candidate_filter=candidate_filter),
        random_policy(random.Random(seed), candidate_filter),
    ]
