"""The entire model-controlled surface: five bounded weights and a tie rule."""

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

    def __post_init__(self) -> None:
        data = {"weights": self.weights, "tie_break": self.tie_break}
        errors = sorted(_VALIDATOR.iter_errors(data), key=lambda error: str(error.path))
        if errors:
            raise ValueError("invalid policy: " + errors[0].message)
        # JSON Schema numeric bounds alone do not reject NaN in all libraries.
        if any(not math.isfinite(value) for value in self.weights.values()):
            raise ValueError("policy weights must be finite")
        object.__setattr__(self, "weights", {name: float(self.weights[name]) for name in FEATURES})

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "tie_break": self.tie_break}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        errors = sorted(_VALIDATOR.iter_errors(data), key=lambda error: str(error.path))
        if errors:
            raise ValueError("invalid policy: " + errors[0].message)
        return cls(data["weights"], data.get("tie_break", "job_id"))


def random_policy(rng: random.Random) -> Policy:
    return Policy(
        {name: round(rng.uniform(-10, 10), 6) for name in FEATURES},
        rng.choice(TIE_BREAKS),
    )


def initial_policies(seed: int) -> list[Policy]:
    """The identical three starting candidates for every search organization."""
    return [
        Policy(),
        Policy({name: -1.0 if name == "remaining_work" else 0.0 for name in FEATURES}),
        random_policy(random.Random(seed)),
    ]
