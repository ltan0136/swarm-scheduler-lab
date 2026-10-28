"""Immutable input data and disjoint deterministic instance generation.

JobShopLib supplies the public FT06 benchmark; its optimum metadata is
deliberately not part of our instance representation or model-facing inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

from job_shop_lib import JobShopInstance
from job_shop_lib.benchmarking import load_benchmark_instance


@dataclass(frozen=True)
class Operation:
    machine: int
    duration: int

    def __post_init__(self) -> None:
        if type(self.machine) is not int or self.machine < 0:
            raise ValueError("machine must be a nonnegative integer")
        if type(self.duration) is not int or self.duration <= 0:
            raise ValueError("duration must be a positive integer")


@dataclass(frozen=True)
class Instance:
    name: str
    jobs: tuple[tuple[Operation, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("instance name must be nonempty")
        if not self.jobs or any(not job for job in self.jobs):
            raise ValueError("instances must contain nonempty jobs")
        if any(not isinstance(op, Operation) for job in self.jobs for op in job):
            raise ValueError("jobs must contain Operation objects")
        object.__setattr__(self, "jobs", tuple(tuple(job) for job in self.jobs))

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_machines(self) -> int:
        return 1 + max(op.machine for job in self.jobs for op in job)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "jobs": [
                [{"machine": op.machine, "duration": op.duration} for op in job]
                for job in self.jobs
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instance:
        if not isinstance(data, dict) or set(data) != {"name", "jobs"}:
            raise ValueError("instance requires exactly name and jobs")
        if not isinstance(data["jobs"], list):
            raise ValueError("jobs must be an array")
        jobs = []
        for job in data["jobs"]:
            if not isinstance(job, list):
                raise ValueError("each job must be an array")
            operations = []
            for op in job:
                if not isinstance(op, dict) or set(op) != {"machine", "duration"}:
                    raise ValueError("operation requires exactly machine and duration")
                operations.append(Operation(op["machine"], op["duration"]))
            jobs.append(tuple(operations))
        return cls(data["name"], tuple(jobs))

    def to_job_shop_lib(self) -> JobShopInstance:
        return JobShopInstance.from_matrices(
            [[op.duration for op in job] for job in self.jobs],
            [[op.machine for op in job] for job in self.jobs],
            name=self.name,
        )


def load_ft06() -> Instance:
    """Load the packaged benchmark data, omitting all solution metadata."""
    instance = load_benchmark_instance("ft06")
    return Instance(
        "ft06",
        tuple(
            tuple(Operation(op.machines[0], op.duration) for op in job)
            for job in instance.jobs
        ),
    )


def make_datasets(
    seed: int,
    train_count: int = 8,
    test_count: int = 8,
    jobs: int = 6,
    machines: int = 6,
    duration_max: int = 20,
) -> dict[str, list[Instance]]:
    """Generate train/test using independent streams, not sequential slices.

    Each job visits each machine once in a shuffled order. Changing the number
    of training instances cannot change the test instances. Stream separation
    and collision rejection also prevent identical train/test problems.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for name, value in {
        "train_count": train_count,
        "test_count": test_count,
        "jobs": jobs,
        "machines": machines,
        "duration_max": duration_max,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    result: dict[str, list[Instance]] = {"train": [], "test": [], "benchmark": [load_ft06()]}
    for split, count in (("train", train_count), ("test", test_count)):
        digest = hashlib.sha256(f"swarm-scheduler-lab:{seed}:{split}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        for index in range(count):
            generated = []
            for _ in range(jobs):
                route = list(range(machines))
                rng.shuffle(route)
                generated.append(tuple(Operation(m, rng.randint(1, duration_max)) for m in route))
            result[split].append(Instance(f"{split}_{seed}_{index:03d}", tuple(generated)))
    # Very small configurable spaces can contain only one problem. Rather
    # than silently overlap or loop forever, fail with an actionable error.
    train_problems = {instance.jobs for instance in result["train"]}
    if any(instance.jobs in train_problems for instance in result["test"]):
        raise ValueError("generated splits overlap; increase instance size or duration_max, or use another seed")
    return result
