import random

import pytest

from swarm_scheduler_lab.instances import Instance, Operation, load_ft06, make_datasets
from swarm_scheduler_lab.policy import Policy, initial_policies, random_policy
from swarm_scheduler_lab.scheduling import InvalidSchedule, dispatch, validate
from swarm_scheduler_lab.solver import solve_reference


@pytest.fixture
def tiny():
    return Instance("tiny", ((Operation(0, 2), Operation(1, 1)), (Operation(1, 2), Operation(0, 1))))


def test_valid_schedule_and_recomputed_makespan(tiny):
    schedule = dispatch(tiny, Policy())
    result = validate(tiny, schedule)
    assert result == {"valid": True, "makespan": max(row["end"] for row in schedule)}
    # SPT can choose a short future-ready operation before a longer immediate
    # one; a valid serial schedule need not be an optimal schedule.
    assert result["makespan"] == 6
    optimal = [
        {"job": 0, "operation": 0, "machine": 0, "start": 0, "end": 2, "duration": 2},
        {"job": 0, "operation": 1, "machine": 1, "start": 2, "end": 3, "duration": 1},
        {"job": 1, "operation": 0, "machine": 1, "start": 0, "end": 2, "duration": 2},
        {"job": 1, "operation": 1, "machine": 0, "start": 2, "end": 3, "duration": 1},
    ]
    assert validate(tiny, optimal)["makespan"] == 3


@pytest.mark.parametrize("violation", ["missing", "duplicate", "duration", "machine", "precedence", "overlap", "negative", "noninteger", "unknown"])
def test_rejects_typical_violations(tiny, violation):
    schedule = dispatch(tiny, Policy())
    if violation == "missing":
        schedule.pop()
    elif violation == "duplicate":
        schedule.append(schedule[0].copy())
    elif violation == "duration":
        schedule[0]["duration"] += 1
    elif violation == "machine":
        schedule[0]["machine"] = 1
    elif violation == "precedence":
        schedule[1]["start"] = 0
        schedule[1]["end"] = 1
    elif violation == "overlap":
        schedule[3]["start"] = 1
        schedule[3]["end"] = 2
        # Keep precedence valid, so this specifically exercises capacity.
        schedule[2]["start"] = 0
        schedule[2]["end"] = 1
        schedule[2]["duration"] = 1
        tiny = Instance("tiny", ((Operation(0, 2), Operation(1, 1)), (Operation(1, 1), Operation(0, 1))))
    elif violation == "negative":
        schedule[0]["start"] = -1
        schedule[0]["end"] = 1
    elif violation == "noninteger":
        schedule[0]["start"] = 0.0
    elif violation == "unknown":
        schedule[0]["job"] = 10
    with pytest.raises(InvalidSchedule):
        validate(tiny, schedule)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 10.01, -10.01, True, "1"])
def test_policy_rejects_invalid_numeric_values(bad):
    payload = Policy().to_dict()
    payload["weights"]["processing_time"] = bad
    with pytest.raises(ValueError):
        Policy.from_dict(payload)


def test_policy_schema_unknown_missing_and_code_fields():
    for payload in [
        {},
        {"weights": {}},
        {**Policy().to_dict(), "python": "print('bad')"},
        {**Policy().to_dict(), "tie_break": "arbitrary_python"},
    ]:
        with pytest.raises(ValueError):
            Policy.from_dict(payload)
    defaults = Policy.from_dict({"weights": Policy().weights})
    assert defaults.tie_break == "job_id"


def test_policy_and_dispatch_are_deterministic(tiny):
    assert initial_policies(42) == initial_policies(42)
    assert random_policy(random.Random(7)) == random_policy(random.Random(7))
    for policy in initial_policies(42):
        assert dispatch(tiny, policy) == dispatch(tiny, Policy.from_dict(policy.to_dict()))
        assert validate(tiny, dispatch(tiny, policy))["valid"]


def test_dataset_split_streams_and_serialization():
    first = make_datasets(42, 2, 2)
    assert first == make_datasets(42, 2, 2)
    assert first["test"] == make_datasets(42, 3, 2)["test"]
    assert {i.jobs for i in first["train"]}.isdisjoint(i.jobs for i in first["test"])
    assert Instance.from_dict(first["train"][0].to_dict()) == first["train"][0]
    assert "55" not in str(first["benchmark"][0].to_dict())


def test_ft06_reference_proves_known_optimum():
    instance = load_ft06()
    assert instance.num_jobs == instance.num_machines == 6
    result = solve_reference(instance, 5.0, 0)
    assert result["status"] == "OPTIMAL"
    assert result["objective"] == result["best_bound"] == 55
    assert result["proven_optimal"]
    assert validate(instance, result["schedule"])["makespan"] == 55


def test_solver_timeout_does_not_claim_optimality():
    instance = make_datasets(9, 1, 1)["train"][0]
    result = solve_reference(instance, 1e-9, 0)
    assert result["status"] == "UNKNOWN"
    assert result["objective"] is None
    assert not result["proven_optimal"]
    assert result["best_bound"] > 0
