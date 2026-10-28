"""Trusted dispatch code, followed by an independent constraint validator."""

from __future__ import annotations

from typing import Any

from job_shop_lib.dispatching import Dispatcher
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

from .instances import Instance
from .policy import FEATURES, Policy


class InvalidSchedule(ValueError):
    """A proposed schedule failed independent validation."""


def dispatch(instance: Instance, policy: Policy) -> list[dict[str, int]]:
    """Run a deterministic serial schedule-generation scheme via JobShopLib.

    Eligible operations are the next unscheduled operation of each job. Their
    start is the maximum of job and machine availability. At every decision,
    each feature is divided by its maximum among eligible operations (or zero
    if its maximum is zero). Machine load includes all unscheduled operations
    assigned to that machine, including the current operation. Remaining work
    and remaining operation counts also include the current operation.
    """
    # Validate even policies constructed directly rather than through JSON.
    policy = Policy.from_dict(policy.to_dict())
    library_instance = instance.to_job_shop_lib()

    def rule(dispatcher: Dispatcher):
        eligible = dispatcher.available_operations()
        machine_load = [0] * instance.num_machines
        remaining_work = []
        remaining_operations = []
        for job_id, job in enumerate(instance.jobs):
            remaining = job[dispatcher.job_next_operation_index[job_id] :]
            remaining_work.append(sum(op.duration for op in remaining))
            remaining_operations.append(len(remaining))
            for op in remaining:
                machine_load[op.machine] += op.duration
        features = [
            {
                "processing_time": op.duration,
                "remaining_work": remaining_work[op.job_id],
                "remaining_operations": remaining_operations[op.job_id],
                "earliest_start": dispatcher.earliest_start_time(op),
                "machine_load": machine_load[op.machines[0]],
            }
            for op in eligible
        ]
        maxima = {name: max(row[name] for row in features) for name in FEATURES}

        def rank(index: int):
            op = eligible[index]
            row = features[index]
            score = sum(
                policy.weights[name] * (row[name] / maxima[name] if maxima[name] else 0.0)
                for name in FEATURES
            )
            tie = 0
            if policy.tie_break == "shortest_processing_time":
                tie = op.duration
            elif policy.tie_break == "most_work_remaining":
                tie = -remaining_work[op.job_id]
            return score, tie, op.job_id, op.position_in_job

        return eligible[min(range(len(eligible)), key=rank)]

    # No library pruning: earliest_start remains a searchable preference.
    schedule = DispatchingRuleSolver(rule, ready_operations_filter=None)(library_instance)
    records = [
        {
            "job": scheduled.job_id,
            "operation": scheduled.position_in_job,
            "machine": scheduled.machine_id,
            "start": scheduled.start_time,
            "end": scheduled.end_time,
            "duration": scheduled.operation.duration,
        }
        for machine in schedule.schedule
        for scheduled in machine
    ]
    return sorted(records, key=lambda row: (row["job"], row["operation"]))


def validate(instance: Instance, schedule: list[dict[str, Any]]) -> dict[str, Any]:
    """Recheck input durations, completeness, precedence and machine capacity.

    This function does not use JobShopLib, policy scores, solver metadata or a
    caller-provided makespan. Invalid solutions are rejected before scoring.
    Intervals are half-open: an operation may start when another ends.
    """
    if not isinstance(schedule, list):
        raise InvalidSchedule("schedule must be an array")
    expected = {(j, k) for j, job in enumerate(instance.jobs) for k in range(len(job))}
    rows: dict[tuple[int, int], dict[str, int]] = {}
    by_machine: dict[int, list[dict[str, int]]] = {}
    required = {"job", "operation", "machine", "start", "end", "duration"}
    for row in schedule:
        if not isinstance(row, dict) or set(row) != required:
            raise InvalidSchedule("operation record must have exactly the required fields")
        if any(type(row[key]) is not int for key in required):
            raise InvalidSchedule("schedule fields must be integers")
        key = (row["job"], row["operation"])
        if key not in expected:
            raise InvalidSchedule(f"unknown operation {key}")
        if key in rows:
            raise InvalidSchedule(f"duplicate operation {key}")
        operation = instance.jobs[key[0]][key[1]]
        if row["machine"] != operation.machine:
            raise InvalidSchedule(f"wrong machine for {key}")
        if row["duration"] != operation.duration or row["end"] - row["start"] != operation.duration:
            raise InvalidSchedule(f"wrong duration for {key}")
        if row["start"] < 0:
            raise InvalidSchedule(f"negative start for {key}")
        rows[key] = row
        by_machine.setdefault(row["machine"], []).append(row)
    if set(rows) != expected:
        raise InvalidSchedule("schedule is incomplete")
    for job_id, job in enumerate(instance.jobs):
        for index in range(1, len(job)):
            if rows[(job_id, index)]["start"] < rows[(job_id, index - 1)]["end"]:
                raise InvalidSchedule(f"job precedence violation for {(job_id, index)}")
    for machine, intervals in by_machine.items():
        intervals.sort(key=lambda row: (row["start"], row["end"]))
        if any(right["start"] < left["end"] for left, right in zip(intervals, intervals[1:])):
            raise InvalidSchedule(f"machine overlap on {machine}")
    return {"valid": True, "makespan": max(row["end"] for row in rows.values())}
