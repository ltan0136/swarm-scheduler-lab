"""Independent time-bounded CP-SAT references, never training feedback."""

from __future__ import annotations

import math
from typing import Any

from ortools.sat.python import cp_model

from .instances import Instance
from .scheduling import validate


def solve_reference(instance: Instance, time_limit: float, seed: int) -> dict[str, Any]:
    """Solve the standard job-shop formulation with OR-Tools CP-SAT.

    UNKNOWN reports a valid structural lower bound and no incumbent. FEASIBLE
    has an incumbent and a valid bound, but cannot assert proven optimality.
    """
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("solver time limit must be positive and finite")
    horizon = sum(op.duration for job in instance.jobs for op in job)
    structural_bound = max(
        max(sum(op.duration for op in job) for job in instance.jobs),
        max(
            sum(op.duration for job in instance.jobs for op in job if op.machine == machine)
            for machine in range(instance.num_machines)
        ),
    )
    model = cp_model.CpModel()
    starts, ends = {}, {}
    intervals: dict[int, list[Any]] = {}
    for job_id, job in enumerate(instance.jobs):
        for index, operation in enumerate(job):
            key = job_id, index
            starts[key] = model.new_int_var(0, horizon, f"start_{job_id}_{index}")
            ends[key] = model.new_int_var(0, horizon, f"end_{job_id}_{index}")
            interval = model.new_interval_var(
                starts[key], operation.duration, ends[key], f"op_{job_id}_{index}"
            )
            intervals.setdefault(operation.machine, []).append(interval)
            if index:
                model.add(starts[key] >= ends[(job_id, index - 1)])
    for machine_intervals in intervals.values():
        model.add_no_overlap(machine_intervals)
    makespan = model.new_int_var(structural_bound, horizon, "makespan")
    model.add_max_equality(makespan, [ends[(j, len(job) - 1)] for j, job in enumerate(instance.jobs)])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed % (2**31 - 1)
    status = solver.solve(model)
    result: dict[str, Any] = {
        "status": solver.status_name(status),
        "objective": None,
        "best_bound": structural_bound,
        "proven_optimal": status == cp_model.OPTIMAL,
        "wall_time": solver.wall_time,
        "schedule": None,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = [
            {
                "job": job_id,
                "operation": index,
                "machine": operation.machine,
                "start": solver.value(starts[(job_id, index)]),
                "end": solver.value(ends[(job_id, index)]),
                "duration": operation.duration,
            }
            for job_id, job in enumerate(instance.jobs)
            for index, operation in enumerate(job)
        ]
        result.update(
            objective=validate(instance, schedule)["makespan"],
            best_bound=max(structural_bound, solver.best_objective_bound),
            schedule=schedule,
        )
    elif status == cp_model.UNKNOWN:
        result["best_bound"] = max(structural_bound, solver.best_objective_bound)
    return result
