"""Budgeted training-only rule search and the three organization modes."""

from __future__ import annotations

import copy
import multiprocessing
import random
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

from .backends import MockBackend, OpenAICompatibleBackend, make_prompt, parse_candidate
from .budget import BudgetExceeded, BudgetLedger
from .instances import Instance
from .policy import CANDIDATE_FILTERS, Policy, initial_policies, random_policy
from .scheduling import dispatch, validate


# pruned_* baselines always use JobShopLib's default candidate pruning, so a
# textbook rule stays visible whatever search space the experiment configures.
METHODS = (
    "single", "independent", "cooperative", "shortest_processing_time",
    "most_work_remaining", "pruned_shortest_processing_time",
    "pruned_most_work_remaining", "random_search",
)
AGENT_METHODS = METHODS[:3]
DIRECTIONS = (
    "A: favor short current operations, then refine from your own feedback",
    "B: favor jobs with more remaining work, then refine from your own feedback",
    "C: explore random weights and mutations, guided by your own feedback",
)


def _evaluate_one(args: tuple[Instance, dict]) -> dict:
    instance, policy_data = args
    try:
        schedule = dispatch(instance, Policy.from_dict(policy_data))
        result = validate(instance, schedule)
        return {"instance": instance.name, "valid": True, "makespan": result["makespan"]}
    except (ValueError, RuntimeError, AssertionError) as error:
        # Invalid schedules never receive a numerical score or a leaderboard place.
        return {"instance": instance.name, "valid": False, "error": type(error).__name__}


def exchange_ring(islands: list[list[dict]], round_number: int, interval: int = 2) -> list[dict]:
    """Simultaneous training-winner migration; never replaces local history."""
    if len(islands) != 3:
        raise ValueError("The ring requires exactly three islands")
    if type(interval) is not int or interval < 1:
        raise ValueError("exchange interval must be a positive integer")
    if round_number <= 0 or round_number % interval:
        return []
    if any(not island for island in islands):
        return []
    winners = [copy.deepcopy(min(group, key=lambda row: row["train_mean"])) for group in islands]
    events = []
    for source, winner in enumerate(winners):
        destination = (source + 1) % 3
        imported = copy.deepcopy(winner)
        imported["imported_from"] = source
        imported["imported_at_round"] = round_number
        islands[destination].append(imported)
        events.append({
            "round": round_number, "source": source, "destination": destination,
            "policy": copy.deepcopy(winner["policy"]), "train_mean": winner["train_mean"],
            "source_evaluation": winner["evaluation"],
            "receiver_local_candidates_retained": sum("imported_from" not in row for row in islands[destination]),
        })
    return events


def run_search(
    method: str, train: list[Instance], *, seed: int, backend: str = "mock",
    max_calls: int = 12, max_tokens: int = 24000, max_output_tokens: int = 512,
    max_evaluations: int = 15, cpu_workers: int = 1, model: str | None = None,
    base_url: str | None = None, api_key_env: str = "SWARM_API_KEY",
    request_timeout: float = 60, max_retries: int = 0,
    candidate_filter: str = "none", exchange_interval: int = 2,
) -> dict:
    """Search using only training instances; holdout/reference data are absent.

    A policy evaluation is one candidate dispatched over *all* training instances.
    Instance-level scheduling attempts are recorded separately. The CPU pool has
    a fixed size; calls are deliberately ordered to keep mock runs reproducible.
    Every explicit subsequent attempt, including after failure, reserves a call.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown search method: {method}")
    if backend not in {"mock", "real"}:
        raise ValueError("backend must be mock or real")
    if not train or len({instance.name for instance in train}) != len(train):
        raise ValueError("Training instances must be nonempty and have unique names")
    if type(cpu_workers) is not int or cpu_workers < 1:
        raise ValueError("cpu_workers must be a positive integer")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if candidate_filter not in CANDIDATE_FILTERS:
        raise ValueError(f"candidate_filter must be one of {', '.join(CANDIDATE_FILTERS)}")
    if type(exchange_interval) is not int or exchange_interval < 1:
        raise ValueError("exchange_interval must be a positive integer")
    ledger = BudgetLedger(max_calls, max_tokens, max_evaluations)
    started = time.perf_counter()
    cpu_started = time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_started = children.ru_utime + children.ru_stime
    initial = initial_policies(seed, candidate_filter)
    pruned = initial_policies(seed, "pruned")
    static_baselines = {
        "shortest_processing_time": initial[0],
        "most_work_remaining": initial[1],
        "pruned_shortest_processing_time": pruned[0],
        "pruned_most_work_remaining": pruned[1],
    }
    history: list[dict] = []
    attempts: list[dict] = []
    islands: list[list[dict]] = [[] for _ in range(1 if method == "single" else 3)]
    exchanges: list[dict] = []
    best: dict | None = None
    valid_schedules = 0
    stop_reason = "completed"
    rng = random.Random(seed + 912_451)

    # A separate seeded generator per island preserves its private random stream.
    generators = []
    if method in AGENT_METHODS:
        for island_index in range(len(islands)):
            if backend == "mock":
                generators.append(MockBackend(seed * 1009 + island_index))
            else:
                generators.append(OpenAICompatibleBackend(
                    model=model, base_url=base_url, api_key_env=api_key_env,
                    request_timeout=request_timeout,
                ))

    def token_fields() -> tuple[int | None, int]:
        snapshot = ledger.snapshot()
        if snapshot["calls"] == 0:
            return 0, 0
        if snapshot["unknown_usage_calls"] or snapshot["pending_calls"]:
            total = None
        else:
            total = (snapshot["input_tokens"] or 0) + (snapshot["output_tokens"] or 0)
        return total, snapshot["reserved_or_charged_tokens"]

    pool_context = ProcessPoolExecutor(
        max_workers=cpu_workers, mp_context=multiprocessing.get_context("spawn")
    ) if cpu_workers > 1 else nullcontext(None)
    with pool_context as pool:
        def evaluate(policy: Policy, island_index: int | None, source: str, round_number: int = 0) -> dict:
            nonlocal best, valid_schedules
            evaluation = ledger.reserve_evaluation()
            policy_data = policy.to_dict()
            tasks = [(instance, policy_data) for instance in train]
            ledger.record_instance_schedules(len(tasks))
            results = list(pool.map(_evaluate_one, tasks)) if pool else [_evaluate_one(task) for task in tasks]
            valid_count = sum(result["valid"] for result in results)
            valid_schedules += valid_count
            valid = valid_count == len(train)
            mean = sum(row["makespan"] for row in results) / len(results) if valid else None
            total_tokens, charged = token_fields()
            row = {
                "evaluation": evaluation, "cumulative_evaluations": evaluation,
                "island": island_index, "source": source, "round": round_number, "policy": policy_data,
                "valid": valid, "train_mean": mean,
                "train_makespans": {row["instance"]: row.get("makespan") for row in results},
                "cumulative_tokens": total_tokens, "cumulative_charged_tokens": charged,
            }
            if valid and (best is None or mean < best["train_mean"]):
                best = row
            row["best_train_mean"] = best["train_mean"] if best else None
            history.append(row)
            if valid and island_index is not None:
                islands[island_index].append(row)
            return row

        if method in static_baselines:
            evaluate(static_baselines[method], None, "static_baseline")
        else:
            # Identical initial three candidates consume three evaluations in all
            # adaptive methods and random search. No privileged single-mode seed.
            for index, policy in enumerate(initial):
                try:
                    island_index = 0 if method == "single" else index
                    evaluate(policy, island_index, "initial")
                except BudgetExceeded as error:
                    stop_reason = str(error)
                    break

            if method == "random_search":
                while ledger.snapshot()["evaluations"] < max_evaluations:
                    evaluate(random_policy(rng, candidate_filter), None, "random")
            elif method in AGENT_METHODS:
                # Quotient/remainder split never gives an island an implicit
                # extra retry budget. Defaults are exactly 12 / (3 * 4).
                count = len(islands)
                quotas = [max_calls // count + (index < max_calls % count) for index in range(count)]
                island_calls = [0] * count
                stop_all = False
                for round_index in range(max(quotas, default=0)):
                    round_attempted = False
                    for island_index, quota in enumerate(quotas):
                        if island_calls[island_index] >= quota or not islands[island_index]:
                            continue
                        if ledger.snapshot()["evaluations"] >= max_evaluations:
                            stop_reason = "candidate evaluation budget exhausted"
                            stop_all = True
                            break
                        local_history = islands[island_index]
                        direction = (
                            "one continuous agent: combine short-operation, remaining-work, and random-weight exploration"
                            if method == "single" else DIRECTIONS[island_index]
                        )
                        prompt = make_prompt(local_history, direction, candidate_filter)
                        # A UTF-8 byte upper-bound plus protocol overhead is much
                        # more conservative than a chars/4 token estimate. Real
                        # providers must honor max_tokens; overruns are flagged.
                        input_bound = len(prompt.encode("utf-8")) + 256
                        for retry in range(max_retries + 1):
                            if island_calls[island_index] >= quota:
                                break
                            try:
                                reservation = ledger.reserve_call(input_bound, max_output_tokens)
                            except BudgetExceeded as error:
                                stop_reason = str(error)
                                stop_all = True
                                break
                            island_calls[island_index] += 1
                            round_attempted = True
                            generation = None
                            try:
                                generation = generators[island_index].generate(prompt, local_history, max_output_tokens)
                                candidate = parse_candidate(generation.content, candidate_filter)
                            except Exception as error:
                                ledger.finish_call(
                                    reservation,
                                    input_tokens=generation.input_tokens if generation else None,
                                    output_tokens=generation.output_tokens if generation else None,
                                    failed=True,
                                )
                                attempts.append({
                                    "call": reservation.number, "round": round_index + 1,
                                    "island": island_index, "retry": retry, "success": False,
                                    "error": type(error).__name__,
                                })
                                if ledger.snapshot()["provider_token_overrun_calls"]:
                                    stop_reason = "provider exceeded reserved token bounds; further generation halted"
                                    stop_all = True
                                    break
                                continue
                            ledger.finish_call(
                                reservation,
                                input_tokens=generation.input_tokens,
                                output_tokens=generation.output_tokens,
                            )
                            attempts.append({
                                "call": reservation.number, "round": round_index + 1,
                                "island": island_index, "retry": retry, "success": True,
                            })
                            evaluate(candidate, island_index, "generated", round_index + 1)
                            if ledger.snapshot()["provider_token_overrun_calls"]:
                                stop_reason = "provider exceeded reserved token bounds; further generation halted"
                                stop_all = True
                            break
                        if stop_all:
                            break
                    if not round_attempted:
                        break
                    # A migrant arriving after the last generation round could
                    # never influence a proposal, so it is not sent at all.
                    later_round = (
                        any(calls < quota for calls, quota in zip(island_calls, quotas))
                        and ledger.snapshot()["evaluations"] < max_evaluations
                    )
                    if method == "cooperative" and not stop_all and later_round:
                        exchanges.extend(exchange_ring(islands, round_index + 1, exchange_interval))
                    if stop_all:
                        break

    if best is None:
        raise RuntimeError("No valid candidate exists; no leaderboard result can be emitted")
    snapshot = ledger.snapshot()
    snapshot.update({
        "wall_time": time.perf_counter() - started,
        "cpu_time": time.process_time() - cpu_started,
        "cpu_workers": cpu_workers,
        "valid_schedules": valid_schedules,
        "validity_rate": valid_schedules / snapshot["instance_schedules"],
        "max_output_tokens": max_output_tokens,
        "max_retries": max_retries,
    })
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    snapshot["cpu_time"] += children.ru_utime + children.ru_stime - child_cpu_started
    if backend == "mock" and method in AGENT_METHODS and snapshot["calls"]:
        snapshot["token_accounting"] = "estimated_mock"
    return {
        "method": method, "backend": backend, "seed": seed,
        "best_policy": best["policy"], "best_train_mean": best["train_mean"],
        "history": history, "attempts": attempts, "resources": snapshot,
        "exchanges": exchanges, "stop_reason": stop_reason,
        "islands": [{
            "index": index,
            "history": rows,
            "local_candidate_count": sum("imported_from" not in row for row in rows),
        } for index, rows in enumerate(islands)] if method in AGENT_METHODS else [],
        "selection_split": "train",
        "objective": "mean_training_makespan",
    }
