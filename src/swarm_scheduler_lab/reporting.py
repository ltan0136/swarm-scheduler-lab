"""Build inspectable offline reports from saved experiment results.

This module consumes results only. It never proposes a policy, changes a
selection, or feeds held-out measurements back to a search agent.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "swarm-scheduler-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _number(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            return "unknown"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _summary_rows(result: dict) -> list[dict]:
    rows = []
    for run in result.get("runs", []):
        for method in run.get("methods", []):
            for split in ("train", "test", "benchmark"):
                finals = [r for r in method.get("final", []) if r["split"] == split]
                if not finals:
                    continue
                valid = [r for r in finals if r.get("valid") and r.get("makespan") is not None]
                spans = [r["makespan"] for r in valid]
                resources = method.get("resources", {})
                rows.append(
                    {
                        "seed": run["seed"],
                        "backend": method.get("backend", result["config"]["backend"]),
                        "method": method["method"],
                        "split": split,
                        "valid": len(valid),
                        "total": len(finals),
                        "valid_rate": len(valid) / len(finals),
                        "best_makespan": min(spans) if spans else None,
                        "mean_makespan": statistics.mean(spans) if spans else None,
                        "worst_makespan": max(spans) if spans else None,
                        "calls": resources.get("calls"),
                        "failures": resources.get("failures"),
                        "evaluations": resources.get("evaluations"),
                        "input_tokens": resources.get("input_tokens"),
                        "output_tokens": resources.get("output_tokens"),
                        "unknown_usage_calls": resources.get("unknown_usage_calls"),
                        "wall_time": resources.get("wall_time"),
                        "cpu_time": resources.get("cpu_time"),
                        "resources_json": json.dumps(resources, sort_keys=True),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_rows(result: dict) -> list[dict]:
    rows = []
    for run in result.get("runs", []):
        for method in run.get("methods", []):
            for item in method.get("history", []):
                rows.append(
                    {
                        "seed": run["seed"],
                        "backend": method.get("backend", result["config"]["backend"]),
                        "method": method["method"],
                        "evaluation": item.get("evaluation"),
                        "train_mean": item.get("train_mean"),
                        "best_train_mean": item.get("best_train_mean"),
                        "cumulative_tokens": item.get("cumulative_tokens"),
                        "cumulative_charged_tokens": item.get("cumulative_charged_tokens"),
                        "island": item.get("island"),
                        "round": item.get("round"),
                        "source": item.get("source"),
                        "policy_json": json.dumps(item.get("policy"), sort_keys=True),
                        "train_makespans_json": json.dumps(item.get("train_makespans")),
                    }
                )
    return rows


def _plot_convergence(result: dict, path: Path) -> None:
    runs = result.get("runs", [])
    fig, axes = plt.subplots(
        max(1, len(runs)), 1, figsize=(9, max(4, 3.7 * len(runs))), squeeze=False
    )
    for ax, run in zip(axes.flat, runs):
        for method in run.get("methods", []):
            points = [
                item
                for item in method.get("history", [])
                if item.get("best_train_mean") is not None
            ]
            if points:
                ax.step(
                    [p["evaluation"] for p in points],
                    [p["best_train_mean"] for p in points],
                    where="post",
                    label=method["method"],
                )
        ax.set_title(f"Training objective / seed {run['seed']} ({result['config']['backend']})")
        ax.set_xlabel("Cumulative candidate evaluations (one candidate over the full training set)")
        ax.set_ylabel("Best training mean makespan ↓")
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_tokens(result: dict, path: Path) -> bool:
    """Only use returned metering; unknown calls never become zero tokens."""
    if result["config"]["backend"] != "real":
        return False
    series = []
    for run in result.get("runs", []):
        for method in run.get("methods", []):
            # Baselines have no model calls and do not define a token trajectory.
            if not method.get("resources", {}).get("calls"):
                continue
            points = [
                p
                for p in method.get("history", [])
                if p.get("cumulative_tokens") is not None
                and p.get("best_train_mean") is not None
            ]
            if points and any(p["cumulative_tokens"] > 0 for p in points):
                series.append((run, method, points))
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, method, points in series:
        ax.step(
            [p["cumulative_tokens"] for p in points],
            [p["best_train_mean"] for p in points],
            where="post",
            label=f"{method['method']} / seed {run['seed']}",
        )
    ax.set_xlabel("Cumulative provider-reported input + output tokens")
    ax.set_ylabel("Best training mean makespan ↓")
    ax.set_title("Real backend: provider-reported token measurements only")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _select_gantt(result: dict) -> tuple[dict, dict, list[dict]] | None:
    """Choose a method by training objective, never by its held-out scores."""
    for run in result.get("runs", []):
        candidates = [
            m
            for m in run.get("methods", [])
            if m.get("best_train_mean") is not None and m.get("best_policy") is not None
        ]
        if not candidates:
            continue
        winner = min(candidates, key=lambda m: (m["best_train_mean"], m["method"]))
        finals = winner.get("final", [])
        schedules = []
        for split in ("benchmark", "test"):
            choices = sorted(
                [r for r in finals if r["split"] == split and r.get("valid") and r.get("schedule")],
                key=lambda r: r["instance"],
            )
            if choices:
                schedules.append(choices[0])
        if schedules:
            return run, winner, schedules
    return None


def _plot_gantt(result: dict, path: Path) -> bool:
    chosen = _select_gantt(result)
    if chosen is None:
        return False
    run, winner, schedules = chosen
    fig, axes = plt.subplots(
        len(schedules), 1, figsize=(11, 3.5 * len(schedules)), squeeze=False
    )
    for ax, final in zip(axes.flat, schedules):
        machine_ids = sorted({o["machine"] for o in final["schedule"]})
        for op in final["schedule"]:
            ax.barh(
                op["machine"],
                op["end"] - op["start"],
                left=op["start"],
                height=0.7,
                color=plt.get_cmap("tab20")(op["job"] % 20),
                edgecolor="white",
                linewidth=0.6,
            )
            if op["end"] - op["start"] > 0:
                ax.text(
                    (op["start"] + op["end"]) / 2,
                    op["machine"],
                    f"J{op['job']}.{op['operation']}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
        ax.set_yticks(machine_ids, [f"M{m}" for m in machine_ids])
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_title(
            f"{final['instance']} / {final['split']} / makespan {final['makespan']}\n"
            f"{winner['method']} selected on training mean, seed {run['seed']} "
            f"({result['config']['backend']})"
        )
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _proof_label(final: dict, reference: dict | None) -> str:
    if not final.get("valid"):
        return "invalid — rejected"
    if final["instance"].lower() == "ft06" and final["makespan"] == 55:
        return "matches known optimum (55)"
    if reference and reference.get("status") == "OPTIMAL" and reference.get("proven_optimal"):
        if final["makespan"] == reference.get("objective"):
            return "proven_optimal (matches CP-SAT proof)"
        return "CP-SAT optimum proven; dispatch result above optimum"
    return "current valid solution; optimality unproven"


def _markdown(result: dict, rows: list[dict], has_tokens: bool, has_gantt: bool) -> str:
    mock = result["config"]["backend"] == "mock"
    # Results saved before these options existed used the same fixed behavior.
    candidate_filter = result["config"].get("candidate_filter", "none")
    exchange_interval = result["config"].get("exchange_interval", 2)
    exchange_rounds = sorted({
        event["round"]
        for run in result.get("runs", [])
        for method in run.get("methods", [])
        if method["method"] == "cooperative"
        for event in method.get("exchanges", [])
    })
    lines = [
        "# Swarm scheduling experiment report",
        "",
        (
            "> **MOCK / OFFLINE FUNCTIONAL DEMONSTRATION. No real model was called; "
            "these results are not empirical evidence about LLM swarm performance.**"
            if mock
            else "> **REAL MODEL RUN.** Provider, configuration and metering are recorded below."
        ),
        "",
        "This experiment searches restricted JSON dispatching rules. Lower mean training "
        "makespan is better. The held-out test set is evaluated only after policy selection; "
        "it never supplies agent feedback, migrations, or policy selection.",
        "",
        "A candidate evaluation means running one policy across every training instance. "
        "Initial candidates consume this budget. Methods share the same initial candidates. "
        "Model call failures and retries consume the call budget. Baselines make no model calls.",
        "",
        f"Searched rule space candidate filter: `{candidate_filter}` "
        + (
            "(every job's next operation is ranked, even one that cannot start yet). "
            if candidate_filter == "none"
            else "(JobShopLib's default dominated + non-immediate pruning). "
        )
        + "The `pruned_*` baselines always use JobShopLib's default pruning.",
        "",
        "## Run configuration",
        "",
        "```json",
        json.dumps(result.get("config", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Final selected-policy measurements",
        "",
        "Training, held-out generated instances, and public FT06 are separate below. "
        "`best` is the smallest makespan among instances in that split, not a new "
        "cross-instance optimization target. Compare means only for the same data split and seed.",
        "",
        "| Seed | Method | Split | Valid / total | Best | Mean | Worst |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['split']} | {row['valid']} / "
            f"{row['total']} | {_number(row['best_makespan'])} | {_number(row['mean_makespan'])} "
            f"| {_number(row['worst_makespan'])} |"
        )
    lines += [
        "",
        "## Actual search resources",
        "",
        "Budgets apply separately to each method and seed; CPU and wall times vary by machine. "
        "Unknown token usage is not reported as zero. Token reservations are a conservative "
        "budget control, not measured provider consumption. Complete resource records are in "
        "`results.json` and `summary.csv` (repeated per split for convenience).",
        "",
        "| Seed | Method | Calls | Failures | Candidate evaluations | Input tokens | Output tokens "
        "| Unknown-usage calls | Wall seconds | CPU seconds | Token accounting |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in result.get("runs", []):
        for method in run.get("methods", []):
            r = method.get("resources", {})
            lines.append(
                f"| {run['seed']} | {method['method']} | {_number(r.get('calls'))} | "
                f"{_number(r.get('failures'))} | {_number(r.get('evaluations'))} | "
                f"{_number(r.get('input_tokens'))} | {_number(r.get('output_tokens'))} | "
                f"{_number(r.get('unknown_usage_calls'))} | {_number(r.get('wall_time'))} | "
                f"{_number(r.get('cpu_time'))} | {_number(r.get('token_accounting'))} |"
            )
    lines += [
        "",
        "## Search traces",
        "",
        "![Best training mean makespan by cumulative evaluations](convergence.png)",
        "",
    ]
    if has_tokens:
        lines += [
            "![Best training mean makespan by provider-reported tokens](tokens.png)",
            "",
            "The token plot includes only observations with known cumulative provider metering. "
            "A missing usage response makes later cumulative actual usage unknown; the "
            "charged reservation total must not be interpreted as actual tokens.",
            "",
        ]
    else:
        lines += [
            (
                "Real-model token comparison: **not run** (mock backend). Mock token values "
                "are estimates for exercising budget accounting."
                if mock
                else "No provider-reported cumulative token trace was available; actual token "
                "consumption is unknown. Consult the recorded usage status and budget charges."
            ),
            "",
        ]
    lines += [
        "## Independent OR-Tools references",
        "",
        "An incumbent objective is an upper bound on the minimum makespan. A valid best bound "
        "is a lower bound, not the true optimum. `FEASIBLE`, `UNKNOWN`, or a timeout does not "
        "prove optimality. Each reference records its solver status and time limit in raw results.",
        "",
        "| Seed | Split | Instance | Status | Incumbent upper bound | Valid lower bound "
        "| Proven optimal | Solver wall seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in result.get("runs", []):
        for ref in run.get("references", []):
            proof = ref.get("status") == "OPTIMAL" and ref.get("proven_optimal") is True
            lines.append(
                f"| {run['seed']} | {ref['split']} | {ref['instance']} | {ref.get('status')} | "
                f"{_number(ref.get('objective'))} | {_number(ref.get('best_bound'))} | "
                f"{str(proof).lower()} | {_number(ref.get('wall_time'))} |"
            )
    lines += [
        "",
        "## Per-instance validity and optimality labels",
        "",
        "FT06 has a published known optimum of 55. That value is report metadata and is never "
        "included in an agent prompt. Generated instances are labeled `proven_optimal` only "
        "when a valid final schedule matches a CP-SAT `OPTIMAL` proof.",
        "",
        "| Seed | Method | Split | Instance | Makespan | Interpretation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in result.get("runs", []):
        refs = {(r["split"], r["instance"]): r for r in run.get("references", [])}
        for method in run.get("methods", []):
            for final in method.get("final", []):
                label = _proof_label(final, refs.get((final["split"], final["instance"])))
                lines.append(
                    f"| {run['seed']} | {method['method']} | {final['split']} | "
                    f"{final['instance']} | {_number(final.get('makespan'))} | {label} |"
                )
    lines += ["", "## Representative schedules", ""]
    if has_gantt:
        lines += [
            "![Validated selected-policy schedules](gantt.png)",
            "",
            "The illustrated method is selected solely by the smallest training objective "
            "(method-name tie break) from the first available seed. The first named FT06 and "
            "held-out instances are displayed without optimizing the selection on their scores.",
            "",
        ]
    else:
        lines += ["No valid selected-policy schedule was available for a Gantt chart.", ""]
    lines += [
        "## Interpretation and limits",
        "",
        "These finite-seed, small-instance runs measure this restricted rule space and this "
        "specific budget. They do not establish a general advantage for cooperation. Differences "
        "may be zero or favor another method. Re-run multiple predeclared seeds and examine "
        "held-out results without feeding them back into tuning. Mock search is a deterministic "
        "workflow check, not a substitute for a real LLM experiment.",
        "",
        "`single` keeps one shared history. `independent` keeps three isolated island histories. "
        f"`cooperative` exchanges a training winner around a simultaneous ring every {exchange_interval} "
        "round(s) while retaining local candidates (rounds with a recorded exchange: "
        f"{', '.join(map(str, exchange_rounds)) or 'none'}). "
        "SPT and MWR are fixed non-model baselines in the searched space; `pruned_*` SPT and MWR "
        "match JobShopLib's built-in rules. Random weight search uses the configured comparable "
        "candidate budget.",
        "",
        "## Files and sources",
        "",
        "- `results.json`: full run configuration, generated instances, policies, selected schedules, "
        "reference results and resource records.",
        "- `summary.csv`: per-seed, per-method, per-split final statistics and resources.",
        "- `evaluations.csv`: candidate-by-candidate training trace, policy JSON and token accounting.",
        "- [JobShopLib](https://github.com/Pabloo22/job_shop_lib), "
        "[dispatching rules documentation](https://job-shop-lib.readthedocs.io/en/stable/examples/03-Dispatching-Rules.html), "
        "[OR-Tools job shop guide](https://developers.google.com/optimization/scheduling/job_shop).",
        "",
    ]
    return "\n".join(lines)


def generate_report(result: dict, output_dir: Path | str) -> list[str]:
    """Write reports from existing data, with no search or model calls.

    Returns absolute paths to generated files. The canonical ``results.json``
    is saved by the experiment runner, not changed here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _summary_rows(result)
    summary_fields = [
        "seed", "backend", "method", "split", "valid", "total", "valid_rate",
        "best_makespan", "mean_makespan", "worst_makespan", "calls", "failures",
        "evaluations", "input_tokens", "output_tokens", "unknown_usage_calls", "wall_time",
        "cpu_time", "resources_json",
    ]
    _write_csv(output_dir / "summary.csv", rows, summary_fields)
    _write_csv(
        output_dir / "evaluations.csv",
        _evaluation_rows(result),
        [
            "seed", "backend", "method", "evaluation", "train_mean", "best_train_mean",
            "cumulative_tokens", "cumulative_charged_tokens", "island", "round", "source",
            "policy_json", "train_makespans_json",
        ],
    )
    _plot_convergence(result, output_dir / "convergence.png")
    has_tokens = _plot_tokens(result, output_dir / "tokens.png")
    has_gantt = _plot_gantt(result, output_dir / "gantt.png")
    # Remove only optional chart outputs generated by this module on a previous
    # report invocation, so stale charts cannot imply an unexecuted real run.
    for name, present in (("tokens.png", has_tokens), ("gantt.png", has_gantt)):
        if not present:
            (output_dir / name).unlink(missing_ok=True)
    (output_dir / "report.md").write_text(
        _markdown(result, rows, has_tokens, has_gantt), encoding="utf-8"
    )
    names = ["report.md", "summary.csv", "evaluations.csv", "convergence.png"]
    if has_gantt:
        names.append("gantt.png")
    if has_tokens:
        names.append("tokens.png")
    return [str((output_dir / name).resolve()) for name in names]
