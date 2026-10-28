import copy
import csv

from swarm_scheduler_lab.reporting import (
    _proof_label,
    _select_gantt,
    generate_report,
)


def _result():
    def method(name, training, test):
        return {
            "method": name,
            "backend": "mock",
            "seed": 7,
            "best_policy": {"weights": {"processing_time": 1}, "tie_break": "job_id"},
            "best_train_mean": training,
            "history": [
                {
                    "evaluation": 1,
                    "train_mean": training,
                    "best_train_mean": training,
                    "cumulative_tokens": 100,
                    "policy": {},
                }
            ],
            "resources": {
                "calls": 1,
                "failures": 0,
                "evaluations": 1,
                "input_tokens": 70,
                "output_tokens": 30,
                "unknown_usage_calls": 0,
                "token_accounting": "estimated (mock)",
                "wall_time": 0.1,
                "cpu_time": 0.08,
            },
            "final": [
                {
                    "split": "benchmark",
                    "instance": "ft06",
                    "valid": True,
                    "makespan": 55,
                    "schedule": [
                        {"job": 0, "operation": 0, "machine": 0, "start": 0, "end": 55, "duration": 55}
                    ],
                },
                {
                    "split": "test",
                    "instance": "test_0",
                    "valid": True,
                    "makespan": test,
                    "schedule": [
                        {"job": 0, "operation": 0, "machine": 0, "start": 0, "end": test, "duration": test}
                    ],
                },
                {
                    "split": "train",
                    "instance": "train_0",
                    "valid": False,
                    "makespan": 1,
                    "schedule": [],
                },
            ],
        }

    return {
        "schema_version": 1,
        "config": {"backend": "mock", "seeds": [7]},
        "runs": [
            {
                "seed": 7,
                "datasets": {"train": [], "test": [], "benchmark": []},
                "references": [
                    {
                        "instance": "test_0",
                        "split": "test",
                        "status": "FEASIBLE",
                        "objective": 15,
                        "best_bound": 10,
                        "proven_optimal": False,
                        "wall_time": 0.1,
                    }
                ],
                "methods": [method("single", 20, 30), method("cooperative", 25, 10)],
            }
        ],
    }


def test_report_outputs_keep_mock_and_unproven_results_explicit(tmp_path):
    result = _result()
    before = copy.deepcopy(result)
    outputs = generate_report(result, tmp_path)
    assert result == before
    assert {p.name for p in tmp_path.iterdir()} == {
        "report.md", "summary.csv", "evaluations.csv", "convergence.png", "gantt.png"
    }
    assert all(tmp_path.joinpath(name).stat().st_size > 0 for name in ("convergence.png", "gantt.png"))
    assert len(outputs) == 5
    report = (tmp_path / "report.md").read_text()
    assert "MOCK / OFFLINE FUNCTIONAL DEMONSTRATION" in report
    assert "Real-model token comparison: **not run**" in report
    assert "current valid solution; optimality unproven" in report
    assert "matches known optimum (55)" in report
    with (tmp_path / "summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    invalid_row = next(r for r in rows if r["split"] == "train")
    assert invalid_row["valid_rate"] == "0.0"
    assert invalid_row["best_makespan"] == ""


def test_gantt_selection_uses_training_not_held_out():
    run, winner, finals = _select_gantt(_result())
    assert run["seed"] == 7
    assert winner["method"] == "single"  # cooperative has the better held-out score
    assert {r["split"] for r in finals} == {"benchmark", "test"}


def test_bound_or_feasible_status_is_never_an_optimality_proof():
    final = {"instance": "generated", "valid": True, "makespan": 10}
    assert "unproven" in _proof_label(
        final, {"status": "FEASIBLE", "objective": 10, "best_bound": 10, "proven_optimal": True}
    )
    assert _proof_label(
        final, {"status": "OPTIMAL", "objective": 10, "best_bound": 10, "proven_optimal": True}
    ).startswith("proven_optimal")
    assert "invalid" in _proof_label({**final, "valid": False}, None)


def test_real_token_chart_requires_provider_measurements_and_stale_chart_is_removed(tmp_path):
    result = _result()
    result["config"]["backend"] = "real"
    for method in result["runs"][0]["methods"]:
        method["backend"] = "real"
        method["resources"]["token_accounting"] = "provider-reported"
    generate_report(result, tmp_path)
    assert (tmp_path / "tokens.png").stat().st_size > 0
    assert "provider-reported tokens" in (tmp_path / "report.md").read_text()
    for method in result["runs"][0]["methods"]:
        method["history"][0]["cumulative_tokens"] = None
        method["resources"]["input_tokens"] = None
        method["resources"]["output_tokens"] = None
        method["resources"]["unknown_usage_calls"] = 1
    generate_report(result, tmp_path)
    assert not (tmp_path / "tokens.png").exists()
    assert "actual token consumption is unknown" in (tmp_path / "report.md").read_text()
