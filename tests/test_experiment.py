"""Integration checks at the data boundary and documented command boundary."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from swarm_scheduler_lab import experiment
from swarm_scheduler_lab.experiment import ExperimentConfig, run_experiment
from swarm_scheduler_lab.instances import Instance
from swarm_scheduler_lab.scheduling import validate


def test_all_searches_finish_before_references_and_holdouts(monkeypatch):
    """A holdout cannot affect later methods, later seeds, or migration."""
    config = ExperimentConfig(methods=("single", "cooperative"), seeds=(7, 19),
                              train_count=2, test_count=2, jobs=3, machines=3,
                              max_calls=0, max_evaluations=3, solver_time_limit=0.5)
    searches = []
    original_search = experiment.run_search
    original_reference = experiment.solve_reference
    original_dispatch = experiment.dispatch

    def search(method, train, **kwargs):
        assert all(instance.name.startswith("train_") for instance in train)
        assert "test" not in kwargs and "benchmark" not in kwargs
        searches.append((method, kwargs["seed"]))
        return original_search(method, train, **kwargs)

    def reference(*args, **kwargs):
        assert len(searches) == 4
        return original_reference(*args, **kwargs)

    def final_dispatch(instance, policy):
        assert len(searches) == 4
        return original_dispatch(instance, policy)

    monkeypatch.setattr(experiment, "run_search", search)
    monkeypatch.setattr(experiment, "solve_reference", reference)
    monkeypatch.setattr(experiment, "dispatch", final_dispatch)
    result = run_experiment(config)
    assert len(result["runs"]) == 2
    assert all(len(m["final"]) == 5 for r in result["runs"] for m in r["methods"])
    assert result["resources"]["generation_calls"] == 0


def test_changed_holdout_size_cannot_change_search_outcomes():
    config = ExperimentConfig(methods=("cooperative",), train_count=2, test_count=1,
                              jobs=3, machines=3, max_calls=6, max_evaluations=9,
                              solver_time_limit=0.5)
    first = run_experiment(config)["runs"][0]["methods"][0]
    second = run_experiment(replace(config, test_count=3))["runs"][0]["methods"][0]
    for key in ("best_policy", "best_train_mean", "history", "exchanges"):
        assert first[key] == second[key]


@pytest.mark.parametrize("changes", [
    {"base_url": "https://user:password@example.org/v1"},
    {"base_url": "https://example.org/v1?key=secret"},
    {"solver_time_limit": float("nan")},
    {"max_calls": -1},
    {"candidate_filter": "dominated"},
    {"exchange_interval": 0},
    {"max_evaluations": 2},
    {"seeds": (7, 7)},
])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **changes).validate()


def test_call_budget_above_twelve_is_accepted():
    ExperimentConfig(max_calls=36, max_evaluations=39).validate()


def test_real_requires_explicit_credentials(monkeypatch):
    monkeypatch.delenv("SWARM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SWARM_API_KEY"):
        ExperimentConfig(backend="real", model="explicit-model").validate()


def test_provenance_also_works_without_repository_files(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "swarm_scheduler_lab"
    package.mkdir(parents=True)
    module = package / "experiment.py"
    module.write_text("# installed wheel source\n")
    monkeypatch.setattr(experiment, "__file__", str(module))
    provenance = experiment._source_provenance()
    assert len(provenance["source_tree_sha256"]) == 64


def test_offline_smoke_entrypoint_produces_valid_raw_data_and_images(tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "swarm_scheduler_lab", "smoke", "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert "MOCK" in process.stdout
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["metadata"]["real_model_run"] is False
    assert result["resources"]["generation_calls"] <= 18
    for run in result["runs"]:
        instances = {i["name"]: Instance.from_dict(i) for split in run["datasets"].values() for i in split}
        assert len(run["methods"]) == 8
        for method in run["methods"]:
            assert method["history"]
            for row in method["final"]:
                assert row["valid"]
                assert validate(instances[row["instance"]], row["schedule"])["makespan"] == row["makespan"]
    for name in ("report.md", "summary.csv", "evaluations.csv", "convergence.png", "gantt.png"):
        assert (tmp_path / name).stat().st_size > 100
    assert "MOCK" in (tmp_path / "report.md").read_text()
    regen = tmp_path / "regenerated"
    report = subprocess.run([sys.executable, "-m", "swarm_scheduler_lab", "report", "--input",
                             str(tmp_path / "results.json"), "--output", str(regen)],
                            capture_output=True, text=True, timeout=30)
    assert report.returncode == 0, report.stderr
    assert (regen / "report.md").exists()
