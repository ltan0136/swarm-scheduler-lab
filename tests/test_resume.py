"""CLI regression tests for isolated output directories and resumed searches."""

import json
import subprocess
import sys

import pytest

from swarm_scheduler_lab import cli, experiment


SMALL_RUN = [
    "run", "--train-count", "1", "--test-count", "1", "--jobs", "2", "--machines", "2",
    "--max-calls", "1", "--max-evaluations", "4", "--solver-time-limit", "0.15",
]


def test_default_runs_get_distinct_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = [*SMALL_RUN, "--methods", "shortest_processing_time"]
    assert cli.main(args) == 0
    first, = (tmp_path / "results").glob("run-*/results.json")
    saved = first.read_bytes()
    assert cli.main(args) == 0
    outputs = list((tmp_path / "results").glob("run-*/results.json"))
    assert len(outputs) == 2
    assert first.read_bytes() == saved


def test_explicit_output_protects_existing_files(tmp_path, monkeypatch):
    run_dir = tmp_path / "empty-dir"
    run_dir.mkdir()
    args = [*SMALL_RUN, "--methods", "shortest_processing_time", "--output", str(run_dir)]
    assert cli.main(args) == 0
    saved = (run_dir / "results.json").read_bytes()
    monkeypatch.setattr(experiment, "run_search", lambda *args, **kwargs: pytest.fail("run already exists"))
    assert cli.main(args) != 0
    assert (run_dir / "results.json").read_bytes() == saved


def test_interrupted_run_resumes_with_identical_policies(tmp_path, monkeypatch):
    args = [*SMALL_RUN, "--methods", "single", "random_search", "--seeds", "7", "19"]
    fresh = tmp_path / "uninterrupted"
    assert cli.main([*args, "--output", str(fresh)]) == 0

    run_dir = tmp_path / "interrupted"
    searches = []
    completed = (7, "single")
    second = (7, "random_search")
    original = experiment.run_search

    def interrupted_search(method, train, *, seed, **kwargs):
        searches.append((seed, method))
        if (seed, method) == second:
            # A second process must not acquire an active directory.
            other = subprocess.run(
                [sys.executable, "-m", "swarm_scheduler_lab", "resume", str(run_dir)],
                capture_output=True, text=True, timeout=15,
            )
            assert other.returncode != 0
            raise OSError("interrupted before search returned")
        return original(method, train, seed=seed, **kwargs)

    monkeypatch.setattr(experiment, "run_search", interrupted_search)
    assert cli.main([*args, "--output", str(run_dir)]) != 0
    assert searches == [completed, second]
    assert not (run_dir / "results.json").exists()

    searches.clear()

    def record_search(method, train, *, seed, **kwargs):
        searches.append((seed, method))
        return original(method, train, seed=seed, **kwargs)

    monkeypatch.setattr(experiment, "run_search", record_search)
    assert cli.main(["resume", str(run_dir)]) == 0
    assert searches == [second, (19, "single"), (19, "random_search")]
    expected = json.loads((fresh / "results.json").read_text())
    actual = json.loads((run_dir / "results.json").read_text())
    assert actual["config"] == expected["config"]
    assert actual["resources"]["generation_calls"] == expected["resources"]["generation_calls"]
    for resumed_run, fresh_run in zip(actual["runs"], expected["runs"], strict=True):
        assert resumed_run["seed"] == fresh_run["seed"]
        for resumed, fresh_method in zip(resumed_run["methods"], fresh_run["methods"], strict=True):
            for field in ("method", "best_policy", "history", "final"):
                assert resumed[field] == fresh_method[field]

    # If reporting failed after results.json was committed, resume repairs it
    # without rewriting the canonical results or rerunning any method.
    searches.clear()
    results_before_report = (run_dir / "results.json").read_bytes()
    (run_dir / "report.md").unlink()
    assert cli.main(["resume", str(run_dir)]) == 0
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "results.json").read_bytes() == results_before_report
    assert not searches


def test_invalid_progress_refused_before_any_search(tmp_path, monkeypatch):
    run_dir = tmp_path / "damaged-run"

    def interrupt(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(experiment, "run_search", interrupt)
    assert cli.main([*SMALL_RUN, "--methods", "single", "--output", str(run_dir)]) != 0
    progress = run_dir / "progress.json"
    assert progress.is_file()
    progress.write_bytes(b"{")
    monkeypatch.setattr(experiment, "run_search", lambda *args, **kwargs: pytest.fail("invalid progress"))
    assert cli.main(["resume", str(run_dir)]) != 0
    assert progress.read_bytes() == b"{"
    assert not (run_dir / "results.json").exists()


@pytest.mark.parametrize("change", ["source", "checkpoint"])
def test_incompatible_checkpoint_refused_before_resuming(tmp_path, monkeypatch, change):
    args = [*SMALL_RUN, "--methods", "single", "random_search", "--output", str(tmp_path)]
    original = experiment.run_search

    def interrupt(method, *args, **kwargs):
        if method == "random_search":
            raise OSError("interrupted")
        return original(method, *args, **kwargs)

    monkeypatch.setattr(experiment, "run_search", interrupt)
    assert cli.main(args) != 0
    if change == "source":
        provenance = {**experiment._source_provenance(), "source_tree_sha256": "changed"}
        monkeypatch.setattr(experiment, "_source_provenance", lambda: provenance)
    else:
        unit = tmp_path / "checkpoints" / "7-single.json"
        data = json.loads(unit.read_text())
        data["payload"]["search"]["resources"]["calls"] += 1
        unit.write_text(json.dumps(data))
    monkeypatch.setattr(experiment, "run_search", lambda *args, **kwargs: pytest.fail("incompatible run"))
    assert cli.main(["resume", str(tmp_path)]) != 0
    assert not (tmp_path / "results.json").exists()
