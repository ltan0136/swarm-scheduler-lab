"""Experiment orchestration: freeze training winners before touching holdouts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .instances import make_datasets
from .policy import Policy
from .scheduling import InvalidSchedule, dispatch, validate
from .search import run_search
from .solver import solve_reference

AGENT_METHODS = ("single", "independent", "cooperative")
METHODS = (*AGENT_METHODS, "shortest_processing_time", "most_work_remaining", "random_search")


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "mock"
    methods: tuple[str, ...] = METHODS
    seeds: tuple[int, ...] = (7,)
    data_seed: int = 2026
    train_count: int = 8
    test_count: int = 8
    jobs: int = 6
    machines: int = 6
    duration_max: int = 20
    max_calls: int = 12
    max_tokens: int = 24000
    max_output_tokens: int = 512
    max_evaluations: int = 15
    workers: int = 1
    solver_time_limit: float = 2.0
    timeout: float = 60.0
    max_retries: int = 0
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "SWARM_API_KEY"

    def validate(self) -> None:
        if self.backend not in ("mock", "real"):
            raise ValueError("backend must be mock or real")
        if not self.methods or any(m not in METHODS for m in self.methods):
            raise ValueError("unknown or empty methods")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be unique")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("supply at least one unique random seed")
        for name in ("train_count", "test_count", "jobs", "machines", "duration_max", "max_tokens",
                     "max_output_tokens", "max_evaluations", "workers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.max_calls <= 12:
            raise ValueError("max_calls must be between 0 and 12 per method")
        if self.max_evaluations < 3:
            raise ValueError("max_evaluations must include the three shared initial policies")
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative; every retry consumes a call")
        if any(not math.isfinite(v) or v <= 0 for v in (self.solver_time_limit, self.timeout)):
            raise ValueError("solver time limit and request timeout must be finite and positive")
        if self.backend == "real" and any(m in AGENT_METHODS for m in self.methods):
            if not self.model:
                raise ValueError("real backend requires --model or SWARM_MODEL")
            if not os.environ.get(self.api_key_env):
                raise ValueError(f"real backend requires the {self.api_key_env} environment variable")
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("base URL must be HTTPS and contain no credentials, query or fragment")


def _revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    files = [(f"swarm_scheduler_lab/{path.name}", path) for path in sorted(package.glob("*.py"))]
    # Installed wheels do not retain repository-level build/lock files.
    files += [(name, root / name) for name in ("pyproject.toml", "requirements-lock.txt")
              if (root / name).is_file()]
    digest = hashlib.sha256()
    for relative, path in files:
        digest.update(relative.encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL,
        ))
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"git_worktree_dirty": dirty, "source_tree_sha256": digest.hexdigest()}


def run_experiment(config: ExperimentConfig) -> dict:
    """Search sees *only* train; references and final evaluations run afterward.

    One evaluation is one policy evaluated on every training instance. All
    schedules (training, holdout, benchmark, solver) pass the independent checker.
    Data are fixed across search seeds to allow paired method comparisons.
    """
    config.validate()
    started, cpu_started = time.perf_counter(), time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_started = children.ru_utime + children.ru_stime
    datasets = make_datasets(
        config.data_seed, train_count=config.train_count, test_count=config.test_count,
        jobs=config.jobs, machines=config.machines, duration_max=config.duration_max,
    )
    dataset_json = {split: [i.to_dict() for i in values] for split, values in datasets.items()}
    dataset_hash = hashlib.sha256(json.dumps(dataset_json, sort_keys=True).encode()).hexdigest()
    result = {
        "schema_version": 1,
        "config": asdict(config),
        "metadata": {
            "python": platform.python_version(),
            "dependencies": {p: importlib.metadata.version(p) for p in
                             ("job-shop-lib", "ortools", "matplotlib", "jsonschema")},
            "git_revision": _revision(),
            **_source_provenance(),
            "dataset_sha256": dataset_hash,
            "experiment_kind": "MOCK_FUNCTIONAL_DEMONSTRATION" if config.backend == "mock" else "REAL_MODEL_EXPERIMENT",
            "real_model_run": config.backend == "real" and any(m in AGENT_METHODS for m in config.methods),
            "selection_metric": "mean training makespan (lower is better)",
            "evaluation_unit": "one candidate policy on all training instances",
            "holdout_protocol": "all winners frozen using training feedback before final holdout evaluation",
            "sources": [
                "https://github.com/Pabloo22/job_shop_lib",
                "https://job-shop-lib.readthedocs.io/en/stable/examples/03-Dispatching-Rules.html",
                "https://developers.google.com/optimization/scheduling/job_shop",
            ],
        },
        "runs": [],
    }
    # Complete every search before any holdout schedule or solver reference is
    # computed. Neither references nor benchmark optimum can enter an agent.
    for seed in config.seeds:
        methods = []
        for method in config.methods:
            methods.append(run_search(
                method, datasets["train"], seed=seed, backend=config.backend,
                max_calls=config.max_calls, max_tokens=config.max_tokens,
                max_output_tokens=config.max_output_tokens,
                max_evaluations=config.max_evaluations, cpu_workers=config.workers,
                model=config.model, base_url=config.base_url, api_key_env=config.api_key_env,
                request_timeout=config.timeout, max_retries=config.max_retries,
            ))
        result["runs"].append({"seed": seed, "datasets": dataset_json, "methods": methods})

    reference_started, reference_cpu = time.perf_counter(), time.process_time()
    references = [
        {"instance": instance.name, "split": split,
         **solve_reference(instance, config.solver_time_limit, config.data_seed)}
        for split, instances in datasets.items() for instance in instances
    ]
    result["reference_resources"] = {
        "instances": len(references), "wall_time": time.perf_counter() - reference_started,
        "cpu_time": time.process_time() - reference_cpu,
        "workers": 1, "time_limit_per_instance": config.solver_time_limit,
    }
    for run in result["runs"]:
        run["references"] = references
        for method in run["methods"]:
            final_started, final_cpu = time.perf_counter(), time.process_time()
            policy = Policy.from_dict(method["best_policy"])
            final = []
            for split, instances in datasets.items():
                for instance in instances:
                    try:
                        schedule = dispatch(instance, policy)
                        checked = validate(instance, schedule)
                        final.append({"split": split, "instance": instance.name,
                                      **checked, "schedule": schedule})
                    except InvalidSchedule as exc:
                        # Invalid output is visible, never scored or ranked.
                        final.append({"split": split, "instance": instance.name, "valid": False,
                                      "makespan": None, "schedule": [], "error": str(exc)})
            method["final"] = final
            method["final_resources"] = {
                "schedules": len(final), "wall_time": time.perf_counter() - final_started,
                "cpu_time": time.process_time() - final_cpu,
            }
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["resources"] = {
        "wall_time": time.perf_counter() - started,
        "cpu_time": time.process_time() - cpu_started + children.ru_utime + children.ru_stime - child_cpu_started,
        "generation_calls": sum(m["resources"]["calls"] for r in result["runs"] for m in r["methods"]),
    }
    return result


def save_results(result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "results.json"
    temporary = output_dir / ".results.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
