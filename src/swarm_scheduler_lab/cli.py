"""Command-line entrypoints; never display credentials or model response bodies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .experiment import METHODS, ExperimentConfig, run_experiment, save_results
from .policy import CANDIDATE_FILTERS
from .reporting import generate_report
from .run_files import SearchJournal, default_output, locked_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restricted dispatch-rule search; mock results are functional demos.")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="small fully offline end-to-end check")
    smoke.add_argument("--output", type=Path, help="new or empty directory (default: unique results/smoke-*)")
    run = commands.add_parser("run", help="run agents and/or model-free baselines")
    run.add_argument("--output", type=Path, help="new or empty directory (default: unique results/run-*)")
    run.add_argument("--backend", choices=("mock", "real"), default="mock")
    run.add_argument("--methods", nargs="+", choices=("all", *METHODS), default=["all"])
    run.add_argument("--seeds", nargs="+", type=int, default=[7])
    for name, default in (("data-seed", 2026), ("train-count", 8), ("test-count", 8),
                          ("jobs", 6), ("machines", 6), ("duration-max", 20),
                          ("max-calls", 12), ("max-tokens", 24000), ("max-output-tokens", 512),
                          ("max-evaluations", 15), ("workers", 1), ("max-retries", 0),
                          ("exchange-interval", 2)):
        run.add_argument(f"--{name}", type=int, default=default)
    run.add_argument("--candidate-filter", choices=CANDIDATE_FILTERS, default="none",
                     help="ready-operation set ranked by searched policies (pruned_* baselines always prune)")
    run.add_argument("--budget-unit", choices=("calls", "tokens"), default="calls",
                     help="calls: split --max-calls across islands; tokens: call until --max-tokens "
                          "binds (raise --max-calls/--max-evaluations so they do not)")
    run.add_argument("--solver-time-limit", type=float, default=2.0)
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--model", default=os.environ.get("SWARM_MODEL"))
    run.add_argument("--base-url", default=os.environ.get("SWARM_BASE_URL"))
    run.add_argument("--api-key-env", default="SWARM_API_KEY")
    resume = commands.add_parser("resume", help="continue a run from its completed seed/method searches")
    resume.add_argument("output", type=Path, metavar="DIRECTORY")
    report = commands.add_parser("report", help="regenerate reports without search or model calls")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace, output: Path) -> ExperimentConfig:
    if args.command == "smoke":
        return ExperimentConfig(train_count=2, test_count=2, jobs=3, machines=3,
                                max_calls=6, max_evaluations=9, max_tokens=12000,
                                solver_time_limit=1.0)
    if args.command == "resume":
        values = SearchJournal.load(output).manifest["config"].copy()
    else:
        values = vars(args).copy()
        for key in ("command", "output"):
            values.pop(key)
        if "all" in args.methods:
            values["methods"] = METHODS
    values["methods"] = tuple(values["methods"])
    values["seeds"] = tuple(values["seeds"])
    return ExperimentConfig(**values)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            with locked_directory(args.output, create=True):
                result = json.loads(args.input.read_text(encoding="utf-8"))
                artifacts = generate_report(result, args.output)
            print(f"Generated {len(artifacts)} artifacts in {args.output}")
            return 0
        output = args.output or default_output(args.command)
        resume = args.command == "resume"
        with locked_directory(output, create=not resume):
            if not resume and any(output.iterdir()):
                raise ValueError("Output directory must be empty for a new run; use resume to continue")
            print(f"Run directory: {output.resolve()}", flush=True)
            path = output / "results.json"
            if resume and path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                config = _config(args, output)
                print("MOCK: functional demonstration, not evidence of LLM swarm performance." if config.backend == "mock"
                      else "REAL: generation calls, failures and tokens are budgeted; holdouts stay out of feedback.")
                result = run_experiment(config, output_dir=output, resume=resume)
                save_results(result, output)
            generate_report(result, output)
        for run in result["runs"]:
            for method in run["methods"]:
                print(f"seed={run['seed']} method={method['method']} train_mean={method['best_train_mean']:.3f} "
                      f"calls={method['resources']['calls']} evaluations={method['resources']['evaluations']}")
        print(f"Results: {path}; report: {output / 'report.md'}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        # Only application-generated validation errors get details. OSError may
        # contain network paths; avoid echoing exception text and response bodies.
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"Error: {detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
