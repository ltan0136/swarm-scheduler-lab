"""Persistent run files shared by the CLI and experiment runner (Linux)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def default_output(command: str) -> Path:
    root = Path("results")
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{command}-", dir=root))


@contextmanager
def locked_directory(directory: Path, *, create: bool):
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another command is writing to this output directory") from None
        yield
    finally:
        os.close(fd)


def write_json(path: Path, value: dict) -> None:
    """Replace atomically; a crash cannot leave a half-written checkpoint in place."""
    stream = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".writing-", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class SearchJournal:
    """A manifest identifies one experiment. Each finished search is one atomic file."""

    def __init__(self, directory: Path, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        self.identity = fingerprint(manifest)

    @classmethod
    def load(cls, directory: Path) -> SearchJournal:
        path = directory / "progress.json"
        if not path.is_file():
            raise ValueError("No progress.json found in the run directory")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or not isinstance(saved.get("manifest"), dict):
            raise ValueError("Invalid run manifest")
        manifest = saved["manifest"]
        journal = cls(directory, manifest)
        if manifest.get("version") != 1 or saved.get("sha256") != journal.identity:
            raise ValueError("Unsupported or damaged run manifest")
        return journal

    @classmethod
    def start(cls, directory: Path, config: dict, metadata: dict, *, resume: bool) -> SearchJournal:
        environment = {k: metadata[k] for k in ("source_tree_sha256", "dataset_sha256",
                                                 "dependencies", "python")}
        manifest = {"version": 1, "config": config, "environment": environment}
        if resume:
            journal = cls.load(directory)
            manifest["run_id"] = journal.manifest["run_id"]
            if journal.identity != fingerprint(manifest):
                raise ValueError("Run configuration, source, dependencies or dataset differ from the checkpoint")
            if not (directory / "checkpoints").is_dir():
                raise ValueError("Missing checkpoint directory")
            return journal
        manifest["run_id"] = uuid4().hex
        journal = cls(directory, manifest)
        (directory / "checkpoints").mkdir()
        write_json(directory / "progress.json", {"manifest": manifest, "sha256": journal.identity})
        return journal

    def _path(self, seed: int, method: str) -> Path:
        return self.directory / "checkpoints" / f"{seed}-{method}.json"

    def record(self, result: dict) -> None:
        payload = {"run": self.identity, "search": result}
        write_json(self._path(result["seed"], result["method"]),
                   {"payload": payload, "sha256": fingerprint(payload)})

    def completed(self, seed: int, method: str) -> dict | None:
        path = self._path(seed, method)
        if not path.exists():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        payload = saved.get("payload") if isinstance(saved, dict) else None
        if not isinstance(payload, dict) or saved.get("sha256") != fingerprint(payload):
            raise ValueError("Damaged search checkpoint")
        result = payload.get("search")
        if (payload.get("run") != self.identity or not isinstance(result, dict)
                or result.get("seed") != seed or result.get("method") != method):
            raise ValueError("Search checkpoint belongs to a different run, seed or method")
        return result
