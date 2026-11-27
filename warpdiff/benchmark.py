"""Read a repository's benchmark definition from its trusted base commit."""

import json
import math
import re
import subprocess
import tomllib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath

DEFAULT_GPUS = ("RTX 4090", "A100")
DEFAULT_KERNEL = "kernels/softmax.py"
CONFIG = "warpdiff.toml"
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
GPU_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,47}")


def git_commit(ref: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], text=True,
    ).strip()


def read_blob(commit: str, path: str) -> bytes:
    record = subprocess.check_output(["git", "--literal-pathspecs", "ls-tree", "-z", commit, "--", path])
    if not record:
        raise ValueError(f"{path} is missing from {commit[:12]}")
    metadata, name = record.rstrip(b"\0").split(b"\t", 1)
    mode, kind, blob = metadata.decode().split()
    if kind != "blob" or mode not in {"100644", "100755"} or name.decode() != path:
        raise ValueError(f"{path} must be a regular file in {commit[:12]}")
    size = int(subprocess.check_output(["git", "cat-file", "-s", blob]))
    if size > 250_000:
        raise ValueError(f"{path} exceeds 250 KB")
    return subprocess.check_output(["git", "cat-file", "blob", blob])


def python_path(value, *, module: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a repository-relative Python file")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py" or ":" in value or "\x00" in value:
        raise ValueError(f"Expected a repository-relative Python file, got {value!r}")
    if module and not all(part.isidentifier() for part in path.with_suffix("").parts):
        raise ValueError(f"Python module paths must use identifier names, got {value!r}")
    return path.as_posix()


def gpu_names(values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 4:
        raise ValueError("Choose one to four distinct GPUs")
    if not all(isinstance(value, str) and GPU_NAME.fullmatch(value) and value == value.strip() for value in values):
        raise ValueError("Use the GPU model names shown in Vast")
    models = tuple(value.replace("_", " ") for value in values)
    if len({model.lower() for model in models}) != len(models):
        raise ValueError("Choose one to four distinct GPUs")
    return models


def decimal(value, name: str, *, minimum: float = 0) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number of at least {minimum}")
    return float(value)


def label(value, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 80 or not value.strip() or not value.isprintable():
        raise ValueError(f"{name} must be a short, single-line label")
    return value.strip()


def case_definitions(items) -> list[dict]:
    if not isinstance(items, list) or not 1 <= len(items) <= 16:
        raise ValueError("Configure between 1 and 16 benchmark cases")
    cases = []
    ids = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each case must be a table with id and label")
        key = item.get("id")
        if not isinstance(key, str) or not IDENTIFIER.fullmatch(key) or key in ids:
            raise ValueError("Each case needs a unique lowercase id (letters, digits and hyphens)")
        ids.add(key)
        cases.append({**item, "label": label(item.get("label"), f"Case {key} label")})
    try:
        json.dumps(cases, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Case settings must be finite numbers, strings, booleans, arrays or tables") from error
    return cases


@dataclass(frozen=True)
class Benchmark:
    kernel: str
    files: tuple[str, ...]
    gpus: tuple[str, ...]
    margin_pct: float
    adapter: bytes | None = None
    job: dict | None = None
    metadata: dict | None = None

    def sources(self, commit: str) -> tuple[dict, dict[str, bytes]]:
        files = {}
        for path in self.files:
            source = read_blob(commit, path)
            compile(source, f"{commit[:12]}:{path}", "exec")
            files[path] = source
        revision = {"sha": commit, "kernel_sha256": sha256(files[self.kernel]).hexdigest()}
        if self.job is not None:
            hashes = {path: sha256(source).hexdigest() for path, source in files.items()}
            revision["sources_sha256"] = digest(hashes)
        return revision, files


def digest(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select(base: str, name: str | None, legacy_kernel: str | None) -> Benchmark:
    if name is None:
        kernel = python_path(legacy_kernel or DEFAULT_KERNEL, module=False)
        return Benchmark(kernel, (kernel,), DEFAULT_GPUS, 5.0)

    config = read_blob(base, CONFIG)
    doc = tomllib.loads(config.decode())
    if not name:
        name = doc.get("default")
    profiles = doc.get("benchmarks", {})
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or not isinstance(profiles, dict) or name not in profiles:
        raise ValueError(f"Choose a benchmark from {CONFIG} on the base commit")
    entry = profiles[name]
    if not isinstance(entry, dict):
        raise ValueError(f"Benchmark {name} must be a table")
    title = label(entry.get("name", name), "Benchmark name")
    kernel = python_path(entry.get("kernel"))
    adapter_path = python_path(entry.get("adapter"))
    extras = entry.get("files", [])
    if not isinstance(extras, list) or len(extras) > 12:
        raise ValueError("Benchmark files must be an array with at most 12 additional modules")
    files = tuple(dict.fromkeys((kernel, *(python_path(path) for path in extras))))
    if adapter_path in files:
        raise ValueError("Keep the trusted benchmark adapter separate from the files being compared")
    gpus = gpu_names(entry.get("gpus", DEFAULT_GPUS))
    margin = decimal(entry.get("margin_pct", 5), "margin_pct", minimum=0.01)
    adapter = read_blob(base, adapter_path)
    compile(adapter, f"{base[:12]}:{adapter_path}", "exec")
    job = {
        "kernel": kernel,
        "cases": case_definitions(entry.get("cases")),
        "rtol": decimal(entry.get("rtol", 2e-5), "rtol"),
        "atol": decimal(entry.get("atol", 2e-6), "atol"),
    }
    definition = {**job, "files": files, "adapter_sha256": sha256(adapter).hexdigest()}
    metadata = {
        "id": name, "name": title, "adapter": adapter_path, "gpus": list(gpus),
        "definition_sha256": digest(definition),
    }
    return Benchmark(kernel, files, gpus, margin, adapter=adapter, job=job, metadata=metadata)
