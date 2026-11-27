"""Run two kernel revisions against a shared reference on one CUDA device."""

import argparse
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from types import ModuleType

CASES = ((8192, 1024), (4096, 4096), (257, 1000))
OPS_PER_GRAPH = 24


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def revision_module(directory: Path, kernel: str, version: str):
    package = ModuleType(f"warpdiff_{version}")
    package.__path__ = [str(directory / version)]
    sys.modules[package.__name__] = package
    module = ".".join(Path(kernel).with_suffix("").parts)
    return importlib.import_module(f"{package.__name__}.{module}")


def prepare_softmax(torch, case: dict):
    source = torch.randn(case["shape"], device="cuda", dtype=torch.float32) * 2
    expected = torch.softmax(source, dim=-1)

    def bind(variant):
        output = torch.empty_like(expected)

        def run():
            variant.softmax(source, output)

        return run, output

    return expected, bind


def cuda_graph(torch, operation):
    for _ in range(4):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(OPS_PER_GRAPH):
            operation()
    return graph


def correctness(torch, bound, expected, rtol, atol):
    outputs = [output for _, output in bound.values()]
    if any(output.device != expected.device for output in outputs) or not expected.is_cuda:
        raise ValueError("The reference and both outputs must be on the same CUDA device")
    if len({tensor.data_ptr() for tensor in (*outputs, expected)}) != 3:
        raise ValueError("The reference and both outputs must use separate buffers")
    errors = {}
    passed = True
    for name, (operation, result) in bound.items():
        if result.shape != expected.shape or result.dtype != expected.dtype or result.numel() == 0:
            errors[name] = None
            passed = False
            continue
        if result.is_floating_point() or result.is_complex():
            result.fill_(float("nan"))
        else:
            result.zero_()
        operation()
        error = (result.to(torch.complex128 if result.is_complex() else torch.float64) - expected).abs().max().item()
        errors[name] = error if math.isfinite(error) else None
        passed &= bool(torch.allclose(result, expected, rtol=rtol, atol=atol))
    return {"passed": passed, "max_abs_error": errors}


def gpu_utilization() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip().isdigit():
        return int(result.stdout.strip())
    return None


def execute(args):
    import torch
    import triton

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible NVIDIA GPU")
    torch.manual_seed(488)
    versions = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "vram_mib": torch.cuda.get_device_properties(0).total_memory // (1024**2),
        "cuda": str(torch.version.cuda),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
    }
    if args.manifest:
        settings = json.loads(args.manifest.read_text())
        directory = args.manifest.parent
        prepare = load_module(directory / "adapter.py", "warpdiff_adapter").prepare
        variants = {name: revision_module(directory, settings["kernel"], name) for name in ("base", "head")}
    else:
        settings = {"cases": [{"shape": list(shape)} for shape in CASES], "rtol": 2e-5, "atol": 2e-6}
        variants = {name: load_module(getattr(args, name), f"{name}_kernel") for name in ("base", "head")}

        def prepare(case):
            return prepare_softmax(torch, case)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    randomizer = random.Random(612)
    active_seconds = 0.0
    cases = []
    soak_graphs = None

    def sample(graph):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for definition in settings["cases"]:
        expected, bind = prepare(definition)
        bound = {name: bind(variant) for name, variant in variants.items()}
        check = correctness(torch, bound, expected, settings["rtol"], settings["atol"])
        case = {"correctness": check}
        if "id" in definition:
            case.update(case_id=definition["id"], label=definition["label"])
        shape = definition.get("shape")
        if isinstance(shape, list) and len(shape) == 2 and all(type(size) is int and size > 0 for size in shape):
            case.update(rows=shape[0], columns=shape[1])
        cases.append(case)
        if not check["passed"]:
            continue
        graphs = {name: cuda_graph(torch, operation) for name, (operation, _) in bound.items()}
        for graph in graphs.values():
            for _ in range(5):
                graph.replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in variants}
        for _ in range(args.pairs):
            order = list(variants)
            randomizer.shuffle(order)
            for name in order:
                elapsed = sample(graphs[name])
                active_seconds += elapsed / 1000
                samples[name].append(round(elapsed / OPS_PER_GRAPH, 8))
        case["samples_ms"] = samples
        if soak_graphs is None:
            # Keep the callables and their buffers alive for the extended run.
            soak_graphs = (graphs, bound)

    utilization = []
    if args.active_seconds and soak_graphs and all(case["correctness"]["passed"] for case in cases):
        graphs, _bound = soak_graphs
        sample_utilization_at = 0
        while active_seconds < args.active_seconds:
            for graph in graphs.values():
                active_seconds += sample(graph) / 1000
            if active_seconds >= sample_utilization_at:
                reading = gpu_utilization()
                if reading is not None:
                    utilization.append(reading)
                sample_utilization_at += 30

    return {
        **versions,
        "cases": cases,
        "active_gpu_seconds": round(active_seconds, 3),
        "utilization_samples": len(utilization),
        "median_gpu_utilization_pct": statistics.median(utilization) if utilization else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--pairs", required=True, type=int)
    parser.add_argument("--active-seconds", default=0, type=int)
    args = parser.parse_args()
    if (args.manifest and (args.base or args.head)) or (not args.manifest and not (args.base and args.head)):
        parser.error("Pass either --manifest or both --base and --head")
    result = execute(args)
    print("WARPDIFF_RESULT " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
