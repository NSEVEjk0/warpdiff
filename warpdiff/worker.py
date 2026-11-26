"""Run trusted Triton variants against the same reference and CUDA device."""

import argparse
import importlib.util
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

CASES = ((8192, 1024), (4096, 4096), (257, 1000))
OPS_PER_GRAPH = 24


def load_variant(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.softmax


def cuda_graph(torch, operation, source, output):
    for _ in range(4):
        operation(source, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(OPS_PER_GRAPH):
            operation(source, output)
    return graph


def correctness(torch, variants, source, outputs):
    expected = torch.softmax(source, dim=-1)
    errors = {}
    passed = True
    for name, operation in variants.items():
        result = outputs[name]
        result.fill_(float("nan"))
        operation(source, result)
        error = (result - expected).abs().max().item()
        errors[name] = error if math.isfinite(error) else None
        passed &= bool(torch.allclose(result, expected, rtol=2e-5, atol=2e-6))
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
    variants = {"base": load_variant(args.base, "base_kernel"), "head": load_variant(args.head, "head_kernel")}
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

    for rows, columns in CASES:
        source = torch.randn((rows, columns), device="cuda", dtype=torch.float32) * 2
        outputs = {name: torch.empty_like(source) for name in variants}
        check = correctness(torch, variants, source, outputs)
        case = {"rows": rows, "columns": columns, "correctness": check}
        cases.append(case)
        if not check["passed"]:
            continue
        graphs = {name: cuda_graph(torch, operation, source, outputs[name]) for name, operation in variants.items()}
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
            # Retain the input and output storage for this graph after the loop advances.
            soak_graphs = (graphs, source, outputs)

    utilization = []
    if args.active_seconds and soak_graphs and all(case["correctness"]["passed"] for case in cases):
        graphs, _, _outputs = soak_graphs
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
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=int)
    parser.add_argument("--active-seconds", default=0, type=int)
    result = execute(parser.parse_args())
    print("WARPDIFF_RESULT " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
