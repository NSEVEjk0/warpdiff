"""Turn the worker measurements and provider records into a portable report."""

import json
import math
import random
from pathlib import Path
from statistics import median

SCHEMA_VERSION = 1
MINIMUM_PAIRS = 12


def measure(base: list[float], head: list[float], margin: float) -> dict:
    if len(base) != len(head) or len(base) < MINIMUM_PAIRS:
        raise ValueError(f"Expected at least {MINIMUM_PAIRS} matching timing pairs")
    if any(not math.isfinite(value) or value <= 0 for value in (*base, *head)):
        raise ValueError("Timing samples must be finite and positive")
    ratios = [(candidate / baseline - 1) * 100 for baseline, candidate in zip(base, head)]
    randomizer = random.Random(412)
    estimates = sorted(median(randomizer.choices(ratios, k=len(ratios))) for _ in range(2000))
    low, high = estimates[50], estimates[1949]
    if low > margin:
        status = "regression"
    elif high < -margin:
        status = "improvement"
    elif low >= -margin and high <= margin:
        status = "within-threshold"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "base_ms": round(median(base), 6),
        "head_ms": round(median(head), 6),
        "change_pct": round(median(ratios), 2),
        "interval_pct": [round(low, 2), round(high, 2)],
        "paired_samples": len(base),
    }


def evaluate_card(card: dict, margin: float) -> dict:
    if card.get("error"):
        card["status"] = "unavailable"
        return card
    cases = card.get("cases", [])
    if not cases:
        raise ValueError("Worker returned no benchmark cases")
    for case in cases:
        if not case.get("correctness", {}).get("passed"):
            case["status"] = "incorrect"
            continue
        samples = case["samples_ms"]
        case.update(measure(samples["base"], samples["head"], margin))
        if card.get("offer_hourly_usd") is not None:
            case["steady_state_usd_per_million"] = round(
                case["head_ms"] * card["offer_hourly_usd"] / 3.6, 6
            )
    card["status"] = worst_status(case["status"] for case in cases)
    return card


def worst_status(statuses) -> str:
    seen = set(statuses)
    for status in ("incorrect", "regression", "unavailable", "inconclusive", "improvement"):
        if status in seen:
            return status
    return "within-threshold"


def report_status(cards: list[dict]) -> str:
    if any(card["cleanup"] == "pending" for card in cards):
        return "unavailable"
    return worst_status(card["status"] for card in cards)


def exit_code(status: str) -> int:
    return {"regression": 2, "incorrect": 2, "unavailable": 3, "inconclusive": 3}.get(status, 0)


def combine_reports(earlier: dict, current: dict) -> dict:
    for report in (earlier, current):
        if report.get("schema_version") != SCHEMA_VERSION or report.get("provenance") != "vast":
            raise ValueError("Combining reports requires two live Vast benchmarks with the current schema")
    if earlier["run_id"] == current["run_id"] or earlier.get("sources") or current.get("sources"):
        raise ValueError("Choose two distinct original runs")
    for key in ("head", "kernel", "margin_pct", "image", "repository", "pull_request"):
        if earlier.get(key) != current.get(key):
            raise ValueError(f"Both runs must benchmark the same {key}")
    for report in (earlier, current):
        if any(card.get("cleanup") == "pending" for card in report["cards"]):
            raise ValueError("Resolve any pending rental cleanup before combining reports")

    available = {}
    for report in (earlier, current):
        for card in report["cards"]:
            if card.get("cases") and card.get("status") != "unavailable" and card.get("cleanup") == "verified":
                available[card["requested_gpu"]] = {**card, "source_run_id": report["run_id"]}
    if not {"RTX 4090", "A100"} <= available.keys():
        raise ValueError("The combined report needs measured RTX 4090 and A100 results")
    cards = [available["RTX 4090"], available["A100"]]
    return {
        **current,
        "started_at": min(earlier["started_at"], current["started_at"]),
        "cards": cards,
        "status": report_status(cards),
        "active_gpu_seconds": round(sum(card.get("active_gpu_seconds", 0) for card in cards), 3),
        "sources": [
            {key: report[key] for key in ("run_id", "base", "workflow_url")}
            for report in (earlier, current)
        ],
    }


def charges_for(rows: list[dict], instance_id: int) -> dict | None:
    matched = [row for row in rows if row.get("source") == f"instance-{instance_id}"]
    if not matched:
        return None
    amounts = {"gpu": 0.0, "disk": 0.0, "bwd": 0.0, "bwu": 0.0}
    for row in matched:
        for item in row.get("items", []):
            category = item.get("type")
            if category in amounts:
                amounts[category] += float(item.get("amount", 0))
    return {
        "status": "reported",
        "total_usd": round(sum(float(row["amount"]) for row in matched), 5),
        "compute_usd": round(amounts["gpu"], 5),
        "storage_usd": round(amounts["disk"], 5),
        "transfer_usd": round(amounts["bwd"] + amounts["bwu"], 5),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_summary(report: dict, details_url: str | None = None) -> str:
    baseline = f"Original `{report['base']['sha'][:12]}`"
    if report.get("sources"):
        commits = dict.fromkeys(source["base"]["sha"] for source in report["sources"])
        baseline = "Original " + " / ".join(f"`{commit[:12]}`" for commit in commits)
    lines = [
        "### GPU performance on Vast.ai",
        "",
        f"{baseline} · patched `{report['head']['sha'][:12]}` · "
        f"slowdown cutoff **{report['margin_pct']}%**",
        "",
        "| GPU | Input | Original | Patched | Change | Result |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    format_status = {
        "incorrect": "Correctness failed",
        "unavailable": "Unavailable",
        "regression": "Regression",
        "improvement": "Improvement",
        "inconclusive": "Inconclusive",
        "within-threshold": "Within threshold",
    }
    for card in report["cards"]:
        gpu = card.get("gpu_name") or card["requested_gpu"]
        if card.get("error"):
            lines.append(f"| {gpu} | — | — | — | — | Unavailable |")
            continue
        for case in card.get("cases", []):
            input_label = f"{case['rows']:,} × {case['columns']:,}"
            status = format_status[case["status"]]
            if case["status"] == "incorrect":
                lines.append(f"| {gpu} | {input_label} | — | — | — | {status} |")
            else:
                lines.append(
                    f"| {gpu} | {input_label} | {case['base_ms']:.3f} ms | {case['head_ms']:.3f} ms | "
                    f"{case['change_pct']:+.2f}% | {status} |"
                )
    lines.extend([
        "", "Change is the median percentage across all timing pairs. A positive number means the patched kernel ran slower.",
        "", f"**{report['active_gpu_seconds'] / 60:.2f} minutes** of active GPU time.", "",
    ])
    for card in report["cards"]:
        gpu = card.get("gpu_name") or card["requested_gpu"]
        charges = card.get("charges", {})
        if card.get("instance_id"):
            receipt = f"Vast reported ${charges['total_usd']:.5f} in charges" if charges.get("status") == "reported" else "Vast charges pending"
            cleanup = "Rental deleted." if card["cleanup"] == "verified" else "Rental deletion unconfirmed."
            lines.append(f"- {gpu} · {receipt}. {cleanup}")
    if details_url:
        lines.extend(["", f"[View the workflow run and download timings]({details_url})"])
    for source in report.get("sources", []):
        if source.get("workflow_url") and source["workflow_url"] != details_url:
            lines.append(f"[Earlier benchmark]({source['workflow_url']})")
    return "\n".join(lines)
