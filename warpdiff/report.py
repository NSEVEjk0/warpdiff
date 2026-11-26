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
    lines = [
        "### GPU performance on Vast.ai",
        "",
        f"Baseline `{report['base']['sha'][:12]}` · candidate `{report['head']['sha'][:12]}` · "
        f"regression threshold **{report['margin_pct']}%**",
        "",
        "| GPU | Input | Baseline | Candidate | Change | Result |",
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
    lines.extend(["", "Change is the median paired latency difference. Positive percentages indicate a slowdown."])
    lines.append(f"CUDA-timed work: **{report['active_gpu_seconds'] / 60:.2f} GPU-minutes**.")
    for card in report["cards"]:
        gpu = card.get("gpu_name") or card["requested_gpu"]
        charges = card.get("charges", {})
        receipt = f"${charges['total_usd']:.5f} billed" if charges.get("status") == "reported" else "billing pending"
        if card.get("instance_id"):
            lines.append(f"- {gpu}: {receipt}, cleanup {card['cleanup']}.")
    if details_url:
        lines.extend(["", f"[Full run and JSON evidence]({details_url})"])
    return "\n".join(lines)
