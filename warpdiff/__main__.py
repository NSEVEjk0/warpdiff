"""Compare two trusted revisions on rented GPUs and write a GitHub check summary."""

import argparse
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .report import SCHEMA_VERSION, charges_for, check_summary, evaluate_card, worst_status, write_json
from .vast import APIError, IMAGE, Vast, api_key, ssh

DEFAULT_GPUS = ("RTX 4090", "A100")
GPU_MODELS = {"A100": ("A100 SXM4", "A100 PCIE")}
REMOTE_DIR = "/workspace/warpdiff"


def log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S UTC")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def revision(revision_name: str, kernel: str) -> tuple[dict, bytes]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision_name}^{{commit}}"], text=True,
    ).strip()
    source = subprocess.check_output(["git", "show", f"{commit}:{kernel}"])
    if len(source) > 250_000:
        raise ValueError("Kernel module exceeds 250 KB")
    compile(source, kernel, "exec")
    return {"sha": commit}, source


def make_archive(baseline: bytes, candidate: bytes) -> bytes:
    target = io.BytesIO()
    files = {"base.py": baseline, "head.py": candidate, "worker.py": Path(__file__).with_name("worker.py").read_bytes()}
    with tarfile.open(fileobj=target, mode="w:gz") as bundle:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o600
            bundle.addfile(entry, io.BytesIO(content))
    return target.getvalue()


def find_offers(vast: Vast, args, gpu: str) -> list[dict]:
    duration = args.startup_seconds + args.run_seconds + 120
    offers = []
    for model in GPU_MODELS.get(gpu, (gpu,)):
        offers.extend(vast.offers(model, args.max_hourly, args.max_transfer_rate, duration))
    return sorted(
        (offer for offer in offers if offer.get("machine_id") not in args.exclude_machine),
        key=lambda offer: float(offer["dph_total"]),
    )


def cleanup(vast: Vast, state: Path, record: dict) -> bool:
    known = set(record.get("instance_ids", []))
    label = record["label"]
    for attempt in range(4):
        lookup_ok = False
        try:
            known.update(int(item["id"]) for item in vast.instances_with_label(label))
            lookup_ok = True
        except APIError as error:
            log(f"{label}: could not reconcile rentals by label ({error})")

        remaining = set()
        for instance_id in known:
            try:
                if vast.instance(instance_id) is not None:
                    vast.destroy(instance_id)
                    log(f"{label}: requested destruction of rental {instance_id}")
                if vast.instance(instance_id) is not None:
                    remaining.add(instance_id)
            except APIError as error:
                remaining.add(instance_id)
                log(f"{label}: cleanup verification failed for {instance_id} ({error})")
        if lookup_ok and not remaining and (attempt or known or not record.get("creating")):
            record.update(instance_ids=sorted(known), cleanup="verified")
            write_json(state, record)
            return True
        if attempt != 3:
            time.sleep(2**attempt * 2)
    record.update(instance_ids=sorted(known), cleanup="pending")
    write_json(state, record)
    return False


def wait_for_worker(vast: Vast, instance_id: int, directory: Path, cancelled, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    public_key = (directory / "worker.pub").read_text().strip()
    key_added = False
    last_status = None
    last_error = None
    while not cancelled.is_set() and time.monotonic() < deadline:
        instance = vast.instance(instance_id)
        status = instance.get("actual_status", "pending") if instance else "pending"
        if status != last_status:
            log(f"Rental {instance_id}: {status}")
            last_status = status
        if status in {"exited", "offline", "error", "unknown"}:
            raise RuntimeError(f"Vast rental entered {status}")
        if instance and status == "running" and (
            ((instance.get("ports") or {}).get("22/tcp") and instance.get("public_ipaddr"))
            or (instance.get("ssh_host") and instance.get("ssh_port"))
        ):
            if not key_added:
                vast.attach(instance_id, public_key)
                key_added = True
            try:
                ssh(instance, directory / "worker", directory / "known_hosts", ["true"], 12, cancelled)
                return instance
            except (TimeoutError, RuntimeError) as error:
                if last_error is None:
                    log(f"Rental {instance_id}: {error}")
                last_error = error
        cancelled.wait(5)
    detail = f" ({last_error})" if last_error else ""
    raise TimeoutError(f"Rental {instance_id} did not become SSH-ready in {timeout} seconds{detail}")


def run_card(vast: Vast, args, gpu: str, offers: list[dict], state: Path, run_id: str, archive: bytes, cancelled) -> dict:
    label = f"warpdiff-{run_id}-{state.stem}"
    record = {"label": label, "instance_ids": [], "cleanup": "pending"}
    card = {"requested_gpu": gpu, "label": label}
    instance_id = None
    try:
        if cancelled.is_set() or not offers:
            raise RuntimeError("No eligible offer within the requested price and host limits")
        record["creating"] = True
        write_json(state, record)
        for offer in offers[:5]:
            if cancelled.is_set():
                raise RuntimeError("Run cancelled")
            try:
                instance_id = vast.create(int(offer["id"]), label)
                card["offer_hourly_usd"] = float(offer["dph_total"])
                break
            except APIError as error:
                if error.status not in {404, 409, 410}:
                    raise
        if instance_id is None:
            raise RuntimeError("All selected offers disappeared before renting")
        record["instance_ids"].append(instance_id)
        write_json(state, record)
        card["instance_id"] = instance_id
        card["machine_id"] = offer.get("machine_id")
        log(f"{gpu}: rented {instance_id} at quoted ${card['offer_hourly_usd']:.3f}/hour")

        with tempfile.TemporaryDirectory(prefix="warpdiff-ssh-") as path:
            directory = Path(path)
            identity, known_hosts = directory / "worker", directory / "known_hosts"
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(identity)], check=True)
            instance = wait_for_worker(vast, instance_id, directory, cancelled, args.startup_seconds)
            card["machine_id"] = instance.get("machine_id") or card["machine_id"]
            ssh(instance, identity, known_hosts, ["mkdir", "-p", REMOTE_DIR], 18, cancelled)
            ssh(instance, identity, known_hosts, ["tar", "-xzf", "-", "-C", REMOTE_DIR], 30, cancelled, archive)
            command = [
                "timeout", str(args.run_seconds), "python", "-u", f"{REMOTE_DIR}/worker.py",
                "--base", f"{REMOTE_DIR}/base.py", "--head", f"{REMOTE_DIR}/head.py", "--pairs", str(args.pairs),
                "--active-seconds", str(args.active_seconds),
            ]
            log(f"{gpu}: running paired correctness and timing checks on machine {card['machine_id']}")
            output = ssh(instance, identity, known_hosts, command, args.run_seconds + 8, cancelled).decode()
            marker = "WARPDIFF_RESULT "
            result = next(line[len(marker):] for line in reversed(output.splitlines()) if line.startswith(marker))
            worker = json.loads(result)
            if gpu.lower() not in worker["gpu_name"].lower():
                raise RuntimeError(f"Vast assigned {worker['gpu_name']} in place of {gpu}")
            for key in ("gpu_name", "compute_capability", "vram_mib", "cuda", "torch", "triton", "cases",
                        "active_gpu_seconds", "utilization_samples", "median_gpu_utilization_pct"):
                card[key] = worker[key]
            evaluate_card(card, args.margin)
            log(f"{gpu}: {card['status']}, {card['active_gpu_seconds']:.1f} active GPU seconds")
    except (APIError, RuntimeError, OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        detail = str(error).strip() or "Worker returned no results"
        log(f"{gpu}: {detail}")
        card.update(status="unavailable", error=detail)
    finally:
        if state.exists():
            card["cleanup"] = "verified" if cleanup(vast, state, record) else "pending"
            if card["cleanup"] == "pending":
                log(f"Cleanup pending. Recover with python -m warpdiff cleanup --state-dir {state.parent}")
        else:
            card["cleanup"] = "not-rented"
    return card


def billing(vast: Vast, report: dict, wait_seconds: int) -> None:
    pending = [card for card in report["cards"] if card.get("instance_id")]
    if not pending:
        return
    start = int(datetime.fromisoformat(report["started_at"]).timestamp()) - 86_400
    deadline = time.monotonic() + wait_seconds
    while pending:
        try:
            rows = vast.charges(start, int(time.time()) + 86_400)
        except APIError as error:
            log(f"Billing lookup: {error}")
            rows = []
        for card in pending:
            card["charges"] = charges_for(rows, card["instance_id"]) or {"status": "pending"}
        pending = [card for card in pending if card["charges"]["status"] != "reported"]
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not pending:
            return
        time.sleep(min(12, remaining))


def compare(args) -> int:
    gpus = args.gpu or DEFAULT_GPUS
    kernel = PurePosixPath(args.kernel)
    if kernel.is_absolute() or ".." in kernel.parts or kernel.suffix != ".py" or ":" in str(kernel):
        raise ValueError("Kernel must be a repository-relative Python file")
    if len(gpus) > 4 or len(gpus) != len(set(gpus)):
        raise ValueError("Choose one to four distinct GPUs")
    base, base_source = revision(args.base, str(kernel))
    head, head_source = revision(args.head, str(kernel))
    if base["sha"] == head["sha"] or base_source == head_source:
        raise ValueError("Expected a kernel change between the two Git revisions")
    bundle = make_archive(base_source, head_source)
    vast = Vast(api_key())
    args.run_seconds = max(180, args.active_seconds + 270)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    state = args.state_dir / run_id
    state.mkdir(parents=True, mode=0o700)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    cancelled = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: cancelled.set())

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        if args.dry_run:
            listings = executor.map(lambda gpu: find_offers(vast, args, gpu), gpus)
            found_all = True
            for gpu, offers in zip(gpus, listings):
                found_all &= bool(offers)
                log(f"{gpu}: {len(offers)} eligible offers")
                for offer in offers[:4]:
                    print(f"{gpu}\t{offer.get('gpu_name')}\t{offer['id']}\t${float(offer['dph_total']):.3f}/h")
            return 0 if found_all else 3
        listings = list(executor.map(lambda gpu: find_offers(vast, args, gpu), gpus))
        compute_estimate = sum(float(offers[0]["dph_total"]) for offers in listings if offers)
        compute_estimate *= (args.startup_seconds + args.run_seconds) / 3600
        log(f"Compute and disk estimate for first-choice offers through timeouts: ${compute_estimate:.2f}; transfer may apply")
        if not all(listings):
            missing = ", ".join(gpu for gpu, offers in zip(gpus, listings) if not offers)
            raise RuntimeError(f"No eligible offers for {missing}. No rentals started.")
        futures = [
            executor.submit(run_card, vast, args, gpu, offers, state / f"card-{position}.json", run_id, bundle, cancelled)
            for position, (gpu, offers) in enumerate(zip(gpus, listings), start=1)
        ]
        cards = [future.result() for future in futures]

    report = {
        "schema_version": SCHEMA_VERSION, "provenance": "vast", "run_id": run_id, "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "base": base, "head": head,
        "image": IMAGE, "kernel": str(kernel), "margin_pct": args.margin, "repository": args.repository,
        "pull_request": args.pull_request, "workflow_url": args.workflow_url, "cards": cards,
        "status": worst_status(card["status"] for card in cards),
        "active_gpu_seconds": round(sum(card.get("active_gpu_seconds", 0) for card in cards), 3),
    }
    if any(card["cleanup"] == "pending" for card in cards):
        report["status"] = "unavailable"
    write_json(args.output, report)
    if not cancelled.is_set():
        billing(vast, report, args.billing_wait)
        write_json(args.output, report)
    print(check_summary(report, args.workflow_url))
    return {"regression": 2, "incorrect": 2, "unavailable": 3, "inconclusive": 3}.get(report["status"], 0)


def refresh_charges(args) -> int:
    report = json.loads(args.report.read_text())
    if report.get("provenance") != "vast":
        raise ValueError("Billing refresh needs a live Vast report")
    billing(Vast(api_key()), report, args.wait)
    write_json(args.report, report)
    return 0


def cleanup_saved(args) -> int:
    vast = Vast(api_key())
    records = sorted(args.state_dir.glob("**/card-*.json"))
    if not records:
        raise ValueError(f"No rental records in {args.state_dir}")
    pending = []
    for path in records:
        record = json.loads(path.read_text())
        if not record.get("label", "").startswith("warpdiff-"):
            raise ValueError(f"Unrecognized rental label in {path}")
        if record.get("cleanup") != "verified" and not cleanup(vast, path, record):
            pending.append(path)
    return 3 if pending else 0


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive value")
    return number


def nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("compare", help="Run a paired benchmark on Vast")
    run.add_argument("--base", required=True)
    run.add_argument("--head", required=True)
    run.add_argument("--kernel", default="kernels/softmax.py")
    run.add_argument("--gpu", action="append", help="Repeat to pick GPUs; default: RTX 4090 and A100")
    run.add_argument("--max-hourly", type=positive, default=1.20, help="Maximum quoted USD/hour per rental")
    run.add_argument("--max-transfer-rate", type=positive, default=0.02, help="Maximum USD/GB per direction")
    run.add_argument("--exclude-machine", type=int, action="append", default=[], help="Skip a previously failed Vast host")
    run.add_argument("--startup-seconds", type=nonnegative_integer, default=480)
    run.add_argument("--active-seconds", type=nonnegative_integer, default=0, help="Minimum active benchmark duration per GPU")
    run.add_argument("--pairs", type=nonnegative_integer, default=35)
    run.add_argument("--margin", type=positive, default=5.0)
    run.add_argument("--output", type=Path, default=Path("artifacts/report.json"))
    run.add_argument("--state-dir", type=Path, default=Path(".warpdiff"))
    run.add_argument("--run-id", help="Unique run identifier used for the Vast rental labels")
    run.add_argument("--repository")
    run.add_argument("--pull-request", type=int)
    run.add_argument("--workflow-url")
    run.add_argument("--billing-wait", type=nonnegative_integer, default=30)
    run.add_argument("--dry-run", action="store_true", help="Search matching offers without renting")
    run.set_defaults(action=compare)
    reconcile = commands.add_parser("billing", help="Refresh provider-reported charges on a saved report")
    reconcile.add_argument("--report", type=Path, required=True)
    reconcile.add_argument("--wait", type=nonnegative_integer, default=0)
    reconcile.set_defaults(action=refresh_charges)
    recover = commands.add_parser("cleanup", help="Retry disposal from durable rental records")
    recover.add_argument("--state-dir", type=Path, default=Path(".warpdiff"))
    recover.set_defaults(action=cleanup_saved)
    args = parser.parse_args()
    if args.command == "compare" and (not 12 <= args.pairs <= 200 or args.startup_seconds < 30):
        parser.error("Use 12 to 200 pairs and at least 30 seconds of startup time")
    if args.command == "compare" and args.run_id and not re.fullmatch(r"[a-z0-9-]{1,32}", args.run_id):
        parser.error("Run identifier must be at most 32 lowercase letters, digits or hyphens")
    try:
        return args.action(args)
    except (APIError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        log(str(error))
        return 3


if __name__ == "__main__":
    sys.exit(main())
