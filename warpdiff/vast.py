"""Vast API and SSH transport for disposable GPU workers."""

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

ORIGIN = "https://console.vast.ai"
IMAGE = "pytorch/pytorch@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
DISK_GB = 24


class APIError(RuntimeError):
    def __init__(self, method: str, path: str, status: int | None, reason: str = "", code: str | None = None):
        self.status = status
        self.code = code
        super().__init__(f"Vast {method} {path} ({status or 'connection error'})" + (f": {reason}" if reason else ""))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def api_key() -> str:
    key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY")
    if not key:
        config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        fallback = config / "vastai" / "vast_api_key"
        if fallback.is_file():
            key = fallback.read_text().strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError("Set VASTAI_API_KEY or save it to ~/.config/vastai/vast_api_key")
    return key


class Vast:
    def __init__(self, key: str):
        self._opener = build_opener(NoRedirect())
        self._headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    def request(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        url = ORIGIN + path
        if params:
            url += "?" + urlencode(params)
        headers = dict(self._headers)
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = Request(url, data, headers, method=method)
        for attempt in range(4):
            try:
                with self._opener.open(request, timeout=20) as response:
                    result = json.load(response)
                if result.get("success") is False:
                    raise APIError(method, path, 400, api_reason(result), result.get("error"))
                return result
            except HTTPError as error:
                if error.code == 429 and attempt < 3:
                    retry_after = error.headers.get("Retry-After", "")
                    delay = min(20, int(retry_after)) if retry_after.isdigit() else 2**attempt
                    time.sleep(delay)
                    continue
                try:
                    result = json.load(error)
                except (ValueError, AttributeError):
                    result = {}
                if not isinstance(result, dict):
                    result = {}
                raise APIError(method, path, error.code, api_reason(result), result.get("error")) from None
            except (URLError, TimeoutError, ValueError) as error:
                raise APIError(method, path, None, str(getattr(error, "reason", error))) from None

    def offers(self, gpu: str, rate: float, traffic_rate: float, duration: int) -> list[dict]:
        result = self.request("POST", "/api/v0/bundles/", {
            "type": "on-demand",
            "limit": 24,
            "gpu_name": {"eq": gpu.replace("_", " ")},
            "num_gpus": {"eq": 1},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "verified": {"eq": True},
            "reliability": {"gte": 0.97},
            "disk_space": {"gte": DISK_GB},
            "allocated_storage": DISK_GB,
            "cuda_max_good": {"gte": 12.4},
            "duration": {"gte": duration},
            "dph_total": {"lte": rate},
            "inet_up_cost": {"lte": traffic_rate},
            "inet_down_cost": {"lte": traffic_rate},
            "inet_down": {"gte": 150},
            "direct_port_count": {"gte": 1},
            "order": [["dph_total", "asc"]],
        })
        return [offer for offer in result.get("offers", []) if 0 < float(offer["dph_total"]) <= rate]

    def create(self, offer_id: int, label: str) -> int:
        result = self.request("PUT", f"/api/v0/asks/{offer_id}/", {
            "image": IMAGE,
            "label": label,
            "disk": DISK_GB,
            "runtype": "ssh_direc ssh_proxy",  # Vast CLI's direct mode, with proxy fallback.
            "target_state": "running",
            "cancel_unavail": True,
        })
        return int(result["new_contract"])

    def instances_with_label(self, label: str) -> list[dict]:
        params = {"select_filters": json.dumps({"label": {"eq": label}})}
        matches = []
        while True:
            result = self.request("GET", "/api/v1/instances/", params=params)
            matches.extend(item for item in result.get("instances", []) if item.get("label") == label)
            next_token = result.get("next_token")
            if not next_token:
                return matches
            params["after_token"] = next_token

    def instance(self, instance_id: int) -> dict | None:
        try:
            result = self.request("GET", f"/api/v0/instances/{instance_id}/")
        except APIError as error:
            if error.status == 404:
                return None
            raise
        instance = result.get("instances")
        if isinstance(instance, list):
            return next((item for item in instance if item["id"] == instance_id), None)
        return instance or None

    def attach(self, instance_id: int, public_key: str) -> None:
        self.request("POST", f"/api/v0/instances/{instance_id}/ssh/", {"ssh_key": public_key})

    def destroy(self, instance_id: int) -> None:
        try:
            self.request("DELETE", f"/api/v0/instances/{instance_id}/")
        except APIError as error:
            if error.status != 404:
                raise

    def charges(self, start: int, end: int) -> list[dict]:
        params = {"select_filters": json.dumps({"day": {"gte": start, "lte": end}, "type": {"in": ["instance"]}})}
        rows = []
        while True:
            result = self.request("GET", "/api/v0/charges/", params=params)
            rows.extend(result.get("results", []))
            next_token = result.get("next_token")
            if not next_token:
                return rows
            params["after_token"] = next_token


def api_reason(result: dict) -> str:
    reason = result.get("msg") or result.get("detail") or result.get("error") or ""
    return " ".join(str(reason).split())[:180]


def ssh_command(instance: dict, identity: Path, known_hosts: Path, remote_command: str) -> list[str]:
    ports = (instance.get("ports") or {}).get("22/tcp")
    if ports and instance.get("public_ipaddr"):
        host, port = instance["public_ipaddr"], int(ports[0]["HostPort"])
    else:
        host, port = str(instance["ssh_host"]), int(instance["ssh_port"])
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", host) or not (1 <= port <= 65535):
        raise ValueError("Vast returned invalid SSH connection details")
    return [
        "ssh", "-F", "/dev/null", "-i", str(identity), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-p", str(port), f"root@{host}", remote_command,
    ]


def ssh(instance: dict, identity: Path, known_hosts: Path, command: list[str], timeout: float, cancelled,
        archive: bytes | None = None) -> bytes:
    remote_command = shlex.join(command)
    arguments = ssh_command(instance, identity, known_hosts, remote_command)
    deadline = time.monotonic() + timeout
    with subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            while True:
                remaining = deadline - time.monotonic()
                if cancelled.is_set() or remaining <= 0:
                    raise TimeoutError("Remote command cancelled or timed out")
                try:
                    output, errors = process.communicate(input=archive, timeout=min(remaining, 3))
                    break
                except subprocess.TimeoutExpired:
                    archive = None
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    if process.returncode:
        tail = " ".join(errors.decode(errors="replace").splitlines()[-3:])
        raise RuntimeError(f"SSH worker exited {process.returncode}: {tail[-350:]}")
    return output
